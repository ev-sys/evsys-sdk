"""Local read-only web server for the results UI — stdlib only, no new deps.

Serves a small vendored static SPA plus a JSON API backed by a store's read
methods (a ``LocalStore`` over the ``.evsys`` mirror). Read-only (GET), bound to
127.0.0.1 only (it exposes raw experiment data). Started by ``evsys ui``.
"""

from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any, Callable
from urllib.parse import parse_qs, urlsplit

from . import api

# Ordered (regex, handler) table. Handlers take (store, match, query) → object.
_ROUTES: list[tuple[re.Pattern, Callable]] = []


def _route(pattern: str):
    def deco(fn: Callable) -> Callable:
        _ROUTES.append((re.compile(f"^{pattern}$"), fn))
        return fn
    return deco


@_route(r"/api/experiments")
def _experiments(store, m, q):
    return api.experiments(store)


@_route(r"/api/experiments/(?P<exp_id>[^/]+)/runs")
def _experiment_runs(store, m, q):
    return api.experiment_runs(store, m.group("exp_id"))


@_route(r"/api/runs/(?P<run_id>[^/]+)")
def _run(store, m, q):
    return api.run_detail(store, m.group("run_id"))


@_route(r"/api/runs/(?P<run_id>[^/]+)/metrics")
def _metrics(store, m, q):
    return api.metrics(store, m.group("run_id"))


@_route(r"/api/runs/(?P<run_id>[^/]+)/evals")
def _evals(store, m, q):
    return api.evals(store, m.group("run_id"))


@_route(r"/api/runs/(?P<run_id>[^/]+)/predictions")
def _predictions(store, m, q):
    return api.predictions(
        store, m.group("run_id"),
        limit=int(q.get("limit", ["200"])[0]),
        offset=int(q.get("offset", ["0"])[0]),
        kind=(q.get("kind", [None])[0]),
    )


_MIME = {
    ".html": "text/html", ".js": "application/javascript", ".css": "text/css",
    ".json": "application/json", ".svg": "image/svg+xml",
}


def _static_bytes(name: str) -> bytes | None:
    """Read a vendored static asset by basename (no path traversal)."""
    if "/" in name or "\\" in name or name.startswith("."):
        return None
    try:
        res = resources.files("evsys_sdk.ui") / "static" / name
        if not res.is_file():
            return None
        return res.read_bytes()
    except (FileNotFoundError, ModuleNotFoundError):
        return None


def _make_handler(store: Any) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A002 — quiet
            return

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: Any) -> None:
            self._send(code, json.dumps(obj, default=str).encode(), "application/json")

        def do_GET(self) -> None:  # noqa: N802
            parts = urlsplit(self.path)
            path, query = parts.path, parse_qs(parts.query)

            if path.startswith("/api/"):
                for pattern, fn in _ROUTES:
                    mt = pattern.match(path)
                    if mt:
                        result = fn(store, mt, query)
                        if result is None:
                            return self._json(404, {"error": "not found"})
                        return self._json(200, result)
                return self._json(404, {"error": "unknown endpoint"})

            # Static: "/" → index.html; SPA fallback for unknown non-API paths.
            name = "index.html" if path in ("/", "") else path.lstrip("/")
            data = _static_bytes(name) or _static_bytes("index.html")
            if data is None:
                return self._json(404, {"error": "ui assets not found"})
            ext = "." + name.rsplit(".", 1)[-1] if "." in name else ".html"
            self._send(200, data, _MIME.get(ext, "application/octet-stream"))

    return Handler


def run_server(store: Any, *, host: str = "127.0.0.1", port: int = 6006) -> None:
    """Serve the UI for ``store`` until interrupted. Blocks."""
    httpd = ThreadingHTTPServer((host, port), _make_handler(store))
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()


__all__ = ["run_server"]
