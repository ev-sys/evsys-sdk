"""HTTP client for hosted platform APIs (sandboxes + vendor integrations).

Talks only to our backend — never to E2B or LangSmith directly when the user
has connected credentials on the platform.
"""

from __future__ import annotations

import os
from typing import Any

import requests

from .constants import (
    CONTENT_TYPE_JSON,
    DEFAULT_API_URL,
    DEFAULT_TIMEOUT_S,
    HEADER_AUTHORIZATION,
    HEADER_CONTENT_TYPE,
    EVSYS_API_KEY_ENV,
    EVSYS_API_URL_ENV,
    EVSYS_PROJECT_ID_ENV,
    bearer,
)


class PlatformClientError(RuntimeError):
    pass


class PlatformClient:
    """Authenticated client for ``/api/sandboxes/...`` and ``/api/integrations/...``."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        project_id: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get(EVSYS_API_URL_ENV) or DEFAULT_API_URL).rstrip("/")
        self.api_key = api_key or os.environ.get(EVSYS_API_KEY_ENV)
        self.project_id = project_id or os.environ.get(EVSYS_PROJECT_ID_ENV)
        self.timeout_s = timeout_s
        if session is not None:
            self._session = session
        else:
            self._session = requests.Session()
            if self.api_key:
                self._session.headers.update({
                    HEADER_AUTHORIZATION: bearer(self.api_key),
                    HEADER_CONTENT_TYPE: CONTENT_TYPE_JSON,
                })

    def _require_auth(self) -> None:
        if not self.api_key:
            raise PlatformClientError(f"missing {EVSYS_API_KEY_ENV}")

    def _raise(self, r: requests.Response, label: str) -> None:
        raise PlatformClientError(f"{label} HTTP {r.status_code}: {r.text[:300]}")

    # -- sandboxes ---------------------------------------------------------

    def create_sandbox(
        self,
        *,
        name: str,
        template_id: str = "base",
        auto_start: bool = True,
        timeout_sec: int = 3600,
    ) -> dict[str, Any]:
        self._require_auth()
        body: dict[str, Any] = {
            "name": name,
            "template_id": template_id,
            "auto_start": auto_start,
            "timeout_sec": timeout_sec,
            "project_id": self.project_id,
        }
        r = self._session.post(f"{self.base_url}/api/sandboxes/", json=body, timeout=self.timeout_s)
        if r.status_code >= 400:
            self._raise(r, "create sandbox")
        return r.json().get("sandbox") or r.json()

    def exec_sandbox(
        self,
        sandbox_id: str,
        command: str,
        *,
        cwd: str | None = None,
        timeout_sec: float | None = None,
    ) -> dict[str, Any]:
        self._require_auth()
        body: dict[str, Any] = {"command": command}
        if cwd:
            body["cwd"] = cwd
        if timeout_sec is not None:
            body["timeout_sec"] = timeout_sec
        timeout = self.timeout_s
        if timeout_sec is not None:
            timeout = max(timeout, float(timeout_sec) + 5)
        r = self._session.post(
            f"{self.base_url}/api/sandboxes/{sandbox_id}/exec/",
            json=body,
            timeout=timeout,
        )
        if r.status_code >= 400:
            self._raise(r, "exec sandbox")
        return r.json().get("result") or r.json()

    def stop_sandbox(self, sandbox_id: str, *, kill: bool = True) -> dict[str, Any]:
        self._require_auth()
        r = self._session.post(
            f"{self.base_url}/api/sandboxes/{sandbox_id}/stop/",
            json={"kill": kill},
            timeout=self.timeout_s,
        )
        if r.status_code >= 400:
            self._raise(r, "stop sandbox")
        return r.json().get("sandbox") or r.json()

    # -- integrations ------------------------------------------------------

    def langsmith_status(self) -> dict[str, Any]:
        self._require_auth()
        if not self.project_id:
            raise PlatformClientError(f"missing {EVSYS_PROJECT_ID_ENV}")
        r = self._session.get(
            f"{self.base_url}/api/integrations/langsmith/",
            params={"project_id": self.project_id},
            timeout=self.timeout_s,
        )
        if r.status_code >= 400:
            self._raise(r, "langsmith status")
        return r.json()

    def langsmith_pull(
        self,
        *,
        project_name: str | None = None,
        since: str | None = None,
        limit: int = 100,
        filter_expr: str | None = None,
    ) -> dict[str, Any]:
        self._require_auth()
        if not self.project_id:
            raise PlatformClientError(f"missing {EVSYS_PROJECT_ID_ENV}")
        body: dict[str, Any] = {"project_id": self.project_id, "limit": limit}
        if project_name:
            body["project_name"] = project_name
        if since:
            body["since"] = since
        if filter_expr:
            body["filter"] = filter_expr
        r = self._session.post(
            f"{self.base_url}/api/integrations/langsmith/pull/",
            json=body,
            timeout=max(self.timeout_s, 60.0),
        )
        if r.status_code >= 400:
            self._raise(r, "langsmith pull")
        return r.json()
