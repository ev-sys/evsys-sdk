"""Centralized tinker ServiceClient creation: TINKER_BASE_URL resolution +
re-export so training, inference, and harbor rollouts all target one backend.

Uses a fake ``tinker`` module (injected via sys.modules) so these run without
the real tinker install — make_service_client imports tinker lazily."""

from __future__ import annotations

import os
import sys
import types

import pytest

from evsys_sdk import tinker_service as ts


@pytest.fixture(autouse=True)
def _restore_base_url():
    """make_service_client mutates os.environ[TINKER_BASE_URL] by design (so
    harbor inherits it); ensure that never leaks across tests."""
    saved = os.environ.get(ts.TINKER_BASE_URL_ENV)
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop(ts.TINKER_BASE_URL_ENV, None)
        else:
            os.environ[ts.TINKER_BASE_URL_ENV] = saved


def test_resolve_base_url_precedence(monkeypatch):
    monkeypatch.delenv(ts.TINKER_BASE_URL_ENV, raising=False)
    assert ts.resolve_base_url() is None                              # neither → None
    monkeypatch.setenv(ts.TINKER_BASE_URL_ENV, "http://env-url")
    assert ts.resolve_base_url() == "http://env-url"                  # env when no arg
    assert ts.resolve_base_url("http://arg-url") == "http://arg-url"  # explicit arg wins


def _fake_tinker(monkeypatch) -> dict:
    captured: dict = {}

    class FakeServiceClient:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

    fake = types.ModuleType("tinker")
    fake.ServiceClient = FakeServiceClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tinker", fake)
    return captured


def test_make_service_client_arg_exports_env_and_passes_base_url(monkeypatch):
    monkeypatch.delenv(ts.TINKER_BASE_URL_ENV, raising=False)
    captured = _fake_tinker(monkeypatch)
    ts.make_service_client("http://my-backend")
    assert captured["kwargs"]["base_url"] == "http://my-backend"      # passed to client
    assert os.environ[ts.TINKER_BASE_URL_ENV] == "http://my-backend"  # re-exported for harbor


def test_make_service_client_reads_env_when_no_arg(monkeypatch):
    monkeypatch.setenv(ts.TINKER_BASE_URL_ENV, "http://env-backend")
    captured = _fake_tinker(monkeypatch)
    ts.make_service_client()
    assert captured["kwargs"]["base_url"] == "http://env-backend"     # the env-only OOTB path


def test_make_service_client_bare_when_unset(monkeypatch):
    monkeypatch.delenv(ts.TINKER_BASE_URL_ENV, raising=False)
    captured = _fake_tinker(monkeypatch)
    ts.make_service_client()
    assert "base_url" not in captured["kwargs"]                       # tinker uses its own default
    assert ts.TINKER_BASE_URL_ENV not in os.environ                   # env left untouched
