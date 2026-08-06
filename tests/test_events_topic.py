"""Tests for evsys_sdk.compute.events_topic."""
import json

from evsys_sdk.compute.events_topic import TopicEvents, post_event


def _msg(t, body):
    return json.dumps({"event": "message", "time": t,
                       "message": json.dumps(body)})


def test_post_event_swallows_failures():
    calls = []
    def ok(url, data=None):
        calls.append((url, data)); return b""
    assert post_event("http://t/x", {"kind": "done", "job_id": "j"}, ok)
    assert json.loads(calls[0][1])["kind"] == "done"
    def boom(url, data=None):
        raise IOError("down")
    assert post_event("http://t/x", {"kind": "done"}, boom) is False


def test_poll_parses_and_advances_cursor():
    pages = [
        "\n".join([_msg(10, {"kind": "checkpoint", "job_id": "j", "step": 1}),
                   json.dumps({"event": "open"}),
                   "not json at all",
                   _msg(11, {"kind": "done", "job_id": "j"})]),
        "",
    ]
    seen_urls = []
    def fake(url, data=None):
        seen_urls.append(url)
        return pages.pop(0).encode()
    ev = TopicEvents("http://t/topic", transport=fake)
    first = ev()
    assert [e["kind"] for e in first] == ["checkpoint", "done"]
    assert ev() == []
    assert "since=all" in seen_urls[0]
    assert "since=11" in seen_urls[1]          # cursor advanced


def test_poll_failure_returns_empty():
    def boom(url, data=None):
        raise IOError("down")
    assert TopicEvents("http://t/x", transport=boom)() == []


def test_non_dict_and_chatter_skipped():
    page = "\n".join([_msg(5, {"kind": "checkpoint", "job_id": "a", "step": 2}),
                      json.dumps({"event": "message", "time": 6,
                                  "message": "human chatter"}),
                      json.dumps({"event": "message", "time": 7,
                                  "message": json.dumps([1, 2])})])
    ev = TopicEvents("http://t/x", transport=lambda u, data=None: page.encode())
    out = ev()
    assert len(out) == 1 and out[0]["step"] == 2
