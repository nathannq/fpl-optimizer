"""The FPL API client's timestamped cache, fallbacks and snapshots (no network)."""
from __future__ import annotations

import gzip
import json
from datetime import datetime, timedelta, timezone

import pytest
import requests

from fplopt.data.api import FPLClient


class Clock:
    def __init__(self):
        self.now = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


class FakeSession:
    """Returns queued responses; an Exception in the queue is raised instead."""

    def __init__(self, *responses):
        self.queue = list(responses)
        self.calls = 0

    def get(self, url, headers=None, timeout=None):
        self.calls += 1
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def client(tmp_path, session, clock):
    return FPLClient(cache_dir=tmp_path / "cache", snapshot_dir=tmp_path / "snap", session=session,
                     clock=clock, backoff=0)


def test_fresh_cache_is_served_without_a_request(tmp_path):
    clock, session = Clock(), FakeSession(FakeResponse({"v": 1}))
    c = client(tmp_path, session, clock)
    assert c.fixtures() == {"v": 1}
    clock.now += timedelta(seconds=60)
    assert c.fixtures() == {"v": 1}
    assert session.calls == 1
    entry = c.cache_entry("fixtures/")
    assert entry["fetched_at"] == "2026-10-27T12:00:00+00:00"
    assert entry["url"].endswith("/fixtures/")


def test_expired_cache_is_refetched(tmp_path):
    clock, session = Clock(), FakeSession(FakeResponse({"v": 1}), FakeResponse({"v": 2}))
    c = client(tmp_path, session, clock)
    c.fixtures()
    clock.now += timedelta(seconds=c.ttl["fixtures"] + 1)
    assert c.fixtures() == {"v": 2}
    assert c.cache_entry("fixtures/")["fetched_at"].startswith("2026-10-27T13:00")


def test_network_failure_falls_back_to_stale_cache(tmp_path):
    clock = Clock()
    session = FakeSession(FakeResponse({"v": 1}), *[requests.ConnectionError("down")] * 3)
    c = client(tmp_path, session, clock)
    c.fixtures()
    clock.now += timedelta(days=1)
    assert c.fixtures() == {"v": 1}


def test_network_failure_without_cache_raises(tmp_path):
    c = client(tmp_path, FakeSession(*[requests.ConnectionError("down")] * 3), Clock())
    with pytest.raises(requests.ConnectionError):
        c.fixtures()


def test_server_errors_are_retried_but_not_found_is_not(tmp_path):
    session = FakeSession(FakeResponse({}, 503), FakeResponse({"ok": True}))
    assert client(tmp_path, session, Clock()).fixtures() == {"ok": True}
    assert session.calls == 2
    session = FakeSession(FakeResponse({}, 404))
    with pytest.raises(requests.HTTPError):
        client(tmp_path, session, Clock()).element_summary(1)
    assert session.calls == 1


def test_bootstrap_is_snapshotted_once_per_network_fetch(tmp_path):
    clock, session = Clock(), FakeSession(FakeResponse({"events": []}))
    c = client(tmp_path, session, clock)
    c.bootstrap()
    c.bootstrap()                                   # served from cache: no second snapshot
    snaps = list((tmp_path / "snap" / "bootstrap-static").glob("*.json.gz"))
    assert [s.name for s in snaps] == ["2026-10-27T12-00-00Z.json.gz"]
    with gzip.open(snaps[0], "rt") as fh:
        assert json.load(fh) == {"events": []}


def test_cache_file_names_are_safe(tmp_path):
    c = client(tmp_path, FakeSession(FakeResponse({"standings": {}})), Clock())
    c.league_standings(123, page=2)
    assert [p.name for p in (tmp_path / "cache").iterdir()] == ["leagues-classic_123_standings_page_standings_2.json"]
