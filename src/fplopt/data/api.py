"""Client for the official FPL API with a timestamped on-disk cache.

Every response is stored as ``{"fetched_at": <UTC ISO>, "url": ..., "data": ...}``
so it is always clear how fresh a number is. A request is served from a fresh
cache entry, otherwise from the network, and falls back to a stale cache entry
(with a warning) only if the network fails.

``bootstrap-static`` is also archived (gzipped, one file per fetch) under the
snapshot directory. FPL's own projection ``ep_next`` only exists in that live
payload, so these snapshots are the only clean record of it for later backtests.
"""
from __future__ import annotations

import gzip
import json
import logging
import re
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://fantasy.premierleague.com/api"
HEADERS = {"User-Agent": "fpl-optimizer (personal research project)"}
DEFAULT_TTL = {"bootstrap": 3600, "fixtures": 3600, "element_summary": 43200, "entry": 300, "league": 300}
RETRY_STATUS = {429, 500, 502, 503, 504}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class FPLClient:
    def __init__(
        self,
        cache_dir: str | Path,
        snapshot_dir: str | Path | None = None,
        base_url: str = DEFAULT_BASE_URL,
        ttl: dict[str, int] | None = None,
        timeout: float = 20,
        retries: int = 3,
        backoff: float = 1.0,
        workers: int = 4,
        session: requests.Session | None = None,
        clock: Callable[[], datetime] = utc_now,
    ):
        self.cache_dir = Path(cache_dir)
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
        self.base_url = base_url.rstrip("/")
        self.ttl = {**DEFAULT_TTL, **(ttl or {})}
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.workers = workers
        self.session = session or requests.Session()
        self.clock = clock

    # --- cache -----------------------------------------------------------
    def _cache_path(self, path: str) -> Path:
        return self.cache_dir / (re.sub(r"[^A-Za-z0-9.-]+", "_", path.strip("/")) + ".json")

    def cache_entry(self, path: str) -> dict | None:
        """The cached envelope for ``path`` (``fetched_at``, ``url``, ``data``), if any."""
        file = self._cache_path(path)
        if not file.exists():
            return None
        try:
            return json.loads(file.read_text())
        except (OSError, ValueError):
            log.warning("Ignoring unreadable cache file %s", file)
            return None

    def _age_seconds(self, entry: dict) -> float:
        return (self.clock() - datetime.fromisoformat(entry["fetched_at"])).total_seconds()

    def _write_cache(self, path: str, envelope: dict) -> None:
        file = self._cache_path(path)
        file.parent.mkdir(parents=True, exist_ok=True)
        tmp = file.with_suffix(".tmp")
        tmp.write_text(json.dumps(envelope))
        tmp.replace(file)

    # --- network ---------------------------------------------------------
    def _fetch(self, url: str) -> Any:
        for attempt in range(1, self.retries + 1):
            try:
                resp = self.session.get(url, headers=HEADERS, timeout=self.timeout)
                if resp.status_code in RETRY_STATUS and attempt < self.retries:
                    raise requests.HTTPError(f"HTTP {resp.status_code}")
                resp.raise_for_status()
                return resp.json()
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if attempt == self.retries or (status is not None and status not in RETRY_STATUS):
                    raise
                time.sleep(self.backoff * 2 ** (attempt - 1))
        raise AssertionError("unreachable")

    def get(self, path: str, ttl: int) -> Any:
        data, _ = self._get(path, ttl)
        return data

    def _get(self, path: str, ttl: int) -> tuple[Any, bool]:
        """Return ``(data, fetched_from_network)``."""
        entry = self.cache_entry(path)
        if entry is not None and self._age_seconds(entry) < ttl:
            return entry["data"], False
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            data = self._fetch(url)
        except (requests.RequestException, ValueError) as exc:
            if entry is not None:
                log.warning("Request for %s failed (%s); using cache from %s", path, exc, entry["fetched_at"])
                return entry["data"], False
            raise
        self._write_cache(path, {"fetched_at": self.clock().isoformat(), "url": url, "data": data})
        return data, True

    def _snapshot(self, name: str, data: Any) -> None:
        if self.snapshot_dir is None:
            return
        folder = self.snapshot_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        stamp = self.clock().strftime("%Y-%m-%dT%H-%M-%SZ")
        with gzip.open(folder / f"{stamp}.json.gz", "wt") as fh:
            json.dump(data, fh)

    # --- endpoints -------------------------------------------------------
    def bootstrap(self) -> dict:
        """Players, teams, gameweeks and game settings."""
        data, fresh = self._get("bootstrap-static/", self.ttl["bootstrap"])
        if fresh:
            self._snapshot("bootstrap-static", data)
        return data

    def fixtures(self) -> list[dict]:
        return self.get("fixtures/", self.ttl["fixtures"])

    def element_summary(self, player_id: int) -> dict:
        """A player's per-fixture history this season plus upcoming fixtures."""
        return self.get(f"element-summary/{player_id}/", self.ttl["element_summary"])

    def element_summaries(self, player_ids: Iterable[int]) -> dict[int, dict]:
        """Fetch many element summaries with a small thread pool (cached individually)."""
        ids = list(player_ids)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return dict(zip(ids, pool.map(self.element_summary, ids)))

    def entry(self, team_id: int) -> dict:
        return self.get(f"entry/{team_id}/", self.ttl["entry"])

    def entry_history(self, team_id: int) -> dict:
        return self.get(f"entry/{team_id}/history/", self.ttl["entry"])

    def transfers(self, team_id: int) -> list[dict]:
        """Every transfer the manager has made this season (with the prices paid/received)."""
        return self.get(f"entry/{team_id}/transfers/", self.ttl["entry"])

    def picks(self, team_id: int, gameweek: int) -> dict:
        return self.get(f"entry/{team_id}/event/{gameweek}/picks/", self.ttl["entry"])

    def league_standings(self, league_id: int, page: int = 1) -> dict:
        return self.get(f"leagues-classic/{league_id}/standings/?page_standings={page}", self.ttl["league"])
