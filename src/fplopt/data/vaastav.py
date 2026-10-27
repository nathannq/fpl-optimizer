"""Historical seasons from the community repo vaastav/Fantasy-Premier-League.

Only the four files the pipeline needs are downloaded per season, from a pinned
commit, and every file's SHA-256 is recorded in ``MANIFEST.json`` so a training
run can always be traced back to the exact data it used.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from fplopt.config import HistoryConfig
from fplopt.data.schema import (
    SeasonData,
    mask_invalid_xp,
    normalise_fixtures,
    normalise_matches,
    normalise_players,
    normalise_teams,
)

log = logging.getLogger(__name__)

RAW_URL = "https://raw.githubusercontent.com/{repo}/{sha}/data/{season}/{file}"
COMMIT_URL = "https://api.github.com/repos/{repo}/commits/{ref}"
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def resolve_ref(repo: str, ref: str, session: requests.Session) -> str:
    """Turn a branch or tag into a commit SHA (a full SHA is returned unchanged)."""
    if SHA_RE.match(ref):
        return ref
    resp = session.get(COMMIT_URL.format(repo=repo, ref=ref), timeout=30)
    resp.raise_for_status()
    return resp.json()["sha"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_history(cfg: HistoryConfig, dest: Path, force: bool = False,
                     session: requests.Session | None = None) -> dict:
    """Download the configured seasons into ``dest/<season>/...`` and write a manifest.

    Files already present from the same commit (and unchanged on disk) are skipped.
    """
    session = session or requests.Session()
    sha = resolve_ref(cfg.repo, cfg.ref, session)
    manifest_path = dest / "MANIFEST.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    same_commit = previous.get("sha") == sha
    files: dict[str, str] = {}

    for season in cfg.seasons:
        for name in cfg.files:
            rel = f"{season}/{name}"
            target = dest / rel
            if (not force and same_commit and target.exists()
                    and previous.get("files", {}).get(rel) == _sha256(target)):
                files[rel] = previous["files"][rel]
                continue
            url = RAW_URL.format(repo=cfg.repo, sha=sha, season=season, file=name)
            log.info("Downloading %s", url)
            resp = session.get(url, timeout=60)
            resp.raise_for_status()
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".tmp")
            tmp.write_bytes(resp.content)
            tmp.replace(target)
            files[rel] = _sha256(target)

    manifest = {
        "repo": cfg.repo,
        "ref": cfg.ref,
        "sha": sha,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_season(season: str, raw_dir: Path, xp_valid_min_nonzero_share: float) -> SeasonData:
    """Read one downloaded season into the common schema.

    Only identity columns are taken from ``players_raw.csv``: it is an
    end-of-season snapshot, so its form/ownership/price fields would leak.
    """
    folder = raw_dir / season
    if not folder.exists():
        raise FileNotFoundError(f"{folder} not found - run `fplopt download-history` first")
    players = normalise_players(pd.read_csv(folder / "players_raw.csv"))
    teams = normalise_teams(pd.read_csv(folder / "teams.csv"))
    fixtures = normalise_fixtures(pd.read_csv(folder / "fixtures.csv"))
    matches = normalise_matches(pd.read_csv(folder / "gws" / "merged_gw.csv"), players)
    matches = mask_invalid_xp(matches, xp_valid_min_nonzero_share)
    return SeasonData(season=season, teams=teams, players=players, fixtures=fixtures, matches=matches)
