"""Glue between configuration, data sources and the feature table."""
from __future__ import annotations

import logging

import pandas as pd

from fplopt.config import Config
from fplopt.data.api import FPLClient
from fplopt.data.live import LiveState, load_live_season
from fplopt.data.schema import SeasonData
from fplopt.data.vaastav import load_season
from fplopt.features.build import build_feature_table

log = logging.getLogger(__name__)


def make_client(cfg: Config) -> FPLClient:
    return FPLClient(
        cache_dir=cfg.paths.api_cache,
        snapshot_dir=cfg.paths.snapshots,
        base_url=cfg.live.base_url,
        ttl=cfg.live.ttl_seconds,
        workers=cfg.live.request_workers,
    )


def load_seasons(cfg: Config, live: bool = True,
                 client: FPLClient | None = None) -> tuple[list[SeasonData], LiveState | None]:
    """Historical seasons from disk, plus the current season from the API when ``live``."""
    seasons = [load_season(s, cfg.paths.raw_history, cfg.history.xp_valid_min_nonzero_share)
               for s in cfg.history.seasons]
    state = None
    if live:
        season, state = load_live_season(client or make_client(cfg), cfg.live.season)
        seasons.append(season)
    return seasons, state


def build_features(cfg: Config, live: bool = True, client: FPLClient | None = None) -> pd.DataFrame:
    """Build and save the feature table, including rows for the next gameweek when live."""
    seasons, state = load_seasons(cfg, live=live, client=client)
    upcoming = {state.season: [state.next_gw]} if state and state.next_gw else None
    table = build_feature_table(seasons, cfg, upcoming=upcoming)
    cfg.paths.processed.mkdir(parents=True, exist_ok=True)
    table.to_parquet(cfg.paths.features, index=False)
    log.info("Wrote %s rows to %s", f"{len(table):,}", cfg.paths.features)
    return table
