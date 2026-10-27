"""The current season, built from the official FPL API in the common schema."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from fplopt.data.api import FPLClient
from fplopt.data.schema import (
    SeasonData,
    normalise_fixtures,
    normalise_matches,
    normalise_players,
    normalise_teams,
)

log = logging.getLogger(__name__)

# Live-only player fields used for serving (availability, current price/ownership).
# They have no history, so they are never model features.
LIVE_PLAYER_FIELDS = [
    "team_id", "now_cost", "selected_by_percent", "status",
    "chance_of_playing_next_round", "ep_next", "news",
]


@dataclass
class LiveState:
    season: str
    next_gw: int | None
    last_finished_gw: int | None
    deadline: pd.Timestamp | None
    total_players: int
    bootstrap_fetched_at: str | None


def _event_id(events: list[dict], flag: str) -> int | None:
    return next((e["id"] for e in events if e.get(flag)), None)


def load_live_season(client: FPLClient, season: str) -> tuple[SeasonData, LiveState]:
    """Fetch bootstrap, fixtures and every player's history (all cached)."""
    bootstrap = client.bootstrap()
    events = bootstrap["events"]
    finished = [e["id"] for e in events if e.get("finished")]
    next_gw = _event_id(events, "is_next")
    deadline = next((e["deadline_time"] for e in events if e["id"] == next_gw), None)
    state = LiveState(
        season=season,
        next_gw=next_gw,
        last_finished_gw=max(finished) if finished else None,
        deadline=pd.Timestamp(deadline) if deadline else None,
        total_players=int(bootstrap.get("total_players") or 0),
        bootstrap_fetched_at=(client.cache_entry("bootstrap-static/") or {}).get("fetched_at"),
    )

    raw_players = pd.DataFrame(bootstrap["elements"]).rename(columns={"team": "team_id"})
    for col in ("now_cost", "selected_by_percent", "ep_next", "chance_of_playing_next_round"):
        raw_players[col] = pd.to_numeric(raw_players[col], errors="coerce")
    players = normalise_players(raw_players, extra=LIVE_PLAYER_FIELDS)
    teams = normalise_teams(pd.DataFrame(bootstrap["teams"]))
    fixtures = normalise_fixtures(pd.DataFrame(client.fixtures()))

    log.info("Fetching %d player histories (cached for %ss)", len(players), client.ttl["element_summary"])
    summaries = client.element_summaries(players["element"].tolist())
    history = pd.DataFrame([row for s in summaries.values() for row in s.get("history", [])])
    if "fixture" in history.columns:   # skip fixtures that are not finished (half-played gameweeks)
        history = history[history["fixture"].isin(fixtures.loc[fixtures["finished"], "fixture_id"])]
    matches = normalise_matches(history, players)
    return SeasonData(season=season, teams=teams, players=players, fixtures=fixtures, matches=matches), state
