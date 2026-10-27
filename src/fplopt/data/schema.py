"""The common shape every data source is normalised into.

Historical seasons (vaastav CSVs) and the live season (official API) both become
a :class:`SeasonData`, so the feature pipeline never needs to know where a row
came from.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# FPL element_type -> position. Type 5 is the 2024-25 "assistant manager" chip,
# which is not a player and is dropped.
POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# Per-fixture player stats. Sources that lack a column get NaN (e.g. defensive
# contributions only exist from 2025-26, when they started scoring points).
MATCH_STATS = [
    "total_points", "minutes", "starts", "goals_scored", "assists", "clean_sheets",
    "goals_conceded", "own_goals", "penalties_saved", "penalties_missed", "yellow_cards",
    "red_cards", "saves", "bonus", "bps", "expected_goals", "expected_assists",
    "expected_goal_involvements", "expected_goals_conceded", "defensive_contribution",
]
# Market state recorded for the gameweek of each fixture (price in tenths of £m).
MARKET_COLUMNS = ["value", "selected", "transfers_in", "transfers_out", "transfers_balance"]
MATCH_COLUMNS = [
    "element", "fixture_id", "gw", "kickoff_time", "was_home", "opponent_team",
    *MATCH_STATS, *MARKET_COLUMNS, "xp_fpl",
]
# Stats that were not always recorded. FPL only added xG/xA and starts partway
# through 2022-23 (they read 0 for everyone before that), so a gameweek where one
# of these is zero for every player means "not recorded", not "zero".
SOMETIMES_UNRECORDED = [
    "starts", "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "defensive_contribution",
]
TEAM_COLUMNS = ["team_id", "team_code", "team_name", "team_short"]
PLAYER_COLUMNS = ["element", "code", "web_name", "position"]
FIXTURE_COLUMNS = [
    "fixture_id", "gw", "kickoff_time", "team_h", "team_a", "team_h_score", "team_a_score", "finished",
]


@dataclass
class SeasonData:
    """One season in the common schema.

    teams:    team_id (season-specific), team_code (stable across seasons), names
    players:  element (season-specific id), code (stable across seasons), web_name, position;
              live seasons also carry team_id and availability fields
    fixtures: one row per fixture with its gameweek, kickoff, sides and score
    matches:  one row per player per fixture played so far (stats + market + FPL xP)
    """

    season: str
    teams: pd.DataFrame
    players: pd.DataFrame
    fixtures: pd.DataFrame
    matches: pd.DataFrame


def normalise_teams(raw: pd.DataFrame) -> pd.DataFrame:
    teams = raw.rename(columns={"id": "team_id", "code": "team_code", "name": "team_name", "short_name": "team_short"})
    return teams[TEAM_COLUMNS].astype({"team_id": "int64", "team_code": "int64"}).reset_index(drop=True)


def normalise_players(raw: pd.DataFrame, extra: list[str] | None = None) -> pd.DataFrame:
    """Map element_type to a position and drop non-players (assistant managers)."""
    players = raw.rename(columns={"id": "element"})
    players = players[players["element_type"].isin(POSITIONS)].copy()
    players["position"] = players["element_type"].map(POSITIONS)
    cols = PLAYER_COLUMNS + [c for c in (extra or []) if c in players.columns]
    return players[cols].astype({"element": "int64", "code": "int64"}).reset_index(drop=True)


def normalise_fixtures(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep scheduled fixtures only (an unscheduled postponement has no gameweek)."""
    fx = raw.rename(columns={"id": "fixture_id", "event": "gw"})
    fx = fx[fx["gw"].notna()].copy()
    fx["gw"] = fx["gw"].astype("int64")
    fx["kickoff_time"] = pd.to_datetime(fx["kickoff_time"], utc=True)
    fx["finished"] = fx["finished"].astype(bool)
    for col in ("team_h_score", "team_a_score"):
        fx[col] = pd.to_numeric(fx[col], errors="coerce").astype("float64")
    return fx[FIXTURE_COLUMNS].sort_values(["gw", "kickoff_time", "fixture_id"]).reset_index(drop=True)


def normalise_matches(raw: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Rename, fill missing stat columns with NaN, drop duplicates and non-players."""
    m = raw.rename(columns={"fixture": "fixture_id", "round": "gw", "xP": "xp_fpl"}).copy()
    for col in MATCH_COLUMNS:
        if col not in m.columns:
            m[col] = np.nan
    m = m.drop_duplicates(["element", "fixture_id"])
    m = m[m["element"].isin(players["element"])]
    m["kickoff_time"] = pd.to_datetime(m["kickoff_time"], utc=True)
    m["was_home"] = m["was_home"].astype(bool)
    float_cols = [*MATCH_STATS, *MARKET_COLUMNS, "xp_fpl"]
    m[float_cols] = m[float_cols].apply(pd.to_numeric, errors="coerce").astype("float64")
    m = m.astype({"element": "int64", "fixture_id": "int64", "gw": "int64", "opponent_team": "int64"})
    m = mask_unrecorded_stats(m)
    return m[MATCH_COLUMNS].sort_values(["gw", "kickoff_time", "fixture_id", "element"]).reset_index(drop=True)


def mask_unrecorded_stats(matches: pd.DataFrame) -> pd.DataFrame:
    """Set a stat to NaN in gameweeks where it is zero for every player (see SOMETIMES_UNRECORDED)."""
    out = matches.copy()
    for col in SOMETIMES_UNRECORDED:
        totals = out[col].fillna(0).abs().groupby(out["gw"]).sum()
        out.loc[out["gw"].isin(totals.index[totals == 0]), col] = np.nan
    return out


def mask_invalid_xp(matches: pd.DataFrame, min_nonzero_share: float) -> pd.DataFrame:
    """Set FPL xP to NaN for gameweeks where the source recorded (almost) only zeros."""
    nonzero_share = matches["xp_fpl"].fillna(0).ne(0).groupby(matches["gw"]).mean()
    bad = nonzero_share.index[nonzero_share < min_nonzero_share]
    out = matches.copy()
    out.loc[out["gw"].isin(bad), "xp_fpl"] = np.nan
    return out
