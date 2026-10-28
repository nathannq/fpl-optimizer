"""A small, fully controlled league for tests.

Two seasons of 8 gameweeks with 4 teams and 5 players each. It contains the
awkward cases the pipeline must handle:

* player ids change between seasons; ``code`` stays the same
* team 4 is relegated after season 1 and team 5 is promoted
* in each season one fixture is moved from GW6 into GW5 (a double and a blank)
* player 199 signs for team 1 in season 2 and first appears in GW4
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from fplopt.data.schema import MARKET_COLUMNS, MATCH_STATS, SeasonData, normalise_matches
from fplopt.optimize.rules import SQUAD_QUOTAS

SEASONS = {"2022-23": [1, 2, 3, 4], "2023-24": [1, 2, 3, 5]}
POSITIONS = ["GK", "DEF", "DEF", "MID", "FWD"]
N_GWS = 8
ROUNDS = [[(0, 1), (2, 3)], [(0, 2), (1, 3)], [(0, 3), (1, 2)]]
MOVED_FROM_GW, MOVED_TO_GW = 6, 5
LATE_SIGNING = {"code": 199, "team_code": 1, "season": "2023-24", "from_gw": 4}


def _fixtures(season_idx: int, rng: np.random.Generator) -> pd.DataFrame:
    rows, fid = [], 1
    for gw in range(1, N_GWS + 1):
        for k, (a, b) in enumerate(ROUNDS[(gw - 1) % 3]):
            home, away = (a, b) if (gw + season_idx) % 2 else (b, a)
            moved = gw == MOVED_FROM_GW and k == 0
            rows.append({
                "fixture_id": fid, "gw": MOVED_TO_GW if moved else gw,
                "kickoff_time": pd.Timestamp("2022-08-01", tz="UTC") + pd.Timedelta(days=365 * season_idx + 7 * gw
                                                                                     + (3 if moved else 0) + k),
                "team_h": home + 1, "team_a": away + 1,
                "team_h_score": float(rng.integers(0, 4)), "team_a_score": float(rng.integers(0, 4)),
                "finished": True,
            })
            fid += 1
    return pd.DataFrame(rows).sort_values(["gw", "kickoff_time"], ignore_index=True)


def _players(season: str, season_idx: int, team_codes: list[int]) -> pd.DataFrame:
    rows = []
    for team_id, team_code in enumerate(team_codes, start=1):
        for j, pos in enumerate(POSITIONS):
            rows.append({"code": team_code * 100 + j, "web_name": f"P{team_code}{j}", "position": pos,
                         "team_id": team_id})
    if season == LATE_SIGNING["season"]:
        rows.append({"code": LATE_SIGNING["code"], "web_name": "Signing", "position": "MID",
                     "team_id": team_codes.index(LATE_SIGNING["team_code"]) + 1})
    players = pd.DataFrame(rows)
    players.insert(0, "element", np.arange(len(players)) + 1 + 100 * season_idx)  # ids differ by season
    return players


def _matches(season: str, players: pd.DataFrame, fixtures: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    base_value = {code: 40 + 5 * (code % 7) for code in players["code"]}
    for fx in fixtures.itertuples():
        for team_id, was_home, opp in [(fx.team_h, True, fx.team_a), (fx.team_a, False, fx.team_h)]:
            for p in players[players["team_id"] == team_id].itertuples():
                if p.code == LATE_SIGNING["code"] and fx.gw < LATE_SIGNING["from_gw"]:
                    continue
                minutes = float(rng.choice([0, 25, 90], p=[0.3, 0.2, 0.5]))
                played = minutes > 0
                rows.append({
                    "element": p.element, "fixture": fx.fixture_id, "round": fx.gw,
                    "kickoff_time": fx.kickoff_time, "was_home": was_home, "opponent_team": opp,
                    "total_points": float(rng.integers(1, 13)) if played else 0.0,
                    "minutes": minutes, "starts": float(minutes >= 60),
                    "goals_scored": float(rng.integers(0, 2)) if played else 0.0,
                    "assists": float(rng.integers(0, 2)) if played else 0.0,
                    "clean_sheets": float(rng.integers(0, 2)) if minutes >= 60 else 0.0,
                    "goals_conceded": float(rng.integers(0, 3)) if played else 0.0,
                    "bonus": float(rng.integers(0, 4)) if played else 0.0,
                    "bps": float(rng.integers(0, 40)) if played else 0.0,
                    "saves": 0.0, "own_goals": 0.0, "penalties_saved": 0.0, "penalties_missed": 0.0,
                    "yellow_cards": 0.0, "red_cards": 0.0,
                    "expected_goals": float(rng.uniform(0, 0.8)) if played else 0.0,
                    "expected_assists": float(rng.uniform(0, 0.5)) if played else 0.0,
                    "expected_goals_conceded": float(rng.uniform(0, 2)) if played else 0.0,
                    "defensive_contribution": float(rng.integers(0, 12)) if played else 0.0,
                    "value": float(base_value[p.code] + fx.gw),
                    "selected": float(rng.integers(1_000, 500_000)),
                    "transfers_in": float(rng.integers(0, 5_000)), "transfers_out": float(rng.integers(0, 5_000)),
                    "transfers_balance": float(rng.integers(-5_000, 5_000)),
                    "xP": float(rng.uniform(0, 6)),
                })
    raw = pd.DataFrame(rows)
    raw["expected_goal_involvements"] = raw["expected_goals"] + raw["expected_assists"]
    # market data is recorded once per gameweek, so it is identical across a double gameweek
    per_gw = raw.groupby(["element", "round"])[[*MARKET_COLUMNS, "xP"]].transform("first")
    raw[[*MARKET_COLUMNS, "xP"]] = per_gw
    return raw


def make_league(seed: int = 0, live_from_gw: int | None = None) -> list[SeasonData]:
    """Both seasons. With ``live_from_gw``, season 2 is only played up to the gameweek before it."""
    rng = np.random.default_rng(seed)
    out = []
    for idx, (season, codes) in enumerate(SEASONS.items()):
        teams = pd.DataFrame({"team_id": range(1, 5), "team_code": codes,
                              "team_name": [f"Team {c}" for c in codes], "team_short": [f"T{c}" for c in codes]})
        fixtures = _fixtures(idx, rng)
        players = _players(season, idx, codes)
        matches = normalise_matches(_matches(season, players, fixtures, rng), players)
        out.append(SeasonData(season, teams, players, fixtures, matches))
    if live_from_gw is not None:
        last = out[-1]
        future = last.fixtures["gw"] >= live_from_gw
        last.fixtures.loc[future, ["team_h_score", "team_a_score"]] = np.nan
        last.fixtures.loc[future, "finished"] = False
        last.matches = last.matches[last.matches["gw"] < live_from_gw].reset_index(drop=True)
    return out


def perturb_from(seasons: list[SeasonData], season: str, gw: int, seed: int = 99) -> list[SeasonData]:
    """Copy of ``seasons`` with every result and market value from (season, gw) onwards replaced at random.

    The fixture schedule (who plays whom, when) is left as it is: it is published in advance.
    """
    rng = np.random.default_rng(seed)
    out = copy.deepcopy(seasons)
    for s in out:
        if s.season < season:
            continue
        m_mask = s.matches["gw"] >= gw if s.season == season else pd.Series(True, index=s.matches.index)
        f_mask = s.fixtures["gw"] >= gw if s.season == season else pd.Series(True, index=s.fixtures.index)
        cols = [*MATCH_STATS, *MARKET_COLUMNS, "xp_fpl"]
        s.matches.loc[m_mask, cols] = rng.uniform(0, 100, size=(int(m_mask.sum()), len(cols))).round()
        played = s.fixtures["finished"] & f_mask
        s.fixtures.loc[played, ["team_h_score", "team_a_score"]] = rng.integers(0, 9, size=(int(played.sum()), 2))
    return out


def make_pool(seed: int = 0, clubs: int = 20) -> pd.DataFrame:
    """A full-size league: every club has a complete 2/5/5/3 squad with random prices and projections."""
    rng = np.random.default_rng(seed)
    rows, pid = [], 1
    for club in range(1, clubs + 1):
        for pos in SQUAD_QUOTAS:
            for _ in range(SQUAD_QUOTAS[pos]):
                price = int(rng.integers(40, 130))
                rows.append({"id": pid, "name": f"P{pid}", "position": pos, "club": club, "team": f"C{club}",
                             "price": price, "projection": round(float(rng.uniform(0.5, 1.5) * price / 15), 2),
                             "ownership": float(rng.uniform(0, 40)), "status": "a", "news": ""})
                pid += 1
    pool = pd.DataFrame(rows).set_index("id")
    pool["objective"] = pool["projection"]
    pool["can_buy"] = True
    pool["can_start"] = True
    return pool
