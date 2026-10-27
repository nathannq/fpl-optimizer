"""Long tables stacked across seasons: the inputs to feature engineering.

* ``team_schedule``   - two rows per fixture (one per side), finished or not
* ``player_matches``  - player-fixture rows with stable ids (``code``, ``team_code``)
* ``gameweek_rows``   - player-gameweek rows (double gameweeks summed), plus
                        optional future rows for gameweeks not yet played
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fplopt.data.schema import MARKET_COLUMNS, MATCH_STATS, SeasonData

SQUAD_SIZE = 15


def season_index(seasons: list[SeasonData]) -> dict[str, int]:
    """Chronological order of seasons ('2022-23' < '2023-24' sorts correctly as text)."""
    return {name: i for i, name in enumerate(sorted({s.season for s in seasons}))}


def team_schedule(data: SeasonData) -> pd.DataFrame:
    fx, teams = data.fixtures, data.teams.set_index("team_id")
    sides = []
    for home, team, opp, gf, ga in [(True, "team_h", "team_a", "team_h_score", "team_a_score"),
                                    (False, "team_a", "team_h", "team_a_score", "team_h_score")]:
        sides.append(pd.DataFrame({
            "season": data.season,
            "fixture_id": fx["fixture_id"],
            "gw": fx["gw"],
            "kickoff_time": fx["kickoff_time"],
            "finished": fx["finished"],
            "team_code": fx[team].map(teams["team_code"]),
            "opp_code": fx[opp].map(teams["team_code"]),
            "home": home,
            "goals_for": fx[gf],
            "goals_against": fx[ga],
        }))
    return pd.concat(sides, ignore_index=True).sort_values(["gw", "kickoff_time", "fixture_id", "home"],
                                                           ignore_index=True)


def player_matches(data: SeasonData) -> pd.DataFrame:
    """Matches with the player's stable code/position and the side they played for."""
    m = data.matches.merge(data.players[["element", "code", "web_name", "position"]], on="element", how="inner")
    fx = data.fixtures.set_index("fixture_id")
    team_id = np.where(m["was_home"], m["fixture_id"].map(fx["team_h"]), m["fixture_id"].map(fx["team_a"]))
    m["team_code"] = pd.Series(team_id, index=m.index).map(data.teams.set_index("team_id")["team_code"])
    m.insert(0, "season", data.season)
    return m


def gameweek_rows(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per player per gameweek: stats summed over the gameweek's fixtures.

    Market columns (price, ownership, transfers) are per gameweek in the source,
    so the first value is taken. ``min_count=1`` keeps a stat NaN when the source
    has no data for it, rather than turning it into 0.
    """
    keys = ["season", "code", "gw"]
    g = matches.sort_values("kickoff_time").groupby(keys, sort=False)
    out = g[MATCH_STATS].sum(min_count=1)
    first = g[[*MARKET_COLUMNS, "xp_fpl", "element"]].first()
    last = g[["web_name", "position", "team_code"]].last()
    out = pd.concat([first, last, out], axis=1)
    out["n_matches"] = g.size()
    out["is_future"] = False
    return out.reset_index()


def upcoming_rows(data: SeasonData, gws: list[int]) -> pd.DataFrame:
    """Rows for gameweeks not yet played: every current player whose team has a fixture.

    Needs ``players.team_id`` (only the live season has it). Stats stay NaN.
    """
    team_code = data.players["team_id"].map(data.teams.set_index("team_id")["team_code"])
    players = data.players.assign(team_code=team_code)[["element", "code", "web_name", "position", "team_code"]]
    sched = team_schedule(data)
    playing = sched[sched["gw"].isin(gws)][["gw", "team_code"]].drop_duplicates()
    rows = players.merge(playing, on="team_code", how="inner")
    rows.insert(0, "season", data.season)
    rows["n_matches"] = 0
    rows["is_future"] = True
    return rows


def managers_by_gameweek(gw_rows: pd.DataFrame, schedule: pd.DataFrame) -> pd.Series:
    """Estimated number of FPL managers per (season, gw).

    Every squad holds exactly 15 players, so total ownership / 15 = managers.
    That only holds in gameweeks where every team plays. In a blank gameweek the
    players without a fixture are missing, so both the partial count and the last
    full gameweek's count are lower bounds; the larger is used. A later gameweek
    is never used, as that would leak.
    """
    played = gw_rows[~gw_rows["is_future"]]
    total = played.groupby(["season", "gw"])["selected"].sum() / SQUAD_SIZE
    teams_per_season = schedule.groupby("season")["team_code"].nunique()
    teams_per_gw = schedule.groupby(["season", "gw"])["team_code"].nunique()
    full = teams_per_gw.reindex(total.index).to_numpy() == teams_per_season.reindex(
        total.index.get_level_values("season")).to_numpy()
    last_full = total.where(full).groupby(level="season").ffill()
    return np.fmax(total, last_full).rename("managers")
