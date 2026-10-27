"""The feature table: one row per player per gameweek.

The no-leakage rule
-------------------
Every feature on a gameweek-N row is computed only from:

* match results from gameweeks before N (player form, team strength),
* market data (price, ownership, transfers) recorded for gameweeks before N,
* the published fixture schedule (who plays whom, home/away, how many times).

Results and market data from gameweek N or later never reach a gameweek-N
feature. This is enforced by ``tests/test_leakage.py``, which rewrites all of
that data at random and checks the features don't move.

The pattern used throughout is: compute a "state after gameweek G" (which may
include G), carry it forward over gameweeks with no matches, then shift by one
gameweek so a row for N only ever sees the state after N-1.

Targets (``y_*``) and FPL's own ``xp_fpl`` are stored alongside the features for
training and evaluation, but are not features. ``xp_fpl`` in particular is
recorded after the gameweek in the source data (see ``fplopt.baselines``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fplopt.config import Config
from fplopt.data.schema import SeasonData
from fplopt.features.tables import (
    gameweek_rows,
    managers_by_gameweek,
    player_matches,
    season_index,
    team_schedule,
    upcoming_rows,
)

# feature prefix -> per-match column, averaged over each rolling window
ROLLING_STATS = {
    "pts": "total_points", "mins": "minutes", "starts": "starts", "goals": "goals_scored",
    "assists": "assists", "xg": "expected_goals", "xa": "expected_assists",
    "xgi": "expected_goal_involvements", "bonus": "bonus", "bps": "bps",
    "cs": "clean_sheets", "defcon": "defensive_contribution",
}
# per-90 rates over the longest window: how dangerous a player is when he plays
PER90_STATS = {"xg": "expected_goals", "xa": "expected_assists", "pts": "total_points"}
AVAILABILITY_FEATURES = ["mins_last", "mins_trend", "matches_since_start", "played_last"]
SEASON_FEATURES = ["pts_season_avg", "matches_season"]
MARKET_FEATURES = ["price", "price_change_1gw", "price_change_season",
                   "ownership_pct", "ownership_change_1gw", "net_transfers_pct"]
TEAM_STATS = ["xg_for", "xg_against", "goals_for", "goals_against", "points", "clean_sheet"]
OPP_STATS = ["xg_for", "xg_against", "goals_for", "goals_against", "points"]
CATEGORICAL_FEATURES = ["position"]

ID_COLUMNS = ["season", "gw", "code", "element", "web_name", "position", "team_code", "team_short",
              "kickoff_time", "is_future"]
REFERENCE_COLUMNS = ["xp_fpl"]   # FPL's projection as recorded by the source: look-ahead, never a feature


def horizon_prefix(cfg: Config) -> str:
    return f"fix{cfg.features.fixture_horizon}"


def target_columns(cfg: Config) -> list[str]:
    return ["y_points", "y_haul", f"y_points_next{cfg.targets.multi_gw_horizon}"]


def feature_columns(cfg: Config) -> list[str]:
    """All model inputs, in a stable order (numeric first, then categorical)."""
    windows = cfg.features.rolling_windows
    hp = horizon_prefix(cfg)
    return [
        *[f"{p}_r{w}" for w in windows for p in ROLLING_STATS],
        *[f"{p}_p90_r{max(windows)}" for p in PER90_STATS],
        *AVAILABILITY_FEATURES,
        *SEASON_FEATURES,
        *MARKET_FEATURES,
        *[f"team_{s}" for s in TEAM_STATS],
        "n_fixtures", "home_frac",
        *[f"opp_{s}" for s in OPP_STATS],
        f"{hp}_n", f"{hp}_opp_xg_against", f"{hp}_opp_xg_for",
        *CATEGORICAL_FEATURES,
    ]


# --- helpers ----------------------------------------------------------------

def _carry_forward(df: pd.DataFrame, cols: list[str], has_data: pd.Series, by: str) -> pd.DataFrame:
    """For rows without data, copy every column from the latest earlier row (same ``by``) that had data.

    ``df`` must be sorted by ``by`` then time, with a RangeIndex. Rows with data keep their values exactly.
    """
    n = len(df)
    pos = pd.Series(np.where(has_data.to_numpy(), np.arange(n), np.nan), index=df.index)
    pos = pos.groupby(df[by]).ffill().fillna(-1).astype(int).to_numpy()
    values = df[cols].to_numpy(dtype="float64")
    out = np.full_like(values, np.nan)
    ok = pos >= 0
    out[ok] = values[pos[ok]]
    return pd.DataFrame(out, columns=cols, index=df.index)


def _rolling(g, cols: list[str], window: int, how: str, min_periods: int = 1) -> pd.DataFrame:
    roll = g[cols].rolling(window, min_periods=min_periods)
    return getattr(roll, how)().reset_index(level=0, drop=True).sort_index()


# --- player form ------------------------------------------------------------

def player_state(matches: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Player form after each gameweek (including it), keyed by (code, season, gw).

    Windows run over the player's team fixtures - including ones he didn't play,
    which is what makes them capture rotation and injuries - and continue across
    season boundaries (last season's form is legitimately known at GW1).
    """
    windows = cfg.features.rolling_windows
    wmax = max(windows)
    m = matches.sort_values(["code", "season_idx", "gw", "kickoff_time", "fixture_id"], ignore_index=True)
    g = m.groupby("code", sort=False)
    cols: dict[str, pd.Series] = {}

    stats = list(ROLLING_STATS.values())
    for w in windows:
        mean = _rolling(g, stats, w, "mean")
        for prefix, col in ROLLING_STATS.items():
            cols[f"{prefix}_r{w}"] = mean[col]

    sums = _rolling(g, ["minutes", *PER90_STATS.values()], wmax, "sum")
    enough = sums["minutes"] >= cfg.features.per90_min_minutes
    for prefix, col in PER90_STATS.items():
        cols[f"{prefix}_p90_r{wmax}"] = (sums[col] / sums["minutes"] * 90).where(enough)

    cols["mins_last"] = m["minutes"]
    last3 = _rolling(g, ["minutes"], 3, "sum", min_periods=3)["minutes"]
    last6 = _rolling(g, ["minutes"], 6, "sum", min_periods=6)["minutes"]
    cols["mins_trend"] = last3 / 3 - (last6 - last3) / 3
    # where starts were not recorded, 60+ minutes stands in for a start
    started = pd.Series(np.where(m["starts"].notna(), m["starts"].gt(0), m["minutes"].ge(60)), index=m.index)
    run = started.groupby(m["code"]).cumsum()
    since = m.groupby([m["code"], run]).cumcount() + (run == 0)
    cols["matches_since_start"] = since.where(~started, 0).clip(upper=10).astype("float64")
    cols["played_last"] = m["minutes"].gt(0).astype("float64")

    gs = m.groupby(["code", "season"], sort=False)
    cols["pts_season_avg"] = gs["total_points"].expanding().mean().reset_index(level=[0, 1], drop=True).sort_index()
    cols["matches_season"] = (gs.cumcount() + 1).astype("float64")

    state = pd.concat([m[["code", "season", "gw"]], pd.DataFrame(cols)], axis=1)
    return state.drop_duplicates(["code", "season", "gw"], keep="last")


# --- team strength ----------------------------------------------------------

def team_matches(schedule: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """One row per team per finished fixture with goals, xG for/against and result."""
    xg = (matches.groupby(["season", "fixture_id", "team_code"])["expected_goals"]
          .sum(min_count=1).rename("xg_for").reset_index())
    tm = schedule[schedule["finished"]].merge(xg, on=["season", "fixture_id", "team_code"], how="left")
    opp = xg.rename(columns={"team_code": "opp_code", "xg_for": "xg_against"})
    tm = tm.merge(opp, on=["season", "fixture_id", "opp_code"], how="left")
    gf, ga = tm["goals_for"], tm["goals_against"]
    tm["points"] = np.select([gf > ga, gf == ga], [3.0, 1.0], 0.0)
    tm["clean_sheet"] = ga.eq(0).astype("float64")
    return tm


def team_state_before(schedule: pd.DataFrame, matches: pd.DataFrame, seasons: list[SeasonData],
                      order: dict[str, int], cfg: Config) -> pd.DataFrame:
    """Team strength going into each gameweek, keyed by (season, gw, team_code).

    Rolling means over the team's last ``team_window`` matches (across seasons).
    Teams with no earlier matches in the data start from the promoted-team prior.
    """
    tm = team_matches(schedule, matches)
    tm["season_idx"] = tm["season"].map(order)
    tm = tm.sort_values(["team_code", "season_idx", "gw", "kickoff_time"], ignore_index=True)
    rolled = _rolling(tm.groupby("team_code", sort=False), TEAM_STATS, cfg.features.team_window, "mean")
    after = pd.concat([tm[["season", "gw", "team_code"]], rolled], axis=1)
    after = after.drop_duplicates(["season", "gw", "team_code"], keep="last")

    # every team x every gameweek of each season it played in, so blanks and future gameweeks exist
    grid = pd.concat([
        pd.MultiIndex.from_product(
            [[s.season], sorted(s.fixtures["gw"].unique()), s.teams["team_code"].unique()],
            names=["season", "gw", "team_code"]).to_frame(index=False)
        for s in seasons
    ], ignore_index=True)
    grid["season_idx"] = grid["season"].map(order)
    grid = grid.merge(after, on=["season", "gw", "team_code"], how="left", indicator=True)
    grid = grid.sort_values(["team_code", "season_idx", "gw"], ignore_index=True)
    grid[TEAM_STATS] = _carry_forward(grid, TEAM_STATS, grid["_merge"].eq("both"), by="team_code")
    grid[TEAM_STATS] = grid.groupby("team_code")[TEAM_STATS].shift(1)
    prior = cfg.features.promoted_prior
    grid[TEAM_STATS] = grid[TEAM_STATS].fillna({s: prior[s] for s in TEAM_STATS})
    return grid[["season", "gw", "team_code", *TEAM_STATS]]


def fixture_features(schedule: pd.DataFrame, team_state: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Per (season, gw, team_code): this gameweek's opponents and the next-N-gameweek run.

    Opponent strength is always taken as it stands *before gameweek N*, even for
    fixtures in N+1 and N+2, so no later results are used.
    """
    opp_state = team_state.rename(columns={"team_code": "opp_code", **{s: f"opp_{s}" for s in TEAM_STATS}})
    sched = schedule[["season", "gw", "team_code", "opp_code", "home"]]

    this_gw = sched.merge(opp_state, on=["season", "gw", "opp_code"], how="left")
    agg = {f"opp_{s}": "mean" for s in OPP_STATS}
    out = this_gw.groupby(["season", "gw", "team_code"]).agg(
        n_fixtures=("opp_code", "size"), home_frac=("home", "mean"), **{k: (k, v) for k, v in agg.items()})

    hp = horizon_prefix(cfg)
    ahead = pd.concat([sched.assign(base_gw=sched["gw"] - d) for d in range(cfg.features.fixture_horizon)])
    ahead = ahead.merge(opp_state.rename(columns={"gw": "base_gw"}), on=["season", "base_gw", "opp_code"],
                        how="inner")
    run = ahead.groupby(["season", "base_gw", "team_code"]).agg(
        **{f"{hp}_n": ("opp_code", "size"),
           f"{hp}_opp_xg_against": ("opp_xg_against", "mean"),
           f"{hp}_opp_xg_for": ("opp_xg_for", "mean")})
    run.index = run.index.set_names("gw", level="base_gw")
    return out.join(run, how="outer").reset_index()


# --- market -----------------------------------------------------------------

def market_features(rows: pd.DataFrame, managers: pd.Series) -> pd.DataFrame:
    """Price/ownership from the player's previous gameweek row in the same season.

    ``rows`` must be sorted by (code, season, gw) with a RangeIndex.
    """
    mgr = managers.reindex(pd.MultiIndex.from_arrays([rows["season"], rows["gw"]])).to_numpy()
    own = rows["selected"] / mgr * 100
    net = rows["transfers_balance"] / mgr * 100
    frame = pd.DataFrame({"value": rows["value"], "own": own, "net": net, "code": rows["code"],
                          "season": rows["season"]})
    g = frame.groupby(["code", "season"], sort=False)
    prev, prev2 = g[["value", "own", "net"]].shift(1), g[["value", "own", "net"]].shift(2)
    season_start = g["value"].transform("first")
    return pd.DataFrame({
        "price": prev["value"] / 10,
        "price_change_1gw": (prev["value"] - prev2["value"]) / 10,
        "price_change_season": (prev["value"] - season_start) / 10,
        "ownership_pct": prev["own"],
        "ownership_change_1gw": prev["own"] - prev2["own"],
        "net_transfers_pct": prev["net"],
    }, index=rows.index)


# --- targets ----------------------------------------------------------------

def last_complete_gw(schedule: pd.DataFrame) -> pd.Series:
    """Per season: the last gameweek such that it and every earlier one are fully played."""
    done = schedule.groupby(["season", "gw"])["finished"].all()

    def last(s: pd.Series) -> int:
        s = s.droplevel("season")
        open_gws = s.index[~s.to_numpy()]
        return int(open_gws.min() - 1) if len(open_gws) else int(s.index.max())

    return done.groupby(level="season").apply(last)


def add_targets(rows: pd.DataFrame, schedule: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    h = cfg.targets.multi_gw_horizon
    rows["y_points"] = rows["total_points"].where(~rows["is_future"])
    rows["y_haul"] = (rows["y_points"] >= cfg.targets.haul_points).astype("float64").where(rows["y_points"].notna())
    pts = rows.set_index(["code", "season", "gw"])["y_points"]
    total = pd.Series(0.0, index=rows.index)
    for d in range(h):
        key = pd.MultiIndex.from_arrays([rows["code"], rows["season"], rows["gw"] + d])
        total += pts.reindex(key).fillna(0).to_numpy()    # no row = blank gameweek or gone = 0 points
    complete = rows["gw"] + h - 1 <= rows["season"].map(last_complete_gw(schedule))
    rows[f"y_points_next{h}"] = total.where(complete & ~rows["is_future"])
    return rows


# --- assembly ---------------------------------------------------------------

def build_feature_table(seasons: list[SeasonData], cfg: Config,
                        upcoming: dict[str, list[int]] | None = None) -> pd.DataFrame:
    """Build the full player-gameweek table.

    ``upcoming`` maps a season to gameweeks not yet played (e.g. ``{"2026-27": [9]}``);
    rows are added for them with features but no targets.
    """
    order = season_index(seasons)
    schedule = pd.concat([team_schedule(s) for s in seasons], ignore_index=True)
    matches = pd.concat([player_matches(s) for s in seasons], ignore_index=True)
    matches["season_idx"] = matches["season"].map(order)

    rows = gameweek_rows(matches)
    future = [upcoming_rows(s, upcoming[s.season]) for s in seasons if upcoming and s.season in upcoming]
    if future:   # a gameweek already part-played keeps its played rows
        rows = pd.concat([rows, *future], ignore_index=True).drop_duplicates(["season", "code", "gw"])
    rows["season_idx"] = rows["season"].map(order)
    rows = rows.sort_values(["code", "season_idx", "gw"], ignore_index=True)

    # player form: state after each gameweek -> carried over unplayed gameweeks -> shifted one gameweek
    state = player_state(matches, cfg)
    state_cols = [c for c in state.columns if c not in ("code", "season", "gw")]
    rows = rows.merge(state, on=["code", "season", "gw"], how="left")
    rows[state_cols] = _carry_forward(rows, state_cols, rows["n_matches"].gt(0), by="code")
    cross_season = [c for c in state_cols if c not in SEASON_FEATURES]
    rows[cross_season] = rows.groupby("code")[cross_season].shift(1)
    rows[SEASON_FEATURES] = rows.groupby(["code", "season"])[SEASON_FEATURES].shift(1)
    rows["matches_season"] = rows["matches_season"].fillna(0.0)

    rows = pd.concat([rows, market_features(rows, managers_by_gameweek(rows, schedule))], axis=1)

    team_state = team_state_before(schedule, matches, seasons, order, cfg)
    rows = rows.merge(team_state.rename(columns={s: f"team_{s}" for s in TEAM_STATS}),
                      on=["season", "gw", "team_code"], how="left")
    rows = rows.merge(fixture_features(schedule, team_state, cfg), on=["season", "gw", "team_code"], how="left")

    kickoff = schedule.groupby(["season", "gw", "team_code"])["kickoff_time"].min().rename("kickoff_time")
    rows = rows.merge(kickoff.reset_index(), on=["season", "gw", "team_code"], how="left")
    short = pd.concat([s.teams.assign(season=s.season) for s in seasons])[["season", "team_code", "team_short"]]
    rows = rows.merge(short, on=["season", "team_code"], how="left")

    rows = add_targets(rows, schedule, cfg)
    rows = rows[rows["n_fixtures"].fillna(0) > 0]
    cols = dict.fromkeys([*ID_COLUMNS, *feature_columns(cfg), *target_columns(cfg), *REFERENCE_COLUMNS])
    return rows[list(cols)].sort_values(["season", "gw", "code"], ignore_index=True)
