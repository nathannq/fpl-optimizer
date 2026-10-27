"""Baselines the model has to beat.

* ``last3``  - average points over the player's last 3 team fixtures x fixtures
               this gameweek. Fully pre-deadline; the honest bar.
* ``fpl_xp`` - FPL's own projection as recorded in the historical data (``xP``).

A caveat about ``fpl_xp``: the historical files record xP *after* the gameweek
is played. FPL recomputes its projection from "form", and by then form already
includes that gameweek's points. :func:`xp_lookahead_check` shows this: players
with no minutes in their previous 5 fixtures get a high xP only in gameweeks
they go on to haul. So ``fpl_xp`` is an optimistic, partly look-ahead
reference rather than a fair baseline. A clean ep_next is recorded going forward
from the live API snapshots (see ``fplopt.data.api``).
"""
from __future__ import annotations

import pandas as pd

from fplopt.config import Config
from fplopt.evaluate import differential_metrics, error_metrics
from fplopt.reporting import markdown_table

BASELINES = {"last3": "base_last3", "fpl_xp": "base_fpl_xp"}


def add_baselines(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["base_last3"] = out["pts_r3"].fillna(0) * out["n_fixtures"]   # no history -> 0
    out["base_fpl_xp"] = out["xp_fpl"]
    return out


def xp_lookahead_check(df: pd.DataFrame) -> pd.DataFrame:
    """Mean recorded xP for players with no minutes in their previous 5 fixtures, by outcome.

    A pre-deadline projection cannot tell these players apart by what they
    later scored; a projection that already knows the result can.
    """
    idle = df[(df["mins_r5"] == 0) & df["xp_fpl"].notna() & df["y_points"].notna()]
    groups = {
        "hauled (8+ pts)": idle["y_points"] >= 8,
        "returned (3-7 pts)": idle["y_points"].between(3, 7),
        "played, blanked (1-2 pts)": idle["y_points"].between(1, 2),
    }
    return pd.DataFrame([
        {"outcome in GW": name, "rows": int(mask.sum()), "mean FPL xP": idle.loc[mask, "xp_fpl"].mean()}
        for name, mask in groups.items()
    ])


def baseline_report(df: pd.DataFrame, cfg: Config) -> str:
    """Markdown report of both baselines over the evaluation seasons."""
    ev = cfg.evaluation
    k, own = ev.top_k, ev.ownership_max_pct
    ret, haul = cfg.targets.return_points, cfg.targets.haul_points
    data = add_baselines(df[df["season"].isin(ev.seasons) & df["y_points"].notna()])
    xp_gws = data.groupby(["season", "gw"])["xp_fpl"].transform(lambda s: s.notna().any())
    with_xp = data[xp_gws]
    active = data["mins_r5"] > 0

    coverage = data.groupby("season").agg(gameweeks=("gw", "nunique"), rows=("gw", "size"))
    coverage["gameweeks_with_fpl_xp"] = with_xp.groupby("season")["gw"].nunique().reindex(coverage.index).fillna(0)

    acc_rows = []
    for label, subset in [("all players", data), ("played in last 5", data[active])]:
        acc_rows.append({"rows": label, "baseline": "last3", **error_metrics(subset, "base_last3")})
    for label, subset in [("all players", with_xp), ("played in last 5", with_xp[with_xp["mins_r5"] > 0])]:
        for name, col in BASELINES.items():
            acc_rows.append({"rows": f"{label}, GWs with FPL xP", "baseline": name, **error_metrics(subset, col)})
    accuracy = pd.DataFrame(acc_rows)

    diff_rows = []
    for season, part in [*data.groupby("season"), ("all", data)]:
        diff_rows.append({"season": season, "baseline": "last3",
                          **differential_metrics(part, "base_last3", k, own, ret, haul)})
    for name, col in BASELINES.items():
        diff_rows.append({"season": "GWs with FPL xP", "baseline": name,
                          **differential_metrics(with_xp, col, k, own, ret, haul)})
    differentials = pd.DataFrame(diff_rows)

    return "\n".join([
        "# Phase 1 - baselines",
        "",
        f"Evaluation seasons: {', '.join(ev.seasons)}. A differential is owned by < {own:g}% of managers "
        f"going into the gameweek; a return is >= {ret} points and a haul >= {haul}.",
        "",
        "## Data coverage",
        "",
        markdown_table(coverage.reset_index()),
        "",
        "## Is FPL's recorded xP a fair baseline?",
        "",
        "Players with **zero minutes in their previous 5 fixtures**, grouped by what they went on to score:",
        "",
        markdown_table(xp_lookahead_check(data)),
        "",
        "A projection made before the deadline cannot know which idle players will haul. The recorded xP "
        "does, because the source captured it after the gameweek (FPL's form input already includes "
        "that gameweek). `fpl_xp` is therefore reported as an optimistic, look-ahead reference; "
        "`last3` is the honest baseline.",
        "",
        "## Accuracy (points next gameweek)",
        "",
        markdown_table(accuracy),
        "",
        f"## Differentials: top {k} per gameweek",
        "",
        markdown_table(differentials),
        "",
        "`pool_*` = the average differential that gameweek (what ownership alone implies).",
        "",
    ])
