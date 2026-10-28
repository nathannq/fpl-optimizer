"""Replay a season with the optimizer, once per projection source, and score what really happened.

Each gameweek the optimizer sees only that source's projections and the
prices/ownership going into the gameweek. It makes transfers (free transfer
banking, -4 hits, selling-price rule), picks the XI and captain, and is then
scored on actual points with FPL's automatic substitutions and vice-captain rule.

Simplifications: no chips; transfer prices are the previous gameweek's price
(changes during the week are ignored); no historical injury flags.
"""
from __future__ import annotations

import gzip
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fplopt.config import Config
from fplopt.optimize.projections import objective_points
from fplopt.optimize.rules import XI_MAX, XI_MIN, XI_SIZE, selling_price
from fplopt.optimize.solver import Plan, Squad, recommend, solve, transfer_options

log = logging.getLogger(__name__)
SNAPSHOT_FORMAT = "%Y-%m-%dT%H-%M-%SZ"     # as written by FPLClient._snapshot


# --- scoring ----------------------------------------------------------------

def score_gameweek(plan: Plan, positions: dict[int, str], points: dict[int, float],
                   minutes: dict[int, float]) -> dict:
    """Actual points for a plan, with FPL's automatic substitutions and captaincy rules.

    Bench players come on in order for starters who didn't play, the reserve
    goalkeeper only for the goalkeeper, and an outfield sub is skipped if he
    would leave the XI unable to meet the formation minimums. If the captain
    didn't play, the vice-captain's points are doubled instead.
    """
    played = lambda i: minutes.get(i, 0) > 0                         # noqa: E731
    final = [i for i in plan.xi if played(i)]
    if not any(positions[i] == "GK" for i in final):
        final += [i for i in plan.bench if positions[i] == "GK" and played(i)][:1]
    missing_outfield = sum(1 for i in plan.xi if positions[i] != "GK" and not played(i))
    for b in plan.bench:
        if missing_outfield == 0:
            break
        if positions[b] == "GK" or not played(b):
            continue
        counts = {p: sum(positions[i] == p for i in final + [b]) for p in XI_MIN}
        open_slots = XI_SIZE - len(final) - 1
        needed = sum(max(0, XI_MIN[p] - counts[p]) for p in XI_MIN)
        if needed <= open_slots and all(counts[p] <= XI_MAX[p] for p in XI_MAX):
            final.append(b)
            missing_outfield -= 1
    captain = plan.captain if played(plan.captain) else plan.vice_captain if played(plan.vice_captain) else None
    base = sum(points.get(i, 0.0) for i in final)
    armband = points.get(captain, 0.0) if captain is not None else 0.0
    return {"points": base + armband - plan.hit_cost, "lineup_points": base + armband,
            "captain_points": armband * 2, "autosubs": len([i for i in final if i not in plan.xi])}


# --- data -------------------------------------------------------------------

def ep_next_history(snapshot_dir: Path) -> pd.DataFrame:
    """FPL's pre-deadline ep_next per (season, gw, element) from archived bootstrap snapshots.

    For each gameweek the last snapshot taken before its deadline is used.
    """
    rows = []
    for file in sorted((snapshot_dir / "bootstrap-static").glob("*.json.gz")):
        taken = pd.Timestamp(datetime.strptime(file.name.removesuffix(".json.gz"), SNAPSHOT_FORMAT), tz="UTC")
        with gzip.open(file, "rt") as fh:
            boot = json.load(fh)
        nxt = next((e for e in boot["events"] if e.get("is_next")), None)
        if nxt is None or taken >= pd.Timestamp(nxt["deadline_time"]):
            continue
        year = pd.Timestamp(boot["events"][0]["deadline_time"]).year
        season = f"{year}-{(year + 1) % 100:02d}"
        rows += [{"season": season, "gw": nxt["id"], "element": e["id"], "taken": taken,
                  "ep_next": float(e.get("ep_next") or 0)} for e in boot["elements"]]
    if not rows:
        return pd.DataFrame(columns=["season", "gw", "element", "ep_next"])
    df = pd.DataFrame(rows).sort_values("taken")
    return df.drop_duplicates(["season", "gw", "element"], keep="last").drop(columns="taken")


def season_frame(oos: pd.DataFrame, features: pd.DataFrame, season: str, ep_next: pd.DataFrame | None = None
                 ) -> pd.DataFrame:
    """Everything the replay needs for one season: projections, prices, outcomes."""
    df = oos[oos["season"] == season].copy()
    df["base_last3"] = df["pts_r3"].fillna(0) * df["n_fixtures"]
    outcomes = features.loc[features["season"] == season, ["gw", "code", "y_minutes"]]
    df = df.merge(outcomes, on=["gw", "code"], how="left")
    if ep_next is not None and not ep_next.empty:
        df = df.merge(ep_next[ep_next["season"] == season].drop(columns="season"), on=["gw", "element"], how="left")
    df["price_tenths"] = (df["price"] * 10).round()
    return df


# --- replay -----------------------------------------------------------------

def simulate(frame: pd.DataFrame, projection: str, cfg: Config) -> pd.DataFrame:
    """Play the season with one projection source. One row per gameweek."""
    opt = cfg.optimizer
    meta: dict[int, tuple[str, str, int]] = {}      # code -> (name, position, club), last seen
    last_price: dict[int, int] = {}
    purchase: dict[int, int] = {}
    owned: list[int] = []
    bank, free = opt.budget, 0
    results = []
    for gw in sorted(g for g in frame["gw"].unique() if g >= cfg.backtest.start_gw):
        week = frame[frame["gw"] == gw].set_index("code")
        for code, r in week.iterrows():
            meta[code] = (r["web_name"], r["position"], int(r["team_code"]))
            if pd.notna(r["price_tenths"]):
                last_price[code] = int(r["price_tenths"])
        ids = sorted(set(week.index) | set(owned))
        proj = week[projection].reindex(ids).fillna(0.0)          # no fixture this week = 0
        pool = pd.DataFrame({
            "name": [meta[i][0] for i in ids], "position": [meta[i][1] for i in ids],
            "club": [meta[i][2] for i in ids],
            "price": [last_price.get(i, 0) for i in ids],
            "projection": proj.to_numpy(),
            "can_buy": [i in week.index and pd.notna(week.at[i, "price_tenths"]) for i in ids],
            "can_start": True,
        }, index=ids)
        pool["objective"] = objective_points(pool["projection"], week["ownership_pct"].reindex(ids), cfg)

        if not owned:                                   # first gameweek: pick a squad from scratch
            plan = solve(pool, opt)
            bought, free = (plan.squad if plan else []), 1
        else:
            squad = Squad({i: selling_price(purchase[i], last_price[i]) for i in owned}, bank, free)
            options = transfer_options(pool, squad, opt)
            plan = recommend(options, opt.min_gain_per_transfer) if options else None
            bought = plan.transfers_in if plan else []
            free = min(max(free - (plan.n_transfers if plan else 0), 0) + 1, opt.max_free_transfers)
        if plan is None:
            raise RuntimeError(f"no feasible plan in GW{gw} for {projection}")
        for i in bought:
            purchase[i] = last_price[i]
        owned, bank = plan.squad, plan.bank

        scored = score_gameweek(plan, {i: meta[i][1] for i in ids},
                                week["y_points"].reindex(ids).fillna(0).to_dict(),
                                week["y_minutes"].reindex(ids).fillna(0).to_dict())
        results.append({"gw": gw, "projected": plan.net_points, **scored, "transfers": plan.n_transfers,
                        "hits": plan.hits, "captain": meta[plan.captain][0], "bank": plan.bank})
    return pd.DataFrame(results)


def run_backtest(oos: pd.DataFrame, features: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """All configured seasons x sources. Sources with no data for a season are skipped."""
    ep_next = ep_next_history(cfg.paths.snapshots)
    out = []
    for season in cfg.backtest.seasons:
        frame = season_frame(oos, features, season, ep_next)
        for name, column in cfg.backtest.sources.items():
            if column not in frame or frame[column].notna().groupby(frame["gw"]).any().mean() < 0.9:
                log.info("Skipping %s for %s: no projections for most gameweeks", name, season)
                continue
            log.info("Replaying %s with %s projections", season, name)
            weeks = simulate(frame, column, cfg)
            out.append(weeks.assign(season=season, source=name))
    return pd.concat(out, ignore_index=True)


def backtest_summary(weeks: pd.DataFrame) -> pd.DataFrame:
    """Season totals per source. ``projected_per_gw`` vs ``mean_per_gw`` shows whether the source's
    projections for the team it picked were honest (selecting on noisy projections inflates them)."""
    g = weeks.assign(hit_points=weeks["lineup_points"] - weeks["points"]).groupby(["season", "source"])
    return g.agg(gameweeks=("gw", "size"), total_points=("points", "sum"), mean_per_gw=("points", "mean"),
                 projected_per_gw=("projected", "mean"), transfers=("transfers", "sum"), hits=("hits", "sum"),
                 hit_points=("hit_points", "sum"), captain_points=("captain_points", "sum"),
                 autosubs=("autosubs", "sum")).reset_index()


def head_to_head(weeks: pd.DataFrame, a: str, b: str, seed: int = 0, n_boot: int = 5000) -> pd.DataFrame:
    """Per-season gameweek-paired comparison of two sources' actual points."""
    rows = []
    for season, part in weeks.groupby("season"):
        wide = part.pivot(index="gw", columns="source", values="points")
        if a not in wide or b not in wide:
            continue
        diff = (wide[a] - wide[b]).to_numpy()
        boots = np.random.default_rng(seed).choice(diff, (n_boot, len(diff))).mean(axis=1)
        rows.append({"season": season, "comparison": f"{a} - {b}", "points_per_gw": diff.mean(),
                     "ci_low": np.percentile(boots, 2.5), "ci_high": np.percentile(boots, 97.5),
                     "season_total": diff.sum(), "gw_wins": (diff > 0).mean(), "gw_ties": (diff == 0).mean()})
    return pd.DataFrame(rows)
