"""One call from "team id" to a recommendation, returned as plain data (for the CLI and the web API)."""
from __future__ import annotations

from dataclasses import replace

import pandas as pd

from fplopt.config import Config
from fplopt.optimize.projections import get_source, live_pool
from fplopt.optimize.rules import SQUAD_QUOTAS
from fplopt.optimize.solver import Plan, recommend, solve, transfer_options
from fplopt.optimize.team import ManagerTeam, load_manager_team


def _player(pool: pd.DataFrame, i: int) -> dict:
    r = pool.loc[i]
    return {"id": int(i), "name": r["name"], "team": r["team"], "position": r["position"],
            "price": r["price"] / 10, "ownership": float(r["ownership"]), "projection": round(float(r["projection"]), 2),
            "status": r["status"], "news": r["news"] or ""}


def _pairs(pool: pd.DataFrame, plan: Plan) -> list[dict]:
    """Match each player out with a player in at the same position, for display."""
    out_by_pos = {p: [i for i in plan.transfers_out if pool.at[i, "position"] == p] for p in SQUAD_QUOTAS}
    pairs = []
    for i in plan.transfers_in:
        o = out_by_pos[pool.at[i, "position"]].pop(0)
        pairs.append({"out": _player(pool, o), "in": _player(pool, i),
                      "gain": round(float(pool.at[i, "projection"] - pool.at[o, "projection"]), 2)})
    return pairs


def describe_plan(pool: pd.DataFrame, plan: Plan) -> dict:
    xi = [_player(pool, i) for i in plan.xi]
    formation = "-".join(str(sum(p["position"] == pos for p in xi)) for pos in ("DEF", "MID", "FWD"))
    return {
        "n_transfers": plan.n_transfers, "transfers": _pairs(pool, plan), "hits": plan.hits,
        "hit_cost": plan.hit_cost, "projected_points": round(plan.projected_points, 2),
        "net_points": round(plan.net_points, 2), "bank": plan.bank / 10, "formation": formation,
        "xi": xi, "bench": [_player(pool, i) for i in plan.bench],
        "captain": _player(pool, plan.captain), "vice_captain": _player(pool, plan.vice_captain),
    }


def optimize(cfg: Config, players: pd.DataFrame, team: ManagerTeam | None = None, projection: str | None = None,
             max_transfers: int | None = None, differential_bonus: float | None = None) -> dict:
    """Recommendation for the next gameweek. Without a team: the best new squad within budget."""
    opt = cfg.optimizer
    if differential_bonus is not None:
        cfg = replace(cfg, optimizer=replace(opt, differential_bonus=differential_bonus))
        opt = cfg.optimizer
    source = get_source(projection or opt.projection, cfg)
    pool = live_pool(players, source, cfg, team.squad if team else None)
    result = {"gameweek": int(players["next_gw"].iloc[0]), "projection": source.name,
              "differential_bonus": opt.differential_bonus}
    if team is None:
        plan = solve(pool, opt)
        result.update({"mode": "new squad", "options": [describe_plan(pool, plan)], "recommended": 0})
        return result
    options = transfer_options(pool, team.squad, opt, max_transfers)
    best = recommend(options, opt.min_gain_per_transfer)
    result.update({
        "mode": "transfers",
        "team": {"id": team.team_id, "name": team.team_name, "manager": team.manager,
                 "bank": team.squad.bank / 10, "free_transfers": team.squad.free_transfers,
                 "free_transfers_inferred": team.free_transfers_inferred,
                 "pending_transfers": len(team.pending_transfers), "chips_used": team.chips_used,
                 "squad": [{**_player(pool, i), "selling_price": team.squad.selling_prices[i] / 10,
                            "purchase_price": team.purchase_prices[i] / 10}
                           for i in sorted(team.squad.ids, key=lambda i: -pool.at[i, "projection"])]},
        "options": [describe_plan(pool, p) for p in options],
        "recommended": options.index(best),
    })
    return result


def optimize_team(cfg: Config, client, team_id: int | None = None, free_transfers: int | None = None,
                  bank: float | None = None, **kwargs) -> dict:
    """Load the live player list and (optionally) the manager's team, then :func:`optimize`."""
    players = pd.read_parquet(cfg.paths.live_players)
    team = None
    if team_id is not None:
        team = load_manager_team(client, team_id, client.bootstrap(), cfg.optimizer.max_free_transfers,
                                 free_transfers=free_transfers,
                                 bank=None if bank is None else int(round(bank * 10)))
    return optimize(cfg, players, team, **kwargs)
