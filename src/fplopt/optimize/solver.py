"""Next-gameweek squad optimisation as a mixed-integer program (PuLP + HiGHS).

Decision variables per player: in the squad, in the starting XI, captain.

    maximise   sum(XI points) + captain's points again           (captain doubles)
             + bench_weight * sum(bench points)                  (autosub cover)
             - hit_cost * hits
    subject to 2 GK / 5 DEF / 5 MID / 3 FWD, at most 3 per club,
               XI of 11 with 1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD, captain in the XI,
               money spent on new players <= bank + selling prices of players sold,
               hits >= transfers - free transfers.

"Points" here are the pool's ``objective`` column: projected points, optionally
with a bonus for differentials. Reported points are always the plain projection.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd
import pulp

from fplopt.config import OptimizerConfig
from fplopt.optimize.rules import MAX_PER_CLUB, SQUAD_QUOTAS, SQUAD_SIZE, XI_MAX, XI_MIN, XI_SIZE

# Columns the player pool must have (indexed by player id).
POOL_COLUMNS = ["name", "position", "club", "price", "projection", "objective", "can_buy", "can_start"]


@dataclass
class Squad:
    """The manager's current squad: player id -> selling price, plus bank and free transfers."""

    selling_prices: dict[int, int]
    bank: int
    free_transfers: int

    @property
    def ids(self) -> set[int]:
        return set(self.selling_prices)


@dataclass
class Plan:
    n_transfers: int
    transfers_in: list[int]
    transfers_out: list[int]
    squad: list[int]
    xi: list[int]
    bench: list[int]               # substitution order: reserve GK first, then outfielders
    captain: int
    vice_captain: int
    hits: int
    hit_cost: int
    projected_points: float        # XI + captain again, before hits (plain projection)
    net_points: float              # projected_points - hit_cost
    objective: float               # what was maximised (bonus and bench weight included)
    bank: int                      # money left, tenths of £m
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _check_pool(pool: pd.DataFrame, squad: Squad | None) -> None:
    missing = [c for c in POOL_COLUMNS if c not in pool.columns]
    if missing:
        raise ValueError(f"pool is missing columns {missing}")
    if squad and not squad.ids <= set(pool.index):
        raise ValueError(f"squad players missing from the pool: {sorted(squad.ids - set(pool.index))}")


def solve(pool: pd.DataFrame, cfg: OptimizerConfig, squad: Squad | None = None,
          n_transfers: int | None = None) -> Plan | None:
    """Best squad/XI/captain. With ``squad``: transfers from it (exactly ``n_transfers`` if given).

    Without ``squad``: pick a new squad within ``cfg.budget``. Returns None if infeasible.
    """
    _check_pool(pool, squad)
    ids = list(pool.index)
    owned = squad.ids if squad else set()
    pos, club, price = pool["position"], pool["club"], pool["price"]
    obj = pool["objective"].astype(float)

    prob = pulp.LpProblem("fpl", pulp.LpMaximize)
    s = prob.add_variable_dict("squad", ids, cat="Binary")
    x = prob.add_variable_dict("start", ids, cat="Binary")
    c = prob.add_variable_dict("captain", ids, cat="Binary")

    for i in ids:
        prob += x[i] <= s[i]
        prob += c[i] <= x[i]
        if i not in owned and not pool.at[i, "can_buy"]:
            prob += s[i] == 0
        if not pool.at[i, "can_start"]:
            prob += x[i] == 0
    for p, quota in SQUAD_QUOTAS.items():
        members = [i for i in ids if pos[i] == p]
        prob += pulp.lpSum(s[i] for i in members) == quota
        prob += pulp.lpSum(x[i] for i in members) >= XI_MIN[p]
        prob += pulp.lpSum(x[i] for i in members) <= XI_MAX[p]
    for team in club.unique():
        prob += pulp.lpSum(s[i] for i in ids if club[i] == team) <= MAX_PER_CLUB
    prob += pulp.lpSum(x.values()) == XI_SIZE
    prob += pulp.lpSum(c.values()) == 1

    value = pulp.lpSum(obj[i] * (x[i] + c[i]) + cfg.bench_weight * obj[i] * (s[i] - x[i]) for i in ids)
    hits = None
    if squad:
        buys = pulp.lpSum(s[i] for i in ids if i not in owned)
        hits = prob.add_variable("hits", lowBound=0, upBound=SQUAD_SIZE, cat="Integer")
        prob += hits >= buys - squad.free_transfers
        if n_transfers is not None:
            prob += buys == n_transfers
        spend = pulp.lpSum(int(price[i]) * s[i] for i in ids if i not in owned)
        proceeds = pulp.lpSum(sp * (1 - s[i]) for i, sp in squad.selling_prices.items())
        prob += spend <= squad.bank + proceeds
        value = value - cfg.hit_cost * hits
    else:
        prob += pulp.lpSum(int(price[i]) * s[i] for i in ids) <= cfg.budget
    prob += value

    stats = prob.solve(pulp.HiGHS(msg=False, timeLimit=cfg.solver_time_limit))
    if stats.status_str != "Optimal":
        return None
    return _plan(pool, squad, s, x, c, hits, cfg, stats.objective)


def _plan(pool, squad, s, x, c, hits, cfg, objective) -> Plan:
    picked = [i for i in pool.index if s[i].value() > 0.5]
    xi = [i for i in picked if x[i].value() > 0.5]
    captain = next(i for i in xi if c[i].value() > 0.5)
    proj = pool["projection"].astype(float)
    order = lambda ids: sorted(ids, key=lambda i: (-proj[i], i))     # noqa: E731
    bench_ids = [i for i in picked if i not in xi]
    bench = ([i for i in bench_ids if pool.at[i, "position"] == "GK"]
             + order([i for i in bench_ids if pool.at[i, "position"] != "GK"]))
    vice = next(i for i in order(xi) if i != captain)
    owned = squad.ids if squad else set()
    n_hits = int(round(hits.value())) if hits is not None else 0
    spent = sum(int(pool.at[i, "price"]) for i in picked if i not in owned)
    if squad:
        bank = squad.bank + sum(sp for i, sp in squad.selling_prices.items() if i not in picked) - spent
    else:
        bank = cfg.budget - spent
    projected = float(proj[xi].sum() + proj[captain])
    return Plan(
        n_transfers=len([i for i in picked if i not in owned]) if squad else 0,
        transfers_in=order([i for i in picked if i not in owned]) if squad else [],
        transfers_out=order([i for i in owned if i not in picked]),
        squad=order(picked), xi=order(xi), bench=bench, captain=captain, vice_captain=vice,
        hits=n_hits, hit_cost=n_hits * cfg.hit_cost,
        projected_points=projected, net_points=projected - n_hits * cfg.hit_cost,
        objective=float(objective), bank=int(bank),
    )


def transfer_options(pool: pd.DataFrame, squad: Squad, cfg: OptimizerConfig,
                     max_transfers: int | None = None) -> list[Plan]:
    """The best plan for exactly 0, 1, ..., ``max_transfers`` transfers (infeasible counts skipped)."""
    top = cfg.max_transfers if max_transfers is None else max_transfers
    return [p for n in range(top + 1) if (p := solve(pool, cfg, squad, n_transfers=n)) is not None]


def recommend(options: list[Plan], min_gain_per_transfer: float) -> Plan:
    """Fewest transfers unless more pay their way: each extra transfer must add ``min_gain_per_transfer``."""
    best = min(options, key=lambda p: p.n_transfers)
    for plan in sorted(options, key=lambda p: p.n_transfers):
        if plan.objective >= best.objective + min_gain_per_transfer * (plan.n_transfers - best.n_transfers):
            best = plan
    return best
