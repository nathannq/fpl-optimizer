"""Phase 3: squad rules, transfer-hit maths, selling prices, autosubs, team loading, backtest."""
from __future__ import annotations

import gzip
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from fplopt.optimize.backtest import ep_next_history, score_gameweek, simulate
from fplopt.optimize.projections import objective_points
from fplopt.optimize.rules import (
    MAX_PER_CLUB,
    SQUAD_QUOTAS,
    infer_free_transfers,
    is_available,
    selling_price,
    valid_formation,
)
from fplopt.optimize.solver import Plan, Squad, recommend, solve, transfer_options
from fplopt.optimize.team import load_manager_team

POSITIONS = ["GK", "DEF", "MID", "FWD"]


def make_pool(seed: int = 0, clubs: int = 20) -> pd.DataFrame:
    """A full-size league: every club has a complete 2/5/5/3 squad with random prices and projections."""
    rng = np.random.default_rng(seed)
    rows, pid = [], 1
    for club in range(1, clubs + 1):
        for pos in POSITIONS:
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


@pytest.fixture(scope="module")
def pool():
    return make_pool()


@pytest.fixture(scope="module")
def opt(cfg):
    return cfg.optimizer


def assert_legal(pool: pd.DataFrame, plan: Plan, budget: int | None = None) -> None:
    squad = pool.loc[plan.squad]
    assert len(plan.squad) == 15 and len(set(plan.squad)) == 15
    assert squad["position"].value_counts().to_dict() == SQUAD_QUOTAS
    assert squad["club"].value_counts().max() <= MAX_PER_CLUB
    assert set(plan.xi) <= set(plan.squad) and valid_formation(pool.loc[plan.xi, "position"])
    assert set(plan.bench) == set(plan.squad) - set(plan.xi) and pool.at[plan.bench[0], "position"] == "GK"
    assert plan.captain in plan.xi and plan.vice_captain in plan.xi and plan.captain != plan.vice_captain
    assert plan.bank >= 0
    if budget is not None:
        assert squad["price"].sum() + plan.bank == budget


def test_new_squad_obeys_every_rule(pool, opt):
    plan = solve(pool, opt)
    assert_legal(pool, plan, budget=opt.budget)
    proj = pool["projection"]
    assert plan.captain == proj[plan.xi].idxmax()                       # captain doubles: the best starter
    assert plan.projected_points == pytest.approx(proj[plan.xi].sum() + proj[plan.captain])
    bench_outfield = plan.bench[1:]
    assert list(proj[bench_outfield]) == sorted(proj[bench_outfield], reverse=True)


def test_unavailable_players_are_never_bought_or_started(pool, opt):
    best = solve(pool, opt)
    blocked = pool.copy()
    blocked.loc[best.captain, "can_buy"] = False
    plan = solve(blocked, opt)
    assert best.captain not in plan.squad

    # an owned player who can't play may stay in the squad but not in the XI
    squad = Squad({i: int(pool.at[i, "price"]) for i in best.squad}, bank=best.bank, free_transfers=0)
    injured = blocked.copy()
    injured.loc[best.captain, ["can_buy", "can_start"]] = False
    keep = solve(injured, opt, squad, n_transfers=0)
    assert best.captain in keep.squad and best.captain not in keep.xi


@pytest.fixture(scope="module")
def current(pool, opt):
    """A squad picked on stale projections, so there are transfers worth making on today's."""
    stale = pool.assign(objective=pool["projection"][::-1].to_numpy(), projection=pool["projection"][::-1].to_numpy())
    plan = solve(stale, opt)
    return Squad({i: int(pool.at[i, "price"]) for i in plan.squad}, bank=plan.bank, free_transfers=1)


def test_transfer_hit_maths(pool, opt, current):
    options = {p.n_transfers: p for p in transfer_options(pool, current, opt, max_transfers=3)}
    assert set(options) == {0, 1, 2, 3}
    assert options[0].squad == sorted(current.ids, key=lambda i: (-pool.at[i, "projection"], i))
    assert (options[1].hits, options[2].hits, options[3].hits) == (0, 1, 2)
    for n, plan in options.items():
        assert len(plan.transfers_in) == len(plan.transfers_out) == n
        assert plan.hit_cost == 4 * plan.hits
        assert plan.net_points == pytest.approx(plan.projected_points - plan.hit_cost)
        assert_legal(pool, plan)
    banked = replace(current, free_transfers=2)
    assert solve(pool, opt, banked, n_transfers=2).hits == 0          # two banked transfers are free
    none_free = replace(current, free_transfers=0)
    assert solve(pool, opt, none_free, n_transfers=1).hits == 1


def test_budget_uses_selling_prices_not_current_prices(opt):
    pool = make_pool(seed=3)
    plan = solve(pool, opt)
    star = max(plan.squad, key=lambda i: pool.at[i, "projection"])
    # Today's price is 10 above what the manager can sell for (half the rise is kept by FPL).
    today = int(pool.at[star, "price"])
    sell = today - 10
    squad = Squad({i: (sell if i == star else int(pool.at[i, "price"])) for i in plan.squad}, bank=0,
                  free_transfers=1)
    target = pool.index.difference(plan.squad)[pool.loc[pool.index.difference(plan.squad), "position"]
                                               == pool.at[star, "position"]][0]
    pool = pool.copy()
    pool.loc[target, "projection"] = pool.loc[target, "objective"] = 100.0  # irresistible
    pool.loc[target, "club"] = pool.at[star, "club"]                        # same club: no 3-per-club issue
    pool.loc[target, "price"] = sell + 1                                    # affordable at today's price only
    p = solve(pool, opt, squad, n_transfers=1)
    assert target not in p.squad
    pool.loc[target, "price"] = sell                                        # exactly the selling price
    p = solve(pool, opt, squad, n_transfers=1)
    assert p.transfers_in == [target] and p.transfers_out == [star] and p.bank == 0


def test_selling_price_rule():
    assert selling_price(50, 55) == 52       # keep half of a 0.5m rise, rounded down
    assert selling_price(50, 51) == 50
    assert selling_price(50, 50) == 50
    assert selling_price(50, 47) == 47       # a fall is taken in full


def test_free_transfer_inference():
    chips = {2: "bboost", 3: "wildcard", 5: "freehit"}
    made = {6: 1, 8: 2}
    assert infer_free_transfers(made, chips, started_event=1, next_event=9, max_free=5) == 4
    assert infer_free_transfers({}, {}, started_event=1, next_event=2, max_free=5) == 1
    assert infer_free_transfers({}, {}, started_event=1, next_event=12, max_free=5) == 5    # capped
    assert infer_free_transfers({2: 3}, {}, started_event=1, next_event=3, max_free=5) == 1  # hits: floor at 0


def test_availability_rule():
    assert is_available("a", None, 75)
    assert is_available("d", 75.0, 75) and not is_available("d", 50.0, 75)
    assert not is_available("i", None, 75) and not is_available("s", float("nan"), 75)


def test_recommendation_needs_each_transfer_to_pay():
    def plan(n, obj):
        return Plan(n, [], [], [], [], [], 0, 0, 0, 0, obj, obj, obj, 0)
    assert recommend([plan(0, 50.0), plan(1, 50.3), plan(2, 51.5)], 0.5).n_transfers == 2
    assert recommend([plan(0, 50.0), plan(1, 50.3), plan(2, 50.9)], 0.5).n_transfers == 0


def test_differential_bonus_only_lifts_low_owned_players(cfg):
    bonus_cfg = replace(cfg, optimizer=replace(cfg.optimizer, differential_bonus=0.2))
    out = objective_points(pd.Series([5.0, 5.0]), pd.Series([3.0, 30.0]), bonus_cfg)
    assert out.tolist() == [6.0, 5.0]


def test_autosubs_and_vice_captain():
    pos = {1: "GK", 2: "DEF", 3: "DEF", 4: "DEF", 5: "MID", 6: "MID", 7: "MID", 8: "MID", 9: "FWD", 10: "FWD",
           11: "FWD", 12: "GK", 13: "MID", 14: "DEF", 15: "FWD"}
    plan = Plan(1, [], [], list(pos), xi=list(range(1, 12)), bench=[12, 13, 14, 15], captain=9, vice_captain=5,
                hits=1, hit_cost=4, projected_points=0, net_points=0, objective=0, bank=0)
    pts = {i: 2.0 for i in pos} | {9: 10.0, 5: 6.0, 12: 3.0, 13: 4.0, 14: 5.0}

    # GK 1, DEF 2 and FWD 9 (captain) don't play. GK 12 replaces GK 1; MID 13 and DEF 14 fill the two
    # outfield gaps (3-5-2 is legal); FWD 15 isn't needed. The vice-captain (5) is doubled.
    r = score_gameweek(plan, pos, pts, {i: 90 for i in pos} | {1: 0, 2: 0, 9: 0})
    lineup = 2.0 * 7 + 6.0 + 3.0 + 4.0 + 5.0      # 3,4,6,7,8,10,11 + 5 + 12 + 13 + 14
    assert r["autosubs"] == 3
    assert r["lineup_points"] == lineup + 6.0
    assert r["points"] == lineup + 6.0 - 4         # the hit is taken off

    # Only DEF 2 misses out: MID 13 would leave two defenders, so he is skipped and DEF 14 comes on.
    r = score_gameweek(plan, pos, pts, {i: 90 for i in pos} | {2: 0})
    assert r["autosubs"] == 1
    assert r["lineup_points"] == 2.0 * 8 + 6.0 + 10.0 + 5.0 + 10.0   # captain 9 doubled


def test_ep_next_history_uses_the_last_snapshot_before_each_deadline(tmp_path):
    folder = tmp_path / "bootstrap-static"
    folder.mkdir()
    events = [{"id": 1, "deadline_time": "2026-08-14T17:30:00Z", "is_next": False},
              {"id": 2, "deadline_time": "2026-08-22T10:00:00Z", "is_next": True}]
    for stamp, ep in [("2026-08-20T09-00-00Z", 3.0), ("2026-08-22T09-00-00Z", 4.0), ("2026-08-22T11-00-00Z", 9.0)]:
        with gzip.open(folder / f"{stamp}.json.gz", "wt") as fh:
            json.dump({"events": events, "elements": [{"id": 7, "ep_next": str(ep)}]}, fh)
    hist = ep_next_history(tmp_path)
    assert hist.to_dict("records") == [{"season": "2026-27", "gw": 2, "element": 7, "ep_next": 4.0}]


def test_backtest_replay_follows_the_rules(cfg):
    """Three gameweeks on a full synthetic league: legal squads, banked transfers, no negative bank."""
    rng = np.random.default_rng(5)
    base = make_pool(seed=5).reset_index().rename(columns={"id": "code", "name": "web_name", "club": "team_code"})
    weeks = []
    for gw in (2, 3, 4):
        w = base.assign(gw=gw, element=base["code"], price=base["price"] / 10, ownership_pct=base["ownership"],
                        y_points=rng.integers(0, 10, len(base)).astype(float),
                        y_minutes=rng.choice([0.0, 90.0], len(base), p=[0.2, 0.8]))
        w["lgbm_points"] = w["projection"] * rng.uniform(0.5, 1.5, len(w))
        weeks.append(w)
    frame = pd.concat(weeks, ignore_index=True)
    frame["price_tenths"] = (frame["price"] * 10).round()
    out = simulate(frame, "lgbm_points", cfg)
    assert out["gw"].tolist() == [2, 3, 4]
    assert (out["bank"] >= 0).all() and out.loc[0, "transfers"] == 0
    assert (out["transfers"] <= cfg.optimizer.max_transfers).all()
    assert (out["hits"] <= out["transfers"]).all()


# --- loading a manager's team from the API (fake client) ----------------------

class FakeClient:
    def __init__(self):
        self.summaries = {e: {"history": [{"round": 1, "value": 50 + e}]} for e in range(1, 20)}

    def entry(self, team_id):
        return {"started_event": 1, "name": "Test XI", "player_first_name": "A", "player_last_name": "B"}

    def entry_history(self, team_id):
        return {"current": [{"event": e, "bank": 10 * e, "event_transfers": t}
                            for e, t in [(1, 0), (2, 1), (3, 0), (4, 0)]],
                "chips": [{"name": "freehit", "event": 4}]}

    def picks(self, team_id, gw):
        assert gw == 3, "GW4 was a Free Hit, so the squad comes from GW3"
        return {"picks": [{"element": e} for e in range(1, 16)]}

    def transfers(self, team_id):
        return [
            {"element_in": 2, "element_in_cost": 60, "element_out": 99, "element_out_cost": 55, "event": 2,
             "time": "2026-08-20T10:00:00Z"},
            {"element_in": 3, "element_in_cost": 70, "element_out": 98, "element_out_cost": 70, "event": 4,
             "time": "2026-09-01T10:00:00Z"},                                    # Free Hit: ignored
            {"element_in": 16, "element_in_cost": 45, "element_out": 15, "element_out_cost": 66, "event": 5,
             "time": "2026-09-08T10:00:00Z"},                                    # already made for GW5
        ]

    def element_summary(self, element):
        return self.summaries[element]


def test_manager_team_is_reconstructed_from_public_endpoints():
    bootstrap = {
        "events": [{"id": e, "finished": e <= 4, "is_next": e == 5} for e in range(1, 7)],
        "elements": [{"id": e, "now_cost": 64, "cost_change_start": 0} for e in range(1, 20)],
    }
    team = load_manager_team(FakeClient(), 123, bootstrap, max_free_transfers=5)
    assert team.next_gw == 5 and team.squad_event == 3
    assert 15 not in team.squad.ids and 16 in team.squad.ids                 # pending transfer applied
    assert team.squad.bank == 30 + 66 - 45                                   # GW3 bank + pending sale - buy
    assert team.purchase_prices[2] == 60                                     # bought in GW2
    assert team.purchase_prices[3] == 53                                     # Free Hit buy ignored: initial
    assert team.purchase_prices[16] == 45
    assert team.squad.selling_prices[2] == 62                                # 60 -> 64: keep half the rise
    assert team.squad.selling_prices[16] == 54                               # 45 -> 64: 45 + 9
    # GW2: 1 free, 1 used -> GW3: 1 -> GW4 (Free Hit, kept): 2 -> GW5: 3, minus 1 pending = 2
    assert team.squad.free_transfers == 2
