"""Phase 4: the web API end to end, with a fake FPL client and synthetic live data."""
from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest
import requests
from fastapi.testclient import TestClient

from fplopt.optimize.rules import MAX_PER_CLUB, SQUAD_QUOTAS, valid_formation
from fplopt.web.app import create_app
from synthetic import make_pool

NEXT_GW = 9


@pytest.fixture(scope="module")
def league_pool():
    pool = make_pool(seed=11)
    pool.loc[pool.index[:3], "status"] = "i"              # three injured players
    return pool


class FakeFPL:
    workers = 2

    def __init__(self, pool: pd.DataFrame):
        self.pool = pool
        best = pool.sort_values("projection", ascending=False)
        # a legal squad: the cheapest 2/5/5/3 from distinct clubs
        squad = []
        for pos, n in SQUAD_QUOTAS.items():
            squad += pool[pool["position"] == pos].sort_values("price").index[:n].tolist()
        self.squad = squad
        self.rival = best.index[:15].tolist()

    def bootstrap(self):
        return {"events": [{"id": e, "finished": e < NEXT_GW, "is_next": e == NEXT_GW,
                            "deadline_time": "2026-10-31T11:00:00Z"} for e in range(1, 12)],
                "elements": [{"id": int(i), "now_cost": int(r.price), "cost_change_start": 0}
                             for i, r in self.pool.iterrows()]}

    def cache_entry(self, path):
        return {"fetched_at": "2026-10-28T10:00:00+00:00"}

    def fixtures(self):
        return [{"event": gw, "team_h": h, "team_a": h + 10, "team_h_difficulty": 2, "team_a_difficulty": 4,
                 "kickoff_time": f"2026-11-0{gw - 8}T15:00:00Z"} for gw in (9, 10, 11) for h in range(1, 11)]

    def _check(self, team_id):
        if team_id == 404:
            resp = requests.Response()
            resp.status_code = 404
            raise requests.HTTPError("not found", response=resp)

    def entry(self, team_id):
        self._check(team_id)
        return {"started_event": 1, "name": "Test XI", "player_first_name": "Ann", "player_last_name": "Ager"}

    def entry_history(self, team_id):
        return {"current": [{"event": e, "bank": 20, "event_transfers": 0} for e in range(1, NEXT_GW)], "chips": []}

    def picks(self, team_id, gw):
        return {"picks": [{"element": int(e)} for e in (self.squad if team_id == 1 else self.rival)]}

    def transfers(self, team_id):
        return []

    def element_summary(self, element):
        return {"history": [{"round": 1, "value": int(self.pool.at[element, "price"])}]}

    def league_standings(self, league_id, page=1):
        self._check(league_id)
        return {"league": {"name": "Office League"}, "standings": {"results": [
            {"entry": e, "entry_name": f"Team {e}", "player_name": f"Manager {e}", "rank": k + 1,
             "last_rank": k + 1, "event_total": 50, "total": 600 - k} for k, e in enumerate([1, 2, 3])]}}


@pytest.fixture(scope="module")
def client(cfg, tmp_path_factory, league_pool):
    root = tmp_path_factory.mktemp("web")
    paths = replace(cfg.paths, processed=root / "processed", models=root / "models")
    paths.processed.mkdir()
    pool = league_pool
    live = pd.DataFrame({
        "element": pool.index, "code": pool.index + 1000, "web_name": pool["name"], "position": pool["position"],
        "team_id": pool["club"], "team_short": pool["team"], "now_cost": pool["price"],
        "selected_by_percent": pool["ownership"], "status": pool["status"],
        "chance_of_playing_next_round": [0.0 if s == "i" else float("nan") for s in pool["status"]],
        "ep_next": pool["projection"] * 0.9, "news": "", "season": "2026-27", "next_gw": NEXT_GW,
    })
    live.to_parquet(paths.live_players, index=False)
    preds = pd.DataFrame({
        "code": live["code"], "pred_points": pool["projection"].where(pool["status"] == "a", 0.0).to_numpy(),
        "p_haul": 0.1, "reasons": "xGI per match, last 5 0.5 (+0.3)", "differential_score": pool["projection"].to_numpy(),
        "edge": 0.0,
    })
    preds.to_parquet(paths.predictions, index=False)
    app = create_app(replace(cfg, paths=paths), client=FakeFPL(pool))
    return TestClient(app)


def test_index_serves_the_single_page_app(client):
    r = client.get("/")
    assert r.status_code == 200 and "/static/app.js" in r.text
    assert client.get("/static/styles.css").status_code == 200


def test_status(client, cfg):
    s = client.get("/api/status").json()
    assert s["gameweek"] == NEXT_GW and s["deadline"] == "2026-10-31T11:00:00Z"
    assert s["ownership_max_pct"] == cfg.differential.ownership_max_pct
    assert s["projections"] == ["ep_next", "model"] and s["predictions_ready"]


def test_differentials_are_low_owned_fit_and_ranked(client, league_pool):
    data = client.get("/api/differentials?limit=10&max_ownership=10").json()
    players = data["players"]
    assert 0 < len(players) <= 10
    assert all(p["ownership"] <= 10 and p["status"] == "a" for p in players)   # shown to 1 decimal
    projections = [p["projection"] for p in players]
    assert projections == sorted(projections, reverse=True)
    assert players[0]["fixtures"][0] == {"gw": NEXT_GW, "opponent": players[0]["fixtures"][0]["opponent"],
                                         "home": players[0]["fixtures"][0]["home"],
                                         "difficulty": players[0]["fixtures"][0]["difficulty"]}
    mids = client.get("/api/differentials?position=MID&max_ownership=50").json()["players"]
    assert mids and {p["position"] for p in mids} == {"MID"}
    assert client.get("/api/differentials?position=XX").status_code == 422


def _assert_plan(plan: dict) -> None:
    squad = plan["xi"] + plan["bench"]
    assert len(plan["xi"]) == 11 and len(squad) == 15
    assert valid_formation([p["position"] for p in plan["xi"]])
    assert {p["position"]: sum(q["position"] == p["position"] for q in squad) for p in squad} == SQUAD_QUOTAS
    assert max(pd.Series([p["team"] for p in squad]).value_counts()) <= MAX_PER_CLUB
    assert plan["captain"]["id"] in {p["id"] for p in plan["xi"]}
    assert all(p["status"] == "a" for p in plan["xi"])


def test_team_recommendation(client):
    r = client.get("/api/team/1?max_transfers=2")
    assert r.status_code == 200
    res = r.json()
    assert res["mode"] == "transfers" and res["team"]["name"] == "Test XI"
    assert [o["n_transfers"] for o in res["options"]] == [0, 1, 2]
    assert 0 <= res["recommended"] < len(res["options"])
    for plan in res["options"]:
        _assert_plan(plan)
        assert len(plan["transfers"]) == plan["n_transfers"]
    assert res["options"][0]["xi"][0]["fixtures"], "players carry their upcoming fixtures"
    ep = client.get("/api/team/1?projection=ep_next").json()
    assert ep["projection"] == "ep_next"


def test_new_squad(client, cfg):
    res = client.get("/api/squad").json()
    plan = res["options"][0]
    _assert_plan(plan)
    cost = sum(p["price"] for p in plan["xi"] + plan["bench"])
    assert cost + plan["bank"] == pytest.approx(cfg.optimizer.budget / 10)


def test_league_view(client, league_pool):
    res = client.get("/api/league/77?team_id=1&sample=3").json()
    assert res["league"]["name"] == "Office League"
    assert [s["is_you"] for s in res["standings"]] == [True, False, False]
    assert res["sampled_entries"] == 2                      # rivals: entries 2 and 3
    top = res["template"][0]
    assert top["league_ownership"] == 100.0                 # both rivals own the same squad
    assert {p["id"] for p in res["you"]["threats"]}.isdisjoint(FakeFPL(league_pool).squad)
    assert all(p["league_ownership"] <= 10 for p in res["untapped"])


def test_errors_are_clean_json(client, cfg, tmp_path):
    assert client.get("/api/team/404").status_code == 404
    assert client.get("/api/league/404").json()["detail"].startswith("FPL has no team or league")
    empty = TestClient(create_app(replace(cfg, paths=replace(cfg.paths, processed=tmp_path)), client=None))
    r = empty.get("/api/differentials")
    assert r.status_code == 503 and "refresh" in r.json()["detail"]
