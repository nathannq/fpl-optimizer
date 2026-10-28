"""Data for the web pages, kept separate from the HTTP layer so it can be tested directly."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from fplopt.data.api import FPLClient

FIXTURES_AHEAD = 3


def upcoming_fixtures(fixtures: list[dict], teams: dict[int, str], from_gw: int,
                      n: int = FIXTURES_AHEAD) -> dict[int, list[dict]]:
    """team id -> one entry per gameweek ``from_gw .. from_gw+n-1`` (several for a double, none for a blank).

    ``difficulty`` is FPL's published fixture difficulty (display only; the model doesn't use it).
    """
    out: dict[int, list[dict]] = {t: [] for t in teams}
    for f in sorted(fixtures, key=lambda f: (f.get("event") or 99, f.get("kickoff_time") or "")):
        gw = f.get("event")
        if gw is None or not from_gw <= gw < from_gw + n:
            continue
        for team, opp, home, diff in ((f["team_h"], f["team_a"], True, f.get("team_h_difficulty")),
                                      (f["team_a"], f["team_h"], False, f.get("team_a_difficulty"))):
            out.setdefault(team, []).append({"gw": gw, "opponent": teams.get(opp, "?"), "home": home,
                                             "difficulty": diff})
    return out


def player_card(row: pd.Series, fixtures: dict[int, list[dict]]) -> dict:
    """The fields every page shows for a player."""
    def num(v, digits=2):
        return None if pd.isna(v) else round(float(v), digits)
    return {
        "id": int(row["element"]), "name": row["web_name"], "team": row["team_short"],
        "position": row["position"], "price": num(row["now_cost"] / 10, 1),
        "ownership": num(row["selected_by_percent"], 1),
        "status": row["status"], "news": row["news"] or "",
        "projection": num(row.get("pred_points")), "p_haul": num(row.get("p_haul"), 3),
        "ep_next": num(row.get("ep_next")), "reasons": row.get("reasons") or "",
        "fixtures": fixtures.get(int(row["team_id"]), []),
    }


def merged_players(players: pd.DataFrame, predictions: pd.DataFrame | None) -> pd.DataFrame:
    """Live players with this week's model outputs (players with a blank gameweek get 0)."""
    if predictions is None:
        return players.assign(pred_points=pd.NA, p_haul=pd.NA, reasons="", differential_score=pd.NA)
    cols = ["code", "pred_points", "p_haul", "reasons", "differential_score", "edge"]
    out = players.merge(predictions[cols], on="code", how="left")
    out[["pred_points", "p_haul"]] = out[["pred_points", "p_haul"]].fillna(0.0)
    return out


def differentials(players: pd.DataFrame, fixtures: dict[int, list[dict]], max_ownership: float,
                  limit: int, position: str | None = None, max_price: float | None = None) -> list[dict]:
    """Top players owned by fewer than ``max_ownership``% who are fit to play, by predicted points."""
    pool = players[(players["selected_by_percent"] < max_ownership) & (players["pred_points"] > 0)]
    if position:
        pool = pool[pool["position"] == position]
    if max_price is not None:
        pool = pool[pool["now_cost"] <= max_price * 10]
    pool = pool.sort_values(["pred_points", "element"], ascending=[False, True]).head(limit)
    return [player_card(r, fixtures) for _, r in pool.iterrows()]


def league_view(client: FPLClient, league_id: int, gameweek: int, players: pd.DataFrame,
                fixtures: dict[int, list[dict]], team_id: int | None = None, sample: int = 20,
                max_league_ownership: float = 10.0) -> dict:
    """Standings plus who the league owns: its template, its threats to you, and what nobody has.

    League ownership comes from the current squads of the top ``sample`` entries (plus yours).
    """
    data = client.league_standings(league_id)
    results = data["standings"]["results"]
    entries = [r["entry"] for r in results[:sample]]
    you_in_sample = team_id in entries
    to_fetch = entries + ([team_id] if team_id and not you_in_sample else [])
    with ThreadPoolExecutor(max_workers=client.workers) as pool:
        squads = dict(zip(to_fetch, pool.map(lambda e: [p["element"] for p in client.picks(e, gameweek)["picks"]],
                                             to_fetch)))
    rivals = [e for e in entries if e != team_id]
    counts = Counter(el for e in rivals for el in squads[e])
    by_id = players.set_index("element")

    def card(el: int) -> dict:
        c = player_card(pd.concat([pd.Series({"element": el}), by_id.loc[el]]), fixtures)
        c["league_ownership"] = round(100 * counts.get(el, 0) / max(len(rivals), 1), 1)
        return c

    known = [el for el in counts if el in by_id.index]
    template = sorted(known, key=lambda el: (-counts[el], -by_id.at[el, "pred_points"]))[:15]
    rare = by_id[(by_id["pred_points"] > 0)].index
    untapped = sorted([el for el in rare if 100 * counts.get(el, 0) / max(len(rivals), 1) <= max_league_ownership],
                      key=lambda el: -by_id.at[el, "pred_points"])[:10]
    view = {
        "league": {"id": league_id, "name": data["league"].get("name", f"League {league_id}")},
        "gameweek": gameweek,
        "sampled_entries": len(rivals),
        "standings": [{"rank": r["rank"], "last_rank": r["last_rank"], "entry": r["entry"],
                       "entry_name": r["entry_name"], "manager": r["player_name"], "gw_points": r["event_total"],
                       "total": r["total"], "is_you": r["entry"] == team_id} for r in results],
        "template": [card(el) for el in template],
        "untapped": [card(el) for el in untapped],
    }
    if team_id:
        mine = [el for el in squads[team_id] if el in by_id.index]
        view["you"] = {
            "entry": team_id,
            "your_differentials": sorted((card(el) for el in mine), key=lambda c: c["league_ownership"])[:8],
            "threats": sorted((card(el) for el in known if el not in mine),
                              key=lambda c: -(c["league_ownership"] / 100) * (c["projection"] or 0))[:8],
        }
    return view
