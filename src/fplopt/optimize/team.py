"""A manager's current squad from the public FPL API (no login needed).

The public endpoints don't expose selling prices, so they are reconstructed:
each player's purchase price is the cost of the latest transfer that brought him
in (or his price when the manager joined, for the original squad), and FPL's
selling-price rule is applied to today's price. Transfers made inside a Free Hit
week are ignored, because the squad reverts afterwards.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from fplopt.data.api import FPLClient
from fplopt.optimize.rules import infer_free_transfers, selling_price
from fplopt.optimize.solver import Squad

log = logging.getLogger(__name__)


@dataclass
class ManagerTeam:
    team_id: int
    team_name: str
    manager: str
    next_gw: int
    squad_event: int                    # gameweek whose picks are the current squad
    squad: Squad
    purchase_prices: dict[int, int]
    pending_transfers: list[dict] = field(default_factory=list)   # already made for next_gw
    chips_used: list[str] = field(default_factory=list)
    free_transfers_inferred: bool = True


def _initial_price(client: FPLClient, element: int, event: int, bootstrap_player: dict) -> int:
    """Price when the manager picked his first squad (gameweek ``event``)."""
    for row in client.element_summary(element).get("history", []):
        if row["round"] == event:
            return int(row["value"])
    return int(bootstrap_player["now_cost"] - bootstrap_player.get("cost_change_start", 0))


def load_manager_team(client: FPLClient, team_id: int, bootstrap: dict, max_free_transfers: int,
                      free_transfers: int | None = None, bank: int | None = None) -> ManagerTeam:
    events = bootstrap["events"]
    next_gw = next(e["id"] for e in events if e.get("is_next"))
    last_finished = max(e["id"] for e in events if e.get("finished"))
    players = {p["id"]: p for p in bootstrap["elements"]}

    entry = client.entry(team_id)
    history = client.entry_history(team_id)
    chips = {c["event"]: c["name"] for c in history.get("chips", [])}
    free_hit = {e for e, name in chips.items() if name == "freehit"}
    played = {row["event"]: row for row in history["current"]}

    # The squad going into next_gw is the latest non-Free-Hit squad.
    squad_event = max(e for e in played if e <= last_finished and e not in free_hit)
    picks = [p["element"] for p in client.picks(team_id, squad_event)["picks"]]
    money = played[squad_event]["bank"]

    transfers = sorted(client.transfers(team_id), key=lambda t: t["time"])
    pending = [t for t in transfers if t["event"] == next_gw]
    for t in pending:                       # transfers already made for the coming gameweek
        picks = [t["element_in"] if p == t["element_out"] else p for p in picks]
        money += t["element_out_cost"] - t["element_in_cost"]

    started = entry["started_event"]
    purchase = {}
    for element in picks:
        bought = [t for t in transfers if t["element_in"] == element
                  and t["event"] <= next_gw and t["event"] not in free_hit]
        purchase[element] = (int(bought[-1]["element_in_cost"]) if bought
                             else _initial_price(client, element, started, players[element]))
    selling = {e: selling_price(purchase[e], int(players[e]["now_cost"])) for e in picks}

    inferred = free_transfers is None
    if inferred:
        made = {e: row["event_transfers"] for e, row in played.items()}
        free_transfers = max(infer_free_transfers(made, chips, started, next_gw, max_free_transfers)
                             - len(pending), 0)
    return ManagerTeam(
        team_id=team_id,
        team_name=entry.get("name", ""),
        manager=f"{entry.get('player_first_name', '')} {entry.get('player_last_name', '')}".strip(),
        next_gw=next_gw,
        squad_event=squad_event,
        squad=Squad(selling_prices=selling, bank=int(money if bank is None else bank),
                    free_transfers=int(free_transfers)),
        purchase_prices=purchase,
        pending_transfers=pending,
        chips_used=[f"{name} (GW{e})" for e, name in sorted(chips.items())],
        free_transfers_inferred=inferred,
    )
