"""Plain-text rendering of an optimizer result (the dict from ``service.optimize``)."""
from __future__ import annotations


def _p(p: dict) -> str:
    flag = "" if p["status"] == "a" else f" [{p['status']}]"
    return f"{p['name']} ({p['team']}, £{p['price']:.1f}m, {p['projection']:.1f}){flag}"


def render(result: dict) -> str:
    lines = [f"GW{result['gameweek']} - projection: {result['projection']}"
             + (f", differential bonus +{result['differential_bonus']:.0%}" if result["differential_bonus"] else "")]
    if result["mode"] == "transfers":
        t = result["team"]
        ft = f"{t['free_transfers']} free transfer(s)" + (" (inferred; override with --free-transfers)"
                                                          if t["free_transfers_inferred"] else "")
        lines += [f"{t['name']} ({t['manager']}) - bank £{t['bank']:.1f}m, {ft}"]
        if t["pending_transfers"]:
            lines.append(f"Includes {t['pending_transfers']} transfer(s) already made for this gameweek.")
        lines += ["", "Transfer options:", f"  {'n':>2}  {'hits':>4}  {'projected':>9}  {'net':>6}  {'vs 0':>6}  moves"]
        base = result["options"][0]["net_points"]
        for k, o in enumerate(result["options"]):
            moves = "; ".join(f"{m['out']['name']} -> {m['in']['name']}" for m in o["transfers"]) or "roll"
            mark = " <- recommended" if k == result["recommended"] else ""
            lines.append(f"  {o['n_transfers']:>2}  {o['hits']:>4}  {o['projected_points']:>9.1f}  "
                         f"{o['net_points']:>6.1f}  {o['net_points'] - base:>+6.1f}  {moves}{mark}")
    plan = result["options"][result["recommended"]]
    lines += ["", f"Starting XI ({plan['formation']}), expected {plan['projected_points']:.1f} pts"
              + (f", {plan['net_points']:.1f} after hits" if plan["hits"] else "") + f"; bank £{plan['bank']:.1f}m"]
    for pos in ("GK", "DEF", "MID", "FWD"):
        players = [x for x in plan["xi"] if x["position"] == pos]
        lines.append(f"  {pos:<3} " + ", ".join(_p(x) for x in players))
    lines += [f"  Captain: {_p(plan['captain'])}   Vice: {plan['vice_captain']['name']}",
              "  Bench:   " + ", ".join(f"{k}. {_p(x)}" for k, x in enumerate(plan["bench"], start=1))]
    return "\n".join(lines)
