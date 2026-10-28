"""FPL squad rules and money/transfer accounting. Prices are in tenths of £m (FPL's own unit)."""
from __future__ import annotations

from collections.abc import Iterable

SQUAD_QUOTAS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
SQUAD_SIZE = sum(SQUAD_QUOTAS.values())
XI_SIZE = 11
XI_MIN = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
XI_MAX = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_CLUB = 3
UNAVAILABLE = {"i", "s", "u", "n"}     # injured, suspended, unavailable (left), not eligible (loan)
TRANSFER_KEEPING_CHIPS = {"wildcard", "freehit"}


def selling_price(purchase: int, current: int) -> int:
    """FPL's selling price: you keep half of any rise (rounded down to £0.1m) and take every fall."""
    if current <= purchase:
        return current
    return purchase + (current - purchase) // 2


def infer_free_transfers(transfers_by_event: dict[int, int], chips_by_event: dict[int, str],
                         started_event: int, next_event: int, max_free: int) -> int:
    """Free transfers available for ``next_event``, replayed from the manager's history.

    One free transfer is added before every gameweek after the first (capped at
    ``max_free``); transfers made use them up. In a Wildcard or Free Hit week the
    transfers are free and saved transfers are kept (FPL's rule since 2024-25).
    Special one-off top-ups are not modelled, so callers allow an override.
    """
    free = 0
    for event in range(started_event + 1, next_event + 1):
        free = min(free + 1, max_free)
        if event == next_event:
            break
        if chips_by_event.get(event) in TRANSFER_KEEPING_CHIPS:
            continue
        free = max(free - transfers_by_event.get(event, 0), 0)
    return free


def is_available(status: str | None, chance: float | None, min_chance: float) -> bool:
    """Can a player be bought or started? FPL's flag if set, otherwise his status."""
    if chance is not None and chance == chance:      # not NaN
        return chance >= min_chance
    return status not in UNAVAILABLE


def valid_formation(positions: Iterable[str]) -> bool:
    counts = {p: 0 for p in XI_MIN}
    for p in positions:
        counts[p] += 1
    return (sum(counts.values()) == XI_SIZE
            and all(XI_MIN[p] <= counts[p] <= XI_MAX[p] for p in counts))
