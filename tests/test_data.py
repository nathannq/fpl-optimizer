"""Source normalisation: ids, duplicates, unrecorded stats, manager counts, downloads."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from fplopt.config import HistoryConfig
from fplopt.data.schema import mask_invalid_xp, normalise_matches, normalise_players
from fplopt.data.vaastav import download_history
from fplopt.features.tables import managers_by_gameweek

PLAYERS = pd.DataFrame({"id": [1, 2, 3], "code": [10, 20, 30], "web_name": ["A", "B", "Boss"],
                        "element_type": [3, 4, 5]})


def raw_match(element, fixture, gw, **stats):
    base = {"element": element, "fixture": fixture, "round": gw, "kickoff_time": "2024-08-17T14:00:00Z",
            "was_home": True, "opponent_team": 2, "total_points": 2, "minutes": 90,
            "expected_goals": 0.1, "starts": 1, "xP": 2.0}
    return {**base, **stats}


def test_assistant_managers_are_not_players():
    players = normalise_players(PLAYERS)
    assert players["element"].tolist() == [1, 2]
    assert players["position"].tolist() == ["MID", "FWD"]


def test_duplicate_and_non_player_rows_are_dropped():
    players = normalise_players(PLAYERS)
    raw = pd.DataFrame([raw_match(1, 100, 1), raw_match(1, 100, 1), raw_match(2, 100, 1), raw_match(3, 100, 1)])
    m = normalise_matches(raw, players)
    assert sorted(m["element"]) == [1, 2]
    assert "defensive_contribution" in m and m["defensive_contribution"].isna().all()   # missing column -> NaN


def test_stats_that_are_zero_for_everyone_count_as_unrecorded():
    players = normalise_players(PLAYERS)
    raw = pd.DataFrame([
        raw_match(1, 100, 1, expected_goals=0.0, starts=0), raw_match(2, 100, 1, expected_goals=0.0, starts=0),
        raw_match(1, 101, 2, expected_goals=0.0), raw_match(2, 101, 2, expected_goals=0.4),
    ])
    m = normalise_matches(raw, players).set_index(["gw", "element"])
    assert m.loc[1, "expected_goals"].isna().all() and m.loc[1, "starts"].isna().all()
    assert m.loc[(2, 1), "expected_goals"] == 0.0           # a real zero is kept
    assert (m["minutes"] == 90).all()                        # other stats untouched


def test_fpl_xp_is_masked_in_gameweeks_where_it_was_not_recorded():
    m = pd.DataFrame({"gw": [1, 1, 2, 2], "xp_fpl": [0.0, 0.0, 0.0, 3.5]})
    out = mask_invalid_xp(m, min_nonzero_share=0.2)
    assert out.loc[out["gw"] == 1, "xp_fpl"].isna().all()
    assert out.loc[out["gw"] == 2, "xp_fpl"].tolist() == [0.0, 3.5]


def test_manager_estimate_never_looks_ahead():
    # GW1: all 4 teams play -> exact. GW2: two teams blank -> partial count is a lower bound.
    # GW3: full again. GW2 must not borrow from GW3.
    schedule = pd.DataFrame({"season": "s", "gw": [1] * 4 + [2] * 2 + [3] * 4,
                             "team_code": [1, 2, 3, 4, 1, 2, 1, 2, 3, 4]})
    rows = pd.DataFrame({"season": "s", "gw": [1, 1, 2, 3, 3], "is_future": False,
                         "selected": [1500.0, 1500.0, 900.0, 3000.0, 3000.0]})
    est = managers_by_gameweek(rows, schedule)
    assert est[("s", 1)] == 200
    assert est[("s", 2)] == 200          # max(partial 60, last full 200), not interpolated towards GW3
    assert est[("s", 3)] == 400


class FakeResponse:
    def __init__(self, content=b"", payload=None, status=200):
        self.content, self._payload, self.status_code = content, payload, status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeSession:
    def __init__(self):
        self.urls = []

    def get(self, url, timeout=None):
        self.urls.append(url)
        if "api.github.com" in url:
            return FakeResponse(payload={"sha": "a" * 40})
        return FakeResponse(content=f"col\n{url}\n".encode())


def test_download_is_pinned_recorded_and_idempotent(tmp_path):
    cfg = HistoryConfig(repo="vaastav/Fantasy-Premier-League", ref="master", seasons=("2024-25",),
                        files=("players_raw.csv", "gws/merged_gw.csv"), xp_valid_min_nonzero_share=0.2)
    session = FakeSession()
    manifest = download_history(cfg, tmp_path, session=session)
    assert manifest["sha"] == "a" * 40
    assert set(manifest["files"]) == {"2024-25/players_raw.csv", "2024-25/gws/merged_gw.csv"}
    assert all("/" + "a" * 40 + "/" in u for u in session.urls[1:])     # files come from the pinned commit
    assert json.loads((tmp_path / "MANIFEST.json").read_text())["files"] == manifest["files"]

    session.urls.clear()
    download_history(cfg, tmp_path, session=session)
    assert len(session.urls) == 1                                      # only the ref lookup; files skipped
    assert np.all([(tmp_path / f).exists() for f in manifest["files"]])
