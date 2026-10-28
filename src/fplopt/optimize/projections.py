"""Swappable projection sources and the live player pool the solver works on.

A projection source turns the current player list into expected points for the
next gameweek. The optimizer only reads the result, so sources can be swapped
(``--projection model`` / ``--projection ep_next``) and backtested against each other.
"""
from __future__ import annotations

from typing import Protocol

import pandas as pd

from fplopt.config import Config
from fplopt.optimize.rules import is_available
from fplopt.optimize.solver import Squad


class ProjectionSource(Protocol):
    name: str

    def project(self, players: pd.DataFrame) -> pd.Series:
        """Expected points next gameweek, indexed like ``players`` (0 for a blank gameweek)."""
        ...


class ModelProjection:
    """This project's LightGBM model (``fplopt predict``), already scaled by chance of playing."""

    name = "model"

    def __init__(self, cfg: Config):
        self.path = cfg.paths.predictions

    def project(self, players: pd.DataFrame) -> pd.Series:
        if not self.path.exists():
            raise FileNotFoundError(f"{self.path} not found - run `fplopt predict` first")
        preds = pd.read_parquet(self.path).set_index("code")["pred_points"]
        return players["code"].map(preds).fillna(0.0).rename("projection")


class EpNextProjection:
    """FPL's own ``ep_next`` from the live bootstrap (it already allows for chance of playing)."""

    name = "ep_next"

    def __init__(self, cfg: Config):
        pass

    def project(self, players: pd.DataFrame) -> pd.Series:
        return players["ep_next"].fillna(0.0).astype(float).rename("projection")


SOURCES: dict[str, type] = {"model": ModelProjection, "ep_next": EpNextProjection}


def get_source(name: str, cfg: Config) -> ProjectionSource:
    if name not in SOURCES:
        raise ValueError(f"unknown projection source {name!r}; choose from {sorted(SOURCES)}")
    return SOURCES[name](cfg)


def objective_points(projection: pd.Series, ownership: pd.Series, cfg: Config) -> pd.Series:
    """Projected points with the optional differential bonus applied (used only as the objective)."""
    is_diff = ownership < cfg.differential.ownership_max_pct
    return projection * (1 + cfg.optimizer.differential_bonus * is_diff)


def live_pool(players: pd.DataFrame, source: ProjectionSource, cfg: Config,
              squad: Squad | None = None) -> pd.DataFrame:
    """The solver's player pool from ``live_players.parquet`` rows, indexed by element id."""
    min_chance = cfg.optimizer.min_chance_of_playing
    available = pd.Series([is_available(st, ch, min_chance) for st, ch in
                           zip(players["status"], players["chance_of_playing_next_round"])], index=players.index)
    pool = pd.DataFrame({
        "name": players["web_name"],
        "position": players["position"],
        "club": players["team_id"],
        "team": players["team_short"],
        "price": players["now_cost"].astype(int),
        "ownership": players["selected_by_percent"].astype(float),
        "projection": source.project(players),
        "can_buy": available,
        "can_start": available,
        "status": players["status"],
        "news": players["news"],
    })
    pool.index = players["element"].astype(int).to_numpy()
    pool["objective"] = objective_points(pool["projection"], pool["ownership"], cfg)
    if squad:
        pool["selling_price"] = pd.Series(squad.selling_prices).reindex(pool.index)
    return pool
