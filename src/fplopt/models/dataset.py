"""Model inputs, prediction tasks and time-based folds."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fplopt.config import Config
from fplopt.features.build import CATEGORICAL_FEATURES, feature_columns

POSITION_CATEGORIES = ["GK", "DEF", "MID", "FWD"]


@dataclass(frozen=True)
class Task:
    name: str      # short name used in column names, e.g. "points"
    target: str    # target column in the feature table
    kind: str      # "regression" or "binary"


def tasks(cfg: Config) -> list[Task]:
    h = cfg.targets.multi_gw_horizon
    return [
        Task("points", "y_points", "regression"),
        Task("haul", "y_haul", "binary"),
        Task(f"next{h}", f"y_points_next{h}", "regression"),
    ]


def numeric_features(cfg: Config) -> list[str]:
    return [c for c in feature_columns(cfg) if c not in CATEGORICAL_FEATURES]


def design_matrix(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Feature matrix with a fixed category set for position (so train and predict always agree)."""
    X = df[feature_columns(cfg)].copy()
    X["position"] = pd.Categorical(X["position"], categories=POSITION_CATEGORIES)
    return X


@dataclass(frozen=True)
class Fold:
    test_season: str
    train_seasons: tuple[str, ...]


def walk_forward_folds(seasons: list[str], eval_seasons: list[str]) -> list[Fold]:
    """Each evaluation season is predicted by a model trained on every *earlier* season only."""
    ordered = sorted(set(seasons))
    folds = []
    for season in sorted(eval_seasons):
        earlier = tuple(s for s in ordered if s < season)
        if earlier:
            folds.append(Fold(season, earlier))
    return folds


def early_stopping_split(train: pd.DataFrame) -> pd.Series:
    """Boolean mask of the rows used to pick the number of boosting rounds.

    Time-based like everything else: the latest training season, or the last
    quarter of gameweeks when there is only one season.
    """
    seasons = sorted(train["season"].unique())
    if len(seasons) > 1:
        return train["season"] == seasons[-1]
    gws = sorted(train["gw"].unique())
    return train["gw"] >= gws[int(len(gws) * 0.75)]
