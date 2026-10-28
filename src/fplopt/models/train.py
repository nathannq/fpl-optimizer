"""Training: walk-forward out-of-sample predictions, and the final models used for live picks."""
from __future__ import annotations

import json
import logging
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import pandas as pd

from fplopt.config import Config
from fplopt.features.build import CATEGORICAL_FEATURES, feature_columns
from fplopt.models.dataset import (
    Task,
    design_matrix,
    early_stopping_split,
    numeric_features,
    tasks,
    walk_forward_folds,
)
from fplopt.models.differential import MarketModel
from fplopt.models.estimators import LGBMModel, RidgeModel

log = logging.getLogger(__name__)

FAMILIES = ("ridge", "lgbm")
# Columns carried from the feature table into prediction frames.
CARRY = ["season", "gw", "code", "element", "web_name", "position", "team_code", "team_short",
         "price", "ownership_pct", "n_fixtures", "pts_r3", "mins_r5", "fix3_n", "is_future"]


def make_model(family: str, task: Task, cfg: Config):
    if family == "ridge":
        return RidgeModel(task.kind, cfg.model.ridge_alpha, numeric_features(cfg), CATEGORICAL_FEATURES)
    return LGBMModel(task.kind, cfg.model.lightgbm, cfg.model.seed, cfg.model.early_stopping_rounds)


@dataclass
class ModelBundle:
    """Every model trained on one training set, plus the market model and metadata."""

    models: dict[str, object] = field(default_factory=dict)      # "<family>_<task>" -> model
    market: MarketModel | None = None
    meta: dict = field(default_factory=dict)

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump({"models": self.models, "market": self.market}, folder / "models.joblib")
        (folder / "meta.json").write_text(json.dumps(self.meta, indent=2, default=str) + "\n")

    @classmethod
    def load(cls, folder: Path) -> "ModelBundle":
        if not (folder / "models.joblib").exists():
            raise FileNotFoundError(f"No trained models in {folder} - run `fplopt train` first")
        blob = joblib.load(folder / "models.joblib")
        return cls(blob["models"], blob["market"], json.loads((folder / "meta.json").read_text()))


def fit_bundle(train: pd.DataFrame, cfg: Config) -> ModelBundle:
    bundle = ModelBundle()
    for task in tasks(cfg):
        rows = train[train[task.target].notna()].reset_index(drop=True)
        X, y = design_matrix(rows, cfg), rows[task.target]
        val_mask = early_stopping_split(rows)
        for family in FAMILIES:
            bundle.models[f"{family}_{task.name}"] = make_model(family, task, cfg).fit(X, y, val_mask)
    bundle.market = MarketModel().fit(train)
    bundle.meta = {
        "train_seasons": sorted(train["season"].unique()),
        "train_rows": int(len(train)),
        "lgbm_trees": {k: m.n_estimators_ for k, m in bundle.models.items() if isinstance(m, LGBMModel)},
    }
    return bundle


def predict_frame(bundle: ModelBundle, df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Predictions from every model for the rows of ``df`` (keys and context carried along)."""
    X = design_matrix(df, cfg)
    out = df[[c for c in CARRY if c in df.columns]].copy()
    for name, model in bundle.models.items():
        out[name] = model.predict(X)
    out["implied_points"] = bundle.market.predict(df)
    return out


def walk_forward(table: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Out-of-sample predictions for the dev and test seasons, one fold per season."""
    labelled = table[~table["is_future"]]
    eval_seasons = [*cfg.model.dev_seasons, *cfg.model.test_seasons]
    targets = [t.target for t in tasks(cfg)]
    parts = []
    for fold in walk_forward_folds(labelled["season"].unique().tolist(), eval_seasons):
        log.info("Fold: train on %s -> predict %s", ", ".join(fold.train_seasons), fold.test_season)
        bundle = fit_bundle(labelled[labelled["season"].isin(fold.train_seasons)], cfg)
        test = labelled[labelled["season"] == fold.test_season]
        preds = predict_frame(bundle, test, cfg)
        preds[targets + ["xp_fpl"]] = test[targets + ["xp_fpl"]].to_numpy()
        preds["split"] = "dev" if fold.test_season in cfg.model.dev_seasons else "test"
        parts.append(preds)
    return pd.concat(parts, ignore_index=True)


def fit_final(table: pd.DataFrame, cfg: Config, weights: dict[str, float]) -> ModelBundle:
    """Train on every labelled row (all seasons, current season to date) for live predictions."""
    bundle = fit_bundle(table[~table["is_future"]], cfg)
    bundle.meta.update({
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "differential_weights": weights,
        "ownership_max_pct": cfg.differential.ownership_max_pct,
        "features": feature_columns(cfg),
        "seed": cfg.model.seed,
        "lightgbm_params": cfg.model.lightgbm,
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ("lightgbm", "scikit-learn", "pandas", "numpy")},
        "data_manifest_sha": _manifest_sha(cfg),
    })
    return bundle


def _manifest_sha(cfg: Config) -> str | None:
    manifest = cfg.paths.raw_history / "MANIFEST.json"
    return json.loads(manifest.read_text()).get("sha") if manifest.exists() else None
