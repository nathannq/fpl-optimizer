"""The two model families behind one small interface: ``fit(X, y, val_mask)`` / ``predict(X)``.

``predict`` returns expected points for regression tasks and P(haul) for binary ones.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class RidgeModel:
    """Linear baseline model: median imputation + missing indicators, scaling, one-hot position.

    Ridge regression for points; L2 logistic regression (C = 1/alpha) for hauls.
    """

    def __init__(self, kind: str, alpha: float, numeric: list[str], categorical: list[str]):
        self.kind = kind
        pre = ColumnTransformer([
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                ("scale", StandardScaler()),
            ]), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ])
        est = Ridge(alpha=alpha) if kind == "regression" else LogisticRegression(C=1 / alpha, max_iter=2000)
        self.pipeline = Pipeline([("pre", pre), ("model", est)])

    def fit(self, X: pd.DataFrame, y: pd.Series, val_mask: pd.Series | None = None) -> "RidgeModel":
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.kind == "binary":
            return self.pipeline.predict_proba(X)[:, 1]
        return self.pipeline.predict(X)


class LGBMModel:
    """Gradient-boosted trees (LightGBM). Missing values and position are handled natively.

    With a validation mask, the number of trees is chosen by early stopping on
    those (later-in-time) rows, then the model is refit on all rows.
    Deterministic for a fixed seed.
    """

    def __init__(self, kind: str, params: dict, seed: int, early_stopping_rounds: int):
        self.kind = kind
        self.params = dict(params)
        self.seed = seed
        self.early_stopping_rounds = early_stopping_rounds
        self.n_estimators_: int | None = None

    def _make(self, n_estimators: int):
        cls = lgb.LGBMRegressor if self.kind == "regression" else lgb.LGBMClassifier
        params = {**self.params, "n_estimators": n_estimators}
        return cls(objective="regression" if self.kind == "regression" else "binary",
                   random_state=self.seed, deterministic=True, force_row_wise=True, n_jobs=4,
                   verbose=-1, **params)

    def fit(self, X: pd.DataFrame, y: pd.Series, val_mask: pd.Series | None = None) -> "LGBMModel":
        max_trees = self.params["n_estimators"]
        if val_mask is not None and val_mask.any() and (~val_mask).any():
            probe = self._make(max_trees)
            probe.fit(X[~val_mask], y[~val_mask], eval_X=X[val_mask], eval_y=y[val_mask],
                      callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)])
            self.n_estimators_ = max(int(probe.best_iteration_ or max_trees), 1)
        else:
            self.n_estimators_ = max_trees
        self.model_ = self._make(self.n_estimators_).fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.kind == "binary":
            return self.model_.predict_proba(X)[:, 1]
        return self.model_.predict(X)

    def contributions(self, X: pd.DataFrame) -> np.ndarray:
        """Per-row SHAP values (TreeSHAP): one column per feature plus the base value last.

        In points for regression, in log-odds for the haul classifier.
        """
        return self.model_.predict(X, pred_contrib=True)
