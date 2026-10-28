"""Phase 2 evaluation report: walk-forward results against the baselines."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from fplopt.config import Config
from fplopt.evaluate import differential_metrics, error_metrics, top_k_differentials
from fplopt.models.differential import differential_score
from fplopt.reporting import markdown_table

# label -> column, for every projection compared in the differential evaluation
RANKERS = {
    "last-3 average (baseline)": "base_last3",
    "market-implied (ownership + price)": "implied_points",
    "ridge: points": "ridge_points",
    "LightGBM: points": "lgbm_points",
    "LightGBM: P(haul)": "lgbm_haul",
    "LightGBM: edge over market": "edge",
    "differential score": "differential_score",
}


def evaluation_frame(oos: pd.DataFrame, weights: dict[str, float], cfg: Config) -> pd.DataFrame:
    h = cfg.targets.multi_gw_horizon
    ev = oos.copy()
    ev["base_last3"] = ev["pts_r3"].fillna(0) * ev["n_fixtures"]
    ev["base_fpl_xp"] = ev["xp_fpl"]
    ev[f"base_last3_next{h}"] = ev["pts_r3"].fillna(0) * ev["fix3_n"]
    ev["pred_points"], ev["p_haul"] = ev["lgbm_points"], ev["lgbm_haul"]
    ev["edge"] = ev["pred_points"] - ev["implied_points"]
    ev["differential_score"] = differential_score(ev, weights, cfg.differential.ownership_max_pct)
    return ev


def per_gameweek(df: pd.DataFrame, score: str, cfg: Config) -> pd.DataFrame:
    picks = top_k_differentials(df, score, cfg.evaluation.top_k, cfg.differential.ownership_max_pct)
    hit = picks["y_points"] >= cfg.targets.return_points
    return picks.assign(hit=hit).groupby(["season", "gw"]).agg(precision=("hit", "mean"),
                                                                  mean_points=("y_points", "mean"))


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, col: str, n_boot: int = 5000,
                     seed: int = 0) -> dict[str, float]:
    """Difference in per-gameweek means (a - b) with a 95% CI from resampling gameweeks."""
    diff = (a[col] - b[col]).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    boots = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    return {"diff": float(diff.mean()), "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)), "gw_wins": float((diff > 0).mean()),
            "gw_ties": float((diff == 0).mean())}


def accuracy_table(ev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, part in ev.groupby("split", sort=False):
        for label, col in [("last-3 average", "base_last3"), ("ridge", "ridge_points"),
                           ("LightGBM", "lgbm_points")]:
            rows.append({"split": split, "seasons": ", ".join(sorted(part["season"].unique())),
                         "projection": label, **error_metrics(part, col)})
    return pd.DataFrame(rows)


def xp_comparison_table(ev: pd.DataFrame) -> pd.DataFrame:
    """Same rows for everyone: gameweeks where FPL's xP was recorded."""
    has_xp = ev.groupby(["season", "gw"])["xp_fpl"].transform(lambda s: s.notna().any())
    part = ev[has_xp]
    return pd.DataFrame([
        {"projection": label, "gameweeks": part.groupby(["season", "gw"]).ngroups, **error_metrics(part, col)}
        for label, col in [("last-3 average", "base_last3"), ("LightGBM", "lgbm_points"),
                           ("FPL xP as recorded (look-ahead)", "base_fpl_xp")]
    ])


def haul_table(ev: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    for split, part in ev.groupby("split", sort=False):
        d = part.dropna(subset=["y_haul"])
        y = d["y_haul"].astype(int)
        rows.append({"split": split, "model": "base rate", "auc": 0.5,
                     "brier": brier_score_loss(y, np.full(len(y), y.mean())), "log_loss": log_loss(y, np.full(len(y), y.mean())),
                     "haul_rate": y.mean()})
        rows.append({"split": split, "model": "last-3 average as a score", "auc": roc_auc_score(y, d["base_last3"]),
                     "brier": np.nan, "log_loss": np.nan, "haul_rate": y.mean()})
        for label, col in [("logistic (ridge)", "ridge_haul"), ("LightGBM", "lgbm_haul")]:
            p = d[col].clip(1e-6, 1 - 1e-6)
            rows.append({"split": split, "model": label, "auc": roc_auc_score(y, p),
                         "brier": brier_score_loss(y, p), "log_loss": log_loss(y, p), "haul_rate": y.mean()})
    return pd.DataFrame(rows)


def next_gws_table(ev: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    h = cfg.targets.multi_gw_horizon
    rows = []
    for split, part in ev.groupby("split", sort=False):
        for label, col in [(f"last-3 average x fixtures", f"base_last3_next{h}"), ("ridge", f"ridge_next{h}"),
                           ("LightGBM", f"lgbm_next{h}")]:
            rows.append({"split": split, "projection": label, **error_metrics(part, col, target=f"y_points_next{h}")})
    return pd.DataFrame(rows)


def differential_table(ev: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    k, own = cfg.evaluation.top_k, cfg.differential.ownership_max_pct
    ret, haul = cfg.targets.return_points, cfg.targets.haul_points
    rows = []
    for split, part in ev.groupby("split", sort=False):
        for label, col in RANKERS.items():
            rows.append({"split": split, "ranked by": label, **differential_metrics(part, col, k, own, ret, haul)})
    return pd.DataFrame(rows).drop(columns=["pool_mean_points", "pool_return_rate"])


def by_season_table(ev: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """The headline comparison (baseline vs LightGBM points) season by season."""
    k, own = cfg.evaluation.top_k, cfg.differential.ownership_max_pct
    ret, haul = cfg.targets.return_points, cfg.targets.haul_points
    rows = []
    for season, part in ev.groupby("season"):
        for label in ("last-3 average (baseline)", "LightGBM: points"):
            m = differential_metrics(part, RANKERS[label], k, own, ret, haul)
            rows.append({"season": season, "ranked by": label, "gameweeks": m["gameweeks"],
                         f"precision@{k}": m[f"precision@{k}"], "haul_rate": m["haul_rate"],
                         "mean_points": m["mean_points"]})
    return pd.DataFrame(rows)


def significance_table(ev: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    for split, part in ev.groupby("split", sort=False):
        base = per_gameweek(part, "base_last3", cfg)
        for label in ("LightGBM: points", "LightGBM: P(haul)", "differential score"):
            mine = per_gameweek(part, RANKERS[label], cfg)
            for metric in ("precision", "mean_points"):
                rows.append({"split": split, "ranked by": label, "metric": metric,
                             **paired_bootstrap(mine, base.reindex(mine.index), metric, seed=cfg.model.seed)})
    return pd.DataFrame(rows)


def calibration_table(ev: pd.DataFrame) -> pd.DataFrame:
    part = ev[ev["split"] == "test"]
    bins = [-np.inf, 0.5, 1, 2, 3, 4, 5, 6, np.inf]
    labels = ["< 0.5", "0.5-1", "1-2", "2-3", "3-4", "4-5", "5-6", "6+"]
    grouped = part.groupby(pd.cut(part["lgbm_points"], bins, labels=labels), observed=True)
    return grouped.agg(rows=("y_points", "size"), mean_predicted=("lgbm_points", "mean"),
                       mean_actual=("y_points", "mean")).reset_index(names="predicted points")


def model_report(ev: pd.DataFrame, weights: dict[str, float], grid: pd.DataFrame,
                 importance: dict[str, pd.DataFrame], trees: dict[str, int], cfg: Config) -> str:
    k, own = cfg.evaluation.top_k, cfg.differential.ownership_max_pct
    ret, haul, h = cfg.targets.return_points, cfg.targets.haul_points, cfg.targets.multi_gw_horizon
    pool = ev.dropna(subset=["y_points"])
    pool = pool[pool["ownership_pct"] < own].groupby("split", sort=False)["y_points"].agg(
        mean_points="mean", return_rate=lambda s: (s >= ret).mean()).reset_index()
    w = ", ".join(f"{n} = {v:g}" for n, v in weights.items())
    return "\n".join([
        "# Phase 2 - the model",
        "",
        "## Setup",
        "",
        f"- **Walk-forward by season.** Each season is predicted by models trained only on earlier seasons "
        f"(dev: {', '.join(cfg.model.dev_seasons)}; test: {', '.join(cfg.model.test_seasons)}). The number of "
        "boosting rounds is chosen by early stopping on the latest *training* season. Nothing is shuffled.",
        "- **Dev vs test.** The differential-score weights were chosen on the dev season only. The test seasons "
        "were not used for any decision.",
        f"- **Targets:** points next GW (regression), haul = {haul}+ points (classifier), points over the next "
        f"{h} GWs (regression).",
        f"- **Models:** ridge / L2-logistic (median imputation + missing indicators, scaled) and LightGBM "
        f"(seed {cfg.model.seed}, deterministic). Final-model tree counts: "
        + ", ".join(f"{k_} {v}" for k_, v in trees.items()) + ".",
        f"- **Differentials:** owned by < {own:g}% going into the GW; top {k} per GW; a return is {ret}+ points.",
        "",
        "## Points next gameweek",
        "",
        markdown_table(accuracy_table(ev)),
        "",
        "Same rows, restricted to gameweeks where FPL's xP was recorded. That xP was captured *after* each "
        "gameweek (see Phase 1), so it is an optimistic reference, not a fair competitor:",
        "",
        markdown_table(xp_comparison_table(ev)),
        "",
        "A fair comparison with FPL's `ep_next` needs projections captured before the deadline. The live "
        "pipeline archives them from 2026-27 GW9 onwards (`data/snapshots/`), so that comparison will grow "
        "week by week.",
        "",
        "Calibration of the LightGBM points model on the test seasons:",
        "",
        markdown_table(calibration_table(ev)),
        "",
        f"## Haul probability ({haul}+ points)",
        "",
        markdown_table(haul_table(ev, cfg)),
        "",
        f"## Points over the next {h} gameweeks",
        "",
        markdown_table(next_gws_table(ev, cfg)),
        "",
        "## Differential score",
        "",
        f"score = predicted points + haul_weight x P(haul) + edge_weight x (predicted - market-implied points) "
        f"- price_weight x price, for players owned by < {own:g}%.",
        "",
        f"Chosen on the dev season: **{w}**. Best weight combinations on dev:",
        "",
        markdown_table(grid.head(8)),
        "",
        f"## Differentials: top {k} per gameweek",
        "",
        markdown_table(differential_table(ev, cfg)),
        "",
        "For reference, an average differential (no model) in the same pool:",
        "",
        markdown_table(pool),
        "",
        "Season by season:",
        "",
        markdown_table(by_season_table(ev, cfg)),
        "",
        "### Is it better than the baseline?",
        "",
        "Paired by gameweek against the last-3 average's top 10. `diff` is the mean per-gameweek difference; "
        "the 95% CI comes from resampling gameweeks. `gw_wins` and `gw_ties` are the shares of gameweeks "
        "that were better or equal.",
        "",
        markdown_table(significance_table(ev, cfg)),
        "",
        "## What drives the predictions",
        "",
        "Mean absolute SHAP value (LightGBM's TreeSHAP) on the final model, over the last two seasons. "
        "Points model in points; haul model in log-odds.",
        "",
        "### Points model",
        "",
        markdown_table(importance["points"].head(15)),
        "",
        "### Haul model",
        "",
        markdown_table(importance["haul"].head(15)),
        "",
    ])
