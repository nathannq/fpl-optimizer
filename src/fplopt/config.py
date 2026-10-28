"""Project configuration: one YAML file loaded into typed, read-only dataclasses."""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "config.yaml"


@dataclass(frozen=True)
class Paths:
    raw_history: Path
    api_cache: Path
    snapshots: Path
    processed: Path
    models: Path
    reports: Path

    @property
    def features(self) -> Path:
        return self.processed / "features.parquet"

    @property
    def live_players(self) -> Path:
        return self.processed / "live_players.parquet"

    @property
    def oos_predictions(self) -> Path:
        return self.processed / "oos_predictions.parquet"

    @property
    def predictions(self) -> Path:
        return self.processed / "predictions.parquet"


@dataclass(frozen=True)
class HistoryConfig:
    repo: str
    ref: str
    seasons: tuple[str, ...]
    files: tuple[str, ...]
    xp_valid_min_nonzero_share: float


@dataclass(frozen=True)
class LiveConfig:
    season: str
    base_url: str
    request_workers: int
    ttl_seconds: dict[str, int]


@dataclass(frozen=True)
class FeatureConfig:
    rolling_windows: tuple[int, ...]
    team_window: int
    per90_min_minutes: int
    fixture_horizon: int
    promoted_prior: dict[str, float]


@dataclass(frozen=True)
class TargetConfig:
    haul_points: int
    return_points: int
    multi_gw_horizon: int


@dataclass(frozen=True)
class EvalConfig:
    seasons: tuple[str, ...]
    top_k: int


@dataclass(frozen=True)
class ModelConfig:
    seed: int
    dev_seasons: tuple[str, ...]
    test_seasons: tuple[str, ...]
    early_stopping_rounds: int
    ridge_alpha: float
    lightgbm: dict[str, Any]


@dataclass(frozen=True)
class DifferentialConfig:
    ownership_max_pct: float
    grid: dict[str, list[float]]


@dataclass(frozen=True)
class OptimizerConfig:
    budget: int
    hit_cost: int
    max_free_transfers: int
    max_transfers: int
    min_gain_per_transfer: float
    bench_weight: float
    min_chance_of_playing: float
    differential_bonus: float
    projection: str
    solver_time_limit: int


@dataclass(frozen=True)
class BacktestConfig:
    seasons: tuple[str, ...]
    start_gw: int
    sources: dict[str, str]


@dataclass(frozen=True)
class Config:
    paths: Paths
    history: HistoryConfig
    live: LiveConfig
    features: FeatureConfig
    targets: TargetConfig
    evaluation: EvalConfig
    model: ModelConfig
    differential: DifferentialConfig
    optimizer: OptimizerConfig
    backtest: BacktestConfig


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _build(cls: type, data: dict[str, Any]):
    """Instantiate a dataclass from a YAML mapping, turning lists into tuples."""
    kwargs = {}
    for f in fields(cls):
        value = data[f.name]
        kwargs[f.name] = tuple(value) if isinstance(value, list) else value
    return cls(**kwargs)


def load_config(path: str | Path | None = None) -> Config:
    """Load the YAML config (default ``configs/config.yaml``, or ``$FPLOPT_CONFIG``)."""
    path = Path(path or os.environ.get("FPLOPT_CONFIG") or DEFAULT_CONFIG)
    raw = yaml.safe_load(path.read_text())
    return Config(
        paths=Paths(**{k: _resolve(v) for k, v in raw["paths"].items()}),
        history=_build(HistoryConfig, raw["history"]),
        live=_build(LiveConfig, raw["live"]),
        features=_build(FeatureConfig, raw["features"]),
        targets=_build(TargetConfig, raw["targets"]),
        evaluation=_build(EvalConfig, raw["evaluation"]),
        model=_build(ModelConfig, raw["model"]),
        differential=_build(DifferentialConfig, raw["differential"]),
        optimizer=_build(OptimizerConfig, raw["optimizer"]),
        backtest=_build(BacktestConfig, raw["backtest"]),
    )
