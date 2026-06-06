"""Load and validate configuration from config.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    tickers: list[str]
    start: str
    end: str
    source: str = "yfinance"
    cache_dir: str = "data/raw"


@dataclass
class SimulationConfig:
    horizon_days: int = 252
    n_paths: int = 50000
    dt: float = 1.0
    seed: int = 42


@dataclass
class RegimeConfig:
    n_states: int = 3
    n_iter: int = 200
    init_state: str = "stationary"


@dataclass
class BacktestConfig:
    trailing_window: int = 504
    horizon_days: int = 21
    anchor_step: int = 21
    nominal_levels: list[float] = field(default_factory=lambda: [0.50, 0.80, 0.95])


@dataclass
class RiskConfig:
    var_alpha: float = 0.95
    drawdown_threshold: float = 0.20


@dataclass
class Config:
    data: DataConfig
    simulation: SimulationConfig
    regime: RegimeConfig
    backtest: BacktestConfig
    risk: RiskConfig
    output_dir: str = "outputs"


def _validate(cfg: Config) -> None:
    assert cfg.regime.n_states >= 2, "n_states must be >= 2"
    assert cfg.simulation.n_paths > 0, "n_paths must be > 0"
    assert cfg.simulation.horizon_days > 0, "horizon_days must be > 0"
    assert 0 < cfg.risk.var_alpha < 1, "var_alpha must be in (0,1)"
    for lvl in cfg.backtest.nominal_levels:
        assert 0 < lvl < 1, f"nominal level {lvl} must be in (0,1)"
    assert cfg.regime.init_state in {"stationary", "last", "sampled"}


def load_config(path: str | Path = "config.yaml") -> Config:
    """Load and validate config from a YAML file."""
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    cfg = Config(
        data=DataConfig(**raw["data"]),
        simulation=SimulationConfig(**raw["simulation"]),
        regime=RegimeConfig(**raw["regime"]),
        backtest=BacktestConfig(**raw["backtest"]),
        risk=RiskConfig(**raw["risk"]),
        output_dir=raw.get("output_dir", "outputs"),
    )
    _validate(cfg)
    return cfg
