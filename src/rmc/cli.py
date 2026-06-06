"""Command-line entry point for the rmc package."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def _find_config() -> Path:
    candidates = [Path("config.yaml"), Path(__file__).parent.parent.parent / "config.yaml"]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("config.yaml not found. Run from the project root directory.")


def cmd_fetch(args: argparse.Namespace) -> None:
    from rmc.config import load_config
    from rmc.data.ingest import load_or_download

    cfg = load_config(_find_config())
    logger.info("Downloading data for %s", cfg.data.tickers)
    prices, returns = load_or_download(cfg)
    logger.info("Prices shape: %s, Returns shape: %s", prices.shape, returns.shape)
    logger.info("Done. Data cached to %s", cfg.data.cache_dir)


def cmd_simulate(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from rmc.config import load_config
    from rmc.data.ingest import load_or_download
    from rmc.evaluation.risk import risk_report
    from rmc.models.gbm import fit_gbm, simulate_gbm
    from rmc.models.regime import fit_regime
    from rmc.models.simulate import simulate_regime
    from rmc.visualization.plots import plot_fan_chart, save_figure

    cfg = load_config(_find_config())
    ticker = args.ticker
    model = args.model
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prices, returns = load_or_download(cfg)
    if ticker not in prices.columns:
        logger.error(
            "Ticker %s not in downloaded data. Available: %s", ticker, list(prices.columns)
        )
        sys.exit(1)

    price_series = prices[ticker].dropna()
    ret_series = returns[ticker].dropna()
    ret_arr = ret_series.values
    s0 = float(price_series.iloc[-1])

    rng = np.random.default_rng(cfg.simulation.seed)

    if model == "gbm":
        params = fit_gbm(ret_arr)
        logger.info(
            "GBM params: mu_daily=%.6f, sigma_daily=%.6f", params.mu_daily, params.sigma_daily
        )
        paths = simulate_gbm(
            s0, params, cfg.simulation.horizon_days, cfg.simulation.n_paths, cfg.simulation.dt, rng
        )
    elif model == "regime":
        params, hmm = fit_regime(ret_arr, cfg.regime.n_states, cfg.regime.n_iter, rng)
        logger.info("Regime params: means=%s, sigmas=%s", params.means, params.sigmas)
        rng2 = np.random.default_rng(cfg.simulation.seed)
        paths = simulate_regime(
            s0,
            params,
            cfg.simulation.horizon_days,
            cfg.simulation.n_paths,
            cfg.regime.init_state,
            rng2,
        )
    else:
        logger.error("Unknown model: %s", model)
        sys.exit(1)

    # Fan chart
    import pandas as pd

    last_date = price_series.index[-1]
    future_dates = pd.bdate_range(start=last_date, periods=cfg.simulation.horizon_days + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_fan_chart(paths, dates=future_dates, ax=ax)
    ax.set_title(f"{ticker} — {model.upper()} Fan Chart ({cfg.simulation.horizon_days}d)")
    fan_path = out_dir / f"{ticker}_{model}_fan_chart.png"
    save_figure(fig, fan_path)
    plt.close(fig)
    logger.info("Fan chart saved: %s", fan_path)

    # Risk report
    report = risk_report(paths, cfg.risk.var_alpha, cfg.risk.drawdown_threshold)
    report_path = out_dir / f"{ticker}_{model}_risk_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Risk report saved: %s", report_path)
    for k, v in report.items():
        logger.info("  %s: %.4f", k, v)


def cmd_backtest(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from rmc.config import load_config
    from rmc.data.ingest import load_or_download
    from rmc.evaluation.calibration import coverage_summary, walk_forward_coverage
    from rmc.visualization.plots import plot_model_comparison, save_figure

    cfg = load_config(_find_config())
    ticker = args.ticker
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prices, _ = load_or_download(cfg)
    if ticker not in prices.columns:
        logger.error("Ticker %s not available", ticker)
        sys.exit(1)

    price_series = prices[ticker].dropna()

    logger.info("Running GBM backtest for %s ...", ticker)
    gbm_cov = walk_forward_coverage(price_series, "gbm", cfg)
    gbm_summary = coverage_summary(gbm_cov)

    logger.info("Running Regime backtest for %s ...", ticker)
    reg_cov = walk_forward_coverage(price_series, "regime", cfg)
    reg_summary = coverage_summary(reg_cov)

    logger.info("\nGBM Coverage:\n%s", gbm_summary.to_string(index=False))
    logger.info("\nRegime Coverage:\n%s", reg_summary.to_string(index=False))

    gbm_cov.to_csv(out_dir / f"{ticker}_gbm_coverage.csv", index=False)
    reg_cov.to_csv(out_dir / f"{ticker}_regime_coverage.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 7))
    plot_model_comparison(gbm_summary, reg_summary, ax=ax)
    ax.set_title(f"{ticker} — Calibration Comparison")
    save_figure(fig, out_dir / f"{ticker}_calibration_comparison.png")
    plt.close(fig)
    logger.info("Calibration plot saved.")


def cmd_report(args: argparse.Namespace) -> None:
    """Run all outputs: simulate (both models) + backtest."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from rmc.config import load_config
    from rmc.data.ingest import load_or_download
    from rmc.evaluation.calibration import coverage_summary, walk_forward_coverage
    from rmc.evaluation.risk import conditional_var, risk_report, terminal_returns, value_at_risk
    from rmc.models.gbm import fit_gbm, simulate_gbm
    from rmc.models.regime import decode_states, fit_regime
    from rmc.models.simulate import simulate_regime
    from rmc.visualization.plots import (
        plot_fan_chart,
        plot_model_comparison,
        plot_regimes,
        plot_terminal_distribution,
        save_figure,
    )

    cfg = load_config(_find_config())
    ticker = args.ticker
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prices, returns = load_or_download(cfg)
    if ticker not in prices.columns:
        logger.error("Ticker %s not available", ticker)
        sys.exit(1)

    price_series = prices[ticker].dropna()
    ret_series = returns[ticker].dropna()
    ret_arr = ret_series.values
    s0 = float(price_series.iloc[-1])

    import pandas as pd

    last_date = price_series.index[-1]
    future_dates = pd.bdate_range(start=last_date, periods=cfg.simulation.horizon_days + 1)

    rng = np.random.default_rng(cfg.simulation.seed)

    # GBM simulation + fan chart
    gbm_params = fit_gbm(ret_arr)
    gbm_paths = simulate_gbm(
        s0, gbm_params, cfg.simulation.horizon_days, cfg.simulation.n_paths, cfg.simulation.dt, rng
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_fan_chart(gbm_paths, dates=future_dates, ax=ax)
    ax.set_title(f"{ticker} — GBM Fan Chart")
    save_figure(fig, out_dir / f"{ticker}_gbm_fan_chart.png")
    plt.close(fig)

    # Regime simulation + fan chart + regime history
    rng2 = np.random.default_rng(cfg.simulation.seed)
    reg_params, reg_model = fit_regime(ret_arr, cfg.regime.n_states, cfg.regime.n_iter, rng2)
    rng3 = np.random.default_rng(cfg.simulation.seed)
    reg_paths = simulate_regime(
        s0,
        reg_params,
        cfg.simulation.horizon_days,
        cfg.simulation.n_paths,
        cfg.regime.init_state,
        rng3,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_fan_chart(reg_paths, dates=future_dates, ax=ax)
    ax.set_title(f"{ticker} — Regime-Switching Fan Chart")
    save_figure(fig, out_dir / f"{ticker}_regime_fan_chart.png")
    plt.close(fig)

    # Regime history plot
    states = decode_states(reg_model, ret_arr)
    fig, ax = plt.subplots(figsize=(12, 5))
    plot_regimes(price_series, states, reg_params.state_labels, ax=ax)
    ax.set_title(f"{ticker} — Regime History")
    save_figure(fig, out_dir / f"{ticker}_regime_history.png")
    plt.close(fig)

    # Terminal return distribution (regime)
    rets = terminal_returns(reg_paths)
    var = value_at_risk(rets, cfg.risk.var_alpha)
    cvar = conditional_var(rets, cfg.risk.var_alpha)
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_terminal_distribution(rets, var, cvar, ax=ax)
    ax.set_title(f"{ticker} — Terminal Return Distribution (Regime)")
    save_figure(fig, out_dir / f"{ticker}_terminal_dist.png")
    plt.close(fig)

    # Risk report
    report = risk_report(reg_paths, cfg.risk.var_alpha, cfg.risk.drawdown_threshold)
    with open(out_dir / f"{ticker}_risk_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Backtest
    logger.info("Running GBM backtest ...")
    gbm_cov = walk_forward_coverage(price_series, "gbm", cfg)
    gbm_summary = coverage_summary(gbm_cov)
    logger.info("Running Regime backtest ...")
    reg_cov = walk_forward_coverage(price_series, "regime", cfg)
    reg_summary = coverage_summary(reg_cov)

    fig, ax = plt.subplots(figsize=(7, 7))
    plot_model_comparison(gbm_summary, reg_summary, ax=ax)
    ax.set_title(f"{ticker} — Calibration Comparison")
    save_figure(fig, out_dir / f"{ticker}_calibration_comparison.png")
    plt.close(fig)

    gbm_cov.to_csv(out_dir / f"{ticker}_gbm_coverage.csv", index=False)
    reg_cov.to_csv(out_dir / f"{ticker}_regime_coverage.csv", index=False)

    logger.info("\nGBM Coverage:\n%s", gbm_summary.to_string(index=False))
    logger.info("\nRegime Coverage:\n%s", reg_summary.to_string(index=False))
    logger.info("All outputs saved to %s/", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(prog="rmc", description="Regime Monte Carlo Engine")
    sub = parser.add_subparsers(dest="command", required=True)

    # fetch
    sub.add_parser("fetch", help="Download and cache price data")

    # simulate
    sim_p = sub.add_parser("simulate", help="Run Monte Carlo simulation for one ticker")
    sim_p.add_argument("--ticker", default="SPY")
    sim_p.add_argument("--model", choices=["gbm", "regime"], default="regime")

    # backtest
    bt_p = sub.add_parser("backtest", help="Walk-forward calibration backtest")
    bt_p.add_argument("--ticker", default="SPY")

    # report
    rep_p = sub.add_parser("report", help="Generate all outputs")
    rep_p.add_argument("--ticker", default="SPY")

    args = parser.parse_args()

    if args.command == "fetch":
        cmd_fetch(args)
    elif args.command == "simulate":
        cmd_simulate(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "report":
        cmd_report(args)


if __name__ == "__main__":
    main()
