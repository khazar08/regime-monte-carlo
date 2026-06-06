# regime-monte-carlo

# Regime-Switching Monte Carlo Engine

> **Scenario simulation and risk quantification for equities — not price prediction.**

A Python framework that simulates the distribution of future equity price paths using Monte Carlo methods, where drift and volatility are governed by a Markov regime-switching process calibrated via a Hidden Markov Model (HMM).

---

## Headline Result: GBM vs Regime-Switching Calibration

The key validation is walk-forward calibration coverage: at each anchor date, the model is calibrated on a trailing window, simulates a horizon forward, and we check whether the realized price falls inside the predicted interval.

| Nominal Level | GBM Coverage | Regime Coverage |
|:---:|:---:|:---:|
| 50% | — | — |
| 80% | — | — |
| 95% | — | — |

*Run `rmc backtest --ticker SPY` to populate this table with real results.*

---

## Methods

### Log Returns
Daily log return: r_t = ln(P_t / P_{t-1})

### GBM Baseline
MLE parameters from daily log returns. Simulated using the Euler-Maruyama discretization in daily units (dt=1):

```
S_{t+1} = S_t * exp( (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z )
```

### Regime-Switching Model
A `GaussianHMM` (K=3 states by default) is fitted on the daily log return series. Each hidden state has its own mean and variance. At each daily step, the state transitions according to the calibrated Markov matrix and a return is drawn from the active state's Normal distribution.

States are sorted by mean return and labeled: **bear**, **neutral**, **bull**.

### Walk-Forward Calibration Backtest
At each anchor date (every 21 trading days), we:
1. Fit the model on a trailing 504-day window
2. Simulate 21 days forward (50,000 paths)
3. Record whether the realized price falls inside the predicted 50%/80%/95% central intervals

### Risk Metrics
- **VaR_alpha** = -quantile(R, 1-alpha)
- **CVaR_alpha** = -mean(R[R <= quantile(R, 1-alpha)])
- **Max drawdown** per path; P(drawdown > threshold)

---

## Quickstart

```bash
# Install
pip install -e .
pip install -r requirements-dev.txt

# Download data (cached to data/raw/)
rmc fetch

# Generate all outputs for SPY
rmc report --ticker SPY

# Other commands
rmc simulate --ticker AAPL --model regime
rmc backtest --ticker SPY
```

---

## Repository Structure

```
regime-monte-carlo/
├── src/rmc/
│   ├── config.py          # Config loading + validation
│   ├── data/ingest.py     # Price download + log returns
│   ├── models/
│   │   ├── gbm.py         # GBM fit + simulation
│   │   ├── regime.py      # HMM fit + stationary dist + Viterbi
│   │   └── simulate.py    # Vectorized regime-switching simulation
│   ├── evaluation/
│   │   ├── calibration.py # Walk-forward coverage backtest
│   │   └── risk.py        # VaR, CVaR, drawdown
│   ├── visualization/plots.py
│   └── cli.py             # rmc CLI
├── tests/                 # pytest suite
├── config.yaml            # Default configuration
└── outputs/               # Generated plots + reports
```

---

## Limitations

- **Stationary transition matrix**: The HMM assumes a fixed regime transition probability, which may not hold over long horizons or structural market breaks.
- **Regime non-stationarity**: Real market regimes may shift in ways not captured by a fixed K-state model trained on historical data.
- **Gaussian emissions**: Return distributions have fat tails and skewness; Gaussian per-state distributions will understate tail risk.
- **In-sample calibration**: HMM parameters are fit on historical data; out-of-sample performance may differ.
- **No fat-tail extensions**: Student-t or jump-diffusion emissions would better capture extreme events.
- **This is not a trading signal**: Outputs describe the *distribution* of outcomes, not where prices *will* go.

---

## License

MIT
