# Regime-Switching Monte Carlo Engine

A Python framework that simulates the distribution of future equity price paths using Monte Carlo methods, where drift and volatility are governed by a Markov regime-switching process calibrated via a Hidden Markov Model (HMM).

---

## Headline Result: GBM vs Regime-Switching Calibration

Walk-forward backtest on SPY (2015–2024). At each anchor date the model is calibrated on a trailing 504-day window, simulates 21 trading days forward across 50,000 paths, and we check whether the realized price falls inside the predicted central interval. A well-calibrated model should hit the nominal rate.

| Nominal Level | GBM Coverage | Regime Coverage |
|:---:|:---:|:---:|
| 50% | 60.0% | 61.4% |
| 80% | 89.5% | 89.8% |
| 95% | 94.7% | 98.9% |

Both models overcover at 50% — their central intervals are wider than necessary, reflecting the high day-to-day variance in SPY returns over a decade that includes 2018, 2020, and 2022. At 80% both sit close to nominal. The divergence appears at 95%: GBM lands at 94.7%, essentially calibrated, while the regime model overcovering at 98.9% — the bear state inflates tail vol estimates, producing intervals that are conservatively wide. Neither model is strictly better; GBM is sharper at the 95% level while the regime decomposition gives more interpretable dynamics.

---

## Methods

### Log Returns
Daily log return: `r_t = ln(P_t / P_{t-1})`

### GBM Baseline
MLE parameters from daily log returns. Simulated using the Euler–Maruyama discretization in daily units (dt = 1):

```
S_{t+1} = S_t * exp( (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z )
```

### Regime-Switching Model
A `GaussianHMM` (K = 3 states) is fitted on the daily log return series. Each hidden state carries its own mean and variance. At each daily step, the active state transitions according to the calibrated Markov matrix and a return is drawn from that state's Normal distribution.

States are sorted by mean return and labeled **bear**, **neutral**, **bull**. Fitted on SPY (2015–2024):

| State | Daily Mean | Daily Vol | Stationary Weight |
|:---:|:---:|:---:|:---:|
| Bear | −0.11% | 1.84% | 18% |
| Neutral | +0.04% | 0.73% | 61% |
| Bull | +0.14% | 0.52% | 21% |

### Walk-Forward Calibration Backtest
At each anchor date (every 21 trading days):
1. Fit the model on a trailing 504-day window
2. Simulate 21 days forward (50,000 paths)
3. Record whether the realized price falls inside the predicted 50%/80%/95% central intervals

### Risk Metrics
- **VaR_α** = −quantile(R, 1 − α)
- **CVaR_α** = −mean(R | R ≤ quantile(R, 1 − α))
- Per-path max drawdown; P(max drawdown > threshold)

---

## Quickstart

```bash
pip install -e .
pip install -r requirements-dev.txt

rmc fetch                          # download and cache price data
rmc report --ticker SPY            # fan chart, regime history, risk report
rmc simulate --ticker AAPL --model regime
rmc backtest --ticker SPY
```

---

## Repository Structure

```
regime-monte-carlo/
├── src/rmc/
│   ├── config.py
│   ├── data/ingest.py
│   ├── models/
│   │   ├── gbm.py
│   │   ├── regime.py
│   │   └── simulate.py
│   ├── evaluation/
│   │   ├── calibration.py
│   │   └── risk.py
│   ├── visualization/plots.py
│   └── cli.py
├── tests/
├── config.yaml
├── pyproject.toml
└── requirements.txt
```

---

## Limitations

- **Stationary transition matrix** — the HMM assumes fixed regime transition probabilities, which may break across structural market shifts.
- **Gaussian emissions** — per-state Normal distributions understate tail risk; Student-t or jump-diffusion extensions would handle fat tails better.
- **In-sample calibration** — HMM parameters are estimated on historical data; realized coverage in a different regime environment may diverge.
- **K fixed at training time** — the number of states is a hyperparameter, not inferred from data; different choices meaningfully change the fitted dynamics.

---

## Roadmap / Ideas

- **Student-t emissions** — replace per-state Gaussians with Student-t distributions to better capture fat tails and reduce overcovering at high confidence levels
- **Time-varying transition matrix** — allow the Markov transition probabilities to depend on macro features (VIX, yield curve slope) rather than staying fixed
- **Multi-asset simulation** — extend to correlated asset paths using a regime-dependent covariance matrix, enabling portfolio-level VaR
- **Jump-diffusion layer** — add a Poisson jump process on top of the GBM baseline to model sudden crashes separately from vol regimes
- **Streamlit dashboard** — interactive UI to pick ticker, horizon, and model and render live fan charts and risk metrics in the browser
- **Bayesian HMM** — replace EM with MCMC or variational inference to get uncertainty estimates on the transition matrix itself
- **Option pricing extension** — price European options via Monte Carlo and compare against Black–Scholes closed form as an engine validation

---
