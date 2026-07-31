from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def download_prices(
    tickers: list[str],
    start: str,
    end: str,
    source: str = "yfinance",
    cache_dir: str | None = None,
) -> pd.DataFrame:

    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"{'_'.join(sorted(tickers))}_{start}_{end}.parquet"
        if cache_path.exists():
            logger.info("Loading prices from cache: %s", cache_path)
            return pd.read_parquet(cache_path)

    if source == "yfinance":
        prices = _download_yfinance(tickers, start, end)
    elif source == "csv":
        raise NotImplementedError("CSV source not yet implemented")
    else:
        raise ValueError(f"Unknown source: {source}")

    if prices.empty:
        raise ValueError(f"No price data returned for tickers={tickers}, start={start}, end={end}")

    prices = prices.ffill(limit=5)

    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        prices.to_parquet(cache_path)
        logger.info("Cached prices to %s", cache_path)

    return prices


def _download_yfinance(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices using yfinance."""
    import yfinance as yf

    valid_tickers = []
    frames = []

    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            if data.empty:
                logger.warning("No data for ticker %s — skipping", ticker)
                continue
            close = data["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close.name = ticker
            frames.append(close)
            valid_tickers.append(ticker)
        except Exception as exc:
            logger.warning("Failed to download %s: %s — skipping", ticker, exc)

    if not frames:
        return pd.DataFrame()

    prices = pd.concat(frames, axis=1)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    log_ret = np.log(prices / prices.shift(1)).dropna()
    return log_ret


def load_or_download(config: object) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = download_prices(
        tickers=config.data.tickers,
        start=config.data.start,
        end=config.data.end,
        source=config.data.source,
        cache_dir=config.data.cache_dir,
    )
    returns = compute_log_returns(prices)
    return prices, returns
