#!/usr/bin/env python3
"""Compute cross-sectional Rank IC for daily factors aggregated from hourly data."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ASIA_SHANGHAI = "Asia/Shanghai"
TRADE_END_HOUR = 15
LAGS = [0, 1,2,3,4,5,7,10,15]
MIN_CROSS_SECTION = 20
EPSILON = 1e-12  # Small epsilon to avoid division by zero
# Toggle whether to neutralize alpha101 factors with log price/volume covariates.
NEUTRALIZE_ALPHA101 = True  # Set to True to apply neutralization to alpha# factors as well.
LOG_FEATURES = ["log_price_ma", "log_volume_ma"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing hourly K-line JSON files (default: %(default)s).",
    )
    parser.add_argument(
        "--mapping",
        default="getdata/itemid.txt",
        help="File mapping item IDs to industries (default: %(default)s).",
    )
    parser.add_argument(
        "--heatmap",
        default="rank_ic_heatmap_old.png",
        help="Output path for the Rank IC heatmap (default: %(default)s).",
    )
    parser.add_argument(
        "--output-summary",
        default="rank_ic_summary_old.csv",
        help="CSV path for aggregated IC statistics (default: %(default)s).",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2024-03-01",
        help="Inclusive start date (YYYY-MM-DD) interpreted at 15:00 Asia/Shanghai.",
    )
    parser.add_argument(
        "--end",
        default="2024-12-13",
        type=str,
        help="Inclusive end date (YYYY-MM-DD) interpreted at 15:00 Asia/Shanghai.",
    )
    parser.add_argument(
        "--mad-threshold",
        type=float,
        default=5.0,
        help="MAD threshold for cross-sectional outlier removal (default: %(default)s).",
    )
    parser.add_argument(
        "--analyze-nan",
        action="store_true",
        help="Analyze and report NaN value distribution in factor data.",
    )
    parser.add_argument(
        "--history-days",
        type=int,
        default=90,
        help="Historical lookback days for long-window factors (default: %(default)s).",
    )
    return parser.parse_args()


def load_industry_mapping(path: Path) -> Dict[str, str]:
    """Parse item -> industry mappings from the config file."""
    mapping: Dict[str, str] = {}
    current_industry: str | None = None

    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                current_industry = None
                continue
            if line.startswith("//"):
                current_industry = line[2:].strip() or "UNKNOWN"
                continue
            if current_industry is None:
                continue
            if "：" in line:
                item_id = line.split("：", 1)[0].strip()
            elif ":" in line:
                item_id = line.split(":", 1)[0].strip()
            else:
                item_id = line.split()[0].strip()
            if item_id:
                mapping[item_id] = current_industry
    return mapping


def parse_window_endpoint(value: str | None) -> pd.Timestamp | None:
    """Parse YYYY-MM-DD into a timezone-aware timestamp at 15:00 Asia/Shanghai."""
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(ASIA_SHANGHAI)
    else:
        ts = ts.tz_convert(ASIA_SHANGHAI)
    ts = ts.replace(hour=TRADE_END_HOUR, minute=0, second=0, microsecond=0, nanosecond=0)
    return ts


def load_hourly_json(path: Path) -> pd.DataFrame:
    """Load hourly candles from JSON and return a DataFrame sorted by timestamp."""
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not raw:
        return pd.DataFrame(columns=["timestamp", "o", "h", "l", "c", "v", "turnover"])
    df = pd.DataFrame(raw)
    if "t" not in df:
        raise ValueError(f"{path.name} is missing 't' timestamps.")
    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    df = df.dropna(subset=["t"])
    df = df.sort_values("t").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    expected_cols = {"o", "h", "l", "c", "v", "turnover"}
    missing = expected_cols.difference(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")
    return df[["timestamp", "o", "h", "l", "c", "v", "turnover"]]


def aggregate_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly candles into daily candles ending at 15:00 Beijing time."""
    if df.empty:
        return df
    df = df.copy()
    df["ts_cn"] = df["timestamp"].dt.tz_convert(ASIA_SHANGHAI)
    df["trade_date"] = (df["ts_cn"] - pd.Timedelta(hours=TRADE_END_HOUR)).dt.floor("D")
    grouped = df.groupby("trade_date", sort=True)
    daily = grouped.agg(
        open=("o", "first"),
        high=("h", "max"),
        low=("l", "min"),
        close=("c", "last"),
        volume=("v", "sum"),
        turnover=("turnover", "sum"),
        obs=("timestamp", "count"),
    )
    daily.index = daily.index.tz_localize(None)
    daily.index.name = "date"
    return daily


def compute_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Compute the factor library defined in features.md on daily candles."""
    if daily.empty:
        return daily

    df = daily.copy()
    price_cols = ["open", "high", "low", "close", "volume", "turnover"]
    df[price_cols] = df[price_cols].astype(float)

    eps = 1e-12
    prev_close = df["close"].shift(1)
    log_close = np.log(df["close"].clip(lower=eps))
    log_open = np.log(df["open"].clip(lower=eps))

    # Base returns / momentum
    df["log_return_1"] = log_close.diff()
    df["intraday_return"] = log_close - log_open
    df["momentum_5"] = log_close - log_close.shift(5)
    df["momentum_20"] = log_close - log_close.shift(20)
    df["momentum_40"] = log_close - log_close.shift(40)
    df["momentum_60"] = log_close - log_close.shift(60)

    daily_log_return = df["log_return_1"].copy()
    rolling_std_20 = daily_log_return.rolling(window=20, min_periods=5).std(ddof=0)
    safe_std_20 = rolling_std_20.mask(rolling_std_20.abs() <= eps)
    df["risk_adj_mom_20"] = df["momentum_20"] / safe_std_20

    # Higher-moment risk metrics
    df["return_skew_20"] = daily_log_return.rolling(window=20, min_periods=20).skew()
    df["return_kurt_20"] = daily_log_return.rolling(window=20, min_periods=20).kurt()
    negative_returns = daily_log_return.clip(upper=0.0)
    df["downside_vol_20"] = np.sqrt(
        negative_returns.pow(2).rolling(window=20, min_periods=20).mean()
    )

    low_rolling_20 = df["close"].rolling(window=20, min_periods=5).min()
    high_rolling_20 = df["close"].rolling(window=20, min_periods=5).max()
    donchian_denom = (high_rolling_20 - low_rolling_20).where(
        (high_rolling_20 - low_rolling_20).abs() > eps
    )
    df["donchian_pos_20"] = (df["close"] - low_rolling_20) / donchian_denom

    # Volatility metrics
    hl = df["high"] - df["low"]
    hc = (df["high"] - prev_close).abs()
    lc = (df["low"] - prev_close).abs()
    true_range = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    safe_prev_close = prev_close.where(prev_close.abs() > eps)
    df["true_range_pct_1"] = true_range / safe_prev_close

    ret_sq = daily_log_return.pow(2)
    realized_vol_5 = np.sqrt(ret_sq.rolling(window=5, min_periods=3).sum())
    df["realized_vol_10"] = np.sqrt(ret_sq.rolling(window=10, min_periods=5).sum())
    df["realized_vol_20"] = np.sqrt(ret_sq.rolling(window=20, min_periods=8).sum())

    ratio = (df["high"] / df["low"]).replace([np.inf, -np.inf], np.nan)
    hl_ratio = np.log(ratio).replace([np.inf, -np.inf], np.nan)
    df["parkinson_vol_10"] = np.sqrt(
        hl_ratio.pow(2).rolling(window=10, min_periods=5).mean() / (4 * np.log(2))
    )

    safe_realized_vol_20 = df["realized_vol_20"].where(df["realized_vol_20"].abs() > eps)
    df["vol_ratio_5_20"] = realized_vol_5 / safe_realized_vol_20
    lagged_realized_vol_20 = df["realized_vol_20"].shift(1)
    safe_lagged_realized_vol_20 = lagged_realized_vol_20.where(
        lagged_realized_vol_20.abs() > eps
    )
    df["vol_change_20"] = (df["realized_vol_20"] - lagged_realized_vol_20) / safe_lagged_realized_vol_20

    # Volume / liquidity metrics
    volume_ma_5 = df["volume"].rolling(window=5, min_periods=1).mean()
    volume_ma_20 = df["volume"].rolling(window=20, min_periods=5).mean()
    turnover_ma_5 = df["turnover"].rolling(window=5, min_periods=1).mean()
    turnover_ma_20 = df["turnover"].rolling(window=20, min_periods=5).mean()

    log_volume = np.log(df["volume"].clip(lower=eps))
    log_amount = np.log(df["turnover"].clip(lower=eps))
    vol_mean = log_volume.rolling(window=20, min_periods=5).mean()
    vol_std = log_volume.rolling(window=20, min_periods=5).std(ddof=0)
    amt_mean = log_amount.rolling(window=20, min_periods=5).mean()
    amt_std = log_amount.rolling(window=20, min_periods=5).std(ddof=0)
    df["log_volume_zscore_20"] = (log_volume - vol_mean) / vol_std.replace(0, np.nan)
    df["log_amount_zscore_20"] = (log_amount - amt_mean) / amt_std.replace(0, np.nan)

    delta_log_volume = log_volume.diff()
    df["corr_price_vol_20"] = daily_log_return.rolling(window=20, min_periods=20).corr(
        delta_log_volume
    )

    safe_volume_ma_20 = volume_ma_20.where(volume_ma_20.abs() > eps)
    safe_turnover_ma_20 = turnover_ma_20.where(turnover_ma_20.abs() > eps)
    df["volume_ma_ratio_5_20"] = volume_ma_5 / safe_volume_ma_20
    df["amount_ma_ratio_5_20"] = turnover_ma_5 / safe_turnover_ma_20

    amihud = (daily_log_return.abs() / df["turnover"].where(df["turnover"].abs() > eps))
    df["amihud_illiquidity_20"] = amihud.rolling(window=20, min_periods=5).mean()

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    money_flow = typical_price * df["volume"]
    tp_delta = typical_price.diff()
    positive_flow = money_flow.where(tp_delta > 0, 0.0)
    negative_flow = money_flow.where(tp_delta < 0, 0.0)
    pos_sum_14 = positive_flow.rolling(window=14, min_periods=14).sum()
    neg_sum_14 = negative_flow.rolling(window=14, min_periods=14).sum()
    safe_neg_sum_14 = neg_sum_14.replace(0, eps)
    flow_ratio = pos_sum_14 / safe_neg_sum_14
    df["mfi_14"] = 100 - (100 / (1 + flow_ratio))

    price_diff = df["close"].diff().fillna(0.0)
    direction = np.sign(price_diff)
    obv = (direction * df["volume"]).cumsum()
    df["obv_slope_20"] = (obv - obv.shift(20)) / 20

    vwap = np.where(df["volume"] > 0, df["turnover"] / df["volume"], df["close"])
    df["vwap"] = vwap
    df["vwap_close_gap"] = np.where(vwap != 0, (df["close"] - vwap) / vwap, np.nan)

    # Oscillators & moving averages
    delta = df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    low_min_14 = df["low"].rolling(window=14, min_periods=5).min()
    high_max_14 = df["high"].rolling(window=14, min_periods=5).max()
    denom_14 = (high_max_14 - low_min_14).where((high_max_14 - low_min_14).abs() > eps)
    df["stoch_k_14"] = (df["close"] - low_min_14) / denom_14

    rolling_mean_20 = df["close"].rolling(window=20, min_periods=5).mean()
    rolling_std_20_price = df["close"].rolling(window=20, min_periods=5).std(ddof=0)
    df["bollinger_z_20"] = (df["close"] - rolling_mean_20) / rolling_std_20_price.replace(0, np.nan)

    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["ema_gap_12_26"] = (ema_12 - ema_26) / ema_26.replace({0: np.nan})

    df["price_ma_gap_20"] = (df["close"] - rolling_mean_20) / rolling_mean_20.replace(0, np.nan)
    ma20_lag5 = rolling_mean_20.shift(5)
    df["trend_slope_ma20"] = (rolling_mean_20 - ma20_lag5) / (5 * ma20_lag5.replace({0: np.nan}))

    # ADX 14 calculation
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_14 = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / atr_14.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / atr_14.replace(0, np.nan)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    df["adx_14"] = dx.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    # Candle structure & micro-structure
    full_range = df["high"] - df["low"]
    denom = (full_range + eps)
    df["body_ratio"] = (df["close"] - df["open"]).abs() / denom
    df["upper_shadow_ratio"] = (df["high"] - df[["open", "close"]].max(axis=1)) / denom
    df["lower_shadow_ratio"] = (df[["open", "close"]].min(axis=1) - df["low"]) / denom
    df["clv_1"] = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / denom
    # CMF calculation with zero handling
    mfm = np.where(full_range != 0,
                  ((df["close"] - df["low"]) - (df["high"] - df["close"])) / full_range,
                  0.0)
    mf_volume = mfm * df["volume"]
    mfv_sum_20 = pd.Series(mf_volume, index=df.index).rolling(window=20, min_periods=20).sum()
    volume_sum_20 = df["volume"].rolling(window=20, min_periods=20).sum()
    df["cmf_20"] = np.where(volume_sum_20 != 0, mfv_sum_20 / volume_sum_20, 0.0)
    bullish = (df["close"] > df["open"]).astype(float)
    df["bull_ratio_5"] = bullish.rolling(window=5, min_periods=5).mean()

    # Helper columns for neutralization (3-day averages per spec)
    price_ma_3 = df["close"].rolling(window=3, min_periods=1).mean()
    volume_ma_3 = df["volume"].rolling(window=3, min_periods=1).mean()
    df["price_ma_3"] = price_ma_3
    df["volume_ma_3"] = volume_ma_3
    df["log_price_ma"] = np.log1p(price_ma_3.clip(lower=0))
    df["log_volume_ma"] = np.log1p(volume_ma_3.clip(lower=0))

    df["target_8d"] = df["close"].shift(-8) / df["close"] - 1

    return df


def neutralize_target_by_industry(
    df: pd.DataFrame,
    target_col: str,
    industry_col: str,
) -> pd.Series:
    """Return target residuals after removing industry mean each date."""

    if target_col not in df or industry_col not in df:
        raise KeyError(f"DataFrame must contain '{target_col}' and '{industry_col}'.")

    grouped = df.groupby(["date", industry_col])[target_col]
    industry_mean = grouped.transform("mean")
    return df[target_col] - industry_mean


def _cross_sectional_rank(values: pd.Series, groups: pd.Series) -> pd.Series:
    """Cross-sectional rank scaled to [0, 1] within each group."""
    def _rank(group: pd.Series) -> pd.Series:
        mask = group.notna()
        if not mask.any():
            return group
        valid = group[mask]
        n = len(valid)
        if n == 1:
            scaled = pd.Series(0.5, index=valid.index)
        else:
            ranks = valid.rank(method="average")
            scaled = (ranks - 1) / (len(valid) - 1)
        result = pd.Series(np.nan, index=group.index)
        result.loc[mask] = scaled.values
        return result

    return values.groupby(groups, group_keys=False, sort=False).apply(_rank)


def _time_series_rank(values: pd.Series, groups: pd.Series, window: int) -> pd.Series:
    """Time-series rank of the latest value within a rolling window per group."""

    def _apply(group: pd.Series) -> pd.Series:
        rolling = group.rolling(window=window, min_periods=window)

        def _rank_window(window_series: pd.Series) -> float:
            if window_series.isna().iloc[-1]:
                return np.nan
            valid = window_series.dropna()
            if valid.empty:
                return np.nan
            if len(valid) == 1:
                return 0.5
            rank = valid.rank(method="average").iloc[-1]
            return (rank - 1) / (len(valid) - 1)

        return rolling.apply(_rank_window, raw=False)

    return values.groupby(groups, group_keys=False, sort=False).apply(_apply)


def _grouped_rolling_corr(
    df: pd.DataFrame, col_x: str, col_y: str, window: int, min_periods: int | None = None
) -> pd.Series:
    """Rolling correlation between two columns per item."""
    min_periods = min_periods or window
    grouped = df.groupby("item_id", group_keys=False, sort=False)

    def _corr(group: pd.DataFrame) -> pd.Series:
        """Vectorized rolling corr within a single item_id."""
        pair = group[[col_x, col_y]]
        rolling_pair = pair.rolling(window=window, min_periods=min_periods)
        corr_df = rolling_pair.corr().xs(col_x, level=1)[col_y]

        # Mark zero-variance windows as 0 (same intent as previous impl), keep
        # insufficient-window NaNs untouched.
        std = rolling_pair.std(ddof=0)
        zero_var = (std[col_x] <= 0) | (std[col_y] <= 0)
        corr_df = corr_df.where(~zero_var, 0.0)
        return corr_df

    return grouped.apply(_corr)


def _grouped_rolling_cov(
    df: pd.DataFrame, col_x: str, col_y: str, window: int, min_periods: int | None = None
) -> pd.Series:
    """Rolling covariance between two columns per item."""
    min_periods = min_periods or window
    grouped = df.groupby("item_id", group_keys=False, sort=False)
    return grouped.apply(
        lambda g: g[col_x].rolling(window=window, min_periods=min_periods).cov(g[col_y])
    )


def add_alpha101_features(df: pd.DataFrame, only: Sequence[str] | None = None) -> List[str]:
    """Compute Alpha101-inspired factors #2-#55 using cross-sectional ranks.

    If ``only`` is provided, compute only those alphas (case-insensitive). This
    keeps preprocessing faster when a small subset is needed.
    """

    if df.empty:
        return []

    only_set = {name.lower() for name in only} if only is not None else None

    def needs(name: str) -> bool:
        return only_set is None or name.lower() in only_set

    eps = 1e-12
    alpha_cols: List[str] = []
    tmp_cols: List[str] = []
    date_index = df["date"]
    item_groups = df.groupby("item_id", group_keys=False, sort=False)
    cs_rank_cache: Dict[str, pd.Series] = {}

    # If caller requested no alpha factors, return early.
    if only_set is not None and not any(name.startswith("alpha") for name in only_set):
        return []

    def store_tmp(name: str, series: pd.Series) -> pd.Series:
        df[name] = series
        tmp_cols.append(name)
        return df[name]

    def get_cs_rank(key: str, series: pd.Series) -> pd.Series:
        if key not in cs_rank_cache:
            cs_rank_cache[key] = _cross_sectional_rank(series, date_index)
        return cs_rank_cache[key]

    def _scale_to_abs_sum(series: pd.Series, target: float = 1.0) -> pd.Series:
        """Cross-sectional scale so sum(abs(x)) equals ``target`` per date."""

        def _scale(group: pd.Series) -> pd.Series:
            mask = group.notna()
            if not mask.any():
                return group
            values = group[mask]
            denom = np.abs(values).sum()
            scaled = pd.Series(np.nan, index=group.index)
            if denom > eps:
                scaled.loc[mask] = values * (target / denom)
            else:
                scaled.loc[mask] = 0.0
            return scaled

        return series.groupby(date_index, group_keys=False, sort=False).apply(_scale)

    def _time_since_argmax(values: pd.Series, window: int) -> pd.Series:
        """Rolling argmax position (0 = latest) within each item window."""

        def _apply(group: pd.Series) -> pd.Series:
            rolling = group.rolling(window=window, min_periods=window)

            def _argmax_distance(window_series: pd.Series) -> float:
                if window_series.isna().any():
                    return np.nan
                idx = int(np.argmax(window_series.to_numpy()))
                return window - 1 - idx

            return rolling.apply(_argmax_distance, raw=False)

        return values.groupby(df["item_id"], group_keys=False, sort=False).apply(_apply)

    # Precompute frequently used ranks.
    rank_open = store_tmp("_tmp_rank_open", get_cs_rank("open", df["open"]))
    rank_high = store_tmp("_tmp_rank_high", get_cs_rank("high", df["high"]))
    rank_low = get_cs_rank("low", df["low"])
    rank_close = store_tmp("_tmp_rank_close", get_cs_rank("close", df["close"]))
    rank_volume = store_tmp("_tmp_rank_volume", get_cs_rank("volume", df["volume"]))

    # Alpha#2
    if needs("alpha002"):
        log_volume = np.log(df["volume"].clip(lower=eps))
        delta_log_volume_2 = log_volume.groupby(df["item_id"]).diff(2)
        rank_delta_log_volume = store_tmp(
            "_tmp_rank_delta_log_vol_2",
            _cross_sectional_rank(delta_log_volume_2, date_index),
        )
        close_open_rel = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)
        rank_close_open = store_tmp(
            "_tmp_rank_close_open",
            _cross_sectional_rank(close_open_rel, date_index),
        )
        corr_alpha2 = _grouped_rolling_corr(
            df, "_tmp_rank_delta_log_vol_2", "_tmp_rank_close_open", window=6
        )
        df["alpha002"] = -corr_alpha2
        alpha_cols.append("alpha002")

    # Alpha#3
    if needs("alpha003"):
        corr_alpha3 = _grouped_rolling_corr(
            df, "_tmp_rank_open", "_tmp_rank_volume", window=10
        )
        df["alpha003"] = -corr_alpha3
        alpha_cols.append("alpha003")

    # Alpha#4
    if needs("alpha004"):
        ts_rank_low = _time_series_rank(rank_low, df["item_id"], window=9)
        df["alpha004"] = -ts_rank_low
        alpha_cols.append("alpha004")

    # Alpha#5
    if needs("alpha005"):
        vwap_mean_10 = item_groups["vwap"].transform(
            lambda s: s.rolling(window=10, min_periods=10).mean()
        )
        open_minus_avg = df["open"] - vwap_mean_10
        rank_open_minus = _cross_sectional_rank(open_minus_avg, date_index)
        close_minus_vwap = df["close"] - df["vwap"]
        rank_close_minus = _cross_sectional_rank(close_minus_vwap, date_index)
        df["alpha005"] = rank_open_minus * (-np.abs(rank_close_minus))
        alpha_cols.append("alpha005")

    # Alpha#6 (reused later)
    corr_open_volume_10 = None
    if needs("alpha006") or needs("alpha014"):
        corr_open_volume_10 = _grouped_rolling_corr(df, "open", "volume", window=10)
    if needs("alpha006"):
        df["alpha006"] = -corr_open_volume_10
        alpha_cols.append("alpha006")

    # Alpha#8
    returns = item_groups["close"].transform(lambda s: s.pct_change())
    store_tmp("_tmp_returns", returns)
    if needs("alpha008"):
        sum_open_5 = item_groups["open"].transform(lambda s: s.rolling(window=5, min_periods=5).sum())
        sum_return_5 = item_groups["_tmp_returns"].transform(
            lambda s: s.rolling(window=5, min_periods=5).sum()
        )
        prod_series = sum_open_5 * sum_return_5
        store_tmp("_tmp_alpha8_prod", prod_series)
        prod_delay_10 = item_groups["_tmp_alpha8_prod"].shift(10)
        rank_prod_delta = _cross_sectional_rank(df["_tmp_alpha8_prod"] - prod_delay_10, date_index)
        df["alpha008"] = -rank_prod_delta
        alpha_cols.append("alpha008")

    # Alpha#11
    if needs("alpha011"):
        store_tmp("_tmp_vwap_close_spread", df["vwap"] - df["close"])
        ts_max_3 = item_groups["_tmp_vwap_close_spread"].transform(
            lambda s: s.rolling(window=3, min_periods=3).max()
        )
        ts_min_3 = item_groups["_tmp_vwap_close_spread"].transform(
            lambda s: s.rolling(window=3, min_periods=3).min()
        )
        rank_ts_max = _cross_sectional_rank(ts_max_3, date_index)
        rank_ts_min = _cross_sectional_rank(ts_min_3, date_index)
        delta_volume_3 = item_groups["volume"].diff(3)
        rank_delta_volume_3 = _cross_sectional_rank(delta_volume_3, date_index)
        df["alpha011"] = (rank_ts_max + rank_ts_min) * rank_delta_volume_3
        alpha_cols.append("alpha011")

    # Alpha#12
    delta_volume_1 = item_groups["volume"].diff(1)
    delta_close_1 = item_groups["close"].diff(1)
    if needs("alpha012"):
        df["alpha012"] = np.sign(delta_volume_1) * (-delta_close_1)
        alpha_cols.append("alpha012")

    # Alpha#13
    if needs("alpha013"):
        cov_rank_close_volume = _grouped_rolling_cov(df, "_tmp_rank_close", "_tmp_rank_volume", window=5)
        df["alpha013"] = -_cross_sectional_rank(cov_rank_close_volume, date_index)
        alpha_cols.append("alpha013")

    # Alpha#14
    if needs("alpha014"):
        if corr_open_volume_10 is None:
            corr_open_volume_10 = _grouped_rolling_corr(df, "open", "volume", window=10)
        delta_returns_3 = item_groups["_tmp_returns"].diff(3)
        rank_delta_returns_3 = _cross_sectional_rank(delta_returns_3, date_index)
        df["alpha014"] = (-rank_delta_returns_3) * corr_open_volume_10
        alpha_cols.append("alpha014")

    # Alpha#15
    if needs("alpha015"):
        corr_rank_high_vol = _grouped_rolling_corr(df, "_tmp_rank_high", "_tmp_rank_volume", window=3)
        rank_corr = _cross_sectional_rank(corr_rank_high_vol, date_index)
        sum_rank_corr = rank_corr.groupby(df["item_id"]).transform(
            lambda s: s.rolling(window=3, min_periods=3).sum()
        )
        df["alpha015"] = -sum_rank_corr
        alpha_cols.append("alpha015")

    # Reusable helper columns for later alphas.
    ts_rank_close_10 = store_tmp(
        "_tmp_ts_rank_close_10", _time_series_rank(df["close"], df["item_id"], window=10)
    )
    adv20 = store_tmp(
        "_tmp_adv20",
        df["turnover"].groupby(df["item_id"]).transform(
            lambda s: s.rolling(window=20, min_periods=20).mean()
        ),
    )

    # Alpha#16
    if needs("alpha016"):
        cov_rank_high_volume = _grouped_rolling_cov(df, "_tmp_rank_high", "_tmp_rank_volume", window=5)
        df["alpha016"] = -_cross_sectional_rank(cov_rank_high_volume, date_index)
        alpha_cols.append("alpha016")

    # Alpha#17
    if needs("alpha017"):
        rank_ts_rank_close = _cross_sectional_rank(ts_rank_close_10, date_index)
        delta_delta_close = delta_close_1.groupby(df["item_id"]).diff(1)
        rank_delta_delta_close = _cross_sectional_rank(delta_delta_close, date_index)
        volume_adv_ratio = df["volume"] / adv20.replace({0: np.nan})
        ts_rank_volume_adv = store_tmp(
            "_tmp_ts_rank_volume_adv",
            _time_series_rank(volume_adv_ratio, df["item_id"], window=5),
        )
        rank_ts_rank_ratio = _cross_sectional_rank(ts_rank_volume_adv, date_index)
        df["alpha017"] = (-rank_ts_rank_close) * rank_delta_delta_close * rank_ts_rank_ratio
        alpha_cols.append("alpha017")

    # Alpha#18
    if needs("alpha018"):
        abs_close_open = (df["close"] - df["open"]).abs()
        std_abs_close_open = abs_close_open.groupby(df["item_id"]).transform(
            lambda s: s.rolling(window=5, min_periods=5).std()
        )
        corr_close_open_10 = _grouped_rolling_corr(df, "close", "open", window=10)
        composite = std_abs_close_open + (df["close"] - df["open"]) + corr_close_open_10
        df["alpha018"] = -_cross_sectional_rank(composite, date_index)
        alpha_cols.append("alpha018")

    # Precompute lags used across multiple alphas.
    high_lag1 = item_groups["high"].shift(1)
    close_lag1 = item_groups["close"].shift(1)
    low_lag1 = item_groups["low"].shift(1)

    # Alpha#20
    if needs("alpha020"):
        rank_open_minus_high = _cross_sectional_rank(df["open"] - high_lag1, date_index)
        rank_open_minus_close = _cross_sectional_rank(df["open"] - close_lag1, date_index)
        rank_open_minus_low = _cross_sectional_rank(df["open"] - low_lag1, date_index)
        df["alpha020"] = (-rank_open_minus_high) * rank_open_minus_close * rank_open_minus_low
        alpha_cols.append("alpha020")

    # Alpha#22
    if needs("alpha022"):
        corr_high_volume_5 = _grouped_rolling_corr(df, "high", "volume", window=5)
        delta_corr_high_volume = corr_high_volume_5.groupby(df["item_id"]).diff(5)
        std_close_20 = item_groups["close"].transform(
            lambda s: s.rolling(window=20, min_periods=20).std()
        )
        rank_std_close_20 = _cross_sectional_rank(std_close_20, date_index)
        df["alpha022"] = -(delta_corr_high_volume * rank_std_close_20)
        alpha_cols.append("alpha022")

    # Alpha#23
    if needs("alpha023"):
        mean_high_20 = item_groups["high"].transform(
            lambda s: s.rolling(window=20, min_periods=20).mean()
        )
        delta_high_2 = item_groups["high"].diff(2)
        df["alpha023"] = np.where(mean_high_20 < df["high"], -delta_high_2, 0.0)
        alpha_cols.append("alpha023")

    # Alpha#25
    if needs("alpha025"):
        high_minus_close = df["high"] - df["close"]
        alpha25_raw = (-df["_tmp_returns"]) * adv20 * df["vwap"] * high_minus_close
        df["alpha025"] = _cross_sectional_rank(alpha25_raw, date_index)
        alpha_cols.append("alpha025")

    # Alpha#26
    if needs("alpha026"):
        ts_rank_volume_5 = store_tmp(
            "_tmp_ts_rank_volume_5", _time_series_rank(df["volume"], df["item_id"], window=5)
        )
        ts_rank_high_5 = store_tmp(
            "_tmp_ts_rank_high_5", _time_series_rank(df["high"], df["item_id"], window=5)
        )
        corr_ts_rank = _grouped_rolling_corr(
            df, "_tmp_ts_rank_volume_5", "_tmp_ts_rank_high_5", window=5
        )
        ts_max_corr = corr_ts_rank.groupby(df["item_id"]).transform(
            lambda s: s.rolling(window=3, min_periods=3).max()
        )
        df["alpha026"] = -ts_max_corr
        alpha_cols.append("alpha026")

    # Alpha#30
    if needs("alpha030"):
        close_lag2 = item_groups["close"].shift(2)
        close_lag3 = item_groups["close"].shift(3)
        sign_sum = (
            np.sign(df["close"] - close_lag1)
            + np.sign(close_lag1 - close_lag2)
            + np.sign(close_lag2 - close_lag3)
        )
        rank_sign_sum = _cross_sectional_rank(sign_sum, date_index)
        sum_volume_5 = item_groups["volume"].transform(
            lambda s: s.rolling(window=5, min_periods=5).sum()
        )
        sum_volume_20 = item_groups["volume"].transform(
            lambda s: s.rolling(window=20, min_periods=20).sum()
        )
        df["alpha030"] = ((1.0 - rank_sign_sum) * sum_volume_5) / sum_volume_20.replace({0: np.nan})
        alpha_cols.append("alpha030")

    # Alpha#33
    if needs("alpha033"):
        ratio_open_close = df["open"] / df["close"].replace({0: np.nan})
        df["alpha033"] = _cross_sectional_rank(-(1 - ratio_open_close), date_index)
        alpha_cols.append("alpha033")

    # Alpha#34
    if needs("alpha034"):
        std_returns_2 = item_groups["_tmp_returns"].transform(
            lambda s: s.rolling(window=2, min_periods=2).std()
        )
        std_returns_5 = item_groups["_tmp_returns"].transform(
            lambda s: s.rolling(window=5, min_periods=5).std()
        )
        ratio_std = std_returns_2 / std_returns_5.replace({0: np.nan})
        rank_ratio_std = _cross_sectional_rank(ratio_std, date_index)
        rank_delta_close = _cross_sectional_rank(delta_close_1, date_index)
        df["alpha034"] = _cross_sectional_rank(
            (1 - rank_ratio_std) + (1 - rank_delta_close), date_index
        )
        alpha_cols.append("alpha034")

    # Alpha#38
    if needs("alpha038"):
        rank_close_open_ratio = _cross_sectional_rank(
            df["close"] / df["open"].replace({0: np.nan}), date_index
        )
        rank_ts_rank_close = _cross_sectional_rank(ts_rank_close_10, date_index)
        df["alpha038"] = (-rank_ts_rank_close) * rank_close_open_ratio
        alpha_cols.append("alpha038")

    # Alpha#40
    if needs("alpha040"):
        std_high_10 = item_groups["high"].transform(
            lambda s: s.rolling(window=10, min_periods=10).std()
        )
        rank_std_high_10 = _cross_sectional_rank(std_high_10, date_index)
        corr_high_volume_10 = _grouped_rolling_corr(df, "high", "volume", window=10)
        df["alpha040"] = (-rank_std_high_10) * corr_high_volume_10
        alpha_cols.append("alpha040")

    # Alpha#41
    if needs("alpha041"):
        df["alpha041"] = np.sqrt(df["high"] * df["low"]) - df["vwap"]
        alpha_cols.append("alpha041")

    # Alpha#42
    if needs("alpha042"):
        rank_vwap_minus_close = _cross_sectional_rank(df["vwap"] - df["close"], date_index)
        rank_vwap_plus_close = _cross_sectional_rank(df["vwap"] + df["close"], date_index)
        df["alpha042"] = rank_vwap_minus_close /( rank_vwap_plus_close.replace(0, np.nan))
        alpha_cols.append("alpha042")

    # Alpha#43
    if needs("alpha043"):
        volume_adv_ratio = df["volume"] / adv20.replace({0: np.nan})
        ts_rank_vol_adv_20 = _time_series_rank(
            volume_adv_ratio, df["item_id"], window=20
        )
        delta_close_7 = item_groups["close"].diff(7)
        ts_rank_neg_delta_close_8 = _time_series_rank(-delta_close_7, df["item_id"], window=8)
        df["alpha043"] = ts_rank_vol_adv_20 * ts_rank_neg_delta_close_8
        alpha_cols.append("alpha043")

    # Alpha#44
    if needs("alpha044"):
        corr_high_rank_vol_5 = _grouped_rolling_corr(df, "high", "_tmp_rank_volume", window=5)
        df["alpha044"] = -corr_high_rank_vol_5
        alpha_cols.append("alpha044")

    # Alpha#45
    if needs("alpha045"):
        sum_delay_close_5_20 = item_groups["close"].transform(
            lambda s: s.shift(5).rolling(window=20, min_periods=20).sum()
        )
        avg_delay_close_5_20 = sum_delay_close_5_20 / 20
        rank_avg_delay_close = _cross_sectional_rank(avg_delay_close_5_20, date_index)

        sum_close_5 = store_tmp(
            "_tmp_sum_close_5",
            item_groups["close"].transform(
                lambda s: s.rolling(window=5, min_periods=5).sum()
            ),
        )
        sum_close_20 = store_tmp(
            "_tmp_sum_close_20",
            item_groups["close"].transform(
                lambda s: s.rolling(window=20, min_periods=20).sum()
            ),
        )
        corr_close_volume_2 = _grouped_rolling_corr(df, "close", "volume", window=2, min_periods=2)
        corr_sum5_sum20_2 = _grouped_rolling_corr(
            df, "_tmp_sum_close_5", "_tmp_sum_close_20", window=2, min_periods=2
        )
        rank_corr_sum = _cross_sectional_rank(corr_sum5_sum20_2, date_index)
        df["alpha045"] = -(rank_avg_delay_close * corr_close_volume_2 * rank_corr_sum)
        alpha_cols.append("alpha045")

    # Shared slope difference used by alpha46/49/51
    close_lag10 = item_groups["close"].shift(10)
    close_lag20 = item_groups["close"].shift(20)
    slope_diff = ((close_lag20 - close_lag10) / 10) - ((close_lag10 - df["close"]) / 10)

    # Alpha#46
    if needs("alpha046"):
        df["alpha046"] = np.where(
            slope_diff > 0.25,
            -1.0,
            np.where(slope_diff < 0, 1.0, -(df["close"] - close_lag1)),
        )
        alpha_cols.append("alpha046")

    # Alpha#47
    if needs("alpha047"):
        rank_inv_close = _cross_sectional_rank(1 / df["close"].replace({0: np.nan}), date_index)
        high_minus_close_rank = _cross_sectional_rank(df["high"] - df["close"], date_index)
        sum_high_5 = item_groups["high"].transform(
            lambda s: s.rolling(window=5, min_periods=5).sum()
        )
        avg_high_5 = sum_high_5 / 5
        top = (rank_inv_close * df["volume"]) / adv20.replace({0: np.nan})
        bottom = (df["high"] * high_minus_close_rank) / avg_high_5.replace({0: np.nan})
        rank_vwap_gap_5 = _cross_sectional_rank(
            df["vwap"] - item_groups["vwap"].shift(5), date_index
        )
        df["alpha047"] = (top * bottom) - rank_vwap_gap_5
        alpha_cols.append("alpha047")

    # Alpha#49
    if needs("alpha049"):
        df["alpha049"] = np.where(slope_diff < -0.1, 1.0, -(df["close"] - close_lag1))
        alpha_cols.append("alpha049")

    # Alpha#50
    if needs("alpha050"):
        rank_vwap = store_tmp("_tmp_rank_vwap", get_cs_rank("vwap", df["vwap"]))
        corr_rank_vol_vwap_5 = _grouped_rolling_corr(
            df, "_tmp_rank_volume", "_tmp_rank_vwap", window=5
        )
        rank_corr_rank_vol_vwap = _cross_sectional_rank(corr_rank_vol_vwap_5, date_index)
        ts_max_rank_corr = rank_corr_rank_vol_vwap.groupby(df["item_id"]).transform(
            lambda s: s.rolling(window=5, min_periods=5).max()
        )
        df["alpha050"] = -ts_max_rank_corr
        alpha_cols.append("alpha050")

    # Alpha#51
    if needs("alpha051"):
        df["alpha051"] = np.where(slope_diff < -0.05, 1.0, -(df["close"] - close_lag1))
        alpha_cols.append("alpha051")

    # Alpha#53
    if needs("alpha053"):
        close_low_denom = (df["close"] - df["low"]).replace(0, np.nan)
        base_clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / close_low_denom
        df["alpha053"] = -base_clv.groupby(df["item_id"]).diff(9)
        alpha_cols.append("alpha053")

    # Alpha#54
    if needs("alpha054"):
        numerator_54 = -(df["low"] - df["close"]) * (df["open"] ** 5)
        denominator_54 = (df["low"] - df["high"]) * (df["close"] ** 5)
        df["alpha054"] = numerator_54 / denominator_54.replace(0, np.nan)
        alpha_cols.append("alpha054")

    # Alpha#55
    if needs("alpha055"):
        ts_min_low_12 = item_groups["low"].transform(
            lambda s: s.rolling(window=12, min_periods=12).min()
        )
        ts_max_high_12 = item_groups["high"].transform(
            lambda s: s.rolling(window=12, min_periods=12).max()
        )
        range_denom_12 = (ts_max_high_12 - ts_min_low_12).replace(0, np.nan)
        price_range_pos = (df["close"] - ts_min_low_12) / range_denom_12
        store_tmp(
            "_tmp_rank_price_range_12", _cross_sectional_rank(price_range_pos, date_index)
        )
        corr_rank_range_rank_vol_6 = _grouped_rolling_corr(
            df, "_tmp_rank_price_range_12", "_tmp_rank_volume", window=6
        )
        df["alpha055"] = -corr_rank_range_rank_vol_6
        alpha_cols.append("alpha055")

    # Alpha#60
    if needs("alpha060"):
        range_hl = df["high"] - df["low"]
        safe_range_hl = range_hl.replace(0, np.nan)
        price_bias = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / safe_range_hl
        volume_tilt = price_bias * df["volume"]
        rank_volume_tilt = _cross_sectional_rank(volume_tilt, date_index)
        scaled_volume_tilt = _scale_to_abs_sum(rank_volume_tilt)

        ts_argmax_close_10 = _time_since_argmax(df["close"], window=10)
        rank_argmax_close = _cross_sectional_rank(ts_argmax_close_10, date_index)
        scaled_argmax_close = _scale_to_abs_sum(rank_argmax_close)

        df["alpha060"] = scaled_argmax_close - 2 * scaled_volume_tilt
        alpha_cols.append("alpha060")

    # Alpha#83
    if needs("alpha083"):
        close_ma_5 = item_groups["close"].transform(
            lambda s: s.rolling(window=5, min_periods=5).mean()
        )
        range_norm = (df["high"] - df["low"]) / close_ma_5.replace({0: np.nan})
        delayed_range = range_norm.groupby(df["item_id"]).shift(2)
        rank_delayed_range = _cross_sectional_rank(delayed_range, date_index)
        rank_rank_volume = _cross_sectional_rank(rank_volume, date_index)

        denom_83 = range_norm / (df["vwap"] - df["close"])
        df["alpha083"] = (rank_delayed_range * rank_rank_volume) / denom_83.replace(0, np.nan)
        alpha_cols.append("alpha083")

    # Alpha#101
    if needs("alpha101"):
        df["alpha101"] = (df["close"] - df["open"]) / ((df["high"] - df["low"]) + 0.001)
        alpha_cols.append("alpha101")

    # Clean up temporary helper columns.
    if tmp_cols:
        df.drop(columns=tmp_cols, inplace=True, errors="ignore")

    return alpha_cols


def filter_by_window(
    df: pd.DataFrame,
    start_ts: pd.Timestamp | None,
    end_ts: pd.Timestamp | None,
) -> pd.DataFrame:
    """Filter daily dataframe by session start time defined at 15:00 local."""
    if df.empty or (start_ts is None and end_ts is None):
        return df

    index_tz = df.index.tz_localize(ASIA_SHANGHAI)
    session_start = index_tz + pd.Timedelta(hours=TRADE_END_HOUR)

    mask = pd.Series(True, index=df.index)
    if start_ts is not None:
        mask &= session_start >= start_ts
    if end_ts is not None:
        mask &= session_start <= end_ts

    return df.loc[mask]


def analyze_nan_distribution(
    df: pd.DataFrame,
    factor_cols: Sequence[str],
    output_file: str = "nan_analysis.csv"
) -> None:
    """Analyze and report NaN value distribution by date, asset, and factor."""
    nan_records = []

    print("Analyzing NaN distribution...")

    # Overall NaN summary
    total_cells = len(df) * len(factor_cols)
    nan_count = df[factor_cols].isna().sum().sum()
    print(f"Overall NaN rate: {nan_count}/{total_cells} ({nan_count/total_cells*100:.2f}%)")

    # Per-date NaN analysis
    for date, group in df.groupby("date"):
        date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
        total_assets = len(group)

        for factor in factor_cols:
            nan_assets = group[factor].isna().sum()
            if nan_assets > 0:
                nan_records.append({
                    "date": date_str,
                    "factor": factor,
                    "nan_count": nan_assets,
                    "total_assets": total_assets,
                    "nan_rate": nan_assets / total_assets,
                    "asset_list": group[group[factor].isna()]["item_id"].tolist()
                })

    # Per-asset NaN analysis
    asset_nan_summary = {}
    for item_id, group in df.groupby("item_id"):
        total_obs = len(group)
        nan_by_factor = {}
        for factor in factor_cols:
            nan_count = group[factor].isna().sum()
            if nan_count > 0:
                nan_by_factor[factor] = nan_count

        if nan_by_factor:
            asset_nan_summary[item_id] = {
                "total_obs": total_obs,
                "nan_factors": nan_by_factor,
                "total_nan_count": sum(nan_by_factor.values())
            }

    # Save detailed analysis
    if nan_records:
        nan_df = pd.DataFrame(nan_records)
        nan_df.to_csv(output_file, index=False)
        print(f"Detailed NaN analysis saved to {output_file}")

        # Summary statistics
        print(f"\n=== NaN Summary ===")
        print(f"Dates with NaN: {nan_df['date'].nunique()}")
        print(f"Factors with NaN: {nan_df['factor'].nunique()}")
        print(f"Most problematic factor: {nan_df.groupby('factor')['nan_rate'].mean().idxmax()}")
        print(f"Most problematic date: {nan_df.groupby('date')['nan_rate'].mean().idxmax()}")
    else:
        print("No NaN values found in factor columns!")

    # Asset-level summary
    if asset_nan_summary:
        print(f"\n=== Asset-level NaN Summary ===")
        worst_assets = sorted(asset_nan_summary.items(),
                            key=lambda x: x[1]['total_nan_count'], reverse=True)[:5]
        for item_id, stats in worst_assets:
            print(f"{item_id}: {stats['total_nan_count']} NaNs across {len(stats['nan_factors'])} factors")


def remove_cross_sectional_outliers_mad(
    df: pd.DataFrame,
    factor_cols: Sequence[str],
    n: float = 3.0,
) -> pd.DataFrame:
    """Remove cross-sectional outliers using MAD method with n-sigma threshold."""
    result = df.copy()

    for factor in factor_cols:
        # Process each date's cross-section
        for date, group in result.groupby("date"):
            values = group[factor].dropna()
            if len(values) == 0:
                continue

            median = np.median(values)
            mad = np.median(np.abs(values - median))

            if mad > 0:
                threshold = n * mad * 1.4826  # MAD to sigma conversion
                outlier_mask = np.abs(values - median) > threshold
                # Replace outliers with median (winsorization)
                original_indices = values.index[outlier_mask]
                if len(original_indices) > 0:
                    result.loc[original_indices, factor] = median

    return result


def neutralize_cross_section(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    industry_col: str,
    continuous_cols: Sequence[str],
) -> pd.DataFrame:
    """Neutralize factors by industry + continuous covariates, return residuals."""
    residuals = pd.DataFrame(index=df.index, columns=feature_cols, dtype=float)

    for date, idx in df.groupby("date").groups.items():
        sub = df.loc[idx]
        design = pd.DataFrame(index=sub.index, dtype=float)
        design["intercept"] = 1.0
        for col in continuous_cols:
            if col not in sub:
                raise KeyError(f"Continuous column '{col}' missing for neutralization.")
            design[col] = sub[col]
        '''industry_dummies = pd.get_dummies(sub[industry_col], drop_first=True)
        if not industry_dummies.empty:
            design = pd.concat([design, industry_dummies], axis=1)'''

        design = design.astype(float)
        valid_design_mask = design.notna().all(axis=1)

        for feature in feature_cols:
            values = sub[feature]
            mask = values.notna() & valid_design_mask
            feature_idx = mask.index[mask]
            if feature_idx.empty:
                continue

            X = design.loc[feature_idx].to_numpy(dtype=float, copy=True)
            y = values.loc[feature_idx].to_numpy(dtype=float, copy=True)

            if X.shape[0] <= X.shape[1]:
                print(
                    "⚠️  Neutralization skipped: "
                    f"{date.date()} feature '{feature}' has samples={X.shape[0]} "
                    f"< columns={X.shape[1]}; filling residuals with NaN."
                )
                residuals.loc[feature_idx, feature] = np.nan
                continue
            try:
                beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                resid = y - X @ beta
            except np.linalg.LinAlgError as exc:
                raise RuntimeError(
                    "Neutralization regression singular on "
                    f"{date.date()} for feature '{feature}'."
                ) from exc

            resid_series = pd.Series(resid, index=feature_idx)
            mean = resid_series.mean()
            std = resid_series.std(ddof=0)
            if std and std > 0:
                resid_series = (resid_series - mean) / std
            else:
                resid_series = resid_series - mean

            residuals.loc[feature_idx, feature] = resid_series

    return residuals


def compute_rank_ic(
    df: pd.DataFrame,
    factor_cols: Sequence[str],
    target_col: str,
    lags: Sequence[int],
    *,
    prelagged: bool = False,
) -> pd.DataFrame:
    """Compute cross-sectional Rank IC for each factor / lag combination.

    If ``prelagged`` is True, assumes columns named ``{factor}_lag{lag}`` already
    exist in ``df`` and skips computing shifts. This allows retaining lag values
    that depend on history outside the analysis window.
    """
    ic_records: List[Dict[str, object]] = []
    grouped_items = df.groupby("item_id", sort=False)

    if not prelagged:
        lagged_cache = {}
        for lag in lags:
            lagged = grouped_items[factor_cols].shift(lag)
            lagged.columns = [f"{c}_lag{lag}" for c in factor_cols]
            lagged_cache[lag] = lagged
        lagged_df = pd.concat(lagged_cache.values(), axis=1)
        df = pd.concat([df, lagged_df], axis=1)

    for date, sub in df.groupby("date"):
        target = sub[target_col]
        if target.notna().sum() < MIN_CROSS_SECTION:
            continue

        target_rank = target.rank(method="average")
        target_valid = target_rank.notna()
        if target_valid.sum() < MIN_CROSS_SECTION:
            continue

        for factor in factor_cols:
            factor_name = factor.replace("_neutral", "").replace("_n", "")
            for lag in lags:
                col_name = f"{factor}_lag{lag}"
                if col_name not in sub:
                    continue
                values = sub[col_name]
                mask = target_valid & values.notna()
                if mask.sum() < MIN_CROSS_SECTION:
                    continue
                factor_rank = values[mask].rank(method="average")
                aligned_target = target_rank[mask]
                if factor_rank.std(ddof=0) == 0 or aligned_target.std(ddof=0) == 0:
                    continue
                corr = factor_rank.corr(aligned_target)
                if not math.isnan(corr):
                    ic_records.append(
                        {"date": date, "factor": factor_name, "lag": lag, "ic": corr}
                    )

    if not ic_records:
        return pd.DataFrame(columns=["factor", "lag", "mean", "std", "count"])

    ic_df = pd.DataFrame(ic_records)
    summary = (
        ic_df.groupby(["factor", "lag"])["ic"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )
    return summary


def plot_heatmap(matrix: pd.DataFrame, output_path: Path) -> None:
    """Render and save a heatmap for the given matrix."""
    if matrix.empty:
        raise ValueError("Heatmap matrix is empty; nothing to plot.")

    data = matrix.to_numpy()
    vmax = np.nanmax(np.abs(data))
    vmax = 0.01 if np.isnan(vmax) or vmax == 0 else vmax

    fig, ax = plt.subplots(figsize=(1.5 + matrix.shape[1], 0.5 + 0.4 * matrix.shape[0]))
    cmap = plt.get_cmap("coolwarm")
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Factor")
    ax.set_title("Rank IC Mean by Factor / Lag")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iat[i, j]
            if np.isnan(value):
                continue
            text_color = "white" if abs(value) > vmax * 0.6 else "black"
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", color=text_color)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Rank IC")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    mapping_path = Path(args.mapping)
    heatmap_path = Path(args.heatmap)
    summary_path = Path(args.output_summary)

    start_ts = parse_window_endpoint(args.start)
    end_ts = parse_window_endpoint(args.end)
    if start_ts and end_ts and start_ts > end_ts:
        raise ValueError("Start timestamp must be earlier than or equal to end timestamp.")

    # Calculate extended start date for data loading to support long-window factors
    extended_start_ts = None
    if start_ts is not None:
        extended_start_ts = start_ts - pd.Timedelta(days=args.history_days)

    print(f"Loading data from {extended_start_ts} to {end_ts} for factor calculation (history: {args.history_days} days)")
    print(f"Analysis will be performed on user-specified range: {start_ts} to {end_ts}")

    industry_mapping = load_industry_mapping(mapping_path)

    records: List[pd.DataFrame] = []
    feature_columns: List[str] = []

    for json_path in sorted(data_dir.glob("*.json")):
        item_id = json_path.stem
        hourly_df = load_hourly_json(json_path)
        daily_df = aggregate_to_daily(hourly_df)
        feature_df = compute_features(daily_df)
        # Filter to the analysis window (extended start to end) to load sufficient data
        feature_df = filter_by_window(feature_df, extended_start_ts, end_ts)
        if not feature_df.empty:
            feature_df["target_8d"] = feature_df["close"].shift(-8) / feature_df["close"] - 1
        if feature_df.empty:
            continue
        feature_df["item_id"] = item_id
        feature_df["industry"] = industry_mapping.get(item_id, "UNKNOWN")
        records.append(feature_df.reset_index())

        if not feature_columns:
            feature_columns = [
                col
                for col in feature_df.columns
                if col
                not in {
                    "item_id",
                    "industry",
                    "price_ma_3",
                    "volume_ma_3",
                    "target_8d",
                    "obs",
                    "vwap",
                }
                and col not in {"open", "high", "low", "close", "volume", "turnover"}
                and feature_df[col].dtype.kind in {"f", "i"}
            ]

    if not records:
        raise RuntimeError(f"No usable daily data found under {data_dir}.")

    all_df = pd.concat(records, ignore_index=True)
    all_df["date"] = pd.to_datetime(all_df["date"])
    all_df = all_df.sort_values(["date", "item_id"]).reset_index(drop=True)

    print(f"Loaded {len(all_df)} records with extended history for alpha factor computation")

    # Compute alpha101 factors FIRST using complete historical data
    alpha_feature_cols = add_alpha101_features(all_df)
    feature_columns.extend(alpha_feature_cols)

    # Cross-sectional MAD outlier removal before neutralization
    print(f"Applying MAD outlier removal with threshold {args.mad_threshold}...")
    all_df = remove_cross_sectional_outliers_mad(all_df, feature_columns, n=args.mad_threshold)

    # Decide which columns to neutralize; alpha101 factors can be optionally skipped.
    alpha_set = set(alpha_feature_cols)
    features_to_neutralize = [
        col
        for col in feature_columns
        if col not in LOG_FEATURES and (NEUTRALIZE_ALPHA101 or col not in alpha_set)
    ]
    continuous_covariates = LOG_FEATURES
    neutralized = neutralize_cross_section(
        all_df, features_to_neutralize, industry_col="industry", continuous_cols=continuous_covariates
    )

    # Standardize log covariates as standalone features (no neutralization)
    standardized_logs = {}
    for col in LOG_FEATURES:
        if col in all_df.columns:
            def _zscore(series: pd.Series) -> pd.Series:
                mean = series.mean()
                std = series.std(ddof=0)
                return (series - mean) / std if std and std > 0 else series - mean

            standardized_logs[col] = all_df.groupby("date")[col].transform(_zscore)

    neutral_suffix_map = {}
    for col in feature_columns:
        new_col = f"{col}_n"
        if col in features_to_neutralize:
            all_df[new_col] = neutralized[col]
        elif col in standardized_logs:
            all_df[new_col] = standardized_logs[col]
        else:
            all_df[new_col] = all_df[col]
        neutral_suffix_map[col] = new_col

    factor_cols = list(neutral_suffix_map.values())

    # Generate lagged columns BEFORE filtering to preserve early-window lags
    grouped_items = all_df.groupby("item_id", sort=False)
    lagged_frames = []
    for lag in LAGS:
        lagged = grouped_items[factor_cols].shift(lag)
        lagged.columns = [f"{c}_lag{lag}" for c in factor_cols]
        lagged_frames.append(lagged)
    if lagged_frames:
        all_df = pd.concat([all_df, *lagged_frames], axis=1)

    # Filter to user-specified output range AFTER lag generation
    if start_ts is not None or end_ts is not None:
        all_df_ts = all_df["date"].dt.tz_localize(ASIA_SHANGHAI) + pd.Timedelta(hours=TRADE_END_HOUR)
        mask = pd.Series(True, index=all_df.index)
        if start_ts is not None:
            mask &= all_df_ts >= start_ts
        if end_ts is not None:
            mask &= all_df_ts <= end_ts
        all_df = all_df.loc[mask]

    print(f"Filtered to {len(all_df)} records in user-specified output range")
    #all_df["target_8d"] = neutralize_target_by_industry(all_df, "target_8d", "industry")

    # Analyze NaN distribution if requested
    if args.analyze_nan:
        analyze_nan_distribution(all_df, feature_columns, output_file="nan_analysis.csv")

    summary = compute_rank_ic(
        all_df,
        factor_cols,
        target_col="target_8d",
        lags=LAGS,
        prelagged=True,
    )

    if summary.empty:
        raise RuntimeError("No Rank IC values were computed. Check data coverage.")

    summary.to_csv(summary_path, index=False)

    matrix = (
        summary.pivot(index="factor", columns="lag", values="mean")
        .reindex([col.replace("_n", "") for col in factor_cols])
        .rename(index=lambda x: x.replace("_n", ""))
        .reindex(columns=LAGS)
    )

    plot_heatmap(matrix, heatmap_path)

    top_lag0 = (
        summary.loc[summary["lag"] == 0]
        .sort_values("mean", ascending=False)
        .head(10)
    )

    print("Top factors by mean Rank IC (lag=0):")
    for _, row in top_lag0.iterrows():
        print(
            f"  {row['factor']:<24s} mean={row['mean']:.4f} "
            f"std={row['std']:.4f} count={int(row['count'])}"
        )
    print(f"\nSaved summary to {summary_path} and heatmap to {heatmap_path}.")


if __name__ == "__main__":
    main()
