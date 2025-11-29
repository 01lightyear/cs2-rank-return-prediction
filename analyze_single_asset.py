#!/usr/bin/env python3
"""Single-asset technical study on 6-hour candles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone, tzinfo as TzInfo
from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from mplfinance.original_flavor import candlestick_ohlc
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

FONT_CANDIDATES = [
    "SimHei",
    "Microsoft YaHei",
    "WenQuanYi Micro Hei",
    "Arial Unicode MS",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Noto Sans CJK",
    "Source Han Sans",
    "Noto Serif CJK SC",
    "Noto Serif CJK",
    "Source Han Serif SC",
]

FONT_SEARCH_DIRS = [
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
    Path.home() / ".local/share/fonts",
]

FONT_FILE_PATTERNS = [
    "**/NotoSansCJK*.ttc",
    "**/NotoSansCJK*.otf",
    "**/SourceHanSans*.otf",
    "**/SourceHanSans*.ttc",
    "**/SourceHanSerif*.otf",
    "**/SourceHanSerif*.ttc",
    "**/NotoSerifCJK*.ttc",
    "**/NotoSerifCJK*.otf",
    "**/WenQuanYi*.ttf",
    "**/SimHei*.ttf",
]


def _register_candidate_fonts() -> None:
    registered = 0
    for base in FONT_SEARCH_DIRS:
        if not base.exists():
            continue
        for pattern in FONT_FILE_PATTERNS:
            for path in base.glob(pattern):
                try:
                    font_manager.fontManager.addfont(str(path))
                    registered += 1
                except OSError:
                    continue

    if registered:
        try:
            font_manager._load_fontmanager(try_read_cache=False)  # type: ignore[attr-defined]
        except Exception:
            pass


def _configure_chinese_font() -> None:
    available_fonts = font_manager.fontManager.ttflist
    chosen_name: Optional[str] = None
    for candidate in FONT_CANDIDATES:
        candidate_lower = candidate.lower()
        for font in available_fonts:
            if candidate_lower in font.name.lower():
                chosen_name = font.name
                break
        if chosen_name is not None:
            break

    if chosen_name is None:
        print("⚠️ 未找到本地中文字体，图表中文可能无法正常显示。")
        return

    existing = list(rcParams.get("font.sans-serif", []))
    rcParams["font.family"] = ["sans-serif"]
    rcParams["font.sans-serif"] = [chosen_name] + existing
    print(f"✅ 已检测到中文字体：{chosen_name}")


_register_candidate_fonts()
_configure_chinese_font()

rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_new"


def Date(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
    tzinfo: Optional[TzInfo] = None,
) -> datetime:
    """Return a timezone-aware datetime interpreted in Asia/Shanghai unless overridden."""

    resolved_tz = tzinfo or ZoneInfo("Asia/Shanghai")
    dt = datetime(year, month, day, hour, minute, second, tzinfo=resolved_tz)
    return dt.astimezone(timezone.utc)


# ---- User inputs ---------------------------------------------------------
TARGET_ID = "2322"

# Set to Date(...) if you want to trim the study window, or leave as None for full range.
# Inputs are interpreted as Beijing time (Asia/Shanghai) and converted to UTC internally.
DATE_START: Optional[datetime] = Date(2025, 11, 10, 0, 0, 0)
DATE_END: Optional[datetime] = Date(2025, 11, 21, 22, 0, 0)

# Indicator configuration expressed in number of 6H candles.
CMF_PERIOD = 28
MFI_PERIOD = 21
RSI_PERIOD = 21
SMA_FAST_PERIOD = 20
SMA_SLOW_PERIOD = 60
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9
ADX_PERIOD = 21
ATR_PERIOD = 21
LOOKBACK_PERIOD = max(
    CMF_PERIOD,
    MFI_PERIOD,
    RSI_PERIOD,
    SMA_FAST_PERIOD,
    SMA_SLOW_PERIOD,
    MACD_SLOW_PERIOD,
    MACD_SIGNAL_PERIOD,
    ADX_PERIOD,
    ATR_PERIOD,
)


def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Compute the On-Balance Volume cumulative series."""

    price_change = close.diff()
    direction = price_change.apply(np.sign).fillna(0.0)
    signed_volume = volume * direction
    obv = signed_volume.cumsum()
    if not obv.empty:
        obv.iloc[0] = 0.0
    return obv


@dataclass(frozen=True)
class IndicatorFrame:
    """Keeps the price frame and derived indicators together."""

    df: pd.DataFrame


def load_hourly_kline(item_id: str) -> pd.DataFrame:
    """Load raw hourly candles for a single asset from data_new."""

    file_path = DATA_DIR / f"{item_id}.json"
    if not file_path.exists():
        raise FileNotFoundError(f"找不到指定 ID 的数据文件: {file_path}")

    with file_path.open("r", encoding="utf-8") as fp:
        raw_rows = json.load(fp)

    if not raw_rows:
        raise ValueError(f"文件 {file_path} 为空，无法分析。")

    df = pd.DataFrame(raw_rows)
    expected_cols = {"t", "o", "h", "l", "c", "v"}
    if not expected_cols.issubset(df.columns):
        missing = expected_cols.difference(df.columns)
        raise KeyError(f"数据缺少列: {missing}")

    df = df.rename(
        columns={
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")
    return df


def clip_time_window(df: pd.DataFrame, start: Optional[datetime], end: Optional[datetime]) -> pd.DataFrame:
    """Restrict the frame to the requested window before resampling."""

    clipped = df
    if start is not None:
        clipped = clipped.loc[clipped.index >= start]
    if end is not None:
        clipped = clipped.loc[clipped.index < end]
    return clipped


def summarize_effective_window(
    index: pd.DatetimeIndex,
    requested_start: Optional[datetime],
    requested_end: Optional[datetime],
    *,
    label: str,
) -> None:
    """Print the actual Beijing-time window covered by the supplied index."""

    if index.empty:
        print(f"⚠️ {label} 指定窗口内没有可用于统计的数据。")
        return

    if index.tz is None:
        raise ValueError("summarize_effective_window 需要带时区的 DatetimeIndex。")

    actual_start = index.min()
    actual_end = index.max()
    beijing_tz = ZoneInfo("Asia/Shanghai")
    start_local = actual_start.astimezone(beijing_tz)
    end_local = actual_end.astimezone(beijing_tz)

    summary = (
        f"ℹ️ {label} 实际使用数据区间："
        f"{start_local:%Y-%m-%d %H:%M} — {end_local:%Y-%m-%d %H:%M} (北京时间)"
    )

    notes: list[str] = []
    if requested_start is not None and actual_start > requested_start:
        req_local = requested_start.astimezone(beijing_tz)
        notes.append(f"起点受限（原设 {req_local:%Y-%m-%d %H:%M}）")
    if requested_end is not None and actual_end < requested_end:
        req_local = requested_end.astimezone(beijing_tz)
        notes.append(f"终点受限（原设 {req_local:%Y-%m-%d %H:%M}）")

    if notes:
        summary += "；" + "，".join(notes)

    print(summary)


def resample_to_six_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly data to 6H candles aligned to Beijing midnight."""

    if df.index.tz is None:
        raise ValueError("输入数据索引缺少时区信息，无法进行 6H 聚合。")

    agg_map = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    if "turnover" in df.columns:
        agg_map["turnover"] = "sum"

    local = df.tz_convert(ZoneInfo("Asia/Shanghai"))
    six_hour_local = (
        local.resample("6h", label="left", closed="left")
        .agg(agg_map)
        .dropna(subset=["open", "high", "low", "close"])
    )
    six_hour = six_hour_local.tz_convert(timezone.utc)
    six_hour.index.name = df.index.name
    return six_hour


def chaikin_money_flow(df: pd.DataFrame, period: int) -> pd.Series:
    """Chaikin Money Flow over the supplied window length."""

    high = df["high"]
    low = df["low"]
    close = df["close"]
    volume = df["volume"]

    price_range = high - low
    price_range = price_range.replace(0, np.nan)

    mfm = ((close - low) - (high - close)) / price_range
    mfm = mfm.fillna(0.0)
    money_flow = mfm * volume

    vol_sum = volume.rolling(period, min_periods=period).sum()
    flow_sum = money_flow.rolling(period, min_periods=period).sum()
    cmf = flow_sum / vol_sum
    cmf = cmf.where(vol_sum != 0)
    return cmf


def money_flow_index(df: pd.DataFrame, period: int) -> pd.Series:
    """Money Flow Index computed with the standard 14-period logic."""

    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    raw_money_flow = typical_price * df["volume"]

    price_diff = typical_price.diff()
    positive_flow = np.where(price_diff > 0, raw_money_flow, 0.0)
    negative_flow = np.where(price_diff < 0, raw_money_flow, 0.0)

    pos_sum = pd.Series(positive_flow, index=df.index).rolling(period, min_periods=period).sum()
    neg_sum = pd.Series(negative_flow, index=df.index).rolling(period, min_periods=period).sum()

    ratio = pos_sum / neg_sum
    mfi = 100.0 - (100.0 / (1.0 + ratio))

    mfi = mfi.where(neg_sum != 0, 100.0)
    mfi = mfi.where((pos_sum != 0) | (neg_sum != 0), 50.0)
    return mfi


def relative_strength_index(close: pd.Series, period: int) -> pd.Series:
    """Wilder-smoothed RSI."""

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(avg_gain != 0, 0.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    return rsi


def moving_average_convergence_divergence(
    close: pd.Series,
    fast_period: int,
    slow_period: int,
    signal_period: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Classic MACD (DIF/DEA) computation."""

    if slow_period <= fast_period:
        raise ValueError("MACD 慢速周期必须大于快速周期。")

    ema_fast = close.ewm(span=fast_period, adjust=False, min_periods=fast_period).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False, min_periods=slow_period).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False, min_periods=signal_period).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def true_range(df: pd.DataFrame) -> pd.Series:
    """True Range of each candle."""

    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def average_true_range(df: pd.DataFrame, period: int) -> pd.Series:
    """Wilder-smoothed ATR."""

    tr = true_range(df)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return atr


def directional_indicators(df: pd.DataFrame, atr: pd.Series, period: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute +DI, -DI, and ADX."""

    up_move = df["high"].diff()
    down_move = df["low"].shift(1) - df["low"]

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm_smoothed = pd.Series(plus_dm, index=df.index).ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()
    minus_dm_smoothed = pd.Series(minus_dm, index=df.index).ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    plus_di = 100.0 * (plus_dm_smoothed / atr)
    minus_di = 100.0 * (minus_dm_smoothed / atr)

    plus_di = plus_di.replace([np.inf, -np.inf], np.nan)
    minus_di = minus_di.replace([np.inf, -np.inf], np.nan)

    dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return plus_di, minus_di, adx


def enrich_with_indicators(df: pd.DataFrame) -> IndicatorFrame:
    """Compute all requested indicators on the aggregated frame."""

    enriched = df.copy()
    enriched["cmf"] = chaikin_money_flow(enriched, CMF_PERIOD)
    enriched["mfi"] = money_flow_index(enriched, MFI_PERIOD)
    enriched["rsi"] = relative_strength_index(enriched["close"], RSI_PERIOD)
    enriched["sma_fast"] = enriched["close"].rolling(SMA_FAST_PERIOD, min_periods=SMA_FAST_PERIOD).mean()
    enriched["sma_slow"] = enriched["close"].rolling(SMA_SLOW_PERIOD, min_periods=SMA_SLOW_PERIOD).mean()
    enriched["obv"] = on_balance_volume(enriched["close"], enriched["volume"])
    macd_line, macd_signal, macd_hist = moving_average_convergence_divergence(
        enriched["close"],
        MACD_FAST_PERIOD,
        MACD_SLOW_PERIOD,
        MACD_SIGNAL_PERIOD,
    )
    enriched["macd"] = macd_line
    enriched["macd_signal"] = macd_signal
    enriched["macd_hist"] = macd_hist

    atr = average_true_range(enriched, ATR_PERIOD)
    enriched["atr"] = atr
    plus_di, minus_di, adx = directional_indicators(enriched, atr, ADX_PERIOD)
    enriched["+di"] = plus_di
    enriched["-di"] = minus_di
    enriched["adx"] = adx
    return IndicatorFrame(df=enriched)


def plot_indicator_panel(
    indicators: IndicatorFrame,
    title: str,
    xlim: tuple[Optional[datetime], Optional[datetime]] | None = None,
) -> None:
    """Plot price and derived indicators on stacked axes."""

    df = indicators.df.dropna(subset=["close"])
    if df.empty:
        raise ValueError("聚合后的数据为空，检查时间范围或数据文件。")

    # Restrict the frame to the visible window so y-axis ranges follow the current viewport.
    df_window = df
    if xlim is not None:
        left_bound, right_bound = xlim
        if left_bound is not None:
            df_window = df_window.loc[df_window.index >= left_bound]
        if right_bound is not None:
            df_window = df_window.loc[df_window.index <= right_bound]
    if df_window.empty:
        raise ValueError("指定的可视窗口内没有可用数据。")

    x_left = xlim[0] if xlim is not None and xlim[0] is not None else df_window.index.min()
    x_right = xlim[1] if xlim is not None and xlim[1] is not None else df_window.index.max()

    fig, axes = plt.subplots(7, 1, figsize=(14, 16), sharex=True, constrained_layout=False)
    price_ax, volume_ax, macd_ax, obv_ax, cmf_ax, momentum_ax, adx_ax = axes

    ohlc = df_window[["open", "high", "low", "close"]].copy()
    ohlc["mdates"] = mdates.date2num(df_window.index.to_pydatetime())
    ohlc_values = ohlc[["mdates", "open", "high", "low", "close"]].to_numpy()
    candle_width = 0.18  # 6 小时蜡烛宽度（天数单位）
    candlestick_ohlc(
        price_ax,
        ohlc_values,
        width=candle_width,
        colorup="#ef5350",
        colordown="#26a69a",
        alpha=0.9,
    )
    price_ax.plot(df_window.index, df_window["sma_fast"], label=f"SMA{SMA_FAST_PERIOD}", color="#ff7f0e", linewidth=1.2)
    price_ax.plot(df_window.index, df_window["sma_slow"], label=f"SMA{SMA_SLOW_PERIOD}", color="#2ca02c", linewidth=1.2)
    price_ax.set_ylabel("Price")
    price_ax.set_title(title)
    price_ax.grid(True, linestyle="--", alpha=0.35)
    price_ax.legend(loc="upper left")

    volume_ax.bar(df_window.index, df_window["volume"], width=candle_width, color="#7f7f7f", alpha=0.6, label="Volume")
    volume_ax.set_ylabel("Volume")
    volume_ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    hist = df_window["macd_hist"]
    hist_colors = np.where(hist >= 0, "#d62728", "#2ca02c")
    macd_ax.bar(df_window.index, hist, width=candle_width, color=hist_colors, alpha=0.6, label="MACD Hist")
    macd_ax.plot(df_window.index, df_window["macd"], label="MACD", color="#9467bd", linewidth=1.2)
    macd_ax.plot(
        df_window.index,
        df_window["macd_signal"],
        label=f"MACD Signal ({MACD_SIGNAL_PERIOD})",
        color="#1f77b4",
        linewidth=1.0,
    )
    macd_ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    macd_ax.set_ylabel("MACD")
    macd_ax.grid(True, linestyle="--", alpha=0.35)
    macd_ax.legend(loc="upper left")

    obv_ax.plot(df_window.index, df_window["obv"], label="OBV", color="#9467bd")
    obv_ax.set_ylabel("OBV")
    obv_ax.grid(True, linestyle="--", alpha=0.35)
    obv_ax.legend(loc="upper left")

    cmf_ax.plot(df_window.index, df_window["cmf"], label=f"CMF ({CMF_PERIOD})", color="#8c564b")
    cmf_ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    cmf_ax.set_ylabel("CMF")
    cmf_ax.grid(True, linestyle="--", alpha=0.35)
    cmf_ax.legend(loc="upper left")

    momentum_ax.plot(df_window.index, df_window["mfi"], label=f"MFI ({MFI_PERIOD})", color="#d62728")
    momentum_ax.plot(df_window.index, df_window["rsi"], label=f"RSI ({RSI_PERIOD})", color="#17becf")
    momentum_ax.axhline(80, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    momentum_ax.axhline(20, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    momentum_ax.axhline(70, color="grey", linestyle=":", linewidth=0.6, alpha=0.5)
    momentum_ax.axhline(30, color="grey", linestyle=":", linewidth=0.6, alpha=0.5)
    momentum_ax.set_ylabel("MFI / RSI")
    momentum_ax.set_ylim(0, 100)
    momentum_ax.grid(True, linestyle="--", alpha=0.35)
    momentum_ax.legend(loc="upper left")

    line_adx, = adx_ax.plot(df_window.index, df_window["adx"], label=f"ADX ({ADX_PERIOD})", color="#bcbd22")
    line_plus_di, = adx_ax.plot(df_window.index, df_window["+di"], label="+DI", color="#2ca02c")
    line_minus_di, = adx_ax.plot(df_window.index, df_window["-di"], label="-DI", color="#d62728")
    adx_ax.set_ylabel("ADX / DI")
    adx_ax.grid(True, linestyle="--", alpha=0.35)

    atr_ax = adx_ax.twinx()
    # Plot ATR on a secondary axis so large ranges do not flatten ADX/DI.
    atr_color = "#1f77b4"
    line_atr, = atr_ax.plot(
        df_window.index,
        df_window["atr"],
        label=f"ATR ({ATR_PERIOD})",
        color=atr_color,
        linestyle="--",
    )
    atr_ax.set_ylabel("ATR")
    atr_ax.tick_params(axis="y", colors=atr_color)
    atr_ax.spines["right"].set_color(atr_color)
    atr_ax.yaxis.label.set_color(atr_color)
    atr_ax.grid(False)

    legend_lines = [line_adx, line_plus_di, line_minus_di, line_atr]
    legend_labels = [line.get_label() for line in legend_lines]
    adx_ax.legend(legend_lines, legend_labels, loc="upper left")

    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    fig.autofmt_xdate()

    for ax in axes:
        ax.set_xlim(x_left, x_right)
    atr_ax.set_xlim(x_left, x_right)

    plt.tight_layout()
    plt.show()


def run_single_asset_study() -> None:
    """Pipeline assembling all steps."""

    hourly = load_hourly_kline(TARGET_ID)
    hourly_window = clip_time_window(hourly, None, DATE_END)
    if hourly_window.empty:
        raise ValueError("选定时间范围内没有原始小时数据。")

    candles_6h = resample_to_six_hours(hourly_window)
    if candles_6h.empty:
        raise ValueError("聚合后的 6 小时数据为空。")

    enriched = enrich_with_indicators(candles_6h).df
    plot_df = enriched
    if DATE_END is not None:
        plot_df = plot_df.loc[plot_df.index < DATE_END]
    if plot_df.empty:
        raise ValueError("指定时间范围内没有可绘制的数据。")

    visible_df = plot_df
    if DATE_START is not None:
        visible_df = visible_df.loc[visible_df.index >= DATE_START]
    if DATE_END is not None:
        visible_df = visible_df.loc[visible_df.index < DATE_END]
    if visible_df.empty:
        raise ValueError("指定时间范围内没有可用的可视数据。")

    if DATE_START is not None:
        first_available = plot_df.index.min()
        if first_available > DATE_START:
            beijing_first = first_available.tz_convert(ZoneInfo("Asia/Shanghai"))
            beijing_start = DATE_START.astimezone(ZoneInfo("Asia/Shanghai"))
            print(
                f"⚠️ 指标计算所需的历史数据不足，首个可用周期为 {beijing_first:%Y-%m-%d %H:%M} "
                f"(北京时)，早于设定起点 {beijing_start:%Y-%m-%d %H:%M} (北京时)。"
            )

    summarize_effective_window(
        visible_df.index,
        DATE_START,
        DATE_END,
        label=f"ID {TARGET_ID} 6H",
    )
    indicators = IndicatorFrame(df=plot_df)

    plot_title = f"ID {TARGET_ID} 6H 指标 (K线/SMA/MACD/OBV/CMF/MFI/RSI/ADX/ATR)"
    plot_indicator_panel(indicators, plot_title, xlim=(DATE_START, DATE_END))


if __name__ == "__main__":
    run_single_asset_study()
