#!/usr/bin/env python3
"""Sector-level technical study on 6-hour candles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import math
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from analyze_single_asset import (
    Date,
    IndicatorFrame,
    clip_time_window,
    enrich_with_indicators,
    load_hourly_kline,
    plot_indicator_panel,
    resample_to_six_hours,
    summarize_effective_window,
)

BASE_DIR = Path(__file__).resolve().parent
ITEM_SPEC_PATH = BASE_DIR / "getdata" / "itemid.txt"


# ---- User inputs ---------------------------------------------------------
TARGET_SECTOR_NAME = "千战"

# Set to Date(...) if you want to trim the study window, or leave as None for full range.
# Inputs are interpreted as Beijing time (Asia/Shanghai) and converted to UTC internally.
DATE_START: Optional[datetime] = Date(2025, 11, 7, 0, 0, 0)
DATE_END: Optional[datetime] = Date(2025, 11, 18, 20, 0, 0)

# Weighting configuration.
WEIGHT_VOLUME_LOOKBACK_HOURS = 288
WEIGHT_VOLUME_LOOKBACK_PERIODS = max(math.ceil(WEIGHT_VOLUME_LOOKBACK_HOURS / 6), 1)


@dataclass(frozen=True)
class SectorMember:
    """Container for sector component metadata."""

    item_id: str
    label: str


def parse_sector_members(spec_path: Path) -> Dict[str, List[SectorMember]]:
    """Parse getdata/itemid.txt into a mapping of sector name -> members."""

    if not spec_path.exists():
        raise FileNotFoundError(f"未找到板块定义文件: {spec_path}")

    mapping: Dict[str, List[SectorMember]] = {}
    current_sector: Optional[str] = None

    with spec_path.open("r", encoding="utf-8") as fp:
        for raw_line in fp:
            line = raw_line.strip()
            if not line:
                current_sector = None
                continue

            if line.startswith("//"):
                current_sector = line[2:].strip()
                if current_sector:
                    mapping.setdefault(current_sector, [])
                continue

            if current_sector is None:
                continue

            separator = "：" if "：" in line else ":"
            if separator not in line:
                continue

            item_id, label = line.split(separator, 1)
            item_id = item_id.strip()
            label = label.strip()
            if not item_id:
                continue

            mapping[current_sector].append(SectorMember(item_id=item_id, label=label))

    return mapping


def prepare_member_candles(member: SectorMember, end: Optional[datetime]) -> pd.DataFrame:
    """Load, trim, and resample a single component to 6H candles with weights."""

    hourly = load_hourly_kline(member.item_id)
    hourly_window = clip_time_window(hourly, None, end)
    if hourly_window.empty:
        raise ValueError(f"资产 {member.item_id} 在选定窗口内没有原始小时数据。")

    six_hour = resample_to_six_hours(hourly_window)
    if six_hour.empty:
        raise ValueError(f"资产 {member.item_id} 在 6 小时聚合后没有有效数据。")

    six_hour.index.name = "timestamp"
    volume_ma = six_hour["volume"].rolling(
        WEIGHT_VOLUME_LOOKBACK_PERIODS,
        min_periods=WEIGHT_VOLUME_LOOKBACK_PERIODS,
    ).mean()
    weight = six_hour["close"] * volume_ma
    fallback_mask = weight.isna()
    if fallback_mask.any():
        # When the rolling window is not fully populated, fall back to current-period volume.
        fallback_weight = six_hour["close"] * six_hour["volume"]
        weight = weight.where(~fallback_mask, fallback_weight)
    six_hour["weight"] = weight
    return six_hour


def aggregate_sector_candles(member_frames: Dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.Series]:
    """Aggregate component candles into a single sector candle series."""

    if not member_frames:
        raise ValueError("没有可用的板块成员数据用于聚合。")

    combined = pd.concat(
        member_frames.values(),
        keys=member_frames.keys(),
        names=["asset_id", "timestamp"],
    )
    combined = combined.sort_index(level="timestamp")
    combined["weight"] = combined["weight"].fillna(0.0)

    weight_series = combined["weight"]
    total_weight = weight_series.groupby(level="timestamp").sum()
    if total_weight.empty:
        raise ValueError("板块成员缺少权重，无法计算聚合 K 线。")

    price_columns = ["open", "high", "low", "close"]
    aggregated = pd.DataFrame(index=total_weight.index)
    for column in price_columns:
        numerator = (combined[column] * weight_series).groupby(level="timestamp").sum()
        aggregated[column] = numerator.div(total_weight).where(total_weight != 0)

    aggregated["volume"] = combined["volume"].groupby(level="timestamp").sum()
    valid_mask = total_weight > 0
    aggregated = aggregated.loc[valid_mask]
    aggregated = aggregated.dropna(subset=["open", "high", "low", "close"])
    aggregated.index.name = "timestamp"

    timestamp_index = weight_series.index.get_level_values("timestamp")
    aligned_totals = total_weight.reindex(timestamp_index)
    weight_values = weight_series.to_numpy(dtype=float)
    total_values = aligned_totals.to_numpy(dtype=float)
    normalized_values = np.divide(
        weight_values,
        total_values,
        out=np.full(weight_values.shape, np.nan, dtype=float),
        where=total_values != 0,
    )
    normalized_weights = pd.Series(
        normalized_values,
        index=weight_series.index,
        name="normalized_weight",
    ).dropna()

    return aggregated, normalized_weights


def compute_average_weights(
    normalized_weights: pd.Series,
    start: Optional[datetime],
    end: Optional[datetime],
) -> pd.Series:
    """Average normalized weights for each component within the analysis window."""

    if normalized_weights.empty:
        return pd.Series(dtype=float, name="average_weight")

    timestamps = normalized_weights.index.get_level_values("timestamp")
    mask = np.ones(len(normalized_weights), dtype=bool)
    if start is not None:
        mask &= timestamps >= start
    if end is not None:
        mask &= timestamps < end

    filtered = normalized_weights[mask].dropna()
    if filtered.empty:
        return pd.Series(dtype=float, name="average_weight")

    averages = filtered.groupby(level="asset_id").mean().sort_values(ascending=False)
    averages.name = "average_weight"
    return averages


def run_sector_study() -> None:
    """Pipeline assembling all steps for a sector-level analysis."""

    sector_mapping = parse_sector_members(ITEM_SPEC_PATH)
    members = sector_mapping.get(TARGET_SECTOR_NAME)
    if not members:
        raise KeyError(f"未在 {ITEM_SPEC_PATH} 中找到板块: {TARGET_SECTOR_NAME}")

    member_frames: Dict[str, pd.DataFrame] = {}
    errors: List[str] = []

    for member in members:
        try:
            member_frames[member.item_id] = prepare_member_candles(member, DATE_END)
        except (FileNotFoundError, ValueError) as exc:
            errors.append(f"{member.item_id}: {exc}")

    if errors:
        joined = "; ".join(errors)
        raise RuntimeError(f"以下资产处理失败: {joined}")

    sector_candles, normalized_weights = aggregate_sector_candles(member_frames)
    if sector_candles.empty:
        raise ValueError("聚合后的板块 6 小时数据为空。")

    enriched = enrich_with_indicators(sector_candles).df
    plot_df = enriched
    if DATE_END is not None:
        plot_df = plot_df.loc[plot_df.index < DATE_END]
    if plot_df.empty:
        raise ValueError("指定区间内没有可用的聚合数据用于绘制。")

    visible_df = plot_df
    if DATE_START is not None:
        visible_df = visible_df.loc[visible_df.index >= DATE_START]
    if DATE_END is not None:
        visible_df = visible_df.loc[visible_df.index < DATE_END]
    if visible_df.empty:
        raise ValueError("指定区间内没有可用于板块分析的数据。")

    if DATE_START is not None:
        first_available = plot_df.index.min()
        if first_available > DATE_START:
            beijing_first = first_available.tz_convert(ZoneInfo("Asia/Shanghai"))
            beijing_start = DATE_START.astimezone(ZoneInfo("Asia/Shanghai"))
            print(
                f"⚠️ 板块组件历史不足，聚合序列最早从 {beijing_first:%Y-%m-%d %H:%M} (北京时) 开始，"
                f"晚于设定起点 {beijing_start:%Y-%m-%d %H:%M} (北京时)。"
            )

    summarize_effective_window(
        visible_df.index,
        DATE_START,
        DATE_END,
        label=f"{TARGET_SECTOR_NAME} 板块 6H",
    )
    indicators = IndicatorFrame(df=plot_df)
    plot_title = f"{TARGET_SECTOR_NAME} 板块 6H 指标 (K线/SMA/MACD/OBV/CMF/MFI/RSI/ADX/ATR)"
    plot_indicator_panel(indicators, plot_title, xlim=(DATE_START, DATE_END))

    averages = compute_average_weights(normalized_weights, DATE_START, DATE_END)
    member_lookup = {member.item_id: member.label for member in members}
    if averages.empty:
        print("⚠️ 指定区间内无法计算板块平均权重。")
    else:
        print("板块各资产平均权重（归一化后数值）：")
        for asset_id, value in averages.items():
            label = member_lookup.get(asset_id, "")
            name_part = f"（{label}）" if label else ""
            print(f"  {asset_id}{name_part}: {value:.4f}")


if __name__ == "__main__":
    run_sector_study()
