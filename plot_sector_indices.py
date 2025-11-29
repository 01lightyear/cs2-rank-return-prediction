#!/usr/bin/env python3
"""Plot normalized performance curves for all defined sectors."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable, Optional, Sequence
from zoneinfo import ZoneInfo

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from analyze_sector import (
    ITEM_SPEC_PATH,
    SectorMember,
    aggregate_sector_candles,
    parse_sector_members,
    prepare_member_candles,
)
from analyze_single_asset import Date, clip_time_window, summarize_effective_window


# ---- User inputs ---------------------------------------------------------

# Leave as None to include every sector defined in itemid.txt.
TARGET_SECTORS: Optional[Sequence[str]] = None

# Set to Date(...) if you want to limit the window, or leave as None for full range.
# Inputs are interpreted as Beijing time (Asia/Shanghai) and converted to UTC internally.
DATE_START: Optional[datetime] = Date(2025, 11, 9, 0, 0, 0)
DATE_END: Optional[datetime] = Date(2025, 11, 18, 20, 0, 0)


def _collect_sector_names(mapping: Dict[str, Iterable[SectorMember]]) -> Sequence[str]:
    if TARGET_SECTORS is not None:
        return list(TARGET_SECTORS)

    return sorted(mapping.keys())


def _build_sector_index(
    name: str,
    members: Sequence[SectorMember],
    end: Optional[datetime],
    start: Optional[datetime],
) -> Optional[pd.Series]:
    """Aggregate member candles and return a normalized close series."""

    member_frames: Dict[str, pd.DataFrame] = {}
    errors: list[str] = []

    for member in members:
        try:
            member_frames[member.item_id] = prepare_member_candles(member, end)
        except (FileNotFoundError, ValueError) as exc:
            errors.append(f"{member.item_id}: {exc}")

    if errors:
        joined = "；".join(errors)
        print(f"⚠️ 板块 {name} 有成员处理失败，已跳过：{joined}")
        return None

    sector_candles, _ = aggregate_sector_candles(member_frames)
    if sector_candles.empty:
        print(f"⚠️ 板块 {name} 聚合后无有效数据，已跳过。")
        return None

    window = clip_time_window(sector_candles, start, end)
    if window.empty:
        print(f"⚠️ 板块 {name} 在指定区间内没有可用数据，已跳过。")
        return None

    baseline_close = window["close"].iloc[0]
    if pd.isna(baseline_close) or baseline_close == 0:
        print(f"⚠️ 板块 {name} 基准收盘价无效，已跳过。")
        return None

    summarize_effective_window(
        window.index,
        start,
        end,
        label=f"板块 {name}",
    )
    normalized = window["close"] / baseline_close * 100.0
    normalized.name = name
    return normalized


def _plot_indices(series_map: Dict[str, pd.Series]) -> None:
    """Render the normalized curves on a shared chart."""

    if not series_map:
        print("⚠️ 没有可绘制的板块指数曲线。")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    formatter = mdates.DateFormatter("%m-%d %H:%M", tz=ZoneInfo("Asia/Shanghai"))

    ordered_items = sorted(series_map.items(), key=lambda item: item[0])
    cmap = plt.get_cmap("turbo", len(ordered_items) or 1)

    for idx, (name, series) in enumerate(ordered_items):
        local_index = series.index.tz_convert(ZoneInfo("Asia/Shanghai"))
        color = cmap(idx)
        ax.plot(local_index, series.values, label=name, color=color)

    ax.set_title("板块归一化指数 (基准=100)")
    ax.set_ylabel("指数值")
    ax.legend(loc="upper left", ncol=2)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.show()


def run_sector_index_plot() -> None:
    """Main entry point for computing and plotting sector indices."""

    sector_mapping = parse_sector_members(ITEM_SPEC_PATH)
    if not sector_mapping:
        raise RuntimeError(f"在 {ITEM_SPEC_PATH} 中没有找到任何板块定义。")

    target_names = _collect_sector_names(sector_mapping)
    if not target_names:
        print("⚠️ 没有指定要处理的板块。")
        return

    series_map: Dict[str, pd.Series] = {}
    for name in target_names:
        members = sector_mapping.get(name)
        if not members:
            print(f"⚠️ 板块 {name} 在配置中没有成员，已跳过。")
            continue

        normalized = _build_sector_index(name, members, DATE_END, DATE_START)
        if normalized is not None:
            series_map[name] = normalized

    _plot_indices(series_map)


if __name__ == "__main__":
    run_sector_index_plot()
