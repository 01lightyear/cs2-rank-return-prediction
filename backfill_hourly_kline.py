#!/usr/bin/env python3
"""
Backfill hourly K-line history by paging backwards with maxTime windows.

This script reuses the lookup helpers from get_hourly_kline.py, sends repeated
type=1 requests with decreasing maxTime values, keeps only on-the-hour candles,
and merges the historical data into data_new/*.json.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import requests

from get_hourly_kline import (  # noqa: WPS301 - reuse existing helpers
    API_URL,
    DATA_DIR,
    HEADERS,
    ItemInfo,
    load_all_items_cache,
    load_item_ids,
    load_itemid_market_map,
    normalise_records,
    safe_filename,
)

DELAY_SECONDS = 3.7
RETRY_SECONDS = 10
MILLIS_PER_HOUR = 3_600_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill hourly K-line data by walking maxTime backwards",
    )
    parser.add_argument(
        "--min-date",
        help="Stop once we cover this date (UTC, YYYY-MM-DD).",
    )
    parser.add_argument(
        "--min-timestamp",
        type=int,
        help="Stop once timestamps (seconds since epoch) fall below this value.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=24,
        help="Safety cap for maxTime paging per item (default: 24 requests).",
    )
    parser.add_argument(
        "--item-id",
        action="append",
        help="Only backfill the specified itemId (can repeat the flag).",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=DELAY_SECONDS,
        help="Delay between items to avoid hammering the API.",
    )
    parser.add_argument(
        "--page-sleep-seconds",
        type=float,
        default=3.7,
        help="Delay between paging requests for the same item.",
    )
    return parser.parse_args()


def determine_stop_ts(args: argparse.Namespace) -> Optional[int]:
    if args.min_timestamp is not None:
        return args.min_timestamp * 1000

    if args.min_date:
        dt = datetime.strptime(args.min_date, "%Y-%m-%d")
        dt_utc = datetime(
            dt.year,
            dt.month,
            dt.day,
            tzinfo=timezone.utc,
        )
        return int(dt_utc.timestamp() * 1000)

    return None


def fetch_page(type_val: str, max_time: Optional[int], max_retries: int = 3) -> Optional[List[List[Optional[float]]]]:
    params = {
        "timestamp": str(int(datetime.now(timezone.utc).timestamp() * 1000)),
        "type": "1",
        "platform": "ALL",
        "specialStyle": "",
        "typeVal": type_val,
    }
    if max_time is not None:
        params["maxTime"] = str(int(max_time))

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(API_URL, headers=HEADERS, params=params, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
        except requests.RequestException as exc:
            print(f"    ❌ 请求失败({attempt}/{max_retries}): {exc}")
        except json.JSONDecodeError as exc:
            print(f"    ❌ 解析响应失败({attempt}/{max_retries}): {exc}")
        else:
            if not payload.get("success"):
                print(f"    ❌ API 返回错误: {payload.get('errorMsg')}")
            else:
                data = payload.get("data") or []
                if not data:
                    print("    ⚠️ API 返回空数据")
                else:
                    return data

        if attempt < max_retries:
            time.sleep(RETRY_SECONDS)

    return None


def normalise_hourly(records: Iterable[List[Optional[float]]]) -> List[Dict[str, float]]:
    cleaned: Dict[int, Dict[str, float]] = {}
    for entry in normalise_records(records):
        try:
            ts_ms = int(entry["t"])
        except (ValueError, TypeError):
            continue
        if ts_ms % MILLIS_PER_HOUR != 0:
            continue
        cleaned[ts_ms] = entry
    return [cleaned[key] for key in sorted(cleaned)]


def load_existing(filepath: str) -> Tuple[Dict[int, Dict[str, float]], Optional[int], Optional[int]]:
    if not os.path.exists(filepath):
        return {}, None, None

    try:
        with open(filepath, "r", encoding="utf-8") as fp:
            raw = json.load(fp)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"    ⚠️ 读取旧数据失败，忽略并重建: {exc}")
        return {}, None, None

    hourly_records: Dict[int, Dict[str, float]] = {}
    for item in raw:
        try:
            ts_ms = int(item.get("t"))
        except (TypeError, ValueError):
            continue
        if ts_ms % MILLIS_PER_HOUR != 0:
            continue
        hourly_records[ts_ms] = {
            "t": ts_ms,
            "o": float(item.get("o", 0.0)),
            "c": float(item.get("c", 0.0)),
            "h": float(item.get("h", 0.0)),
            "l": float(item.get("l", 0.0)),
            "v": float(item.get("v", 0.0)),
            "turnover": float(item.get("turnover", 0.0)),
        }

    if not hourly_records:
        return {}, None, None

    sorted_ts = sorted(hourly_records)
    return hourly_records, sorted_ts[0], sorted_ts[-1]


def merge_records(existing: Dict[int, Dict[str, float]], new_records: Iterable[Dict[str, float]]) -> None:
    for record in new_records:
        ts_ms = int(record["t"])
        existing[ts_ms] = record


def save_records(filepath: str, records: Dict[int, Dict[str, float]]) -> None:
    if not records:
        print("    ⚠️ 最终无可写入数据，跳过保存")
        return

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    ordered = [records[key] for key in sorted(records)]
    with open(filepath, "w", encoding="utf-8") as fp:
        json.dump(ordered, fp, ensure_ascii=False, indent=2)

    first_ts = ordered[0]["t"]
    last_ts = ordered[-1]["t"]
    start_dt = datetime.fromtimestamp(first_ts / 1000, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)
    print(
        f"    ✅ 已保存 {len(ordered)} 条记录，范围 {start_dt.isoformat()} - {end_dt.isoformat()}",
    )


def backfill_item(
    info: ItemInfo,
    item_id: str,
    display_name: str,
    stop_ts_ms: Optional[int],
    max_iterations: int,
    page_sleep: float,
) -> None:
    filename = f"{safe_filename(item_id)}.json"
    filepath = os.path.join(DATA_DIR, filename)

    records, existing_min, existing_max = load_existing(filepath)
    existing_count = len(records)

    if existing_min is not None:
        print(
            f"  ℹ️ 当前已有 {existing_count} 条整点数据 "
            f"[{datetime.fromtimestamp(existing_min / 1000, tz=timezone.utc).isoformat()}"
            f" -> {datetime.fromtimestamp(existing_max / 1000, tz=timezone.utc).isoformat()}]",
        )
    else:
        print("  ℹ️ 当前没有整点数据，将从最新窗口开始抓取")

    max_time = existing_min // 1000 if existing_min is not None else None
    iterations = 0
    reached_stop = False
    last_window_min: Optional[int] = None

    while iterations < max_iterations:
        iterations += 1
        page = fetch_page(info.type_val, max_time)
        if not page:
            break

        hourly = normalise_hourly(page)
        if not hourly:
            print("    ⚠️ 此页没有整点数据，结束回溯")
            break

        merge_records(records, hourly)

        window_min = min(int(row["t"]) for row in hourly)
        window_max = max(int(row["t"]) for row in hourly)
        print(
            f"    🔄 第 {iterations} 页: {len(hourly)} 条 "
            f"[{datetime.fromtimestamp(window_min / 1000, tz=timezone.utc).isoformat()} "
            f"-> {datetime.fromtimestamp(window_max / 1000, tz=timezone.utc).isoformat()}]",
        )

        if stop_ts_ms is not None and window_min <= stop_ts_ms:
            reached_stop = True
            break

        if last_window_min is not None and window_min >= last_window_min:
            print("    ⚠️ 最旧时间戳未继续向前推进，结束回溯")
            break

        last_window_min = window_min
        max_time = window_min // 1000
        time.sleep(page_sleep)

    if reached_stop:
        print("    ✅ 已达到设定的最早时间阈值")

    if len(records) == existing_count:
        print("  ℹ️ 未发现新增整点数据，保持原文件不变")
        return

    save_records(filepath, records)


def main() -> None:
    args = parse_args()
    stop_ts_ms = determine_stop_ts(args)

    mapping = load_all_items_cache()
    item_id_pairs = load_item_ids()
    id_to_market = load_itemid_market_map()

    if not mapping or not item_id_pairs or not id_to_market:
        print("❌ 无法继续，缺少必要的映射数据")
        return

    if args.item_id:
        wanted = set(args.item_id)
        item_id_pairs = [pair for pair in item_id_pairs if pair[0] in wanted]
        missing = wanted - {pair[0] for pair in item_id_pairs}
        for item_id in sorted(missing):
            print(f"⚠️ 未在 itemid.txt 中找到指定 itemId: {item_id}")

    total = len(item_id_pairs)
    for idx, (item_id, local_name) in enumerate(item_id_pairs, 1):
        market_hash_name = id_to_market.get(item_id)
        if not market_hash_name:
            print(f"[{idx}/{total}] 跳过，itemId 未匹配 marketHashName: {item_id}")
            continue

        info = mapping.get(market_hash_name)
        if not info:
            print(f"[{idx}/{total}] 跳过，未找到 all_items_cache 映射: {market_hash_name}")
            continue

        display_name = local_name or info.display_name
        print(f"[{idx}/{total}] {market_hash_name} ({display_name})")
        backfill_item(info, item_id, display_name, stop_ts_ms, args.max_iterations, args.page_sleep_seconds)

        if idx < total:
            time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
