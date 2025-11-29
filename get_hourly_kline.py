#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""按 watchlist 抓取小时 K 线到 JSON 的简化脚本。"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests

# 基础路径
# 项目根目录
BASE_DIR = "/home/user/csdata"

# 配置
ALL_ITEMS_CACHE_FILE = os.path.join(BASE_DIR, "all_items_cache.json")
DATA_DIR = os.path.join(BASE_DIR, "data_new")
ITEM_IDS_FILE = os.path.join(BASE_DIR, "getdata", "itemid.txt")
ITEM_ID_MARKET_MAP_FILE = os.path.join(BASE_DIR, "getdata", "itemid_market_map.json")

API_URL = "https://api.steamdt.com/user/steam/category/v1/kline"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Referer": "https://steamdt.com/",
}

DELAY_SECONDS = 3.7
RETRY_SECONDS = 15


@dataclass
class ItemInfo:
    """保存 watchlist 物品的基础信息。"""

    market_hash_name: str
    type_val: str
    display_name: str


def load_all_items_cache() -> Dict[str, ItemInfo]:
    """读取 all_items_cache.json，返回 marketHashName -> ItemInfo 的映射。"""

    if not os.path.exists(ALL_ITEMS_CACHE_FILE):
        print(f"❌ 找不到 all_items_cache: {ALL_ITEMS_CACHE_FILE}")
        return {}

    try:
        with open(ALL_ITEMS_CACHE_FILE, "r", encoding="utf-8") as fp:
            raw_items = json.load(fp)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"❌ 加载 all_items_cache.json 失败: {exc}")
        return {}

    mapping: Dict[str, ItemInfo] = {}
    for item in raw_items:
        market_hash_name = item.get("marketHashName")
        if not market_hash_name:
            continue

        display_name = item.get("name") or market_hash_name

        type_val = None
        for platform in item.get("platformList", []):
            if platform.get("name") == "C5" and platform.get("itemId"):
                type_val = str(platform["itemId"])
                break

        if not type_val:
            continue

        mapping[market_hash_name] = ItemInfo(
            market_hash_name=market_hash_name,
            type_val=type_val,
            display_name=display_name,
        )

    print(f"✅ 已加载 {len(mapping)} 个物品的映射")
    return mapping


def load_item_ids() -> List[tuple[str, str]]:
    """读取 itemid.txt，返回 (item_id, 中文名) 列表，保持文件顺序。"""

    if not os.path.exists(ITEM_IDS_FILE):
        print(f"❌ 找不到 item 列表文件: {ITEM_IDS_FILE}")
        return []

    items: List[tuple[str, str]] = []
    with open(ITEM_IDS_FILE, "r", encoding="utf-8") as fp:
        for raw_line in fp:
            line = raw_line.strip()
            if not line or line.startswith("//"):
                continue

            # 同时兼容半角/全角冒号
            if "：" in line:
                key, value = line.split("：", 1)
            elif ":" in line:
                key, value = line.split(":", 1)
            else:
                print(f"  ⚠️ 无法解析行（缺少冒号）: {line}")
                continue

            item_id = key.strip()
            display_name = value.strip()
            if not item_id:
                print(f"  ⚠️ 忽略空 ID 行: {line}")
                continue

            items.append((item_id, display_name))

    print(f"✅ itemid.txt 中共 {len(items)} 个物品 ID")
    return items


def load_itemid_market_map() -> Dict[str, str]:
    """读取 itemid_market_map.json，返回 item_id -> market hash name 映射。"""

    if not os.path.exists(ITEM_ID_MARKET_MAP_FILE):
        print(f"❌ 找不到 itemid_market_map: {ITEM_ID_MARKET_MAP_FILE}")
        return {}

    try:
        with open(ITEM_ID_MARKET_MAP_FILE, "r", encoding="utf-8") as fp:
            mapping = json.load(fp)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"❌ 加载 itemid_market_map.json 失败: {exc}")
        return {}

    print(f"✅ 已加载 {len(mapping)} 条 itemId -> marketHashName 映射")
    return mapping


def safe_filename(name: str) -> str:
    """将中文名清洗成可用的文件名。"""

    keep = []
    for ch in name:
        if ch in "\\/:*?\"<>|":
            keep.append("_")
        else:
            keep.append(ch)
    return "".join(keep)


def fetch_kline(type_val: str, max_retries: int = 3) -> Optional[List[List[Optional[float]]]]:
    """从 API 获取指定 typeVal 的 K 线数据，失败时会按 RETRY_SECONDS 重试。"""

    params = {
        "timestamp": str(int(datetime.now(timezone.utc).timestamp() * 1000)),
        "type": "1",
        "platform": "ALL",
        "specialStyle": "",
        "typeVal": type_val,
    }

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(API_URL, headers=HEADERS, params=params, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
        except requests.RequestException as exc:
            print(f"  ❌ 请求失败({attempt}/{max_retries}): {exc}")
        except json.JSONDecodeError as exc:
            print(f"  ❌ 解析响应失败({attempt}/{max_retries}): {exc}")
        else:
            if not payload.get("success"):
                print(f"  ❌ API 返回错误信息: {payload.get('errorMsg')}")
            else:
                data = payload.get("data") or []
                if not data:
                    print("  ⚠️ 返回数据为空")
                else:
                    # 去掉最后一条即时采样记录
                    trimmed = data[:-1] if len(data) > 1 else []
                    return trimmed

        if attempt < max_retries:
            print(f"  ⏳ {RETRY_SECONDS} 秒后重试...")
            time.sleep(RETRY_SECONDS)

    return None


def normalise_records(records: List[List[Optional[float]]]) -> List[Dict[str, Optional[float]]]:
    """将原始列表转成带字段的字典，时间戳统一成毫秒，并填充空值。"""

    normalised = []
    for entry in records:
        if len(entry) < 7:
            continue

        ts_raw = int(entry[0])
        open_price = entry[1]
        close_price = entry[2]
        high_price = entry[3]
        low_price = entry[4]
        volume = entry[5]
        turnover = entry[6]

        def _fallback(value: Optional[float]) -> float:
            return float(value) if value not in (None, "") else 0.0

        normalised.append(
            {
                "t": ts_raw * 1000,
                "o": _fallback(open_price),
                "c": _fallback(close_price),
                "h": _fallback(high_price),
                "l": _fallback(low_price),
                "v": _fallback(volume),
                "turnover": _fallback(turnover),
            }
        )

    normalised.sort(key=lambda item: item["t"])
    return normalised


def save_to_file(item_id: str, display_name: str, records: List[Dict[str, Optional[float]]]) -> None:
    """以 itemId 作为文件名保存 JSON 数据。"""

    if not records:
        print("  ⚠️ 无可保存数据，跳过")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    filename = f"{safe_filename(item_id)}.json"
    filepath = os.path.join(DATA_DIR, filename)

    existing: List[Dict[str, Optional[float]]] = []
    last_ts: Optional[int] = None
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as fp:
                existing = json.load(fp)
            if existing:
                last_ts = int(existing[-1].get("t", 0))
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            print(f"  ⚠️ 读取旧数据失败，覆盖写入: {exc}")
            existing = []
            last_ts = None

    if last_ts is not None:
        new_records = [row for row in records if int(row.get("t", 0)) > last_ts]
    else:
        new_records = records

    if not new_records:
        print("  ℹ️ 没有比现有数据更新的记录，保持原文件不变。")
        return

    merged_records = (existing or []) + new_records

    with open(filepath, "w", encoding="utf-8") as fp:
        json.dump(merged_records, fp, ensure_ascii=False, indent=2)

    start_source = merged_records[0]["t"] if merged_records else records[0]["t"]
    end_source = merged_records[-1]["t"] if merged_records else records[-1]["t"]
    start_dt = datetime.fromtimestamp(start_source / 1000, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_source / 1000, tz=timezone.utc)

    print(f"  ✅ 已保存 {len(new_records)} 条新增数据 -> {filepath}")
    print(f"     时间范围: {start_dt.astimezone().strftime('%Y-%m-%d %H:%M:%S')} - {end_dt.astimezone().strftime('%Y-%m-%d %H:%M:%S')}")


def process_item(item_info: ItemInfo, item_id: str, display_name: str) -> None:
    """抓取单个物品并写出 JSON。"""

    print(f"开始处理: {item_info.market_hash_name} ({display_name})")
    records_raw = fetch_kline(item_info.type_val)

    if records_raw is None:
        print("  ❌ 最终仍未获取成功")
        return

    cleaned_records = normalise_records(records_raw)
    if not cleaned_records:
        print("  ⚠️ 清洗后无有效数据")
        return

    save_to_file(item_id, display_name, cleaned_records)


def main() -> None:
    mapping = load_all_items_cache()
    item_id_pairs = load_item_ids()
    id_to_market = load_itemid_market_map()

    if not mapping or not item_id_pairs or not id_to_market:
        print("❌ 无法继续，缺少必要的映射数据")
        return

    for idx, (item_id, local_name) in enumerate(item_id_pairs, 1):
        market_hash_name = id_to_market.get(item_id)
        if not market_hash_name:
            print(f"[{idx}/{len(item_id_pairs)}] 跳过，itemId 未找到对应的 marketHashName: {item_id}")
            continue

        info = mapping.get(market_hash_name)
        if not info:
            print(f"[{idx}/{len(item_id_pairs)}] 跳过，未找到 all_items_cache 映射: {market_hash_name}")
            continue

        # 记录显示名称用于日志，若为空 fallback 到缓存内的名称
        display_name = local_name or info.display_name

        print(f"[{idx}/{len(item_id_pairs)}] {market_hash_name}")
        process_item(info, item_id, display_name)

        if idx < len(item_id_pairs):
            print(f"等待 {DELAY_SECONDS} 秒后继续...\n")
            time.sleep(DELAY_SECONDS)


if __name__ == "__main__":
    main()
