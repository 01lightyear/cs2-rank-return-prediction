#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
物品数据时间连续性检查脚本
检查 data_new 目录下以纯数字 ID 命名的 json 文件时间戳是否符合规范：
1. 时间戳是否为北京时间整点
2. 相邻记录时间戳相差 1 小时
"""

import json
import os
import glob
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Tuple

ONE_HOUR_MS = 60 * 60 * 1000


def find_missing_timestamps(data: List[Dict[str, Any]]) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    找出可通过线性插值填补的缺失时间戳以及无法被整除的间隔。
    返回:
        missing_timestamps: 需要插值的时间戳（毫秒）
        irregular_gaps: 无法整除的间隔列表，元素为 (前一个时间戳, 当前时间戳)
    """
    missing_timestamps: List[int] = []
    irregular_gaps: List[Tuple[int, int]] = []

    for i in range(1, len(data)):
        prev = data[i - 1]
        curr = data[i]
        if 't' not in prev or 't' not in curr:
            continue

        try:
            prev_ts = int(prev['t'])
            curr_ts = int(curr['t'])
        except (ValueError, TypeError):
            continue

        interval = curr_ts - prev_ts
        if interval <= ONE_HOUR_MS:
            continue

        if interval % ONE_HOUR_MS != 0:
            irregular_gaps.append((prev_ts, curr_ts))
            continue

        missing_count = interval // ONE_HOUR_MS - 1
        for step in range(1, missing_count + 1):
            missing_timestamps.append(prev_ts + ONE_HOUR_MS * step)

    return missing_timestamps, irregular_gaps


def interpolate_value(prev_value: Any, next_value: Any, ratio: float) -> Any:
    """对数值进行线性插值，无法转换为浮点数时返回前一个值。"""
    try:
        prev_f = float(prev_value)
        next_f = float(next_value)
    except (TypeError, ValueError):
        return prev_value
    return prev_f + (next_f - prev_f) * ratio


def create_interpolated_entry(
    prev_entry: Dict[str, Any],
    next_entry: Dict[str, Any],
    missing_ts: int,
    ratio: float,
) -> Dict[str, Any]:
    """生成线性插值后的数据点。"""
    interpolated = {'t': str(missing_ts)}

    for field in ['o', 'h', 'l', 'c', 'v']:
        if field in prev_entry and field in next_entry:
            interpolated[field] = interpolate_value(prev_entry[field], next_entry[field], ratio)
        elif field in prev_entry:
            interpolated[field] = prev_entry[field]
        elif field in next_entry:
            interpolated[field] = next_entry[field]

    extra_keys = set(prev_entry.keys()).union(next_entry.keys()) - {'t', 'o', 'h', 'l', 'c', 'v'}
    for key in extra_keys:
        interpolated[key] = prev_entry.get(key, next_entry.get(key))

    return interpolated


def fill_missing_points(file_path: str) -> int:
    """对指定文件内的缺失时间点进行线性插值填补，返回插值数量。"""
    data = load_json_file(file_path)
    if not data:
        print(f"   • 无法加载数据，跳过填补: {os.path.basename(file_path)}")
        return 0

    new_data: List[Dict[str, Any]] = []
    inserted = 0

    for i in range(len(data) - 1):
        prev_entry = data[i]
        next_entry = data[i + 1]
        new_data.append(prev_entry)

        if 't' not in prev_entry or 't' not in next_entry:
            continue

        try:
            prev_ts = int(prev_entry['t'])
            next_ts = int(next_entry['t'])
        except (ValueError, TypeError):
            continue

        interval = next_ts - prev_ts
        if interval <= ONE_HOUR_MS or interval % ONE_HOUR_MS != 0:
            continue

        missing_count = interval // ONE_HOUR_MS - 1
        for step in range(1, missing_count + 1):
            ratio = step / (missing_count + 1)
            missing_ts = prev_ts + ONE_HOUR_MS * step
            interpolated_entry = create_interpolated_entry(prev_entry, next_entry, missing_ts, ratio)
            new_data.append(interpolated_entry)
            inserted += 1

    if data:
        new_data.append(data[-1])

    if inserted:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)
        print(f"   • 已填补 {inserted} 个缺失点: {os.path.basename(file_path)}")
    else:
        print(f"   • 未发现可填补的缺失点: {os.path.basename(file_path)}")

    return inserted


def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                print(f"❌ 文件格式不是数组: {file_path}")
                return []
    except Exception as e:
        print(f"❌ 读取文件 {file_path} 失败: {e}")
        return []


def timestamp_to_beijing_time(ts_str: str) -> datetime:
    """将毫秒时间戳转换为北京时间"""
    ts = int(ts_str)
    # 先转为UTC时间
    utc_time = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
    # 转为北京时间 (UTC+8)
    beijing_time = utc_time + timedelta(hours=8)
    return beijing_time


def is_valid_time_point(beijing_time: datetime) -> bool:
    """检查时间是否为北京时间整点"""
    return beijing_time.minute == 0 and beijing_time.second == 0


def check_timestamp_continuity(file_path: str) -> Tuple[bool, List[str], List[int]]:
    """检查时间戳连续性"""
    print(f"🔍 检查时间戳连续性: {os.path.basename(file_path)}")

    # 加载数据
    data = load_json_file(file_path)
    if not data:
        return False, ["无法读取文件或文件为空"], []

    if len(data) < 2:
        return True, ["数据量不足2条，无法检查连续性"], []

    issues = []
    intervals = []
    valid_time_points = []
    invalid_time_points = []

    # 检查每个时间戳
    for i, item in enumerate(data):
        if 't' not in item:
            issues.append(f"第 {i+1} 条记录缺少时间戳字段")
            continue

        ts_str = item['t']
        try:
            ts = int(ts_str)
            beijing_time = timestamp_to_beijing_time(ts_str)

            # 检查是否为有效时间点
            if is_valid_time_point(beijing_time):
                valid_time_points.append((i+1, beijing_time))
            else:
                invalid_time_points.append((i+1, beijing_time))
                issues.append(f"第 {i+1} 条记录不是标准时间点: {beijing_time.strftime('%Y-%m-%d %H:%M:%S')}")

            # 检查间隔（从第二条记录开始）
            if i > 0:
                prev_ts = int(data[i-1]['t'])
                interval = ts - prev_ts
                intervals.append(interval)

                if interval != ONE_HOUR_MS:  # 1小时 = 3600000毫秒
                    prev_time = timestamp_to_beijing_time(data[i-1]['t'])
                    curr_time = beijing_time
                    issues.append(f"第 {i} 条记录间隔异常: {interval} 毫秒 (应为{ONE_HOUR_MS}毫秒)")
                    issues.append(f"  从 {prev_time.strftime('%Y-%m-%d %H:%M:%S')} 到 {curr_time.strftime('%Y-%m-%d %H:%M:%S')}")

        except (ValueError, TypeError) as e:
            issues.append(f"第 {i+1} 条记录时间戳格式错误: {ts_str}")

    missing_timestamps, irregular_gaps = find_missing_timestamps(data)
    for prev_ts, curr_ts in irregular_gaps:
        prev_time = timestamp_to_beijing_time(str(prev_ts))
        curr_time = timestamp_to_beijing_time(str(curr_ts))
        issues.append(
            f"存在无法整除1小时的时间间隔: 从 {prev_time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"到 {curr_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

    # 统计信息
    print(f"📊 时间点统计:")
    print(f"   • 有效时间点: {len(valid_time_points)} 个")
    print(f"   • 无效时间点: {len(invalid_time_points)} 个")

    if invalid_time_points:
        print(f"❌ 无效时间点示例:")
        for idx, time_point in invalid_time_points[:3]:  # 只显示前3个
            print(f"   • 第{idx}条: {time_point.strftime('%Y-%m-%d %H:%M:%S')}")
        if len(invalid_time_points) > 3:
            print(f"   • ... 还有 {len(invalid_time_points) - 3} 个无效时间点")

    if intervals:
        unique_intervals = set(intervals)
        if len(unique_intervals) == 1 and ONE_HOUR_MS in unique_intervals:
            print(f"✅ 时间戳间隔检查通过: {len(data)} 条记录，间隔均为1小时")
        else:
            interval_counts = {value: intervals.count(value) for value in unique_intervals}
            print(f"⚠️  发现间隔问题，间隔分布: {interval_counts}")

    if missing_timestamps:
        print(f"⏳ 发现 {len(missing_timestamps)} 个可通过线性插值填补的缺失时间点")
        preview = missing_timestamps[:3]
        for ts in preview:
            bt = timestamp_to_beijing_time(str(ts))
            print(f"   • 缺失时间点: {bt.strftime('%Y-%m-%d %H:%M:%S')}")
        if len(missing_timestamps) > len(preview):
            print(f"   • ... 还有 {len(missing_timestamps) - len(preview)} 个缺失点")

    return len(issues) == 0, issues, missing_timestamps


def get_item_files() -> List[str]:
    """获取所有物品文件（仅匹配纯数字 ID 命名的 JSON）"""
    pattern = os.path.join('data_new', '*.json')
    all_files = glob.glob(pattern)

    item_files = []
    for file in all_files:
        basename = os.path.basename(file)
        name, ext = os.path.splitext(basename)

        if ext != '.json':
            continue

        if not name.isdigit():
            continue

        item_files.append(file)

    return sorted(item_files)


def main():
    """主函数"""
    print("🚀 开始检查物品数据时间连续性...")

    # 获取所有物品文件
    item_files = get_item_files()

    if not item_files:
        print("❌ 未找到任何物品文件")
        return

    print(f"📁 找到 {len(item_files)} 个物品文件")

    # 检查所有文件
    passed_files = []
    failed_files = []

    missing_summary: Dict[str, List[int]] = {}

    for file_path in item_files:
        print(f"\n{'='*60}")
        is_valid, issues, missing_timestamps = check_timestamp_continuity(file_path)

        if is_valid:
            print(f"✅ {os.path.basename(file_path)}: 时间戳检查通过")
            passed_files.append(file_path)
        else:
            print(f"❌ {os.path.basename(file_path)}: 发现 {len(issues)} 个问题:")
            for issue in issues[:10]:  # 只显示前10个问题
                print(f"   • {issue}")
            if len(issues) > 10:
                print(f"   • ... 还有 {len(issues) - 10} 个问题")
            failed_files.append(file_path)

        if missing_timestamps:
            missing_summary[file_path] = missing_timestamps

    if missing_summary:
        print("\n⏱ 可通过线性插值填补的缺失时间点汇总:")
        total_missing = 0
        for file_path, timestamps in missing_summary.items():
            total_missing += len(timestamps)
            sample_times = [
                timestamp_to_beijing_time(str(ts)).strftime('%Y-%m-%d %H:%M:%S')
                for ts in timestamps[:3]
            ]
            preview = '，'.join(sample_times)
            if len(timestamps) > 3:
                preview += f"，... 等 {len(timestamps)} 个时间点"
            print(f"   • {os.path.basename(file_path)} 缺失 {len(timestamps)} 个时间点 -> {preview}")
        print(f"   • 总计缺失时间点: {total_missing} 个")

    # 最终统计
    print(f"\n{'='*60}")
    print(f"📊 检查完成统计:")
    print(f"   • 总文件数: {len(item_files)}")
    print(f"   • 通过检查: {len(passed_files)} 个")
    print(f"   • 未通过检查: {len(failed_files)} 个")

    if failed_files:
        print(f"\n❌ 未通过检查的文件:")
        for file_path in failed_files:
            print(f"   • {os.path.basename(file_path)}")

    print(f"\n{'='*60}")
    if len(failed_files) == 0:
        print("🎉 所有文件的时间戳检查都通过了！")
    else:
        print(f"⚠️  有 {len(failed_files)} 个文件存在问题，请检查上述输出")
    print(f"{'='*60}")

    if missing_summary:
        confirm = input("\n是否使用线性插值填补这些缺失数据点？(y/N): ").strip().lower()
        if confirm in {'y', 'yes', '是', '好', 'ok'}:
            print("\n🔧 开始填补缺失时间点...")
            total_inserted = 0
            for file_path in missing_summary:
                inserted = fill_missing_points(file_path)
                total_inserted += inserted
            print(f"\n✅ 填补完成，总共新增 {total_inserted} 个数据点")
        else:
            print("\nℹ️ 已取消填补操作，原始数据未做改动")


if __name__ == "__main__":
    main()
