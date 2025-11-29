#!/usr/bin/env python3
"""Run inference for a specific cross-sectional date using the trained XGBoost model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "xgboost is not installed. Activate /home/user/miniforge3/envs/cs and pip install xgboost."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_DATASET = PROJECT_ROOT / "TBD" / "factor_dataset.parquet"
DEFAULT_MODEL = PROJECT_ROOT / "TBD" / "xgb_rank_model.json"


from rank_ic_analysis import add_alpha101_features, parse_window_endpoint  # noqa: E402
from TBD.preprocess_xgb import (  # noqa: E402  pylint: disable=wrong-import-position
    attach_neutralized_factors,
    build_panel,
    compute_target_rank,
    create_lagged_columns,
    drop_weak_cross_sections,
    parse_factor_lag_file,
)


def load_item_name_map(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not path.exists():
        return mapping
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("//"):
                continue
            separator = "：" if "：" in line else ":"
            if separator not in line:
                continue
            item_id, name = line.split(separator, 1)
            item_id = item_id.strip()
            name = name.strip()
            if item_id and name:
                mapping[item_id] = name
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="2025-08-17", help="Target date (YYYY-MM-DD) for inference.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data_new")
    parser.add_argument("--mapping", type=Path, default=PROJECT_ROOT / "getdata" / "itemid.txt")
    parser.add_argument("--features-file", type=Path, default=PROJECT_ROOT / "TBD" / "features.md")
    parser.add_argument("--history-days", type=int, default=90, help="How many past days to include when computing lagged factors.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--save-csv",
        type=Path,
        default=None,
        help="Optional path to dump predictions vs. actual ranks.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Optional preprocessed dataset (Parquet). If provided and exists, skips rebuilding factors.",
    )
    parser.add_argument(
        "--require-target",
        action="store_true",
        help="Require target columns (target_rank_pct/target_8d). If absent, inference will error.",
    )
    return parser.parse_args()


def load_cross_section_from_dataset(dataset: Path, date: pd.Timestamp) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(dataset)
    df["date"] = pd.to_datetime(df["date"])
    cross = df.loc[df["date"] == date].copy()
    if cross.empty:
        raise RuntimeError(f"Date {date.date()} not found in dataset {dataset}.")
    excluded = {"date", "item_id"}
    feature_cols = [col for col in cross.columns if col not in excluded]
    target_cols = {"target_rank_pct", "target_8d", "target_rank_label"}
    missing_targets = target_cols.difference(cross.columns)
    if missing_targets:
        for col in missing_targets:
            cross[col] = np.nan
    # Remove target columns from features if they slipped in.
    feature_cols = [col for col in feature_cols if col not in target_cols]
    return cross, feature_cols


def build_cross_section_from_raw(
    args: argparse.Namespace, target_date: pd.Timestamp, require_target: bool
) -> tuple[pd.DataFrame, list[str]]:
    combos = parse_factor_lag_file(args.features_file)
    unique_factors = sorted({combo.name for combo in combos})

    end_ts = parse_window_endpoint(args.date)
    if end_ts is None:
        raise ValueError("Invalid --date provided.")
    start_ts = end_ts - pd.Timedelta(days=args.history_days)

    panel = build_panel(args.data_dir, args.mapping, start_ts, end_ts)
    add_alpha101_features(panel)
    panel = drop_weak_cross_sections(panel)
    panel = attach_neutralized_factors(panel, unique_factors)
    feature_cols = create_lagged_columns(panel, combos)
    panel["target_rank_pct"] = compute_target_rank(panel)

    required = ["date", "item_id"] + feature_cols
    panel = panel.dropna(subset=feature_cols)
    cross = panel.loc[panel["date"] == target_date, required + ["target_rank_pct", "target_8d"]].copy()
    if cross.empty:
        raise RuntimeError(
            f"No cross-sectional data available for {target_date.date()} - check history window or data coverage."
        )
    if require_target and (cross["target_rank_pct"].isna().all() or cross["target_8d"].isna().all()):
        raise RuntimeError("Target data unavailable for requested date while --require-target is set.")
    return cross, feature_cols


def build_dmatrix(frame: pd.DataFrame, feature_cols: list[str]) -> xgb.DMatrix:
    return xgb.DMatrix(frame[feature_cols], feature_names=feature_cols)


def load_model(path: Path) -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(path)
    return booster


def predict_scores(booster: xgb.Booster, matrix: xgb.DMatrix) -> np.ndarray:
    best_iteration = getattr(booster, "best_iteration", None)
    if best_iteration is not None and best_iteration >= 0:
        return booster.predict(matrix, iteration_range=(0, best_iteration + 1))
    return booster.predict(matrix)


def main() -> None:
    args = parse_args()
    target_date = pd.to_datetime(args.date)

    if args.dataset and args.dataset.exists():
        cross_section, feature_cols = load_cross_section_from_dataset(args.dataset, target_date)
    elif DEFAULT_DATASET.exists():
        cross_section, feature_cols = load_cross_section_from_dataset(DEFAULT_DATASET, target_date)
    else:
        cross_section, feature_cols = build_cross_section_from_raw(args, target_date, args.require_target)
    if args.require_target and (
        cross_section.get("target_rank_pct") is None or cross_section["target_rank_pct"].isna().all()
    ):
        raise RuntimeError("Target columns missing but --require-target is set. Provide historical dataset with labels.")

    if not args.model.exists():
        raise FileNotFoundError(f"Model file {args.model} not found.")

    model = load_model(args.model)
    matrix = build_dmatrix(cross_section, feature_cols)
    preds = predict_scores(model, matrix)
    prediction_rank = pd.Series(preds, index=cross_section.index)
    cross_section["pred_score"] = prediction_rank
    cross_section["pred_rank_pct"] = prediction_rank.rank(pct=True)

    name_map = load_item_name_map(args.mapping)
    cross_section["item_name"] = cross_section["item_id"].map(name_map).fillna(cross_section["item_id"])
    cross_section["actual_return"] = cross_section.get("target_8d", np.nan)

    cross_section = cross_section.sort_values("pred_score", ascending=False)
    display_cols = ["item_id", "item_name", "pred_score", "pred_rank_pct"]
    if "target_rank_pct" in cross_section:
        display_cols.append("target_rank_pct")
    if "actual_return" in cross_section:
        display_cols.append("actual_return")
    print(f"Inference results for {target_date.date()} (n={len(cross_section)}):")
    print(cross_section[display_cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if args.save_csv:
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        cross_section[display_cols].to_csv(args.save_csv, index=False)
        print("Saved inference table to", args.save_csv)


if __name__ == "__main__":
    main()
