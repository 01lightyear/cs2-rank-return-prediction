#!/usr/bin/env python3
"""Live inference for a single cross-sectional date without relying on labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


from rank_ic_analysis import add_alpha101_features, parse_window_endpoint  # noqa: E402
from TBD.preprocess_xgb import (  # noqa: E402  pylint: disable=wrong-import-position
    attach_neutralized_factors,
    build_panel,
    create_lagged_columns,
    drop_weak_cross_sections,
    parse_factor_lag_file,
)


DEFAULT_MODEL = PROJECT_ROOT / "TBD" / "xgb_rank_model.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="2025-11-28", help="Target date (YYYY-MM-DD) for inference.")
    parser.add_argument("--history-days", type=int, default=90, help="Lookback window for building lags.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data_new")
    parser.add_argument("--mapping", type=Path, default=PROJECT_ROOT / "getdata" / "itemid.txt")
    parser.add_argument("--features-file", type=Path, default=PROJECT_ROOT / "TBD" / "features.md")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--save-csv", type=Path, default=None)
    return parser.parse_args()


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


def prepare_cross_section(args: argparse.Namespace, target_date: pd.Timestamp) -> tuple[pd.DataFrame, list[str]]:
    combos = parse_factor_lag_file(args.features_file)
    unique_factors = sorted({combo.name for combo in combos})

    end_ts = parse_window_endpoint(args.date)
    if end_ts is None:
        raise ValueError("Invalid --date provided.")
    # Use end_ts as the anchor; build_panel will extend by history_days internally.
    panel = build_panel(
        args.data_dir,
        args.mapping,
        end_ts,
        end_ts,
        history_days=args.history_days,
    )
    add_alpha101_features(panel)
    panel = drop_weak_cross_sections(panel)
    panel = attach_neutralized_factors(panel, unique_factors)
    panel, feature_cols = create_lagged_columns(panel, combos)

    missing_mask = panel[feature_cols].isna().any(axis=1)
    missing_rows = panel.loc[(panel["date"] == target_date) & missing_mask, ["item_id"] + feature_cols]
    if not missing_rows.empty:
        name_map = load_item_name_map(args.mapping)
        detail_lines = []
        for _, row in missing_rows.iterrows():
            nan_cols = [col for col in feature_cols if pd.isna(row[col])]
            if not nan_cols:
                continue
            preview = ", ".join(nan_cols[:5])
            if len(nan_cols) > 5:
                preview += ", ..."
            item_id = str(row["item_id"])
            name = name_map.get(item_id)
            display_id = f"{item_id} ({name})" if name else item_id
            detail_lines.append(f"  - {display_id}: {preview}")
        print(f"Skipping {len(detail_lines)} items with NaN features on {target_date.date()}:")
        print("\n".join(detail_lines))

    panel = panel.dropna(subset=feature_cols)
    cross = panel.loc[panel["date"] == target_date, ["date", "item_id"] + feature_cols].copy()
    if cross.empty:
        raise RuntimeError(
            f"No cross-sectional data available for {target_date.date()} - check data coverage or history window."
        )
    return cross, feature_cols


def build_dmatrix(frame: pd.DataFrame, feature_cols: list[str]) -> xgb.DMatrix:
    return xgb.DMatrix(frame[feature_cols], feature_names=feature_cols)


def load_model(path: Path) -> xgb.Booster:
    if not path.exists():
        raise FileNotFoundError(f"Model file {path} not found. Train a model first.")
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

    cross_section, feature_cols = prepare_cross_section(args, target_date)
    model = load_model(args.model)
    matrix = build_dmatrix(cross_section, feature_cols)
    preds = predict_scores(model, matrix)
    cross_section["pred_score"] = preds
    cross_section["pred_rank_pct"] = cross_section["pred_score"].rank(pct=True)

    name_map = load_item_name_map(args.mapping)
    cross_section["item_name"] = cross_section["item_id"].map(name_map).fillna(cross_section["item_id"])

    cross_section = cross_section.sort_values("pred_score", ascending=False)
    display_cols = ["item_id", "item_name", "pred_score", "pred_rank_pct"]
    print(f"Live inference results for {target_date.date()} (n={len(cross_section)}):")
    print(cross_section[display_cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if args.save_csv:
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        cross_section[display_cols].to_csv(args.save_csv, index=False)
        print("Saved predictions to", args.save_csv)


if __name__ == "__main__":
    main()
