#!/usr/bin/env python3
"""Generate SHAP-based explanations for the trained XGBoost rank model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Tuple

import numpy as np
import pandas as pd

try:
    import shap
except ImportError as exc:  # pragma: no cover - guide user to install
    raise SystemExit(
        "shap is not installed. Run '/home/user/miniforge3/envs/cs/bin/python -m pip install shap'."
    ) from exc

try:
    import xgboost as xgb
except ImportError as exc:  # pragma: no cover - fail fast
    raise SystemExit(
        "xgboost is not installed. Activate /home/user/miniforge3/envs/cs and install it via pip."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from TBD.train_xgb import prepare_features  # noqa: E402  pylint: disable=wrong-import-position


DEFAULT_DATASET = Path(__file__).resolve().parent / "factor_dataset.parquet"
DEFAULT_MODEL = Path(__file__).resolve().parent / "xgb_rank_model3.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Factor dataset (Parquet).")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Trained XGBoost model (JSON).")
    parser.add_argument("--date", type=str, default=None, help="Optional single date (YYYY-MM-DD) to analyze.")
    parser.add_argument("--start-date", type=str, default="2025-04-01", help="Inclusive start date for filtering.")
    parser.add_argument("--end-date", type=str, default="2025-10-13", help="Inclusive end date for filtering.")
    parser.add_argument("--max-samples", type=int, default=10000, help="Limit the number of rows passed to SHAP.")
    parser.add_argument("--random-state", type=int, default=17, help="RNG seed for sampling.")
    parser.add_argument("--top-k", type=int, default=50, help="How many features to print in the summary table.")
    parser.add_argument(
        "--save-summary",
        type=Path,
        default=None,
        help="Optional path (CSV or Parquet) to store the mean |SHAP| ranking.",
    )
    parser.add_argument(
        "--save-details",
        type=Path,
        default=None,
        help="Optional path to dump per-sample SHAP values (columns: meta + feature contributions).",
    )
    parser.add_argument(
        "--save-abs-shap-ranking",
        type=Path,
        default=Path("shap_rank.csv"),
        help="Optional path to export all features ranked by mean |SHAP| (descending).",
    )
    return parser.parse_args()


def parse_date(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.to_datetime(value)
    return pd.Timestamp(ts.date())


def load_dataset(
    path: Path,
    focus_date: pd.Timestamp | None,
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp | None,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset {path} not found. Run preprocess_xgb.py first.")

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])

    if focus_date is not None:
        df = df.loc[df["date"] == focus_date]
    else:
        if start_date is not None:
            df = df.loc[df["date"] >= start_date]
        if end_date is not None:
            df = df.loc[df["date"] <= end_date]

    if df.empty:
        raise RuntimeError("No rows left after applying date filters.")

    df = df.sort_values(["date", "item_id"]).reset_index(drop=True)
    return df


def maybe_sample(df: pd.DataFrame, max_samples: int, random_state: int) -> pd.DataFrame:
    if max_samples is None or max_samples <= 0 or len(df) <= max_samples:
        return df
    return df.sample(n=max_samples, random_state=random_state).sort_values(["date", "item_id"]).reset_index(drop=True)


def load_booster(path: Path) -> xgb.Booster:
    if not path.exists():
        raise FileNotFoundError(f"Model file {path} not found. Train a model first.")
    booster = xgb.Booster()
    booster.load_model(path)
    return booster


def predict_scores(booster: xgb.Booster, features: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    matrix = xgb.DMatrix(features, feature_names=feature_cols)
    best_iteration = getattr(booster, "best_iteration", None)
    if best_iteration is not None and best_iteration >= 0:
        return booster.predict(matrix, iteration_range=(0, best_iteration + 1))
    return booster.predict(matrix)


def compute_shap_matrix(
    booster: xgb.Booster, features: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, float]:
    explainer = shap.TreeExplainer(booster)
    explanation = explainer(features, check_additivity=False)

    shap_values = np.asarray(explanation.values, dtype=np.float64)
    base_values = np.asarray(explanation.base_values, dtype=np.float64).reshape(-1)
    expected_value = float(np.mean(base_values))

    return shap_values, base_values, expected_value


def summarize_shap(shap_matrix: np.ndarray, feature_cols: list[str]) -> pd.DataFrame:
    abs_mean = np.abs(shap_matrix).mean(axis=0)
    mean_vals = shap_matrix.mean(axis=0)
    std_vals = shap_matrix.std(axis=0)
    summary = pd.DataFrame(
        {
            "feature": feature_cols,
            "mean_abs_shap": abs_mean,
            "mean_shap": mean_vals,
            "std_shap": std_vals,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    return summary


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)


def main() -> None:
    args = parse_args()
    focus_date = parse_date(args.date)
    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)

    if focus_date is not None and (start_date is not None or end_date is not None):
        print("Note: --date overrides --start-date/--end-date filters.")

    df = load_dataset(args.dataset, focus_date, start_date, end_date)
    feature_cols = prepare_features(df)
    meta_cols = ["date", "item_id"]
    optional_targets = [col for col in ["target_rank_pct", "target_8d"] if col in df.columns]
    ordered_cols = meta_cols + optional_targets + feature_cols
    working_df = df[ordered_cols]

    sampled_df = maybe_sample(working_df, args.max_samples, args.random_state)
    feature_frame = sampled_df[feature_cols].astype(np.float32)

    booster = load_booster(args.model)
    shap_matrix, base_values, expected_value = compute_shap_matrix(booster, feature_frame)

    preds = predict_scores(booster, feature_frame, feature_cols)

    summary = summarize_shap(shap_matrix, feature_cols)
    top_k = min(len(summary), max(1, args.top_k))

    print(
        f"Explained {len(feature_frame)} samples spanning "
        f"{sampled_df['date'].min().date()} -> {sampled_df['date'].max().date()} "
        f"({sampled_df['date'].nunique()} dates)."
    )
    print(f"Model expected value (base prediction): {expected_value:.6f}")
    print("Top features by mean |SHAP| contribution:")
    print(summary.head(top_k).to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    if args.save_summary:
        save_table(summary, args.save_summary)
        print("Saved SHAP summary to", args.save_summary)

    if args.save_abs_shap_ranking:
        ranking = summary.reset_index(drop=True).copy()
        ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
        save_table(ranking, args.save_abs_shap_ranking)
        print("Saved abs SHAP ranking to", args.save_abs_shap_ranking)

    if args.save_details:
        shap_df = pd.DataFrame(shap_matrix, columns=feature_cols)
        shap_df.insert(0, "base_value", base_values)
        shap_df.insert(0, "prediction", preds)
        if "target_rank_pct" in sampled_df.columns:
            shap_df.insert(0, "target_rank_pct", sampled_df["target_rank_pct"].values)
        if "target_8d" in sampled_df.columns:
            shap_df.insert(0, "target_8d", sampled_df["target_8d"].values)
        shap_df.insert(0, "item_id", sampled_df["item_id"].values)
        shap_df.insert(0, "date", sampled_df["date"].values)
        save_table(shap_df, args.save_details)
        print("Saved per-sample SHAP details to", args.save_details)


if __name__ == "__main__":
    main()
