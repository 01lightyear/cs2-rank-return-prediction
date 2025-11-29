#!/usr/bin/env python3
"""Build neutralized lagged factor matrix for XGBoost training."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

# Reuse the battle-tested feature engineering helpers.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rank_ic_analysis import (  # noqa: E402  pylint: disable=wrong-import-position
    aggregate_to_daily,
    add_alpha101_features,
    compute_features,
    filter_by_window,
    load_hourly_json,
    load_industry_mapping,
    neutralize_cross_section,
    neutralize_target_by_industry,
    parse_window_endpoint,
)


ASIA_SHANGHAI = "Asia/Shanghai"
CONTINUOUS_COVARIATES = ["log_price_ma","log_volume_ma"]
MIN_CROSS_SECTION = 20


@dataclass(frozen=True)
class FactorLag:
    """Container describing a (factor, lag) pair."""

    name: str
    lag: int


def parse_factor_lag_file(path: Path) -> List[FactorLag]:
    """Read factor/lag pairs from Markdown table or CSV content.

    The file can be either the legacy whitespace table (rank, factor, lag) or
    the CSV-style output used by selected_factors.csv (see TBD/features.md).
    """

    combos: List[FactorLag] = []

    # Peek at the first non-empty line to determine parsing mode.
    with path.open(encoding="utf-8") as handle:
        lines = [line for line in handle if line.strip()]

    if lines and "," in lines[0]:
        df = pd.read_csv(path)
        if "factor" not in df.columns or "lag" not in df.columns:
            raise ValueError(
                f"CSV factor file {path} must contain 'factor' and 'lag' columns."
            )
        for _, row in df.iterrows():
            try:
                lag = int(row["lag"])
            except (TypeError, ValueError):
                continue
            combos.append(FactorLag(name=str(row["factor"]), lag=lag))
    else:
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            # First column is rank index, second is factor name, third is lag.
            try:
                int(parts[0])
                lag = int(parts[2])
            except ValueError:
                continue
            factor = parts[1]
            combos.append(FactorLag(name=factor, lag=lag))

    if not combos:
        raise ValueError(f"No factor/lag definitions parsed from {path}.")

    return combos


def ensure_columns_exist(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for neutralization: {missing}")


def compute_target_rank(df: pd.DataFrame) -> pd.Series:
    """Return cross-sectional percentile ranks of target within each date."""

    def _rank_pct(values: pd.Series) -> pd.Series:
        valid = values.notna()
        if valid.sum() < MIN_CROSS_SECTION:
            return pd.Series(np.nan, index=values.index)
        ranked = values.rank(method="average", pct=True)
        ranked[~valid] = np.nan
        return ranked

    return df.groupby("date")["target_8d"].transform(_rank_pct)


RANK_BUCKET_BINS = (-np.inf, 0.4, 0.7, 0.9,0.97, np.inf)
RANK_BUCKET_LABELS = [0.0, 1.0, 3.0, 7.0, 10.0]


def bucketize_rank(series: pd.Series) -> pd.Series:
    """Map percentile ranks into discrete buckets for NDCG training."""

    return pd.cut(
        series,
        bins=RANK_BUCKET_BINS,
        labels=RANK_BUCKET_LABELS,
        right=False,
        include_lowest=True,
    ).astype(float)


def build_panel(
    data_dir: Path,
    mapping_path: Path,
    start_ts: pd.Timestamp | None,
    end_ts: pd.Timestamp | None,
    history_days: int,
) -> pd.DataFrame:
    """Load hourly JSONs, aggregate to daily, and attach meta columns."""

    extended_start = start_ts - pd.Timedelta(days=history_days) if start_ts is not None else None
    mapping = load_industry_mapping(mapping_path)
    records: List[pd.DataFrame] = []

    for json_path in sorted(data_dir.glob("*.json")):
        hourly_df = load_hourly_json(json_path)
        daily_df = aggregate_to_daily(hourly_df)
        features_df = compute_features(daily_df)
        features_df = filter_by_window(features_df, extended_start, end_ts)
        if features_df.empty:
            continue
        features_df = features_df.copy()
        features_df["item_id"] = json_path.stem
        features_df["industry"] = mapping.get(json_path.stem, "UNKNOWN")
        records.append(features_df.reset_index())

    if not records:
        raise RuntimeError(f"No usable daily data in {data_dir} for the requested window.")

    panel = pd.concat(records, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["date", "item_id"]).reset_index(drop=True)
    return panel


def attach_neutralized_factors(
    panel: pd.DataFrame,
    unique_factors: Iterable[str],
) -> pd.DataFrame:
    """Neutralize selected factors and append *_n columns to the panel."""

    unique_factors = list(unique_factors)
    ensure_columns_exist(panel, CONTINUOUS_COVARIATES)
    ensure_columns_exist(panel, unique_factors)

    residuals = neutralize_cross_section(
        panel,
        feature_cols=unique_factors,
        industry_col="industry",
        continuous_cols=CONTINUOUS_COVARIATES,
    )

    for factor in unique_factors:
        panel[f"{factor}_n"] = residuals[factor]

    return panel


def create_lagged_columns(
    panel: pd.DataFrame, combos: Sequence[FactorLag]
) -> Tuple[pd.DataFrame, List[str]]:
    """Generate lagged versions of neutralized factors per item without fragmenting."""

    grouped = panel.groupby("item_id", sort=False)
    feature_cols: List[str] = []
    lagged_data: dict[str, pd.Series] = {}

    for combo in combos:
        base_col = f"{combo.name}_n"
        if base_col not in panel:
            raise KeyError(f"Neutralized base column '{base_col}' missing.")
        col_name = f"{combo.name}_lag{combo.lag}"
        lagged_series = grouped[base_col].shift(combo.lag)
        lagged_series.name = col_name
        lagged_data[col_name] = lagged_series
        feature_cols.append(col_name)

    lagged_df = pd.DataFrame(lagged_data, index=panel.index)
    panel = pd.concat([panel, lagged_df], axis=1)

    return panel, feature_cols


def drop_weak_cross_sections(panel: pd.DataFrame) -> pd.DataFrame:
    counts = panel.groupby("date")["item_id"].transform("count")
    mask = counts >= MIN_CROSS_SECTION
    return panel.loc[mask].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data_new", type=Path)
    parser.add_argument("--mapping", default="getdata/itemid.txt", type=Path)
    parser.add_argument(
        "--features-file", default=PROJECT_ROOT / "TBD" / "features.md", type=Path
    )
    parser.add_argument(
        "--output",
        default=PROJECT_ROOT / "TBD" / "factor_dataset.parquet",
        type=Path,
        help="Path to save the processed panel (Parquet).",
    )
    parser.add_argument("--start", type=str, default="2025-03-01")
    parser.add_argument("--end", type=str, default="2025-11-19")
    parser.add_argument(
        "--history-days",
        type=int,
        default=90,
        help="Historical lookback days to keep for long-window factors and lag generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    combos = parse_factor_lag_file(args.features_file)
    unique_factors = sorted({combo.name for combo in combos})

    start_ts = parse_window_endpoint(args.start)
    end_ts = parse_window_endpoint(args.end)
    if start_ts and end_ts and start_ts > end_ts:
        raise ValueError("Start timestamp must not be after end timestamp.")

    panel = build_panel(args.data_dir, args.mapping, start_ts, end_ts, history_days=args.history_days)
    alpha_needed = [name for name in unique_factors if name.lower().startswith("alpha")]
    add_alpha101_features(panel, only=alpha_needed if alpha_needed else None)
    panel = drop_weak_cross_sections(panel)

    panel = attach_neutralized_factors(panel, unique_factors)
    panel, feature_cols = create_lagged_columns(panel, combos)

    # Final time filter after lag generation to retain edge lags
    if start_ts is not None or end_ts is not None:
        panel_ts = panel["date"].dt.tz_localize(ASIA_SHANGHAI) + pd.Timedelta(hours=15)
        mask = pd.Series(True, index=panel.index)
        if start_ts is not None:
            mask &= panel_ts >= start_ts
        if end_ts is not None:
            mask &= panel_ts <= end_ts
        panel = panel.loc[mask]

    panel["target_rank_pct"] = compute_target_rank(panel)
    panel["target_rank_label"] = bucketize_rank(panel["target_rank_pct"])

    required_cols = ["target_rank_pct", "target_rank_label", "target_8d"] + feature_cols
    panel = panel.dropna(subset=required_cols)

    # Keep only the columns needed downstream.
    final_cols = [
        "date",
        "item_id",
        "target_8d",
        "target_rank_pct",
        "target_rank_label",
    ] + feature_cols
    final_df = panel[final_cols].sort_values(["date", "item_id"])

    if final_df.empty:
        raise RuntimeError("Processed dataset is empty after filtering and lagging.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(args.output, index=False)

    print(
        f"Saved {len(final_df):,} rows with {len(feature_cols)} lagged factors to {args.output}."
    )


if __name__ == "__main__":
    main()
