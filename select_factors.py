#!/usr/bin/env python3
"""Select factor/lag pairs ranked by IC metric while enforcing correlation limits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from rank_ic_analysis import (
    ASIA_SHANGHAI,
    aggregate_to_daily,
    add_alpha101_features,
    compute_features,
    filter_by_window,
    load_hourly_json,
    load_industry_mapping,
    neutralize_cross_section,
    parse_window_endpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metric-csv",
        default="rank_ic_sorted_by_metric.csv",
        help="CSV sorted by metric=ICIR/sqrt(count) (default: %(default)s).",
    )
    parser.add_argument(
        "--data-dir",
        default="data_new",
        help="Directory containing hourly JSON files (default: %(default)s).",
    )
    parser.add_argument(
        "--mapping",
        default="getdata/itemid.txt",
        help="Industry mapping file used during neutralization (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default="selected_factors.csv",
        help="Where to store the selected factor list (default: %(default)s).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Maximum allowed absolute correlation between selected factors (default: %(default)s).",
    )
    parser.add_argument(
        "--min-overlap",
        type=int,
        default=8,
        help="Minimum non-NaN observations required to consider a factor/lag series (default: %(default)s).",
    )
    parser.add_argument(
        "--max-selected",
        type=int,
        default=300,
        help="Optional cap on number of selected factor/lag pairs.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2025-03-01",
        help="Inclusive start date (YYYY-MM-DD) interpreted at 15:00 Asia/Shanghai.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2025-10-13",
        help="Inclusive end date (YYYY-MM-DD) interpreted at 15:00 Asia/Shanghai.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print decisions for skipped factors.",
    )
    parser.add_argument(
        "--history-days",
        type=int,
        default=90,
        help="Historical lookback days to keep for long-window factors and lag generation.",
    )
    parser.add_argument(
        "--preselect",
        nargs="*",
        default=[],#["log_volume_ma,0","log_volume_ma,3","log_volume_ma,7","log_volume_ma,15","log_price_ma,0","log_price_ma,3","log_price_ma,7","log_price_ma,15"],
        help="Factor/lag pairs to force include before greedy selection, format factor,lag (e.g. log_price_ma,7).",
    )
    return parser.parse_args()


def load_metric_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected_cols = {"factor", "lag", "mean", "std", "count", "metric"}
    missing_cols = expected_cols.difference(df.columns)
    if missing_cols:
        raise ValueError(f"{path} missing required columns: {sorted(missing_cols)}")
    df = df.sort_values("metric", ascending=False).reset_index(drop=True)
    return df


def build_feature_panel(
    data_dir: Path,
    mapping_path: Path,
    start_ts: pd.Timestamp | None,
    end_ts: pd.Timestamp | None,
    required_factors: Sequence[str],
    history_days: int,
) -> Tuple[pd.DataFrame, List[str]]:
    extended_start = start_ts - pd.Timedelta(days=history_days) if start_ts is not None else None

    mapping = load_industry_mapping(mapping_path)
    records: List[pd.DataFrame] = []
    feature_columns: List[str] = []

    for json_path in sorted(data_dir.glob("*.json")):
        hourly_df = load_hourly_json(json_path)
        daily_df = aggregate_to_daily(hourly_df)
        feature_df = compute_features(daily_df)
        feature_df = filter_by_window(feature_df, extended_start, end_ts)
        if feature_df.empty:
            continue
        feature_df["item_id"] = json_path.stem
        feature_df["industry"] = mapping.get(json_path.stem, "UNKNOWN")
        records.append(feature_df.reset_index())

        if not feature_columns:
            excluded = {
                "item_id",
                "industry",
                "price_ma_3",
                "volume_ma_3",
                "log_price_ma",
                "log_volume_ma",
                "target_8d",
                "obs",
                "vwap",
            }
            raw_cols = [
                col
                for col in feature_df.columns
                if col not in excluded
                and col
                not in {
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "turnover",
                }
                and feature_df[col].dtype.kind in {"f", "i"}
            ]
            # Allow explicitly requested factors even if normally excluded.
            feature_columns = list(
                {
                    *raw_cols,
                    *(col for col in excluded if col in required_factors),
                }
            )

    if not records:
        raise RuntimeError(f"No usable data found under {data_dir}")

    all_df = pd.concat(records, ignore_index=True)
    all_df["date"] = pd.to_datetime(all_df["date"])
    all_df = all_df.sort_values(["date", "item_id"]).reset_index(drop=True)

    alpha_cols = add_alpha101_features(all_df)
    feature_columns.extend(alpha_cols)

    required = set(required_factors)
    if required:
        missing = sorted(required.difference(feature_columns))
        if missing:
            print(
                f"Warning: {len(missing)} requested factors not found in features: {', '.join(missing)}",
                file=sys.stderr,
            )
        factor_cols = [col for col in feature_columns if col in required]
        if not factor_cols:
            raise RuntimeError("None of the requested factors exist in computed features.")
    else:
        factor_cols = feature_columns

    neutralized = neutralize_cross_section(
        all_df,
        feature_cols=factor_cols,
        industry_col="industry",
        continuous_cols=("log_price_ma", "log_volume_ma"),
    )

    for col in factor_cols:
        all_df[f"{col}_n"] = neutralized[col]

    return all_df, factor_cols


def parse_preselected(values: Sequence[str]) -> List[Tuple[str, int]]:
    """Parse preselected factor/lag pairs from CLI input."""
    pairs: List[Tuple[str, int]] = []
    for raw in values:
        cleaned = raw.replace("；", ";")
        for token in cleaned.split(";"):
            token = token.strip()
            if not token:
                continue
            if "," not in token:
                raise ValueError(f"Expected factor,lag format but got '{token}'.")
            factor, lag_str = [part.strip() for part in token.split(",", 1)]
            if not factor or not lag_str:
                raise ValueError(f"Expected factor,lag format but got '{token}'.")
            try:
                lag = int(lag_str)
            except ValueError as exc:
                raise ValueError(f"Invalid lag '{lag_str}' in '{token}'.") from exc
            pairs.append((factor, lag))
    return pairs


def build_lagged_matrix(
    panel: pd.DataFrame,
    factors: Sequence[str],
    lags: Iterable[int],
) -> pd.DataFrame:
    factor_cols = [f"{name}_n" for name in factors]
    missing = [col for col in factor_cols if col not in panel.columns]
    if missing:
        raise KeyError(f"Missing neutralized columns: {missing}")

    grouped = panel.groupby("item_id", sort=False)
    lagged_frames: List[pd.DataFrame] = []

    for lag in sorted(set(lags)):
        shifted = grouped[factor_cols].shift(lag)
        rename_map = {
            col: f"{col[:-2]}_lag{lag}" if col.endswith("_n") else f"{col}_lag{lag}"
            for col in shifted.columns
        }
        lagged_frames.append(shifted.rename(columns=rename_map))

    lagged_df = pd.concat(lagged_frames, axis=1)
    return lagged_df


def greedy_select(
    metrics: pd.DataFrame,
    factor_values: Dict[str, pd.Series],
    threshold: float,
    min_overlap: int,
    max_selected: int | None,
    verbose: bool = False,
    preselected: Sequence[Tuple[str, int]] | None = None,
) -> List[Dict[str, object]]:
    selected_keys: List[str] = []
    selections: List[Dict[str, object]] = []
    metric_lookup = {(row.factor, int(row.lag)): row for row in metrics.itertuples(index=False)}

    def build_selection(row: object, non_na: int) -> Dict[str, object]:
        return {
            "factor": row.factor,
            "lag": int(row.lag),
            "mean": row.mean,
            "std": row.std,
            "count": int(row.count),
            "metric": row.metric,
            "non_na": non_na,
        }

    preselected = preselected or []
    for factor, lag in preselected:
        key = f"{factor}_lag{lag}"
        row = metric_lookup.get((factor, lag))
        if row is None:
            print(f"Preselect {key} skipped: not found in metric table.", file=sys.stderr)
            continue

        series = factor_values.get(key)
        if series is None:
            print(f"Preselect {key} skipped: series not found in factor matrix.", file=sys.stderr)
            continue

        non_na = int(series.notna().sum())
        if non_na < min_overlap:
            print(
                f"Preselect {key} skipped: only {non_na} non-NaN observations (need {min_overlap}).",
                file=sys.stderr,
            )
            continue

        selected_keys.append(key)
        selections.append(build_selection(row, non_na))

        if max_selected is not None and len(selected_keys) >= max_selected:
            return selections

    for row in metrics.itertuples(index=False):
        key = f"{row.factor}_lag{int(row.lag)}"

        if key in selected_keys:
            continue

        series = factor_values.get(key)
        if series is None:
            if verbose:
                print(f"Skip {key}: series not found in factor matrix.")
            continue

        non_na = int(series.notna().sum())
        if non_na < min_overlap:
            if verbose:
                print(f"Skip {key}: only {non_na} non-NaN observations (need {min_overlap}).")
            continue

        keep = True
        for sel_key in selected_keys:
            corr = series.corr(factor_values[sel_key])
            if np.isnan(corr):
                continue
            if abs(corr) > threshold:
                keep = False
                if verbose:
                    print(
                        f"Skip {key}: corr={corr:.3f} with {sel_key} exceeds threshold {threshold:.2f}."
                    )
                break

        if not keep:
            continue

        selected_keys.append(key)
        selections.append(build_selection(row, non_na))

        if max_selected is not None and len(selected_keys) >= max_selected:
            break

    return selections


def main() -> None:
    args = parse_args()
    metric_path = Path(args.metric_csv)
    data_dir = Path(args.data_dir)
    mapping_path = Path(args.mapping)
    output_path = Path(args.output)

    metrics = load_metric_table(metric_path)
    requested_factors = sorted(metrics["factor"].unique().tolist())
    requested_lags = sorted(metrics["lag"].unique().tolist())

    start_ts = parse_window_endpoint(args.start)
    end_ts = parse_window_endpoint(args.end)

    try:
        preselected = parse_preselected(args.preselect)
    except ValueError as exc:
        raise SystemExit(f"Invalid --preselect value: {exc}") from exc

    panel, factor_cols = build_feature_panel(
        data_dir=data_dir,
        mapping_path=mapping_path,
        start_ts=start_ts,
        end_ts=end_ts,
        required_factors=requested_factors,
        history_days=args.history_days,
    )

    lagged_df = build_lagged_matrix(panel, factors=factor_cols, lags=requested_lags)

    # Final time filter after lag generation to retain edge lags
    if start_ts is not None or end_ts is not None:
        panel_ts = panel["date"].dt.tz_localize(ASIA_SHANGHAI) + pd.Timedelta(hours=15)
        mask = pd.Series(True, index=panel.index)
        if start_ts is not None:
            mask &= panel_ts >= start_ts
        if end_ts is not None:
            mask &= panel_ts <= end_ts
        panel = panel.loc[mask]
        lagged_df = lagged_df.loc[mask]

    desired_columns = sorted(
        {f"{row.factor}_lag{int(row.lag)}" for row in metrics.itertuples(index=False)}
    )
    vectors = {col: lagged_df[col] for col in desired_columns if col in lagged_df.columns}
    if not vectors:
        raise RuntimeError("No matching factor/lag series were found in the computed panel.")

    selections = greedy_select(
        metrics=metrics,
        factor_values=vectors,
        threshold=args.threshold,
        min_overlap=args.min_overlap,
        max_selected=args.max_selected,
        verbose=args.verbose,
        preselected=preselected,
    )

    if not selections:
        raise RuntimeError("No factors satisfied the correlation and coverage constraints.")

    output_df = pd.DataFrame(selections)
    output_df = output_df.sort_values("metric", ascending=False).reset_index(drop=True)
    output_df.to_csv(output_path, index=False)

    print(f"Selected {len(selections)} factor/lag pairs (threshold={args.threshold:.2f}).")
    print(output_df)
    print(f"\nSaved selection to {output_path}")


if __name__ == "__main__":
    main()
