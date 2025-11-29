#!/usr/bin/env python3
"""Train an XGBoost model on neutralized lagged factors and evaluate accuracy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except ImportError as exc:  # pragma: no cover - fail fast with clear message
    raise SystemExit(
        "xgboost is not installed in the active environment. "
        "Install it via 'pip install xgboost' inside /home/user/miniforge3/envs/cs."  # noqa: E501
    ) from exc


DEFAULT_DATASET = Path(__file__).resolve().parent / "factor_dataset.parquet"
TARGET_LABEL_COL = "target_rank_label"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path(__file__).resolve().parent / "xgb_rank_model.json",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path(__file__).resolve().parent / "xgb_rank_metrics.json",
    )
    parser.add_argument("--top-frac", type=float, default=0.05)
    parser.add_argument(
        "--truth-frac",
        type=float,
        default=0.1,
        help="Fraction of true top/bottom ranks used when scoring accuracy.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--n-estimators", type=int, default=1000)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample", type=float, default=0.8)
    parser.add_argument("--early-stopping-rounds", type=int, default=250)
    parser.add_argument(
        "--grid-config",
        type=Path,
        default=Path("TBD/xgb_grid_sample.json"),
        help="Optional JSON file describing parameter overrides for grid search.",
    )
    return parser.parse_args()


def split_by_date(
    df: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1
) -> Dict[str, pd.DataFrame]:
    dates = np.sort(df["date"].unique())
    if len(dates) < 3:
        raise ValueError("Need at least 3 unique dates to form train/val/test splits.")

    train_end = max(1, int(np.floor(len(dates) * train_ratio)))
    val_end = max(train_end + 1, int(np.floor(len(dates) * (train_ratio + val_ratio))))
    if val_end >= len(dates):
        val_end = len(dates) - 1
    if train_end >= val_end:
        train_end = val_end - 1
    if train_end <= 0 or val_end <= train_end:
        raise ValueError("Unable to form sequential date splits with current ratios.")

    train_dates = dates[:train_end]
    val_dates = dates[train_end:val_end]
    test_dates = dates[val_end:]

    if not len(val_dates) or not len(test_dates):
        raise ValueError("Validation or test split ended up empty. Adjust ratios or data window.")

    return {
        "train": df[df["date"].isin(train_dates)].copy(),
        "val": df[df["date"].isin(val_dates)].copy(),
        "test": df[df["date"].isin(test_dates)].copy(),
    }


def precision_at_frac(true_pct: np.ndarray, preds: np.ndarray, pred_frac: float, truth_frac: float) -> float:
    if len(true_pct) == 0:
        return float("nan")

    pred_frac = min(max(pred_frac, 0.0), 0.5)
    truth_frac = min(max(truth_frac, 0.0), 0.5)
    if pred_frac == 0 or truth_frac == 0:
        raise ValueError("top_frac must be > 0")

    k = max(1, int(np.floor(len(preds) * pred_frac)))
    order = np.argsort(preds)[::-1]
    top_idx = order[:k]

    positives = true_pct >= 1 - truth_frac
    hits = positives[top_idx].astype(float)
    return float(hits.mean())


def precision_bottom_frac(
    true_pct: np.ndarray, preds: np.ndarray, pred_frac: float, truth_frac: float
) -> float:
    if len(true_pct) == 0:
        return float("nan")

    pred_frac = min(max(pred_frac, 0.0), 0.5)
    truth_frac = min(max(truth_frac, 0.0), 0.5)
    if pred_frac == 0 or truth_frac == 0:
        raise ValueError("top_frac must be > 0")

    k = max(1, int(np.floor(len(preds) * pred_frac)))
    order = np.argsort(preds)
    bottom_idx = order[:k]

    positives = true_pct <= truth_frac
    hits = positives[bottom_idx].astype(float)
    return float(hits.mean())


def spearman_corr(true_pct: np.ndarray, preds: np.ndarray) -> float:
    mask = np.isfinite(true_pct) & np.isfinite(preds)
    if mask.sum() < 2:
        return float("nan")
    truth = pd.Series(true_pct[mask])
    preds_series = pd.Series(preds[mask])
    truth_rank = truth.rank(method="average")
    pred_rank = preds_series.rank(method="average")
    return float(truth_rank.corr(pred_rank, method="pearson"))


def prepare_features(df: pd.DataFrame) -> List[str]:
    excluded = {"date", "item_id", "target_8d", "target_rank_pct", TARGET_LABEL_COL}
    feature_cols = [col for col in df.columns if col not in excluded]
    if not feature_cols:
        raise ValueError("No feature columns detected in dataset.")
    return feature_cols


def build_model_params(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg",
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "n_estimators": args.n_estimators,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample,
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
        "early_stopping_rounds": args.early_stopping_rounds,
    }


def build_dmatrix(
    frame: pd.DataFrame,
    feature_cols: List[str],
    require_min_group_size: bool,
) -> tuple[xgb.DMatrix, np.ndarray]:
    ordered = frame.sort_values(["date", "item_id"]).reset_index(drop=True)
    matrix = xgb.DMatrix(
        ordered[feature_cols],
        label=ordered[TARGET_LABEL_COL],
        feature_names=feature_cols,
    )
    group_sizes = ordered.groupby("date", sort=False).size().tolist()
    if require_min_group_size and any(size < 2 for size in group_sizes):
        raise ValueError(
            "Each train/val date group must contain at least two items for ranking objective."
        )
    matrix.set_group(group_sizes)
    truth = ordered["target_rank_pct"].to_numpy()
    return matrix, truth


def format_top_metric_name(pred_frac: float, truth_frac: float) -> str:
    return f"top_acc@{pred_frac:.2f}_truth@{truth_frac:.2f}"


def make_top_acc_metric(
    pred_frac: float, truth_frac: float, truth_lookup: Dict[int, np.ndarray]
):
    """Return a custom_metric callback computing precision_at_frac using pred_frac vs truth_frac."""

    metric_name = format_top_metric_name(pred_frac, truth_frac)

    def _metric(predt: np.ndarray, dmatrix: xgb.DMatrix):
        top_frac = min(max(pred_frac, 0.0), 0.5)
        truth = truth_lookup.get(id(dmatrix))
        if truth is None or len(truth) == 0:
            return metric_name, float("nan")
        score = precision_at_frac(truth, predt, top_frac, truth_frac)
        return metric_name, score

    return _metric


def predict_scores(model: xgb.Booster, matrix: xgb.DMatrix) -> np.ndarray:
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is not None and best_iteration >= 0:
        return model.predict(matrix, iteration_range=(0, best_iteration + 1))
    return model.predict(matrix)


def train_model(
    matrices: Dict[str, xgb.DMatrix],
    truths: Dict[str, np.ndarray],
    model_params: Dict[str, object],
    top_frac: float,
    truth_frac: float,
) -> xgb.Booster:
    params = model_params.copy()
    num_boost_round = int(params.pop("n_estimators"))
    early_stopping_rounds = params.pop("early_stopping_rounds")
    truth_lookup = {id(matrix): truths[name] for name, matrix in matrices.items()}
    top_metric_name = format_top_metric_name(top_frac, truth_frac)
    callbacks = [
        xgb.callback.EarlyStopping(
            rounds=early_stopping_rounds,
            save_best=True,
            data_name="val",
            metric_name=top_metric_name,
            maximize=True,
        )
    ]

    booster = xgb.train(
        params=params,
        dtrain=matrices["train"],
        num_boost_round=num_boost_round,
        evals=[(matrices["train"], "train"), (matrices["val"], "val")],
        callbacks=callbacks,
        custom_metric=make_top_acc_metric(top_frac, truth_frac, truth_lookup),
        maximize=True,
        verbose_eval=50,
    )

    return booster


def evaluate_model(
    model: xgb.Booster,
    matrices: Dict[str, xgb.DMatrix],
    truths: Dict[str, np.ndarray],
    pred_frac: float,
    truth_frac: float,
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for split_name, matrix in matrices.items():
        preds = predict_scores(model, matrix)
        truth = truths[split_name]
        precision_top = precision_at_frac(truth, preds, pred_frac, truth_frac)
        precision_bottom = precision_bottom_frac(truth, preds, pred_frac, truth_frac)
        corr = spearman_corr(truth, preds)
        metrics[split_name] = {
            "accuracy_top_frac": precision_top,
            "accuracy_bottom_frac": precision_bottom,
            "spearman": corr,
        }
    return metrics


def load_grid_config(path: Path | None) -> List[Dict[str, object]]:
    if path is None:
        return []
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise TypeError("Grid config must be a JSON list of parameter dictionaries.")
    combos: List[Dict[str, object]] = []
    for entry in data:
        if not isinstance(entry, dict):
            raise TypeError("Each grid config entry must be a dictionary.")
        combos.append(entry)
    if not combos:
        raise ValueError("Grid config list is empty.")
    return combos


def train_with_grid(
    matrices: Dict[str, xgb.DMatrix],
    truths: Dict[str, np.ndarray],
    base_params: Dict[str, object],
    pred_frac: float,
    truth_frac: float,
    grid: List[Dict[str, object]],
) -> tuple[xgb.Booster, Dict[str, Dict[str, float]], Dict[str, object], List[dict]]:
    best_model: xgb.Booster | None = None
    best_metrics: Dict[str, Dict[str, float]] | None = None
    best_params: Dict[str, object] | None = None
    best_score = -np.inf
    results: List[dict] = []

    for idx, override in enumerate(grid):
        merged_params = {**base_params, **override}
        print(f"Grid search {idx + 1}/{len(grid)} params={override}")
        model = train_model(matrices, truths, merged_params, pred_frac, truth_frac)
        metrics = evaluate_model(model, matrices, truths, pred_frac, truth_frac)
        val_score = metrics["val"]["accuracy_top_frac"]
        results.append({
            "index": idx,
            "params": merged_params,
            "metrics": metrics,
        })
        print(
            "  -> val "
            f"accuracy@{pred_frac:.2f}_truth@{truth_frac:.2f}={val_score:.4f}"
        )
        if np.isnan(val_score):
            continue
        if val_score > best_score:
            best_score = val_score
            best_model = model
            best_metrics = metrics
            best_params = merged_params

    if best_model is None or best_metrics is None or best_params is None:
        raise RuntimeError("Grid search did not yield a valid model.")

    return best_model, best_metrics, best_params, results


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset {args.dataset} not found. Run preprocess_xgb.py first.")

    df = pd.read_parquet(args.dataset)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "item_id"]).reset_index(drop=True)

    feature_cols = prepare_features(df)
    splits = split_by_date(df)
    matrices: Dict[str, xgb.DMatrix] = {}
    truths: Dict[str, np.ndarray] = {}
    for name, frame in splits.items():
        matrix, truth = build_dmatrix(
            frame,
            feature_cols,
            require_min_group_size=(name in {"train", "val"}),
        )
        matrices[name] = matrix
        truths[name] = truth

    base_params = build_model_params(args)
    grid = load_grid_config(args.grid_config)
    grid_results: List[dict] = []

    if grid:
        model, metrics, best_params, grid_results = train_with_grid(
            matrices, truths, base_params, args.top_frac, args.truth_frac, grid
        )
    else:
        model = train_model(matrices, truths, base_params, args.top_frac, args.truth_frac)
        metrics = evaluate_model(model, matrices, truths, args.top_frac, args.truth_frac)
        best_params = base_params

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(args.model_output)

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_params": best_params,
        "metrics": metrics,
    }
    if grid_results:
        payload["grid_results"] = grid_results
    with args.metrics_output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print("Saved model to", args.model_output)
    print("Evaluation metrics:")
    for split_name, values in metrics.items():
        top_acc = values["accuracy_top_frac"]
        bottom_acc = values["accuracy_bottom_frac"]
        corr = values["spearman"]
        print(
            "  "
            f"{split_name}: top_acc@{args.top_frac:.2f}_truth@{args.truth_frac:.2f}={top_acc:.4f} "
            f"bottom_acc@{args.top_frac:.2f}_truth@{args.truth_frac:.2f}={bottom_acc:.4f} "
            f"spearman={corr:.4f}"
        )

    print("Metrics written to", args.metrics_output)
    if grid_results:
        print(f"Grid search evaluated {len(grid_results)} combinations; best params saved in metrics JSON.")


if __name__ == "__main__":
    main()
