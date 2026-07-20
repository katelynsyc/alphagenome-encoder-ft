#!/usr/bin/env python
"""Evaluate an encoder-only AlphaGenome Jores checkpoint on the held-out test split.

Computes per-condition (cold/dark/light/warm/maize) regression metrics between
actual and predicted enrichment, and saves per-sample predictions to CSV so a
separate plotting script (plot_jores_scatter.py) can build actual-vs-predicted
scatterplots without re-running the model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from alphagenome_encoder_ft import (
    AlphaGenomeEncoderModel,
    TrainConfig,
    create_dataloader,
    create_jores_splits,
)

# Order must match JoresMPRADataset._targets (see mydata.py).
CONDITION_NAMES = ["cold", "dark", "light", "warm", "maize"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an encoder-only AlphaGenome Jores checkpoint")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--input_tsv", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use_amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=None)
    return parser


def _load_config_from_checkpoint(checkpoint_path: Path) -> tuple[TrainConfig, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw_config = checkpoint.get("config")
    if raw_config is None:
        raise ValueError(f"Checkpoint does not contain a serialized config: {checkpoint_path}")
    return TrainConfig.from_dict(raw_config), checkpoint


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float64)

    start = 0
    while start < sorted_values.shape[0]:
        end = start + 1
        while end < sorted_values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def compute_pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.size < 2:
        return float("nan")

    true_centered = y_true - y_true.mean()
    pred_centered = y_pred - y_pred.mean()
    denom = np.linalg.norm(true_centered) * np.linalg.norm(pred_centered)
    if denom == 0.0:
        return float("nan")
    return float(np.dot(true_centered, pred_centered) / denom)


def compute_spearmanr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.size < 2:
        return float("nan")
    return compute_pearsonr(_average_ranks(y_true), _average_ranks(y_pred))


@torch.no_grad()
def collect_predictions(
    model,
    data_loader,
    *,
    device: torch.device,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (y_true, y_pred), each shape (N, len(CONDITION_NAMES))."""
    model.eval()

    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []

    for sequences, targets in data_loader:
        sequences = sequences.to(device)
        targets = targets.to(device).float()
        organism_idx = torch.zeros(sequences.shape[0], dtype=torch.long, device=device)

        if use_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                preds = model(sequences, organism_idx)
        else:
            preds = model(sequences, organism_idx)

        all_targets.append(targets.detach().cpu().numpy())
        all_predictions.append(preds.detach().float().cpu().numpy())

    num_conditions = len(CONDITION_NAMES)
    if not all_targets:
        return (
            np.empty((0, num_conditions), dtype=np.float32),
            np.empty((0, num_conditions), dtype=np.float32),
        )

    return (
        np.concatenate(all_targets, axis=0).astype(np.float32, copy=False),
        np.concatenate(all_predictions, axis=0).astype(np.float32, copy=False),
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Per-condition regression metrics, plus their means across conditions."""
    metrics: dict[str, Any] = {"n_samples": int(y_true.shape[0])}

    per_condition: dict[str, dict[str, float]] = {}
    for i, name in enumerate(CONDITION_NAMES):
        true_i = y_true[:, i]
        pred_i = y_pred[:, i]
        residual = pred_i - true_i
        mse = float(np.mean(np.square(residual))) if true_i.size else float("nan")
        per_condition[name] = {
            "mse": mse,
            "rmse": float(math.sqrt(mse)) if not math.isnan(mse) else float("nan"),
            "mae": float(np.mean(np.abs(residual))) if true_i.size else float("nan"),
            "pearsonr": compute_pearsonr(true_i, pred_i),
            "spearmanr": compute_spearmanr(true_i, pred_i),
        }
    metrics["per_condition"] = per_condition

    for stat in ("mse", "rmse", "mae", "pearsonr", "spearmanr"):
        values = [per_condition[name][stat] for name in CONDITION_NAMES]
        metrics[f"mean_{stat}"] = float(np.mean(values))

    return metrics


def save_predictions(path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["index"]
        for name in CONDITION_NAMES:
            header += [f"{name}_true", f"{name}_pred"]
        writer.writerow(header)
        for idx in range(y_true.shape[0]):
            row: list[Any] = [idx]
            for col in range(len(CONDITION_NAMES)):
                row += [float(y_true[idx, col]), float(y_pred[idx, col])]
            writer.writerow(row)


def main() -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path).resolve()
    if not checkpoint_path.exists():
        parser.error(f"Checkpoint not found: {checkpoint_path}")

    config, checkpoint = _load_config_from_checkpoint(checkpoint_path)

    if args.input_tsv is not None:
        config.data.input_tsv = args.input_tsv
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers
    if args.pin_memory is not None:
        config.data.pin_memory = args.pin_memory
    if args.use_amp is not None:
        config.runtime.use_amp = args.use_amp
    if args.device is not None:
        config.runtime.device = args.device

    if not config.data.input_tsv:
        parser.error("data.input_tsv must be present in the checkpoint config or provided via --input_tsv")

    device = torch.device(config.runtime.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    default_output_dir = checkpoint_path.parent / f"{checkpoint_path.stem}_test_eval"
    output_dir = Path(args.output_dir).resolve() if args.output_dir is not None else default_output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = AlphaGenomeEncoderModel.from_checkpoint(checkpoint_path, device=device)

    _, _, test_dataset = create_jores_splits(
        config.data.input_tsv,
        seed=config.runtime.seed,
        sequence_length=config.data.sequence_length,
        reverse_complement=config.data.reverse_complement,
        rc_prob=config.data.rc_prob,
        random_shift=config.data.random_shift,
        shift_prob=config.data.shift_prob,
        max_shift=config.data.max_shift,
    )
    test_loader = create_dataloader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    y_true, y_pred = collect_predictions(
        model,
        test_loader,
        device=device,
        use_amp=config.runtime.use_amp,
    )
    metrics = compute_metrics(y_true, y_pred)
    metrics.update(
        {
            "checkpoint_path": str(checkpoint_path),
            "output_dir": str(output_dir),
            "save_mode": checkpoint.get("save_mode"),
            "checkpoint_stage": checkpoint.get("stage"),
            "checkpoint_epoch": checkpoint.get("epoch"),
        }
    )

    predictions_path = output_dir / "test_predictions.csv"
    metrics_path = output_dir / "test_metrics.json"

    save_predictions(predictions_path, y_true, y_pred)
    with open(metrics_path, "w") as handle:
        json.dump(metrics, handle, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved metrics to {metrics_path}")
    return metrics


if __name__ == "__main__":
    main()
