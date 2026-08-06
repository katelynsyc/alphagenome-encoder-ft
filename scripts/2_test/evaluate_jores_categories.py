#!/usr/bin/env python
"""Evaluate the Jores checkpoint across media-5.xlsx's design-validation categories.

metadata/jores_design_validation.tsv (media-5.xlsx's test-set rows, annotated with an
`experiment` column) mixes five categories per row: `evolution: ...` (plantGREP in
silico evolution), `TFBS shuffling: TF_##`, `TFBS insertion: <a>, <b>, <c>`,
`unmodified control`, and `validation of ACR sequence library`. Scoring every row
against all 5 Jores conditions the way evaluate_jores.py does is fine for an overall
number, but hides two things this script surfaces instead:

  - evolution rows should mostly be judged on the condition(s) they were actually
    selected for (see evolution_objectives.py) -- both the on-target subset (plotted
    by plot_jores_categories.py as "Evolved Condition-Specific Sequences") and its
    complement, the off-target subset ("Off-Target Evolved Sequences"), to see how much
    a directed-evolution run's gains generalize vs. overfit to its own objective.
  - TFBS shuffling/insertion rows ("Perturbed Library Sequences") are grouped by which
    perturbation was applied -- shuffling, or insertion of 1/2/3 binding sites -- since
    those are mechanistically different edits to the same background sequence.

One forward pass over the whole file produces (y_true, y_pred); metadata parsed from
each row's `experiment` string picks out which of these groups it belongs to. Saves an
enriched predictions CSV (category/target_conditions/total_rounds/perturbation columns
alongside the usual {condition}_true/{condition}_pred) and an evaluation_metrics.json
with the same per-condition mse/rmse/mae/pearsonr shape as evaluate_jores.py, computed
separately for the overall set, evolution on-/off-target, and each perturbation group --
so plot_jores_categories.py can build its three scatterplots without re-running the
model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from alphagenome_encoder_ft import (
    AlphaGenomeEncoderModel,
    create_dataloader,
    create_jores_splits,
    load_config_from_checkpoint,
)
from alphagenome_encoder_ft.evolution_objectives import (
    target_conditions_for_experiment,
    total_evolution_rounds,
)

from evaluate_jores import CONDITION_NAMES, collect_predictions, compute_metrics, compute_pearsonr

_TFBS_INSERTION_RE = re.compile(r"^TFBS insertion:\s*(.+)$")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_TSV = REPO_ROOT / "metadata" / "jores_design_validation.tsv"


def classify_row(experiment: str) -> dict[str, Any]:
    """Per-row category + the extra metadata that category needs downstream.

    Raises on an experiment string that doesn't match any known category, rather than
    silently dropping it -- same "fail loudly on unrecognized vocabulary" stance as
    evolution_objectives.target_conditions_for_experiment.
    """
    experiment = experiment.strip()

    if experiment.startswith("evolution:"):
        return {
            "category": "evolution",
            "target_conditions": target_conditions_for_experiment(experiment),
            "total_rounds": total_evolution_rounds(experiment),
            "perturbation": "",
        }

    if experiment.startswith("TFBS shuffling"):
        return {"category": "tfbs_shuffling", "target_conditions": (), "total_rounds": None, "perturbation": "shuffling"}

    match = _TFBS_INSERTION_RE.match(experiment)
    if match is not None:
        slots = [token.strip() for token in match.group(1).split(",")]
        if len(slots) != 3:
            raise ValueError(f"Expected 3 comma-separated TFBS insertion slots, got: {experiment!r}")
        n_inserted = sum(1 for slot in slots if slot != "none")
        return {
            "category": "tfbs_insertion",
            "target_conditions": (),
            "total_rounds": None,
            "perturbation": f"insertion_{n_inserted}",
        }

    if experiment == "unmodified control":
        return {"category": "unmodified_control", "target_conditions": (), "total_rounds": None, "perturbation": ""}

    if experiment == "validation of ACR sequence library":
        return {"category": "validation_of_acr", "target_conditions": (), "total_rounds": None, "perturbation": ""}

    raise ValueError(f"Unrecognized experiment category: {experiment!r}")


def _condition_mask(target_conditions: list[tuple[str, ...] | None], name: str, mode: str) -> np.ndarray:
    """Boolean row-mask for one condition panel.

    target_conditions[i] is None for non-evolution rows (never on- or off-target for
    any condition), or the tuple of conditions that row's evolution objective(s)
    targeted (see evolution_objectives.target_conditions_for_experiment).
    """
    if mode == "on_target":
        return np.array([targets is not None and name in targets for targets in target_conditions])
    if mode == "off_target":
        return np.array([targets is not None and name not in targets for targets in target_conditions])
    raise ValueError(f"Unknown mode: {mode!r}")


def _uniform_masks(mask: np.ndarray, condition_names: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Same row-mask applied to every condition panel (perturbation groups aren't
    scoped to a particular condition the way evolution on-/off-target rows are)."""
    return {name: mask for name in condition_names}


def compute_condition_scoped_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    condition_masks: dict[str, np.ndarray],
    condition_names: tuple[str, ...],
) -> dict[str, Any]:
    """Per-condition regression metrics, each restricted to condition_masks[name] --
    the generalization of evolution_objectives.compute_targeted_metrics to masks that
    don't have to come from "name in target_conditions" (e.g. a perturbation-type mask
    applied uniformly across all 5 conditions)."""
    metrics: dict[str, Any] = {}
    per_condition: dict[str, dict[str, float]] = {}

    for i, name in enumerate(condition_names):
        mask = condition_masks[name]
        true_i = y_true[mask, i]
        pred_i = y_pred[mask, i]
        residual = pred_i - true_i
        mse = float(np.mean(np.square(residual))) if true_i.size else float("nan")
        per_condition[name] = {
            "n_samples": int(mask.sum()),
            "mse": mse,
            "rmse": float(math.sqrt(mse)) if not math.isnan(mse) else float("nan"),
            "mae": float(np.mean(np.abs(residual))) if true_i.size else float("nan"),
            "pearsonr": compute_pearsonr(true_i, pred_i),
        }
    metrics["per_condition"] = per_condition

    for stat in ("mse", "rmse", "mae", "pearsonr"):
        values = [per_condition[name][stat] for name in condition_names]
        metrics[f"mean_{stat}"] = float(np.nanmean(values))

    return metrics


def save_category_predictions(
    path: Path,
    ids: list[str],
    row_meta: list[dict[str, Any]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    condition_names: tuple[str, ...],
) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["index", "id", "category", "target_conditions", "total_rounds", "perturbation"]
        for name in condition_names:
            header += [f"{name}_true", f"{name}_pred"]
        writer.writerow(header)

        for idx in range(y_true.shape[0]):
            meta = row_meta[idx]
            row: list[Any] = [
                idx,
                ids[idx],
                meta["category"],
                ",".join(meta["target_conditions"]),
                meta["total_rounds"] if meta["total_rounds"] is not None else "",
                meta["perturbation"],
            ]
            for col in range(len(condition_names)):
                row += [float(y_true[idx, col]), float(y_pred[idx, col])]
            writer.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument(
        "--input_tsv",
        type=str,
        default=None,
        help=f"Default: {DEFAULT_INPUT_TSV} (media-5.xlsx's design-validation table, all rows set='test')",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use_amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=None)
    return parser


def main() -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path).resolve()
    if not checkpoint_path.exists():
        parser.error(f"Checkpoint not found: {checkpoint_path}")

    input_tsv = Path(args.input_tsv).resolve() if args.input_tsv is not None else DEFAULT_INPUT_TSV.resolve()
    if not input_tsv.exists():
        parser.error(f"Input TSV not found: {input_tsv}")

    config, checkpoint = load_config_from_checkpoint(checkpoint_path)
    config.data.input_tsv = str(input_tsv)
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

    device = torch.device(config.runtime.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    default_output_dir = checkpoint_path.parent / f"{checkpoint_path.stem}_category_eval"
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

    y_true, y_pred = collect_predictions(model, test_loader, device=device, use_amp=config.runtime.use_amp)

    # Same 'set' == 'test' filter, in the same original-file order, as create_jores_splits
    # -- so this lines up row-for-row with (y_true, y_pred) collected with shuffle=False.
    with open(input_tsv, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = [row for row in reader if row["set"] == "test"]

    if len(rows) != y_true.shape[0]:
        raise ValueError(f"Row count mismatch: {len(rows)} metadata rows vs {y_true.shape[0]} predictions")

    ids = [row["id"] for row in rows]
    row_meta = [classify_row(row["experiment"]) for row in rows]

    overall_metrics = compute_metrics(y_true, y_pred)

    target_conditions: list[tuple[str, ...] | None] = [
        meta["target_conditions"] if meta["category"] == "evolution" else None for meta in row_meta
    ]
    on_target_masks = {name: _condition_mask(target_conditions, name, "on_target") for name in CONDITION_NAMES}
    off_target_masks = {name: _condition_mask(target_conditions, name, "off_target") for name in CONDITION_NAMES}
    evolution_on_target = compute_condition_scoped_metrics(y_true, y_pred, on_target_masks, CONDITION_NAMES)
    evolution_off_target = compute_condition_scoped_metrics(y_true, y_pred, off_target_masks, CONDITION_NAMES)

    perturbation_labels = sorted({meta["perturbation"] for meta in row_meta if meta["perturbation"]})
    perturbation_metrics: dict[str, Any] = {}
    for label in perturbation_labels:
        mask = np.array([meta["perturbation"] == label for meta in row_meta])
        perturbation_metrics[label] = compute_condition_scoped_metrics(
            y_true, y_pred, _uniform_masks(mask, CONDITION_NAMES), CONDITION_NAMES
        )
    combined_mask = np.array([meta["perturbation"] != "" for meta in row_meta])
    perturbation_metrics["combined"] = compute_condition_scoped_metrics(
        y_true, y_pred, _uniform_masks(combined_mask, CONDITION_NAMES), CONDITION_NAMES
    )
    # All 3 insertion counts pooled into one series -- plot_pearson_comparison.py's
    # perturbed_library chart compares this against "shuffling" (a mechanistically
    # different edit) rather than lumping everything into "combined".
    insertion_mask = np.array([meta["perturbation"].startswith("insertion_") for meta in row_meta])
    perturbation_metrics["insertion"] = compute_condition_scoped_metrics(
        y_true, y_pred, _uniform_masks(insertion_mask, CONDITION_NAMES), CONDITION_NAMES
    )

    metrics: dict[str, Any] = {
        "n_samples": int(y_true.shape[0]),
        "checkpoint_path": str(checkpoint_path),
        "input_tsv": str(input_tsv),
        "output_dir": str(output_dir),
        "save_mode": checkpoint.get("save_mode"),
        "checkpoint_stage": checkpoint.get("stage"),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "overall": overall_metrics,
        "evolution_on_target": evolution_on_target,
        "evolution_off_target": evolution_off_target,
        "perturbation": perturbation_metrics,
    }

    predictions_path = output_dir / "evaluation_predictions.csv"
    metrics_path = output_dir / "evaluation_metrics.json"

    save_category_predictions(predictions_path, ids, row_meta, y_true, y_pred, CONDITION_NAMES)
    with open(metrics_path, "w") as handle:
        json.dump(metrics, handle, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved metrics to {metrics_path}")
    return metrics


if __name__ == "__main__":
    main()
