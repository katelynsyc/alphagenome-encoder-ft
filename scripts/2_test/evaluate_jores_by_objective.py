#!/usr/bin/env python
"""Evaluate an encoder-only AlphaGenome Jores checkpoint per evolution objective.

Like evaluate_jores.py, but scores each condition only on the subset of test rows
whose evolution objective(s) actually targeted that condition (see
alphagenome_encoder_ft.evolution_objectives) instead of averaging in conditions
nobody selected the sequence on. Meant for an --input_tsv built from the "evolution"
category of media-5.xlsx that carries an `experiment` column (e.g.
metadata/evolution_only_eval.tsv) -- modelling_data_tamsACR.tsv has no such column.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from alphagenome_encoder_ft import (
    AlphaGenomeEncoderModel,
    create_dataloader,
    create_jores_splits,
    load_config_from_checkpoint,
)
from alphagenome_encoder_ft.evolution_objectives import (
    compute_targeted_metrics,
    load_target_conditions,
    save_targeted_predictions,
)

from evaluate_jores import CONDITION_NAMES, build_arg_parser, collect_predictions


def main() -> dict[str, Any]:
    parser = build_arg_parser()
    parser.description = "Evaluate an encoder-only AlphaGenome Jores checkpoint per evolution objective"
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path).resolve()
    if not checkpoint_path.exists():
        parser.error(f"Checkpoint not found: {checkpoint_path}")

    config, checkpoint = load_config_from_checkpoint(checkpoint_path)

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

    default_output_dir = checkpoint_path.parent / f"{checkpoint_path.stem}_test_eval_by_objective"
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

    target_conditions = load_target_conditions(config.data.input_tsv)
    metrics = compute_targeted_metrics(y_true, y_pred, target_conditions, condition_order=tuple(CONDITION_NAMES))
    metrics.update(
        {
            "checkpoint_path": str(checkpoint_path),
            "output_dir": str(output_dir),
            "input_tsv": str(config.data.input_tsv),
            "save_mode": checkpoint.get("save_mode"),
            "checkpoint_stage": checkpoint.get("stage"),
            "checkpoint_epoch": checkpoint.get("epoch"),
        }
    )

    predictions_path = output_dir / "test_predictions.csv"
    metrics_path = output_dir / "test_metrics_by_objective.json"

    save_targeted_predictions(predictions_path, y_true, y_pred, target_conditions, condition_order=tuple(CONDITION_NAMES))
    with open(metrics_path, "w") as handle:
        json.dump(metrics, handle, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved metrics to {metrics_path}")
    return metrics


if __name__ == "__main__":
    main()
