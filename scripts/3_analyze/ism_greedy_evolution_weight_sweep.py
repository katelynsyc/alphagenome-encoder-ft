#!/usr/bin/env python
"""Sweep --other_condition_weight for ism_greedy_evolution.py and save one result
file pair per (sequence, weight) combination, so plot_weight_sweep_tradeoff.py can
chart the tradeoff between hitting target_warm and preserving cold/dark/light/maize
at their round-0 baseline.

Why a separate script rather than shelling out to ism_greedy_evolution.py once per
weight: that would reload the ~450M-param checkpoint from disk every call. Here the
model loads once and evolve_sequence() (imported unchanged from
ism_greedy_evolution.py) is called once per (sequence, weight) pair against the same
loaded model -- same greedy search, same stop conditions ("converged" / "max_iterations"),
just repeated with different other_condition_weight values.

Output layout: <output_dir>/<safe_id>_w<weight:g>_history.tsv and
_summary.json, e.g. Zm-1631_rev_w0.5_history.tsv -- distinct from
ism_greedy_evolution.py's own <safe_id>_history.tsv (no weight suffix), so a sweep
never clobbers a single-weight run in the same directory.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from alphagenome_encoder_ft import AlphaGenomeEncoderModel, load_config_from_checkpoint
from alphagenome_encoder_ft.mydata import JORES_ADAPTER_DOWN, JORES_ADAPTER_UP, _read_custom_tsv

from ism_greedy_evolution import _sanitize_filename, evolve_sequence, save_result
from saturation_mutagenesis import TangermemeWrapper


def build_arg_parser() -> argparse.ArgumentParser:
    # Same flags as ism_greedy_evolution.py's build_arg_parser, except
    # --other_condition_weight becomes a list here (default sweeps 0 -> 3, spanning
    # and going past the "0 to 2" range asked for) since that's the axis being swept.
    parser = argparse.ArgumentParser(
        description="Sweep other_condition_weight for the greedy ISM evolution search, reusing one "
                    "loaded model across every (sequence, weight) run -- see plot_weight_sweep_tradeoff.py "
                    "to chart the resulting warm-vs-preservation tradeoff."
    )
    parser.add_argument(
        "--checkpoint_path", type=str,
        default="/grid/koo/home/kachu/projects/alphagenome-encoder-ft/results/e898939e/df4406c4716cd2cf/stage2/best.pt",
    )
    parser.add_argument("--input_tsv", type=str, default=None,
                         help="Defaults to the checkpoint's saved config.data.input_tsv.")
    parser.add_argument("--sequence_ids", type=str, nargs="+",
                         default=["Zm-1631_rev", "Zm-16206_fwd"],
                         help="ids from input_tsv's 'id' column to evolve independently.")
    parser.add_argument("--target_warm", type=float, nargs="+", required=True,
                         help="Desired enrichment_warm level(s), one value for every --sequence_ids "
                              "entry or exactly one per sequence (see ism_greedy_evolution.py's own "
                              "--target_warm help for the matching rule).")
    parser.add_argument("--other_condition_weights", type=float, nargs="+",
                         default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
                         help="other_condition_weight values to sweep, reusing the same loaded "
                              "model and sequence across all of them.")
    parser.add_argument("--max_iterations", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use_adapters", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ism_batch_size", type=int, default=32)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if len(args.target_warm) == 1:
        target_warm_by_id = {seq_id: args.target_warm[0] for seq_id in args.sequence_ids}
    elif len(args.target_warm) == len(args.sequence_ids):
        target_warm_by_id = dict(zip(args.sequence_ids, args.target_warm))
    else:
        raise ValueError(
            f"--target_warm got {len(args.target_warm)} value(s) but --sequence_ids has "
            f"{len(args.sequence_ids)}; pass either one target (used for every sequence) or "
            "exactly one target per sequence, in the same order."
        )

    checkpoint_path = Path(args.checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    t_load_start = time.time()
    config, _ = load_config_from_checkpoint(checkpoint_path)
    input_tsv = args.input_tsv or config.data.input_tsv
    if not input_tsv:
        raise ValueError("input_tsv must be provided or present in the checkpoint's config.data.input_tsv")

    device = torch.device(args.device or config.runtime.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    model = AlphaGenomeEncoderModel.from_checkpoint(checkpoint_path, device=device)
    wrapped = TangermemeWrapper(model)
    print(f"Loaded model + checkpoint in {time.time() - t_load_start:.1f}s")

    left_adapter = JORES_ADAPTER_UP if args.use_adapters else ""
    right_adapter = JORES_ADAPTER_DOWN if args.use_adapters else ""

    rows_by_id = {row["id"]: row for row in _read_custom_tsv(input_tsv)}
    missing = [seq_id for seq_id in args.sequence_ids if seq_id not in rows_by_id]
    if missing:
        raise KeyError(f"sequence id(s) not found in {input_tsv}: {missing}")

    output_dir = (
        Path(args.output_dir).resolve() if args.output_dir is not None
        else checkpoint_path.parent / f"{checkpoint_path.stem}_greedy_evolution_weight_sweep"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    n_runs = len(args.sequence_ids) * len(args.other_condition_weights)
    print(f"Sweeping {len(args.other_condition_weights)} weight(s) x {len(args.sequence_ids)} sequence(s) "
          f"= {n_runs} run(s): weights={args.other_condition_weights}")

    t_sweep_start = time.time()
    run_idx = 0
    for seq_id in args.sequence_ids:
        insert_seq = rows_by_id[seq_id]["sequence"].strip().upper()
        safe_id = _sanitize_filename(seq_id)
        for weight in args.other_condition_weights:
            run_idx += 1
            print(f"--- [{run_idx}/{n_runs}] {seq_id} other_condition_weight={weight:g} ---", flush=True)
            result = evolve_sequence(
                wrapped, seq_id, insert_seq, left_adapter, right_adapter,
                target_warm=target_warm_by_id[seq_id],
                max_iterations=args.max_iterations,
                other_condition_weight=weight,
                ism_batch_size=args.ism_batch_size,
            )
            save_result(output_dir, result, filename_stem=f"{safe_id}_w{weight:g}")

    print(f"Swept {n_runs} run(s) in {time.time() - t_sweep_start:.1f}s "
          f"(total wall time {time.time() - t_load_start:.1f}s). Results in {output_dir}")


if __name__ == "__main__":
    main()
