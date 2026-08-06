#!/usr/bin/env python
"""Record every candidate mutation's loss tried during each round of
ism_greedy_evolution.py's greedy search, for the first --n_rounds rounds.

Each round of that search runs full saturation mutagenesis (every position x every
base) on the current sequence and greedily accepts whichever single candidate
mutation minimizes the loss (see ism_greedy_evolution.py's module docstring for the
loss definition). This script re-runs the exact same evolve_sequence() loop
(imported unchanged, not duplicated) but additionally records the *entire* per-round
candidate-loss grid -- not just the winning mutation's loss -- via evolve_sequence's
on_round_candidates hook, so the distribution of losses across a round's search space
(not only its minimum) can be inspected with plot_ism_round_loss_distribution.py.

Stops after --n_rounds rounds, or earlier if the search converges first (see
ism_greedy_evolution.py's "converged" stop reason) -- a converged round has no
mutation left that beats the current sequence, so later rounds would just repeat it.

Output: one <safe_id>_round_candidate_losses.tsv per sequence in --output_dir, long
format with columns id, round, position, base, loss, accepted (accepted=True marks
the one candidate each round that evolve_sequence actually applied).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from alphagenome_encoder_ft import AlphaGenomeEncoderModel, load_config_from_checkpoint
from alphagenome_encoder_ft.mydata import JORES_ADAPTER_DOWN, JORES_ADAPTER_UP, _read_custom_tsv

from ism_greedy_evolution import BASES, _sanitize_filename, evolve_sequence
from saturation_mutagenesis import TangermemeWrapper


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record per-round candidate-mutation losses from ism_greedy_evolution.py's "
                    "greedy search, for the first --n_rounds rounds."
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
    parser.add_argument("--n_rounds", type=int, default=10,
                         help="Number of rounds of saturation mutagenesis to record candidate "
                              "losses for (the search may converge and stop earlier).")
    parser.add_argument("--other_condition_weight", type=float, default=0.5,
                         help="Same meaning as in ism_greedy_evolution.py.")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use_adapters", action=argparse.BooleanOptionalAction, default=True,
                         help="Same meaning as in ism_greedy_evolution.py.")
    parser.add_argument("--ism_batch_size", type=int, default=32,
                         help="Forward-pass batch size within a single round's ISM call.")
    return parser


def record_round_losses(
    wrapped: TangermemeWrapper,
    seq_id: str,
    insert_seq: str,
    left_adapter: str,
    right_adapter: str,
    target_warm: float,
    n_rounds: int,
    other_condition_weight: float,
    ism_batch_size: int,
) -> list[dict]:
    """Runs evolve_sequence for up to n_rounds rounds and returns one row per
    (round, position, base) candidate tried, via its on_round_candidates hook."""
    rows: list[dict] = []
    accepted_by_round: dict[int, tuple[int, int]] = {}  # round -> (position, base_idx)

    def on_round_candidates(round_idx: int, losses: torch.Tensor) -> None:
        # losses: (4, W) -- every base x every position candidate this round.
        flat_idx = int(torch.argmin(losses))
        best_char, best_pos = divmod(flat_idx, losses.shape[1])
        accepted_by_round[round_idx] = (best_pos, best_char)
        for base_idx in range(losses.shape[0]):
            for pos in range(losses.shape[1]):
                rows.append({
                    "id": seq_id,
                    "round": round_idx,
                    "position": pos,
                    "base": BASES[base_idx],
                    "loss": losses[base_idx, pos].item(),
                    "accepted": (pos, base_idx) == (best_pos, best_char),
                })

    evolve_sequence(
        wrapped, seq_id, insert_seq, left_adapter, right_adapter,
        target_warm=target_warm,
        max_iterations=n_rounds,
        other_condition_weight=other_condition_weight,
        ism_batch_size=ism_batch_size,
        on_round_candidates=on_round_candidates,
    )
    return rows


def save_round_losses(output_dir: Path, seq_id: str, rows: list[dict]) -> Path:
    stem = _sanitize_filename(seq_id)
    output_path = output_dir / f"{stem}_round_candidate_losses.tsv"
    with open(output_path, "w") as handle:
        columns = ["id", "round", "position", "base", "loss", "accepted"]
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join(str(row[col]) for col in columns) + "\n")
    print(f"[{seq_id}] wrote {output_path.name} ({len(rows)} candidate rows)")
    return output_path


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
        else checkpoint_path.parent / f"{checkpoint_path.stem}_round_candidate_losses"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    t_run_start = time.time()
    for seq_id in args.sequence_ids:
        insert_seq = rows_by_id[seq_id]["sequence"].strip().upper()
        rows = record_round_losses(
            wrapped, seq_id, insert_seq, left_adapter, right_adapter,
            target_warm=target_warm_by_id[seq_id],
            n_rounds=args.n_rounds,
            other_condition_weight=args.other_condition_weight,
            ism_batch_size=args.ism_batch_size,
        )
        save_round_losses(output_dir, seq_id, rows)

    print(f"Recorded candidate losses for {len(args.sequence_ids)} sequence(s) in "
          f"{time.time() - t_run_start:.1f}s (total wall time {time.time() - t_load_start:.1f}s). "
          f"Results in {output_dir}")


if __name__ == "__main__":
    main()
