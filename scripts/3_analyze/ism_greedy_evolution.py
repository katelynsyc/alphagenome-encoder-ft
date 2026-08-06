#!/usr/bin/env python
"""Greedy, single-nucleotide-at-a-time in-silico evolution of a Jores MPRA insert,
driven by exhaustive ISM (in-silico saturation mutagenesis) at each round.

Each round runs full saturation mutagenesis (every position x every base) on the
CURRENT sequence via tangermeme.saturation_mutagenesis, scores every resulting
single-mutant candidate (including the "identity" candidate -- i.e. no change) with
the loss below, and greedily accepts whichever single mutation minimizes it:

    loss = max(target_warm - pred_warm, 0)^2                              # one-sided:
                                                                            # zero once
                                                                            # warm >= target
         + other_condition_weight * sum_{c in cold,dark,light,maize} (pred_c - baseline_c)^2

baseline_c is that condition's PREDICTED level on the original, unmutated sequence
(fixed once at round 0) -- "keeping the existing levels of expression in all other
conditions" means staying close to what the model already predicted for this specific
sequence, not to some other reference. The warm term is a one-sided hinge (via clamp)
rather than a plain squared distance so the search stops pulling on warm the moment the
target is reached, instead of continuing to overshoot at the expense of the other four
conditions.

Because every position's identity (no-op) substitution is included among the round's
candidates -- tangermeme fills those in with the reference prediction y0 -- the search
has a natural stopping rule: if the best-scoring candidate is no better than the current
sequence's own loss, no single mutation helps anymore, and the algorithm has converged
to a local optimum. Combined with max_iterations (a second, unconditional stop), this
gives three possible stop reasons: "converged", "max_iterations", or (implicitly, if
target_warm is easy to reach) still running when warm first crosses target -- see
target_reached_round in the saved summary, which is tracked independently of when the
loop actually stops.

By default no flanking adapters are added -- the insert is scored and mutated as a
bare ~170bp sequence. Pass --use_adapters to prepend/append the 15bp Jores adapters
(JORES_ADAPTER_UP/DOWN) before scoring, matching the padded construct some other
scripts in this repo use; even then, only the insert itself is ever mutated (the
adapters are fixed library sequence, not something a real bare-insert design varies),
via saturation_mutagenesis's start/end args.

max_iterations default of 100: Jores et al.'s own wet-lab directed-evolution rows
(metadata/evolution_only_eval.tsv) never carry a single-objective lineage past round=12
(see src/alphagenome_encoder_ft/evolution_objectives.py). 100 is beyond
that real-world ceiling for the extra constraint this script adds (preserving 4 other
conditions, not just chasing one), while still stopping short of exhaustively
mutating every position in the 170bp insert.

--random_topk_steps/--random_topk/--n_paths implement the alternative branching search
from Taskiran et al. 2024 (Nature 632:1035-1044), Methods: "To explore the alternative
in silico sequence evolution paths besides choosing the best mutation (greedy
algorithm), we chose the top 20 mutations on each sequence for every incremental step
... We followed this procedure for five incremental mutational steps." Concretely, for
each of the first --random_topk_steps rounds (m, paper value 5), instead of taking
argmin over every candidate substitution, we rank the real single-base substitutions
(excluding each position's own identity/no-op) by loss and sample uniformly among the
--random_topk (k, paper value 20) lowest-loss ones; that mutation is applied
unconditionally, i.e. the "converged" early-stop does not apply while step <= m, since
forcing a move every round is how the branching tree of paths gets built. After round
m, the search falls back to plain greedy argmin (with its usual convergence check) for
the remaining rounds up to --max_iterations. Running this whole procedure --n_paths
times per sequence (each restarting from the same input CRE, since unlike the paper we
are not starting from a random sequence) yields that many independent final sequences;
see the "<id>_paths_summary.tsv" file for which of them reached target_warm.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from collections.abc import Callable

import torch
from tangermeme.saturation_mutagenesis import saturation_mutagenesis

from alphagenome_encoder_ft import AlphaGenomeEncoderModel, load_config_from_checkpoint
from alphagenome_encoder_ft.mydata import JORES_ADAPTER_DOWN, JORES_ADAPTER_UP, _read_custom_tsv
from alphagenome_pytorch.utils.sequence import onehot_tensor_to_sequence, sequence_to_onehot

# saturation_mutagenesis.py lives alongside this script -- see its own docstring's
# notebook snippet for the same sibling-import pattern.
from saturation_mutagenesis import CONDITION, TangermemeWrapper

BASES = "ACGT"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Greedy ISM-driven evolution: push warm expression toward a target "
                    "while preserving cold/dark/light/maize at each sequence's own baseline."
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
                         help="Desired enrichment_warm level(s) (same scale as input_tsv). Pass "
                              "one value to use it for every --sequence_ids entry, or exactly as "
                              "many values as --sequence_ids to give each sequence its own target, "
                              "matched by position (e.g. --sequence_ids A B --target_warm -1.0 -0.5 "
                              "targets A at -1.0 and B at -0.5).")
    parser.add_argument("--max_iterations", type=int, default=100,
                         help="Stop after this many accepted mutations even if the target "
                              "hasn't been reached (see module docstring for the default).")
    parser.add_argument("--other_condition_weight", type=float, default=0.5,
                         help="Weight on the sum-of-squared deviation of cold/dark/light/maize "
                              "from their round-0 baseline, relative to the warm-target term.")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use_adapters", action=argparse.BooleanOptionalAction, default=True,
                         help="Prepend/append the 15bp Jores adapters before scoring. Off by "
                              "default -- sequences are designed as bare inserts, with no "
                              "assumed cloning context. The adapters are never mutated either way.")
    parser.add_argument("--ism_batch_size", type=int, default=32,
                         help="Forward-pass batch size within a single round's ISM call.")
    parser.add_argument("--random_topk_steps", type=int, default=0,
                         help="For each of the first this-many rounds (m in Taskiran et al. 2024's "
                              "top-20 branching procedure), sample uniformly from the --random_topk "
                              "lowest-loss real mutations instead of taking the single best one. "
                              "0 (default) disables this and runs plain greedy throughout, "
                              "matching the script's previous behavior.")
    parser.add_argument("--random_topk", type=int, default=20,
                         help="Size of the uniformly-sampled-from pool for each of the first "
                              "--random_topk_steps rounds (k in Taskiran et al. 2024; paper value 20).")
    parser.add_argument("--n_paths", type=int, default=1,
                         help="Number of independent evolution paths to run per sequence, each "
                              "restarting from the same input CRE. Use with --random_topk_steps > 0 "
                              "to sample many alternative branches (e.g. --n_paths 100) rather than "
                              "the single deterministic greedy path.")
    parser.add_argument("--seed", type=int, default=None,
                         help="Seed for the top-k uniform sampling, for reproducible paths. "
                              "Unset draws fresh randomness every run.")
    return parser


def _sanitize_filename(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)


def evolve_sequence(
    wrapped: TangermemeWrapper,
    seq_id: str,
    insert_seq: str,
    left_adapter: str,
    right_adapter: str,
    target_warm: float,
    max_iterations: int,
    other_condition_weight: float,
    ism_batch_size: int,
    on_round_candidates: Callable[[int, torch.Tensor], None] | None = None,
    random_topk_steps: int = 0,
    random_topk: int = 20,
    rng: torch.Generator | None = None,
) -> dict[str, Any]:
    """Run one evolution path for one sequence. Returns a JSON-able result dict
    with the full per-round history plus the final evolved sequence.

    on_round_candidates, if given, is called once per round as
    on_round_candidates(round_idx, losses) with the full (4, W) candidate-loss grid
    saturation_mutagenesis scores that round (every base x every position), right
    before the round's winning mutation is applied -- lets a caller (e.g.
    ism_round_candidate_losses.py) record the whole search space per round without
    duplicating this loop.

    random_topk_steps/random_topk/rng implement the Taskiran et al. 2024 branching
    variant (see module docstring): for round_idx < random_topk_steps, the applied
    mutation is sampled uniformly from the random_topk lowest-loss real substitutions
    (identity/no-op candidates excluded) instead of taking argmin, and is applied
    unconditionally -- no convergence check on those rounds. rng is a torch.Generator
    used for that sampling; pass one seeded generator across paths/rounds for
    reproducibility, or leave None for fresh randomness each call."""
    construct = f"{left_adapter}{insert_seq}{right_adapter}"
    onehot = sequence_to_onehot(construct).astype("float32")  # (L, 4)
    X = torch.from_numpy(onehot).transpose(0, 1).unsqueeze(0)  # (1, 4, L)
    organism_idx = torch.zeros(1, dtype=torch.long)  # unused by the encoder-only forward pass

    start, end = len(left_adapter), len(left_adapter) + len(insert_seq)
    other_names = [name for name in CONDITION if name != "warm"]
    other_idxs = [CONDITION[name] for name in other_names]
    warm_idx = CONDITION["warm"]

    baseline_other: torch.Tensor | None = None

    def loss_of(pred: torch.Tensor) -> torch.Tensor:
        """pred: (..., 5) -> (...,) loss. Requires baseline_other to already be set."""
        warm_term = torch.clamp(target_warm - pred[..., warm_idx], min=0.0) ** 2
        other_term = ((pred[..., other_idxs] - baseline_other) ** 2).sum(dim=-1)
        return warm_term + other_condition_weight * other_term

    history: list[dict[str, Any]] = []
    target_reached_round: int | None = None
    stop_reason = "max_iterations"
    round_idx = 0
    t_start = time.time()

    while True:
        y0, y_hat = saturation_mutagenesis(
            wrapped, X, args=(organism_idx,), start=start, end=end,
            raw_outputs=True, batch_size=ism_batch_size,
        )
        y0 = y0[0]        # (5,) -- current sequence's actual prediction
        y_hat = y_hat[0]  # (4, W, 5) -- every single-base substitution's prediction, W = end-start

        if baseline_other is None:
            baseline_other = y0[other_idxs].clone()  # fixed for every subsequent round

        current_loss = loss_of(y0).item()
        levels = {name: float(y0[idx].item()) for name, idx in CONDITION.items()}
        print(
            f"[{seq_id}] round={round_idx} loss={current_loss:.6f} "
            + " ".join(f"{name}={value:.4f}" for name, value in levels.items()),
            file=sys.stderr, flush=True,
        )
        history.append({"round": round_idx, **levels, "loss": current_loss, "mutation": None, "phase": None})

        if target_reached_round is None and levels["warm"] >= target_warm:
            target_reached_round = round_idx

        if round_idx >= max_iterations:
            stop_reason = "max_iterations"
            break

        losses = loss_of(y_hat)  # (4, W)
        if on_round_candidates is not None:
            on_round_candidates(round_idx, losses)

        if round_idx < random_topk_steps:
            # Taskiran et al. 2024 branching step: sample uniformly from the top-k
            # lowest-loss REAL mutations (identity/no-op candidates masked out, since
            # "top 20 mutations" means actual substitutions), applied unconditionally --
            # no convergence check here, since forcing a move every round is how the
            # tree of alternative paths gets built.
            current_bases = X[0, :, start:end].argmax(dim=0)  # (W,)
            is_identity = torch.arange(4).unsqueeze(1) == current_bases.unsqueeze(0)  # (4, W)
            masked_losses = losses.masked_fill(is_identity, float("inf")).reshape(-1)
            k = min(random_topk, masked_losses.numel() - int(is_identity.sum()))
            _, topk_flat_idx = torch.topk(masked_losses, k=k, largest=False)
            choice = int(torch.randint(len(topk_flat_idx), (1,), generator=rng))
            flat_idx = int(topk_flat_idx[choice])
            phase = "random_topk"
        else:
            # Plain greedy acceptance: only ever take a strictly loss-improving move.
            flat_idx = int(torch.argmin(losses))
            phase = "greedy"

        best_char, best_pos_rel = divmod(flat_idx, losses.shape[1])
        best_loss = losses.reshape(-1)[flat_idx].item()

        if phase == "greedy" and best_loss >= current_loss - 1e-9:
            stop_reason = "converged"
            break

        best_pos = best_pos_rel + start
        ref_char = int(X[0, :, best_pos].argmax())
        X[0, :, best_pos] = 0.0
        X[0, best_char, best_pos] = 1.0
        history[-1]["mutation"] = f"{BASES[ref_char]}{best_pos_rel}{BASES[best_char]}"
        history[-1]["phase"] = phase
        round_idx += 1

    elapsed = time.time() - t_start
    print(f"[{seq_id}] stopped after {round_idx} round(s): {stop_reason} "
          f"(target_reached_round={target_reached_round}) in {elapsed:.1f}s", file=sys.stderr, flush=True)

    final_construct = onehot_tensor_to_sequence(X[0].transpose(0, 1))
    final_insert = final_construct[start:end]

    return {
        "id": seq_id,
        "target_warm": target_warm,
        "other_condition_weight": other_condition_weight,
        "max_iterations": max_iterations,
        "n_rounds": round_idx,
        "stop_reason": stop_reason,
        "target_reached_round": target_reached_round,
        "elapsed_seconds": elapsed,
        "initial_sequence": insert_seq,
        "final_sequence": final_insert,
        "initial_levels": history[0],
        "final_levels": history[-1],
        "history": history,
    }


def save_result(output_dir: Path, result: dict[str, Any], filename_stem: str | None = None) -> None:
    """filename_stem overrides the default sanitized-id stem -- used by
    ism_greedy_evolution_weight_sweep.py to fan one id out into one file pair per
    swept other_condition_weight without them clobbering each other."""
    stem = filename_stem or _sanitize_filename(result["id"])

    history_path = output_dir / f"{stem}_history.tsv"
    with open(history_path, "w") as handle:
        columns = ["round", "cold", "dark", "light", "warm", "maize", "loss", "mutation", "phase"]
        handle.write("\t".join(columns) + "\n")
        for row in result["history"]:
            handle.write("\t".join(str(row[col]) for col in columns) + "\n")

    summary = {k: v for k, v in result.items() if k != "history"}
    summary_path = output_dir / f"{stem}_summary.json"
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"[{result['id']}] wrote {history_path.name} and {summary_path.name}")


def save_paths_summary(output_dir: Path, seq_id: str, path_rows: list[dict[str, Any]]) -> None:
    """One row per --n_paths run for seq_id, so the many independent final sequences
    from the random-topk branching mode (or repeated plain-greedy runs) can be
    filtered down to whichever ones reached target_warm without re-parsing every
    per-path summary.json."""
    stem = _sanitize_filename(seq_id)
    summary_path = output_dir / f"{stem}_paths_summary.tsv"
    columns = ["path", "final_sequence", "final_warm", "target_warm", "target_reached", "n_rounds", "stop_reason"]
    with open(summary_path, "w") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in path_rows:
            handle.write("\t".join(str(row[col]) for col in columns) + "\n")
    n_reached = sum(1 for row in path_rows if row["target_reached"])
    print(f"[{seq_id}] wrote {summary_path.name}: {n_reached}/{len(path_rows)} path(s) reached target_warm")


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
        else checkpoint_path.parent / f"{checkpoint_path.stem}_greedy_evolution"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = torch.Generator().manual_seed(args.seed) if args.seed is not None else torch.Generator()

    t_evolve_start = time.time()
    for seq_id in args.sequence_ids:
        insert_seq = rows_by_id[seq_id]["sequence"].strip().upper()
        path_rows: list[dict[str, Any]] = []
        for path_idx in range(args.n_paths):
            if args.n_paths > 1:
                print(f"--- [{seq_id}] path {path_idx + 1}/{args.n_paths} ---", flush=True)
            result = evolve_sequence(
                wrapped, seq_id, insert_seq, left_adapter, right_adapter,
                target_warm=target_warm_by_id[seq_id],
                max_iterations=args.max_iterations,
                other_condition_weight=args.other_condition_weight,
                ism_batch_size=args.ism_batch_size,
                random_topk_steps=args.random_topk_steps,
                random_topk=args.random_topk,
                rng=rng,
            )
            filename_stem = f"{_sanitize_filename(seq_id)}_path{path_idx:03d}" if args.n_paths > 1 else None
            save_result(output_dir, result, filename_stem=filename_stem)
            path_rows.append({
                "path": path_idx,
                "final_sequence": result["final_sequence"],
                "final_warm": result["final_levels"]["warm"],
                "target_warm": result["target_warm"],
                "target_reached": result["final_levels"]["warm"] >= result["target_warm"],
                "n_rounds": result["n_rounds"],
                "stop_reason": result["stop_reason"],
            })
        if args.n_paths > 1:
            save_paths_summary(output_dir, seq_id, path_rows)

    print(f"Evolved {len(args.sequence_ids)} sequence(s) x {args.n_paths} path(s) in "
          f"{time.time() - t_evolve_start:.1f}s (total wall time {time.time() - t_load_start:.1f}s). "
          f"Results in {output_dir}")


if __name__ == "__main__":
    main()
