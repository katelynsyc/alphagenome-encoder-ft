"""Reusable encoder-only training primitives."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import Subset

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from .config import OptimConfig, TrainConfig
from .model import AlphaGenomeEncoderModel


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    # torch.load(..., map_location=...) remaps every tensor in the checkpoint -- including
    # these RNG-state bytes -- onto that device. torch.get_rng_state()/get_rng_state_all()
    # always represent RNG state as CPU ByteTensors regardless of which device captured it, and
    # torch.set_rng_state() requires exactly that, so force back to CPU before restoring.
    torch.set_rng_state(state["torch"].cpu())
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["torch_cuda"]])


def _set_dataset_epoch(dataset, epoch: int | float) -> None:
    """Call dataset.set_epoch(epoch) if the dataset supports it, unwrapping one level of
    Subset first (create_random_splits/create_deng_splits wrap datasets in Subset). No-op for
    dataset types that don't implement set_epoch (e.g. LentiMPRADataset/DeepSTARRDataset).

    epoch is rounded to an int here: run_training_stage's epoch_number is start_epoch +
    epoch_idx + 1, and stage 2's start_epoch (stage 1's best_epoch) can be a non-integer float
    when the best validation event landed mid-epoch -- but np.random.default_rng requires a
    plain int seed, and epoch_number always represents one whole epoch regardless of start_epoch's
    fractional offset, so rounding loses nothing semantically.
    """
    target = dataset.dataset if isinstance(dataset, Subset) else dataset
    set_epoch = getattr(target, "set_epoch", None)
    if set_epoch is not None:
        set_epoch(int(round(epoch)))


def _autocast_context(device: torch.device, use_amp: bool):
    if use_amp and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _default_loss_fn(preds: Tensor, targets: Tensor) -> Tensor:
    return F.mse_loss(preds.float(), targets.float())


def _unpack_batch(batch: tuple[Tensor, ...]) -> tuple[Tensor, Tensor, Tensor | None]:
    """Datasets may yield (sequences, targets) or (sequences, targets, sample_weight)."""
    if len(batch) == 3:
        return batch
    sequences, targets = batch
    return sequences, targets, None


def _compute_loss( #per sample loss has a barcode-count weight during training only
    preds: Tensor,
    targets: Tensor,
    weights: Tensor | None,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
) -> Tensor:
    if weights is None:
        return loss_fn(preds, targets) #already has reduction as mean, so will give 1 scalar
    # per-sample MSE, weighted by e.g. barcode-count confidence, then averaged.
    per_sample = F.mse_loss(preds.float(), targets.float(), reduction="none") #(B, num_outputs)
    if per_sample.ndim > 1:
        per_sample = per_sample.mean(dim=tuple(range(1, per_sample.ndim))) #collapse over all dims other than batch
    return (per_sample * weights).mean() #multiply per sample loss by barcode weight, average over batch


def _pearson_r(preds: Tensor, targets: Tensor, eps: float = 1e-8) -> Tensor:
    preds = preds.float()
    targets = targets.float()
    if preds.numel() < 2:
        return torch.tensor(float("nan"), device=preds.device)
    preds_centered = preds - preds.mean()
    targets_centered = targets - targets.mean()
    denom = preds_centered.pow(2).sum().sqrt() * targets_centered.pow(2).sum().sqrt()
    return (preds_centered * targets_centered).sum() / (denom + eps)


# per-track pearson when preds/targets are (N, K); returns one scalar per track.
def _pearson_r_per_track(preds: Tensor, targets: Tensor, eps: float = 1e-8) -> list[float]:
    if preds.ndim != 2 or targets.ndim != 2 or preds.shape[1] < 2:
        return []
    preds = preds.float()
    targets = targets.float()
    scores: list[float] = []
    for track in range(preds.shape[1]): #for each tissue type or condition
        r = _pearson_r(preds[:, track], targets[:, track], eps=eps) #take all the preds & targets for N samples for 1 track
        scores.append(float(r.detach().cpu().item())) #add this pearson to the list
    return scores


def _compute_metrics(
    preds: Tensor,
    targets: Tensor,
    metric_fns: dict[str, Callable[[Tensor, Tensor], Tensor | float]] | None,
    track_names: list[str] | None = None,
) -> dict[str, float]:
    functions = metric_fns or {"pearson": _pearson_r}
    metrics: dict[str, float] = {}
    for name, fn in functions.items():
        value = fn(preds, targets)
        if isinstance(value, Tensor):
            value = value.detach().float().cpu().item()
        metrics[name] = float(value)

    # multi-output heads (e.g. DeepSTARR dev+hk, Jores light/dark/warm/cold/maize):
    # also report per-track pearson, keyed by the caller's track_names when given
    # (the caller knows which dataset/condition set is in play; this stays dataset-
    # agnostic) or a generic "pearson_trackN" fallback otherwise.
    per_track = _pearson_r_per_track(preds, targets)
    if per_track:
        if track_names is not None and len(track_names) == len(per_track):
            for name, score in zip(track_names, per_track):
                metrics[f"{name}_pearson"] = score
        else:
            for idx, score in enumerate(per_track):
                metrics[f"pearson_track{idx}"] = score

        # mean of the per-condition pearsons -- matches train_deeptomato.py's mean_pearson
        # (mean of independently-computed per-tissue Pearson correlations), used in place
        # of the pooled/flattened correlation since it's inflated by between-track separation.
        metrics["pearson"] = sum(per_track) / len(per_track)
    return metrics


def _gather_predictions(preds: list[Tensor], targets: list[Tensor]) -> tuple[Tensor, Tensor]:
    pred_tensor = torch.cat(preds, dim=0) if preds else torch.empty(0)
    target_tensor = torch.cat(targets, dim=0) if targets else torch.empty(0)
    return pred_tensor, target_tensor


def set_encoder_trainable(model: AlphaGenomeEncoderModel, trainable: bool) -> None:
    model.set_encoder_trainable(trainable)


def create_optimizer(
    optim_config: OptimConfig,
    params,
    *,
    learning_rate: float | None = None,
) -> torch.optim.Optimizer:
    lr = optim_config.learning_rate if learning_rate is None else learning_rate
    if optim_config.optimizer == "adam":
        return Adam(params, lr=lr, weight_decay=optim_config.weight_decay)
    return AdamW(params, lr=lr, weight_decay=optim_config.weight_decay)


def create_scheduler( #optimizes the learning rate as the model 
    optim_config: OptimConfig,
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
):
    lr_scheduler = optim_config.lr_scheduler
    if lr_scheduler == "constant":
        return None
    if lr_scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max(1, total_epochs))
    if lr_scheduler == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode=optim_config.plateau_mode,
            factor=optim_config.plateau_factor,
            patience=optim_config.plateau_patience,
            min_lr=optim_config.plateau_min_lr,
        )
    raise ValueError(f"Unknown lr_scheduler: {lr_scheduler}")


def scheduler_stepper(name: str):
    if name == "plateau":
        return lambda scheduler, metrics: scheduler.step(metrics["loss"]) if scheduler is not None else None
    return lambda scheduler, metrics: scheduler.step() if scheduler is not None else None


def _num_batches(data_loader) -> int | None:
    try:
        return len(data_loader)
    except TypeError:
        return None


def train_epoch(
    model: AlphaGenomeEncoderModel,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    *,
    epoch: int | None = None,
    loss_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
    metric_fns: dict[str, Callable[[Tensor, Tensor], Tensor | float]] | None = None,
    track_names: list[str] | None = None,
    gradient_accumulation_steps: int = 1,
    use_amp: bool = True,
    train_encoder: bool = False,
    grad_clip: float | None = None,
    show_progress: bool = False,
    batch_end_callback: Callable[[int, int], bool] | None = None,
) -> dict[str, float]:
    """Train for one epoch."""

    device = torch.device(device)
    loss_fn = loss_fn or _default_loss_fn
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be > 0")

    if epoch is not None:
        # Reseed dataset augmentation RNG from (base_seed, epoch) before the DataLoader
        # iterator is created below, so resumed runs reproduce an uninterrupted run's
        # augmentations for this epoch. See PlantMPRADataset.set_epoch().
        _set_dataset_epoch(train_loader.dataset, epoch)

    if train_encoder:
        model.train()
    else:
        model.eval()
        model.head.train() #just train the head

    total_loss = 0.0
    total_samples = 0
    all_preds: list[Tensor] = []
    all_targets: list[Tensor] = []

    optimizer.zero_grad(set_to_none=True)

    num_batches = _num_batches(train_loader)
    batch_iterator = train_loader
    if tqdm is not None and show_progress:
        batch_iterator = tqdm(
            train_loader,
            total=num_batches,
            desc="train",
            leave=False,
        )

    for batch_idx, batch in enumerate(batch_iterator, start=1):
        sequences, targets, weights = _unpack_batch(batch)
        sequences = sequences.to(device)
        targets = targets.to(device).float()
        weights = weights.to(device).float() if weights is not None else None
        organism_idx = torch.zeros(sequences.shape[0], dtype=torch.long, device=device)
        autocast_ctx = _autocast_context(device, use_amp)

        if train_encoder:
            with autocast_ctx:
                preds = model(sequences, organism_idx)
                loss = _compute_loss(preds, targets, weights, loss_fn)
        else:
            with torch.no_grad():
                with autocast_ctx:
                    encoder_output = model.encode(sequences, organism_idx)
            with autocast_ctx:
                preds = model.predict_from_encoder(encoder_output)
                loss = _compute_loss(preds, targets, weights, loss_fn)

        (loss / gradient_accumulation_steps).backward()

        if batch_idx % gradient_accumulation_steps == 0 or batch_idx == len(train_loader):
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.head.parameters(), grad_clip)
                if train_encoder:
                    torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        batch_size = targets.shape[0]
        total_samples += batch_size
        total_loss += loss.detach().float().cpu().item() * batch_size
        all_preds.append(preds.detach().float().cpu())
        all_targets.append(targets.detach().float().cpu())

        if tqdm is not None and show_progress:
            batch_iterator.set_postfix(loss=total_loss / max(1, total_samples))

        if batch_end_callback is not None and not batch_end_callback(batch_idx, len(train_loader)):
            break

    preds_cat, targets_cat = _gather_predictions(all_preds, all_targets)
    metrics = _compute_metrics(preds_cat, targets_cat, metric_fns, track_names=track_names)
    metrics["loss"] = total_loss / max(1, total_samples)
    return metrics


@torch.no_grad()
def evaluate(
    model: AlphaGenomeEncoderModel,
    data_loader,
    device: torch.device | str,
    *,
    loss_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
    metric_fns: dict[str, Callable[[Tensor, Tensor], Tensor | float]] | None = None,
    track_names: list[str] | None = None,
    use_amp: bool = True,
) -> dict[str, float]:
    """Evaluate on a data loader."""

    device = torch.device(device)
    loss_fn = loss_fn or _default_loss_fn
    model.eval()

    total_loss = 0.0
    total_samples = 0
    all_preds: list[Tensor] = []
    all_targets: list[Tensor] = []

    for batch in data_loader:
        sequences, targets, _weights = _unpack_batch(batch)
        sequences = sequences.to(device)
        targets = targets.to(device).float()
        organism_idx = torch.zeros(sequences.shape[0], dtype=torch.long, device=device)
        with _autocast_context(device, use_amp):
            preds = model(sequences, organism_idx)
            loss = loss_fn(preds, targets)

        batch_size = targets.shape[0]
        total_samples += batch_size
        total_loss += loss.detach().float().cpu().item() * batch_size
        all_preds.append(preds.detach().float().cpu())
        all_targets.append(targets.detach().float().cpu())

    preds_cat, targets_cat = _gather_predictions(all_preds, all_targets)
    metrics = _compute_metrics(preds_cat, targets_cat, metric_fns, track_names=track_names)
    metrics["loss"] = total_loss / max(1, total_samples)
    return metrics


def save_checkpoint(
    path: str | Path,
    model: AlphaGenomeEncoderModel,
    *,
    config: TrainConfig,
    save_mode: str,
    stage: str,
    epoch: int,
    metrics: dict[str, Any] | None = None,
    training_state: dict[str, Any] | None = None,
) -> Path:
    """Save a checkpoint following the repo checkpoint contract.

    ``training_state`` is optional and additive: when provided (only ``last.pt`` writes do
    this -- see ``run_training_stage``), the checkpoint also carries everything needed to
    resume training (optimizer/scheduler state, RNG state, epoch/early-stopping bookkeeping,
    history) on top of the usual model-weights-only payload. ``best.pt`` writes omit it and
    stay exactly as small as before.
    """

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_name(f"{checkpoint_path.name}.{os.getpid()}.tmp")

    payload: dict[str, Any] = {
        "save_mode": save_mode,
        "stage": stage,
        "epoch": epoch,
        "config": config.to_dict(),
        "head_type": config.head.head_type,
        "head_state_dict": model.head.state_dict(),
        "head_config": config.head_kwargs(),
        "construct_config": config.construct_config(),
        "metrics": metrics or {},
    }

    if save_mode == "minimal":
        payload["encoder_state_dict"] = model.encoder.state_dict()
    elif save_mode == "full":
        payload["model_state_dict"] = model.state_dict()
    elif save_mode != "head":
        raise ValueError(f"Unknown save_mode: {save_mode}")

    if training_state is not None:
        payload["training_state"] = training_state

    # Write to a process-unique temp file first, then atomically rename into place. torch.save
    # streams directly into whatever file it's given with no atomicity of its own -- writing
    # straight to checkpoint_path means a SIGKILL mid-write (e.g. a slow_nice preemption landing
    # during a slow GPFS write) leaves a truncated file under the "last.pt" name permanently,
    # since it's overwritten in place every epoch with nothing older to fall back to. Renaming a
    # fully-written tmp file over it instead means checkpoint_path is always either the old,
    # complete checkpoint or the new, complete one -- os.replace is a single atomic filesystem
    # rename, never a partial state, even if killed mid-rename. The pid suffix keeps two
    # processes that briefly both hold this checkpoint_dir (e.g. an old and new Ray actor
    # overlapping after a restart) from interleaving writes into the same tmp file.
    torch.save(payload, tmp_path)
    os.replace(tmp_path, checkpoint_path)
    return checkpoint_path


# Config knobs that don't affect training correctness and so shouldn't gate a
# fresh restart or split off a new wandb run on their own: checkpoint_dir is
# structurally implied by which last.pt we're even looking at (and can
# legitimately differ from a checkpoint's saved value if the directory was
# later moved/reused), pretrained_weights/save_mode only matter for the very
# first initialization, num_epochs/second_stage_epochs/early_stopping_patience
# are stopping criteria rather than model/data/optimizer parameters -- raising
# them to let an already-resumed, still-improving stage keep training longer
# must not be treated as invalidating its progress -- and the rest are
# logging/dataloader-perf knobs.
_RESUME_CONFIG_IGNORE = {
    ("checkpoint", "checkpoint_dir"),
    ("checkpoint", "pretrained_weights"),
    ("checkpoint", "save_mode"),
    ("logging", "use_wandb"),
    ("logging", "wandb_project"),
    ("logging", "wandb_name"),
    ("runtime", "device"),
    ("data", "num_workers"),
    ("data", "pin_memory"),
    ("stage", "auto_resume"),
    ("stage", "num_epochs"),
    ("stage", "second_stage_epochs"),
    ("stage", "early_stopping_patience"),
}


# stage1 training doesn't depend on either of these -- they only take effect
# once stage 2 starts -- so a stage1 resume shouldn't be invalidated by them.
# stage2's check deliberately does NOT get this carve-out: it stays a superset
# of stage1's must-match fields, so any stage1-relevant change still cascades
# into invalidating stage2 too, on top of catching stage2-only changes.
#
# Dormant since train_ag.py started routing checkpoint_dir through training_run_id():
# that hash already includes both of these fields, so changing either one now sends a
# run to a brand-new (empty) checkpoint folder before this carve-out could ever matter --
# there's no longer a shared last.pt for stage1 to need protecting from a stage2-only
# change. Only re-enable this (and the extra_ignore= line below) if something calls
# run_training_stage/run_two_stage_training directly against a manually-fixed,
# not-per-config checkpoint_dir (e.g. train_mpra.py) and wants stage1 to keep resuming
# across second_stage_lr/scheduler sweeps there.
# _STAGE1_IRRELEVANT = {
#     ("stage", "second_stage_lr"),
#     ("stage", "second_stage_lr_scheduler"),
# }


def resume_config_mismatches(
    current: dict[str, Any],
    checkpointed: dict[str, Any],
    *,
    extra_ignore: set[tuple[str, str]] = frozenset(),
) -> list[str]:
    """Compare a live (fully merged: config file + CLI overrides) config against
    the config a checkpoint was saved with. Returns a human-readable diff for
    every field that changed and isn't in ``_RESUME_CONFIG_IGNORE`` or
    ``extra_ignore``; an empty list means the checkpoint is safe to resume from."""

    ignore = _RESUME_CONFIG_IGNORE | extra_ignore
    mismatches: list[str] = []
    for section, current_values in current.items():
        if not isinstance(current_values, dict):
            continue
        checkpointed_values = checkpointed.get(section, {})
        for key, value in current_values.items():
            if (section, key) in ignore:
                continue
            checkpointed_value = checkpointed_values.get(key)
            if value != checkpointed_value:
                mismatches.append(f"{section}.{key}: checkpoint={checkpointed_value!r} -> current={value!r}")
    return mismatches


def _training_relevant_config(config: TrainConfig) -> dict[str, Any]:
    """Every config field that isn't in ``_RESUME_CONFIG_IGNORE`` -- i.e. everything that
    actually changes what gets trained (data/head/optim/stopping-criteria-adjacent knobs
    excluded), as opposed to logging/dataloader-perf/stopping-criteria knobs. Shared basis
    for both ``stable_run_id`` (wandb display identity) and ``training_run_id`` (on-disk
    checkpoint folder identity) so the two can't silently drift apart."""

    return {
        section: {key: value for key, value in values.items() if (section, key) not in _RESUME_CONFIG_IGNORE}
        for section, values in config.to_dict().items()
        if isinstance(values, dict)
    }


def stable_run_id(checkpoint_dir: str | Path, config: TrainConfig) -> str:
    """A wandb run id that stays the same across a genuine preemption-resume
    (same checkpoint_dir, same training-relevant config) so the wandb curve
    continues, but changes whenever ``resume_config_mismatches`` would trigger
    a fresh restart -- so an intentional parameter change gets its own wandb
    run instead of appending onto an unrelated earlier curve."""

    payload = json.dumps(
        {"checkpoint_dir": str(Path(checkpoint_dir).resolve()), "config": _training_relevant_config(config)},
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha1(payload.encode()).hexdigest()[:12]
    return f"{config.logging.wandb_name}-{digest}"


def training_run_id(config: TrainConfig) -> str:
    """A short id derived from every training-relevant config field (see
    ``_training_relevant_config``): the same value across a genuine preemption-resume of
    the same job (config unchanged), a different one the moment any real hyperparameter
    changes. Meant to be appended as a ``checkpoint_dir`` subfolder -- see callers in
    train_ag.py -- so each distinct parameterization gets its own stage1/stage2
    checkpoints on disk instead of overwriting or resuming onto a previous run's.

    Deliberately excludes ``logging.wandb_name`` (unlike ``stable_run_id``, which is only
    a wandb display id): relabeling a run for wandb shouldn't fragment its on-disk
    checkpoint lineage.
    """

    payload = json.dumps(_training_relevant_config(config), sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def load_checkpoint(
    path: str | Path | dict[str, Any],
    model: AlphaGenomeEncoderModel,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint into ``model``. ``path`` may also be an already-loaded
    checkpoint dict (e.g. one already read once to check resume compatibility),
    to avoid reading a multi-GB file twice."""

    checkpoint = path if isinstance(path, dict) else torch.load(path, map_location=map_location, weights_only=False)
    save_mode = checkpoint["save_mode"]

    if save_mode == "minimal":
        model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
    elif save_mode == "full":
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    elif save_mode != "head":
        raise ValueError(f"Unknown save_mode: {save_mode}")

    model.head.load_state_dict(checkpoint["head_state_dict"])
    return checkpoint


def load_training_state(
    path: str | Path | dict[str, Any],
    model: AlphaGenomeEncoderModel,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a ``last.pt`` checkpoint's full training state: model weights (via
    ``load_checkpoint``), optimizer state (including per-parameter Adam ``exp_avg``/
    ``exp_avg_sq``/``step``), scheduler state, and RNG state, into the given already-constructed
    model/optimizer/scheduler. Returns the bookkeeping the caller needs to resume the epoch
    loop: ``epochs_done``, ``early_stopped``, ``best_monitor``, ``best_epoch``,
    ``evals_without_improvement``, ``history``.
    """

    checkpoint = load_checkpoint(path, model, map_location=map_location)
    training_state = checkpoint.get("training_state")
    if training_state is None:
        raise ValueError(f"Checkpoint {path} has no training_state to resume from")

    optimizer.load_state_dict(training_state["optimizer_state_dict"])
    if scheduler is not None and training_state.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(training_state["scheduler_state_dict"])
    _restore_rng_state(training_state["rng_state"])

    return {
        "epochs_done": training_state["epochs_done"],
        "early_stopped": training_state["early_stopped"],
        "best_monitor": training_state["best_monitor"],
        "best_epoch": training_state["best_epoch"],
        "evals_without_improvement": training_state["evals_without_improvement"],
        "history": training_state["history"],
    }


def _default_scheduler_step(scheduler, metrics: dict[str, float]) -> None:
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(metrics["loss"])
    else:
        scheduler.step()


def _history_template() -> dict[str, list[float]]:
    return {
        "train_loss": [],
        "train_pearson": [],
        "train_epoch": [],
        "val_loss": [],
        "val_pearson": [],
        "val_epoch": [],
        "test_loss": [],
        "test_pearson": [],
        "test_epoch": [],
    }


def _append_stage_history(history: dict[str, list[float]], stage_history: dict[str, list[float]]) -> None:
    for key, values in stage_history.items():
        history.setdefault(key, []).extend(values)


def _format_metric_parts(prefix: str, metrics: dict[str, float]) -> list[str]:
    """Format every metric (loss, pearson, and any per-track pearson_trackN) as
    "{prefix}_{key}=value", so multi-output heads aren't silently collapsed to
    just the pooled "pearson" key."""
    parts = [f"{prefix}_loss={metrics['loss']:.4f}"]
    for key in sorted(metrics):
        if key == "loss":
            continue
        parts.append(f"{prefix}_{key}={metrics[key]:.4f}")
    return parts


def _add_metrics_to_payload(payload: dict[str, Any], prefix: str, metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        payload[f"{prefix}_{key}"] = value


def add_metrics_to_history(history: dict[str, list[float]], prefix: str, metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        if key in {"loss", "pearson"}:
            continue  # already tracked under the fixed f"{prefix}_loss"/f"{prefix}_pearson" keys
        history.setdefault(f"{prefix}_{key}", []).append(value)


def run_training_stage(
    model: AlphaGenomeEncoderModel,
    train_loader,
    *,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    device: torch.device | str,
    num_epochs: int,
    stage: str,
    train_encoder: bool,
    val_loader=None,
    scheduler=None,
    scheduler_step: Callable[[Any, dict[str, float]], None] | None = None,
    loss_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
    metric_fns: dict[str, Callable[[Tensor, Tensor], Tensor | float]] | None = None,
    track_names: list[str] | None = None,
    checkpoint_dir: str | Path | None = None,
    start_epoch: int = 0,
    epoch_callback: Callable[[dict[str, Any]], None] | None = None,
    show_progress: bool = False,
    resume: bool = True,
) -> dict[str, Any]:
    """Run a single training stage.

    Early stopping and best-checkpoint selection are driven by (mean, across-condition)
    pearson -- higher is better -- mirroring train_deeptomato.py's mean_pearson
    selection, rather than validation loss.

    Idempotent/resumable: when ``resume`` is True (the default) and ``checkpoint_dir/last.pt``
    already exists, model/optimizer/scheduler/RNG/history/bookkeeping are restored from it and
    the epoch loop continues from where it left off. If that checkpoint shows the stage already
    finished (early stopped, or ``num_epochs`` already reached), this returns immediately without
    running any epochs. Safe to call repeatedly on the same ``checkpoint_dir``: fresh, mid-way,
    and already-finished are all handled by the same call.
    """

    device = torch.device(device)
    scheduler_step = scheduler_step or _default_scheduler_step
    stage_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if stage_dir is not None:
        stage_dir.mkdir(parents=True, exist_ok=True)

    last_checkpoint_path = stage_dir / "last.pt" if stage_dir is not None else None
    checkpoint_for_resume: dict[str, Any] | None = None
    if resume and last_checkpoint_path is not None and last_checkpoint_path.exists():
        checkpoint_for_resume = torch.load(last_checkpoint_path, map_location=device, weights_only=False)
        # extra_ignore = _STAGE1_IRRELEVANT if stage == "stage1" else frozenset()  # see _STAGE1_IRRELEVANT above
        extra_ignore = frozenset()
        mismatches = resume_config_mismatches(
            config.to_dict(), checkpoint_for_resume.get("config", {}), extra_ignore=extra_ignore
        )
        if mismatches:
            print(f"[{stage}] config changed since {last_checkpoint_path} was written -- starting fresh instead of resuming:")
            for mismatch in mismatches:
                print(f"    {mismatch}")
            checkpoint_for_resume = None

    # Whether this call is continuing the *same* weights lineage that was already on disk
    # (still training, or already finished under an unchanged config) as opposed to starting
    # a fresh lineage (no checkpoint yet, or one invalidated by a config mismatch above).
    # Callers that chain stages together (see run_two_stage_training) need this to know
    # whether a later stage's own checkpoint is still built on top of this stage's output.
    lineage_resumed = checkpoint_for_resume is not None

    if checkpoint_for_resume is not None:
        resumed = load_training_state(checkpoint_for_resume, model, optimizer, scheduler, map_location=device)
        history = resumed["history"]
        best_monitor = resumed["best_monitor"]
        best_epoch = resumed["best_epoch"]
        evals_without_improvement = resumed["evals_without_improvement"]
        epochs_done = resumed["epochs_done"]
        early_stopped = resumed["early_stopped"]
        best_checkpoint_path = stage_dir / "best.pt" if (stage_dir / "best.pt").exists() else None
        if early_stopped or epochs_done >= num_epochs:
            print(f"[{stage}] already complete ({epochs_done} epochs, early_stopped={early_stopped}) -- skipping")
            return {
                "history": history,
                "best_epoch": best_epoch,
                "best_monitor": best_monitor,
                "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path is not None else None,
                "resumed": lineage_resumed,
            }
        print(f"[{stage}] resuming from epoch {epochs_done} ({last_checkpoint_path})")
    else:
        history = _history_template()
        best_monitor = -math.inf
        best_epoch = float(start_epoch)
        best_checkpoint_path = None
        evals_without_improvement = 0
        epochs_done = 0

    for epoch_idx in range(epochs_done, num_epochs):
        epoch_number = start_epoch + epoch_idx + 1
        num_train_batches = len(train_loader)
        if val_loader is not None and config.stage.val_evals_per_epoch > 1:
            val_eval_interval = max(1, num_train_batches // config.stage.val_evals_per_epoch)
            val_eval_points = [i * val_eval_interval for i in range(1, config.stage.val_evals_per_epoch + 1)]
            val_eval_points = [min(point, num_train_batches) for point in val_eval_points]
            val_eval_points = sorted(set(val_eval_points))
        else:
            val_eval_points = [num_train_batches]

        latest_eval_metrics: dict[str, float] = {"loss": math.inf}
        val_metrics: dict[str, float] | None = None
        should_stop = False

        def _validate_if_needed(batch_idx: int, total_batches: int) -> bool:
            nonlocal best_checkpoint_path, best_epoch, best_monitor
            nonlocal evals_without_improvement, latest_eval_metrics, should_stop, val_metrics

            if val_loader is None or batch_idx not in val_eval_points:
                return True

            val_metrics = evaluate(
                model,
                val_loader,
                device,
                loss_fn=loss_fn,
                metric_fns=metric_fns,
                track_names=track_names,
                use_amp=config.runtime.use_amp,
            )
            # ``evaluate`` switches the module to eval mode; restore the active
            # training mode before continuing with the rest of the epoch.
            if train_encoder:
                model.train()
            else:
                model.eval()
                model.head.train()
            current_epoch = start_epoch + epoch_idx + (batch_idx / total_batches)
            history["val_loss"].append(val_metrics["loss"])
            history["val_pearson"].append(val_metrics.get("pearson", float("nan")))
            history["val_epoch"].append(float(current_epoch))
            add_metrics_to_history(history, "val", val_metrics)
            latest_eval_metrics = val_metrics

            if val_metrics["pearson"] > best_monitor:
                best_monitor = val_metrics["pearson"]
                best_epoch = float(current_epoch)
                evals_without_improvement = 0
                if stage_dir is not None:
                    best_checkpoint_path = save_checkpoint(
                        stage_dir / "best.pt",
                        model,
                        config=config,
                        save_mode=config.checkpoint.save_mode,
                        stage=stage,
                        epoch=current_epoch,
                        metrics=val_metrics,
                    )
            else:
                evals_without_improvement += 1

            patience_in_evals = config.stage.early_stopping_patience * config.stage.val_evals_per_epoch
            if evals_without_improvement >= patience_in_evals:
                should_stop = True
                return False
            return True

        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch=epoch_number,
            loss_fn=loss_fn,
            metric_fns=metric_fns,
            track_names=track_names,
            gradient_accumulation_steps=config.optim.gradient_accumulation_steps,
            use_amp=config.runtime.use_amp,
            train_encoder=train_encoder,
            grad_clip=config.optim.gradient_clip,
            show_progress=show_progress,
            batch_end_callback=_validate_if_needed if val_loader is not None else None,
        )
        history["train_loss"].append(train_metrics["loss"])
        history["train_pearson"].append(train_metrics.get("pearson", float("nan")))
        history["train_epoch"].append(float(epoch_number))
        add_metrics_to_history(history, "train", train_metrics)

        if val_loader is None:
            latest_eval_metrics = {"loss": train_metrics["loss"]}
            if train_metrics["pearson"] > best_monitor:
                best_monitor = train_metrics["pearson"]
                best_epoch = float(epoch_number)
                evals_without_improvement = 0
                if stage_dir is not None:
                    best_checkpoint_path = save_checkpoint(
                        stage_dir / "best.pt",
                        model,
                        config=config,
                        save_mode=config.checkpoint.save_mode,
                        stage=stage,
                        epoch=epoch_number,
                        metrics=latest_eval_metrics,
                    )
            else:
                evals_without_improvement += 1

        scheduler_step(scheduler, latest_eval_metrics)

        metrics_parts = [f"[{stage}] epoch {epoch_number}"]
        metrics_parts += _format_metric_parts("train", train_metrics)
        if val_metrics is not None:
            metrics_parts += _format_metric_parts("val", val_metrics)
        print(" | ".join(metrics_parts))

        if epoch_callback is not None:
            payload: dict[str, Any] = {"stage": stage, "epoch": float(epoch_number)}
            _add_metrics_to_payload(payload, "train", train_metrics)
            if val_metrics is not None:
                _add_metrics_to_payload(payload, "val", val_metrics)
            epoch_callback(payload)

        if stage_dir is not None:
            save_checkpoint(
                stage_dir / "last.pt",
                model,
                config=config,
                save_mode=config.checkpoint.save_mode,
                stage=stage,
                epoch=epoch_number,
                metrics=latest_eval_metrics,
                training_state={
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                    "rng_state": _capture_rng_state(),
                    "epochs_done": epoch_idx + 1,
                    "early_stopped": should_stop,
                    "best_monitor": best_monitor,
                    "best_epoch": best_epoch,
                    "evals_without_improvement": evals_without_improvement,
                    "history": history,
                },
            )

        if should_stop:
            break

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_monitor": best_monitor,
        "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path is not None else None,
        "resumed": lineage_resumed,
    }


def run_two_stage_training(
    model: AlphaGenomeEncoderModel,
    train_loader,
    *,
    stage1_optimizer: torch.optim.Optimizer,
    stage2_optimizer_factory: Callable[[AlphaGenomeEncoderModel], torch.optim.Optimizer] | None,
    config: TrainConfig,
    device: torch.device | str,
    val_loader=None,
    stage1_scheduler=None,
    stage2_scheduler_factory: Callable[[torch.optim.Optimizer], Any] | None = None,
    stage1_scheduler_step: Callable[[Any, dict[str, float]], None] | None = None,
    stage2_scheduler_step: Callable[[Any, dict[str, float]], None] | None = None,
    loss_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
    metric_fns: dict[str, Callable[[Tensor, Tensor], Tensor | float]] | None = None,
    track_names: list[str] | None = None,
    epoch_callback: Callable[[dict[str, Any]], None] | None = None,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Run stage 1 head-only training followed by stage 2 encoder+head training."""

    if config.stage.second_stage_lr is None:
        raise ValueError("stage.second_stage_lr must be set for two-stage training")
    if stage2_optimizer_factory is None:
        raise ValueError("stage2_optimizer_factory is required for two-stage training")
    if config.checkpoint.save_mode == "head":
        raise ValueError("head save_mode is not allowed for two-stage training")

    checkpoint_dir = Path(config.checkpoint.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    combined_history = _history_template()

    stage1_dir = checkpoint_dir / "stage1"
    stage2_dir = checkpoint_dir / "stage2"
    stage1_result: dict[str, Any]

    if not config.stage.resume_from_stage2:
        model.set_encoder_trainable(False)
        stage1_result = run_training_stage(
            model,
            train_loader,
            optimizer=stage1_optimizer,
            config=config,
            device=device,
            num_epochs=config.stage.num_epochs,
            stage="stage1",
            train_encoder=False,
            val_loader=val_loader,
            scheduler=stage1_scheduler,
            scheduler_step=stage1_scheduler_step,
            loss_fn=loss_fn,
            metric_fns=metric_fns,
            track_names=track_names,
            checkpoint_dir=stage1_dir,
            epoch_callback=epoch_callback,
            show_progress=show_progress,
            resume=config.stage.auto_resume,
        )
        _append_stage_history(combined_history, stage1_result["history"])
    else:
        stage1_checkpoint = stage1_dir / "best.pt"
        if not stage1_checkpoint.exists():
            raise FileNotFoundError(f"Stage 1 checkpoint not found: {stage1_checkpoint}")
        load_checkpoint(stage1_checkpoint, model)
        stage1_result = {
            "history": _history_template(),
            "best_epoch": 0,
            "best_monitor": -math.inf,
            "best_checkpoint_path": str(stage1_checkpoint),
            "resumed": True,
        }

    best_stage1_path = stage1_result["best_checkpoint_path"] or str(stage1_dir / "best.pt")
    load_checkpoint(best_stage1_path, model)

    # If stage 1 didn't resume its own checkpoint -- e.g. a config change invalidated it and
    # it just retrained from scratch -- its output is a brand-new encoder/head lineage. Any
    # existing stage2 checkpoint was built on top of the *previous* stage1 lineage, so letting
    # stage2 resume it here would immediately overwrite the weights just loaded above with that
    # unrelated lineage's weights, silently discarding what stage1 just produced this run.
    stage2_resume = config.stage.auto_resume and stage1_result.get("resumed", True)
    if config.stage.auto_resume and not stage2_resume:
        print(
            f"[stage2] stage1 restarted from scratch this run -- ignoring {stage2_dir / 'last.pt'} "
            "(it was built on a different stage1 lineage)"
        )

    # head.dropout is a plain float read fresh each forward pass (see MPRAHead._apply_hidden_layers),
    # not baked into the module at construction, so overriding it here takes effect for every
    # stage2 epoch without rebuilding the head or disturbing its already-loaded weights.
    if config.stage.second_stage_dropout is not None:
        model.head.dropout = config.stage.second_stage_dropout

    model.set_encoder_trainable(True)
    stage2_optimizer = stage2_optimizer_factory(model)
    stage2_scheduler = stage2_scheduler_factory(stage2_optimizer) if stage2_scheduler_factory is not None else None
    stage2_result = run_training_stage(
        model,
        train_loader,
        optimizer=stage2_optimizer,
        config=config,
        device=device,
        num_epochs=config.stage.second_stage_epochs,
        stage="stage2",
        train_encoder=True,
        val_loader=val_loader,
        scheduler=stage2_scheduler,
        scheduler_step=stage2_scheduler_step,
        loss_fn=loss_fn,
        metric_fns=metric_fns,
        track_names=track_names,
        checkpoint_dir=stage2_dir,
        start_epoch=stage1_result["best_epoch"],
        epoch_callback=epoch_callback,
        show_progress=show_progress,
        resume=stage2_resume,
    )
    _append_stage_history(combined_history, stage2_result["history"])

    return {
        "history": combined_history,
        "stage1": stage1_result,
        "stage2": stage2_result,
    }
