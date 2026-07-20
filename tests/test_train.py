from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from alphagenome_encoder_ft.config import OptimConfig, TrainConfig
from alphagenome_encoder_ft.heads import JoresMPRAHead, MPRAHead
from alphagenome_encoder_ft.model import AlphaGenomeEncoderModel
import alphagenome_encoder_ft.train as train_module
from alphagenome_encoder_ft.train import create_scheduler, evaluate, load_checkpoint, load_training_state, run_training_stage, run_two_stage_training, save_checkpoint, training_run_id


class DummyAlphaGenome(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(4, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1536),
        )

    def forward(self, sequences, organism_idx, encoder_only=False):
        del organism_idx
        if not encoder_only:
            raise ValueError("Dummy model only supports encoder_only=True")
        batch, length, channels = sequences.shape
        encoded = self.encoder(sequences.reshape(batch * length, channels)).reshape(batch, length, 1536)
        return {"encoder_output": encoded}


def _make_loader():
    torch.manual_seed(0)
    sequences = torch.randn(12, 2, 4)
    targets = sequences.sum(dim=(1, 2))
    return DataLoader(TensorDataset(sequences, targets), batch_size=4, shuffle=False)


def _make_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig.from_dict(
        {
            "data": {
                "input_tsv": "/tmp/mock.tsv",
                "sequence_length": 256,
                "batch_size": 4,
            },
            "head": {
                "pooling_type": "flatten",
                "hidden_sizes": [8],
                "center_bp": 256,
                "dropout": 0.1,
                "activation": "relu",
            },
            "optim": {
                "optimizer": "adam",
                "learning_rate": 1e-2,
                "weight_decay": 0.0,
                "lr_scheduler": "constant",
                "plateau_factor": 0.5,
                "plateau_patience": 2,
                "plateau_mode": "min",
                "plateau_min_lr": 0.0,
                "gradient_accumulation_steps": 1,
            },
            "stage": {
                "num_epochs": 2,
                "early_stopping_patience": 5,
                "val_evals_per_epoch": 1,
                "second_stage_lr": 1e-3,
                "second_stage_epochs": 1,
            },
            "checkpoint": {
                "pretrained_weights": "/tmp/weights.pt",
                "checkpoint_dir": str(tmp_path),
                "save_mode": "minimal",
            },
            "runtime": {
                "use_amp": False,
                "seed": 0,
            },
        }
    )


def _make_model() -> AlphaGenomeEncoderModel:
    model = AlphaGenomeEncoderModel(DummyAlphaGenome(), MPRAHead(pooling_type="flatten", hidden_sizes=8))
    model.initialize_head(sequence_length=2, device="cpu")
    return model


def test_run_training_stage_writes_minimal_checkpoint(tmp_path: Path):
    model = _make_model()
    loader = _make_loader()
    config = _make_config(tmp_path)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-2)

    result = run_training_stage(
        model,
        loader,
        optimizer=optimizer,
        config=config,
        device="cpu",
        num_epochs=2,
        stage="stage1",
        train_encoder=False,
        checkpoint_dir=tmp_path / "stage1",
    )

    assert (tmp_path / "stage1" / "best.pt").exists()
    assert result["best_checkpoint_path"] is not None


def test_two_stage_training_rejects_head_mode(tmp_path: Path):
    model = _make_model()
    loader = _make_loader()
    config = _make_config(tmp_path)
    config.checkpoint.save_mode = "head"
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-2)

    try:
        run_two_stage_training(
            model,
            loader,
            stage1_optimizer=optimizer,
            stage2_optimizer_factory=lambda model_obj: torch.optim.Adam(
                model_obj.trainable_parameters(include_encoder=True),
                lr=1e-3,
            ),
            config=config,
            device="cpu",
        )
    except ValueError as exc:
        assert "head save_mode" in str(exc)
    else:
        raise AssertionError("Expected ValueError for head save_mode")


def test_resume_from_stage2_loads_stage1_checkpoint(tmp_path: Path):
    model = _make_model()
    loader = _make_loader()
    config = _make_config(tmp_path)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-2)

    run_training_stage(
        model,
        loader,
        optimizer=optimizer,
        config=config,
        device="cpu",
        num_epochs=1,
        stage="stage1",
        train_encoder=False,
        checkpoint_dir=tmp_path / "stage1",
    )

    stage2_model = _make_model()
    stage2_config = _make_config(tmp_path)
    stage2_config.stage.resume_from_stage2 = True
    result = run_two_stage_training(
        stage2_model,
        loader,
        stage1_optimizer=torch.optim.Adam(stage2_model.head.parameters(), lr=1e-2),
        stage2_optimizer_factory=lambda model_obj: torch.optim.Adam(
            model_obj.trainable_parameters(include_encoder=True),
            lr=1e-3,
        ),
        config=stage2_config,
        device="cpu",
    )

    assert result["stage2"]["best_checkpoint_path"] is not None


def _run_two_stage(model, loader, config):
    return run_two_stage_training(
        model,
        loader,
        stage1_optimizer=torch.optim.Adam(model.head.parameters(), lr=1e-2),
        stage2_optimizer_factory=lambda model_obj: torch.optim.Adam(
            model_obj.trainable_parameters(include_encoder=True),
            lr=1e-3,
        ),
        config=config,
        device="cpu",
    )


def test_stage1_restart_forces_stage2_to_restart_instead_of_resuming_stale_lineage(tmp_path: Path):
    # Regression test: stage2 must never resume its own checkpoint when stage1 was forced to
    # retrain from scratch this run, since that checkpoint's weights were produced by a
    # *different* stage1 lineage than the one this run just finished.
    model = _make_model()
    loader = _make_loader()
    config = _make_config(tmp_path)
    config.stage.num_epochs = 2
    config.stage.second_stage_epochs = 2

    first_result = _run_two_stage(model, loader, config)
    assert first_result["stage1"]["resumed"] is False  # first-ever run: nothing to resume yet
    assert len(first_result["stage2"]["history"]["train_loss"]) == 2

    # A training-relevant, non-ignored config change (not in _RESUME_CONFIG_IGNORE) forces
    # stage1 to restart from scratch. second_stage_epochs is raised so that, if stage2 wrongly
    # resumed its old checkpoint, it would only run 3 more epochs (2 done + 3 = 5) instead of
    # restarting cleanly for a full fresh run of 5.
    second_model = _make_model()
    second_config = _make_config(tmp_path)
    second_config.stage.num_epochs = 2
    second_config.stage.second_stage_epochs = 5
    second_config.optim.weight_decay = 0.1

    second_result = _run_two_stage(second_model, loader, second_config)

    assert second_result["stage1"]["resumed"] is False
    assert len(second_result["stage2"]["history"]["train_loss"]) == 5
    # The restarted stage2 must not carry over the first run's recorded history.
    assert second_result["stage2"]["history"]["train_loss"][:2] != first_result["stage2"]["history"]["train_loss"]


def test_training_run_id_is_stable_unless_a_relevant_field_changes(tmp_path: Path):
    config = _make_config(tmp_path)
    same_config = _make_config(tmp_path)  # a separate but identical config, e.g. a requeue

    assert training_run_id(config) == training_run_id(same_config)

    changed = _make_config(tmp_path)
    changed.optim.learning_rate = 1e-3
    assert training_run_id(changed) != training_run_id(config)

    # Fields in _RESUME_CONFIG_IGNORE (num_epochs, second_stage_epochs, wandb_name, ...)
    # are stopping-criteria/logging knobs, not hyperparameters -- raising an epoch budget
    # or relabeling a run for wandb shouldn't fragment its on-disk checkpoint lineage.
    ignore_changed = _make_config(tmp_path)
    ignore_changed.stage.num_epochs = 999
    ignore_changed.logging.wandb_name = "some-other-label"
    assert training_run_id(ignore_changed) == training_run_id(config)


def test_run_training_stage_runs_validation_within_each_epoch_and_emits_callbacks(tmp_path: Path):
    model = _make_model()
    train_loader = _make_loader()
    val_loader = _make_loader()
    config = _make_config(tmp_path)
    config.stage.num_epochs = 3
    config.stage.val_evals_per_epoch = 2
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-2)
    epoch_events = []

    result = run_training_stage(
        model,
        train_loader,
        optimizer=optimizer,
        config=config,
        device="cpu",
        num_epochs=3,
        stage="stage1",
        train_encoder=False,
        val_loader=val_loader,
        checkpoint_dir=tmp_path / "stage1",
        epoch_callback=epoch_events.append,
    )

    assert len(result["history"]["train_loss"]) == 3
    assert result["history"]["val_epoch"] == pytest.approx([1 / 3, 2 / 3, 4 / 3, 5 / 3, 7 / 3, 8 / 3])
    assert result["history"]["test_epoch"] == []
    assert [event["epoch"] for event in epoch_events] == [1.0, 2.0, 3.0]
    assert epoch_events[0]["val_loss"] >= 0.0
    assert "test_loss" not in epoch_events[-1]
    assert epoch_events[1]["val_loss"] >= 0.0


def test_run_training_stage_validates_once_per_epoch_when_requested(tmp_path: Path):
    model = _make_model()
    train_loader = _make_loader()
    val_loader = _make_loader()
    config = _make_config(tmp_path)
    config.stage.num_epochs = 2
    config.stage.val_evals_per_epoch = 1
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-2)

    result = run_training_stage(
        model,
        train_loader,
        optimizer=optimizer,
        config=config,
        device="cpu",
        num_epochs=2,
        stage="stage1",
        train_encoder=False,
        val_loader=val_loader,
        checkpoint_dir=tmp_path / "stage1",
    )

    assert result["history"]["val_epoch"] == [1.0, 2.0]


def test_run_training_stage_deduplicates_dense_validation_points(tmp_path: Path):
    model = _make_model()
    train_loader = _make_loader()
    val_loader = _make_loader()
    config = _make_config(tmp_path)
    config.stage.num_epochs = 1
    config.stage.val_evals_per_epoch = 5
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-2)

    result = run_training_stage(
        model,
        train_loader,
        optimizer=optimizer,
        config=config,
        device="cpu",
        num_epochs=1,
        stage="stage1",
        train_encoder=False,
        val_loader=val_loader,
        checkpoint_dir=tmp_path / "stage1",
    )

    assert result["history"]["val_epoch"] == [1 / 3, 2 / 3, 1.0]


def test_run_training_stage_early_stopping_counts_validation_events(tmp_path: Path):
    model = _make_model()
    train_loader = _make_loader()
    val_loader = _make_loader()
    config = _make_config(tmp_path)
    config.stage.num_epochs = 10
    config.stage.early_stopping_patience = 2
    config.stage.val_evals_per_epoch = 3
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-2)

    original_evaluate = train_module.evaluate
    eval_losses = iter([1.0] + [2.0] * 20)

    def fake_evaluate(*args, **kwargs):
        return {"loss": next(eval_losses), "pearson": 0.0}

    train_module.evaluate = fake_evaluate
    try:
        result = run_training_stage(
            model,
            train_loader,
            optimizer=optimizer,
            config=config,
            device="cpu",
            num_epochs=10,
            stage="stage1",
            train_encoder=False,
            val_loader=val_loader,
            checkpoint_dir=tmp_path / "stage1",
        )
    finally:
        train_module.evaluate = original_evaluate

    assert len(result["history"]["val_epoch"]) == 7
    assert result["best_epoch"] == 1 / 3


def test_load_checkpoint_then_evaluate_best_checkpoint(tmp_path: Path):
    model = _make_model()
    train_loader = _make_loader()
    test_loader = _make_loader()
    config = _make_config(tmp_path)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-2)

    result = run_training_stage(
        model,
        train_loader,
        optimizer=optimizer,
        config=config,
        device="cpu",
        num_epochs=2,
        stage="stage1",
        train_encoder=False,
        checkpoint_dir=tmp_path / "stage1",
    )

    load_checkpoint(result["best_checkpoint_path"], model, map_location="cpu")
    metrics = evaluate(model, test_loader, device="cpu")

    assert metrics["loss"] >= 0.0
    assert "pearson" in metrics


def test_create_scheduler_uses_plateau_config():
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.tensor(1.0))], lr=1e-2)
    optim_config = OptimConfig(
        lr_scheduler="plateau",
        plateau_factor=0.25,
        plateau_patience=4,
        plateau_mode="min",
        plateau_min_lr=1e-5,
    )

    scheduler = create_scheduler(optim_config, optimizer, total_epochs=5)

    assert isinstance(scheduler, ReduceLROnPlateau)
    assert scheduler.factor == 0.25
    assert scheduler.patience == 4
    assert scheduler.mode == "min"
    assert scheduler.min_lrs == [1e-5]


def test_train_config_rejects_invalid_plateau_settings():
    try:
        TrainConfig.from_dict(
            {
                "data": {"input_tsv": "/tmp/mock.tsv"},
                "checkpoint": {"pretrained_weights": "/tmp/weights.pt"},
                "optim": {"plateau_factor": 1.0},
            }
        )
    except ValueError as exc:
        assert "optim.plateau_factor" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid plateau_factor")


def test_save_checkpoint_persists_head_type_mpra_default(tmp_path: Path):
    model = _make_model()
    config = _make_config(tmp_path)
    path = save_checkpoint(
        tmp_path / "mpra.pt",
        model,
        config=config,
        save_mode="minimal",
        stage="stage1",
        epoch=1,
        metrics={"pearson": 0.5},
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["head_type"] == "mpra"
    assert payload["head_config"]["num_outputs"] == 1


def test_from_checkpoint_without_head_type_defaults_to_mpra(tmp_path: Path):
    # mimic a pre-PR checkpoint: no head_type field on the payload at all.
    model = _make_model()
    config = _make_config(tmp_path)
    path = save_checkpoint(
        tmp_path / "legacy.pt",
        model,
        config=config,
        save_mode="minimal",
        stage="stage1",
        epoch=1,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload.pop("head_type", None)
    payload["head_config"].pop("head_type", None)
    torch.save(payload, path)

    restored = torch.load(path, map_location="cpu", weights_only=False)
    assert "head_type" not in restored
    # dispatch logic inside AlphaGenomeEncoderModel.from_checkpoint reads
    # checkpoint.get("head_type", ..., "mpra"); re-exercise that path directly here.
    from alphagenome_encoder_ft.config import build_head
    head = build_head(
        restored.get("head_type", restored.get("head_config", {}).get("head_type", "mpra")),
        restored.get("head_config", {}),
    )
    assert isinstance(head, MPRAHead)
    assert not isinstance(head, JoresMPRAHead)


def test_save_checkpoint_persists_head_type_joresmpra(tmp_path: Path):
    # build a joresmpra config and a matching model, assert the saved payload
    # carries the dispatch field.
    config = TrainConfig.from_dict(
        {
            "data": {"input_tsv": "/tmp/mock.tsv", "sequence_length": 256},
            "head": {
                "head_type": "joresmpra",
                "pooling_type": "flatten",
                "hidden_sizes": [8],
                "center_bp": 256,
                "dropout": 0.5,
                "activation": "relu",
                "num_outputs": 2,
            },
            "checkpoint": {
                "pretrained_weights": "/tmp/weights.pt",
                "checkpoint_dir": str(tmp_path),
                "save_mode": "minimal",
            },
            "stage": {"second_stage_lr": 1e-3},
        }
    )
    model = AlphaGenomeEncoderModel(DummyAlphaGenome(), JoresMPRAHead(pooling_type="flatten", hidden_sizes=8))
    model.initialize_head(sequence_length=2, device="cpu")
    path = save_checkpoint(
        tmp_path / "joresmpra.pt",
        model,
        config=config,
        save_mode="minimal",
        stage="stage1",
        epoch=1,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["head_type"] == "joresmpra"
    assert payload["head_config"]["num_outputs"] == 2


def test_last_checkpoint_round_trips_full_training_state(tmp_path: Path):
    model = _make_model()
    loader = _make_loader()
    config = _make_config(tmp_path)
    config.stage.num_epochs = 1
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-2)

    run_training_stage(
        model,
        loader,
        optimizer=optimizer,
        config=config,
        device="cpu",
        num_epochs=1,
        stage="stage1",
        train_encoder=False,
        checkpoint_dir=tmp_path / "stage1",
    )

    last_path = tmp_path / "stage1" / "last.pt"
    assert last_path.exists()

    # last.pt must be a complete, standalone checkpoint: model weights load the same way
    # best.pt's do (no training_state needed for this half)...
    fresh_model = _make_model()
    load_checkpoint(last_path, fresh_model, map_location="cpu")
    for (name, original), (_, restored) in zip(
        model.head.state_dict().items(), fresh_model.head.state_dict().items()
    ):
        assert torch.equal(original, restored), name

    # ...and separately/additionally, optimizer state (including Adam's per-parameter
    # exp_avg/exp_avg_sq/step) round-trips into a freshly constructed optimizer.
    fresh_optimizer = torch.optim.Adam(fresh_model.head.parameters(), lr=1e-2)
    resumed = load_training_state(last_path, fresh_model, fresh_optimizer, map_location="cpu")

    original_state = list(optimizer.state.values())
    restored_state = list(fresh_optimizer.state.values())
    assert len(original_state) == len(restored_state) > 0
    for original_param_state, restored_param_state in zip(original_state, restored_state):
        assert original_param_state["step"] == restored_param_state["step"]
        assert torch.equal(original_param_state["exp_avg"], restored_param_state["exp_avg"])
        assert torch.equal(original_param_state["exp_avg_sq"], restored_param_state["exp_avg_sq"])

    assert resumed["epochs_done"] == 1
    assert resumed["early_stopped"] is False


def test_run_training_stage_resumes_instead_of_restarting(tmp_path: Path):
    model = _make_model()
    loader = _make_loader()
    config = _make_config(tmp_path)
    config.stage.num_epochs = 2
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-2)

    first_result = run_training_stage(
        model,
        loader,
        optimizer=optimizer,
        config=config,
        device="cpu",
        num_epochs=2,
        stage="stage1",
        train_encoder=False,
        checkpoint_dir=tmp_path / "stage1",
    )
    assert len(first_result["history"]["train_loss"]) == 2

    # Simulates a preempted-and-restarted process: brand-new model/optimizer objects,
    # same checkpoint_dir, larger num_epochs (the stage's full original target).
    resumed_model = _make_model()
    resumed_optimizer = torch.optim.Adam(resumed_model.head.parameters(), lr=1e-2)
    resumed_result = run_training_stage(
        resumed_model,
        loader,
        optimizer=resumed_optimizer,
        config=config,
        device="cpu",
        num_epochs=5,
        stage="stage1",
        train_encoder=False,
        checkpoint_dir=tmp_path / "stage1",
    )

    # Continued from epoch 2 to 5 (3 more), not restarted (2 + 5 = 7) and not stuck at 2.
    assert len(resumed_result["history"]["train_loss"]) == 5
    # The first two epochs' recorded history are exactly what was loaded from the
    # checkpoint, not recomputed.
    assert resumed_result["history"]["train_loss"][:2] == first_result["history"]["train_loss"]


def test_run_training_stage_is_idempotent_when_already_complete(tmp_path: Path):
    model = _make_model()
    loader = _make_loader()
    config = _make_config(tmp_path)
    config.stage.num_epochs = 2
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-2)

    first_result = run_training_stage(
        model,
        loader,
        optimizer=optimizer,
        config=config,
        device="cpu",
        num_epochs=2,
        stage="stage1",
        train_encoder=False,
        checkpoint_dir=tmp_path / "stage1",
    )

    second_model = _make_model()
    second_optimizer = torch.optim.Adam(second_model.head.parameters(), lr=1e-2)

    original_train_epoch = train_module.train_epoch

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("train_epoch should not run when the stage is already complete")

    train_module.train_epoch = _fail_if_called
    try:
        second_result = run_training_stage(
            second_model,
            loader,
            optimizer=second_optimizer,
            config=config,
            device="cpu",
            num_epochs=2,
            stage="stage1",
            train_encoder=False,
            checkpoint_dir=tmp_path / "stage1",
        )
    finally:
        train_module.train_epoch = original_train_epoch

    assert second_result["best_checkpoint_path"] == first_result["best_checkpoint_path"]
    assert second_result["best_epoch"] == first_result["best_epoch"]
    assert second_result["history"]["train_loss"] == first_result["history"]["train_loss"]


def test_set_dataset_epoch_accepts_fractional_epoch_numbers():
    # stage 2's start_epoch is stage 1's best_epoch, which is a float whenever the best
    # validation event landed mid-epoch -- np.random.default_rng requires a plain int seed,
    # so epoch_number (start_epoch + epoch_idx + 1) being a non-integer float must not crash.
    from alphagenome_encoder_ft.mydata import JoresMPRADataset

    rows = [
        {
            "sequence": "ACGT" * 43,
            "enrichment_cold": "1.0",
            "enrichment_dark": "2.0",
            "enrichment_light": "3.0",
            "enrichment_warm": "4.0",
            "enrichment_maize": "5.0",
        }
    ]
    dataset = JoresMPRADataset(rows, use_adapters=False, sequence_length=172, seed=42)

    train_module._set_dataset_epoch(dataset, 58.5)  # fractional start_epoch -> fractional epoch_number
    train_module._set_dataset_epoch(dataset, 59.0)  # whole-valued float, as in the reported crash
