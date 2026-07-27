"""Encoder-only AlphaGenome fine-tuning utilities."""

__all__ = [
    "DataConfig",
    "HeadConfig",
    "OptimConfig",
    "StageConfig",
    "CheckpointConfig",
    "LoggingConfig",
    "RuntimeConfig",
    "CachedEmbeddingsConfig",
    "TrainConfig",
    "build_head",
    "load_train_config",
    "merge_train_config",
    "parse_hidden_sizes",
    "PlantMPRADataset",
    "create_random_splits",
    "DengMPRADataset",
    "create_deng_splits",
    "JoresMPRADataset",
    "create_jores_splits",
    "build_species_masks",
    "summarize_species_masks",
    "LentiMPRADataset",
    "DeepSTARRDataset",
    "DEEPSTARR_ADAPTER_UP",
    "DEEPSTARR_ADAPTER_DOWN",
    "create_dataloader",
    "ConstructSpec",
    "LENTIMPRA_BARCODE",
    "LENTIMPRA_LEFT_ADAPTER",
    "LENTIMPRA_PROMOTER",
    "LENTIMPRA_RIGHT_ADAPTER",
    "AlphaGenomeEncoderModel",
    "load_config_from_checkpoint",
    "MPRAHead",
    "JoresMPRAHead",
    "train_epoch",
    "evaluate",
    "run_training_stage",
    "run_two_stage_training",
    "save_checkpoint",
    "load_checkpoint",
    "set_encoder_trainable",
    "create_optimizer",
    "create_scheduler",
    "scheduler_stepper",
    "add_metrics_to_history",
    "stable_run_id",
    "training_run_id",
]


def __getattr__(name: str):
    if name in {"PlantMPRADataset",
                 "create_random_splits",
                "DengMPRADataset",
                 "create_deng_splits",
                "JoresMPRADataset",
                "create_jores_splits",
                "create_dataloader",
                "build_species_masks",
                "summarize_species_masks"}:
        from . import mydata

        return getattr(mydata, name)
    if name in {
        "LentiMPRADataset",
        "DeepSTARRDataset",
        "DEEPSTARR_ADAPTER_UP",
        "DEEPSTARR_ADAPTER_DOWN",
    }:
        from . import data

        return getattr(data, name)
    if name in {
        "ConstructSpec",
        "LENTIMPRA_BARCODE",
        "LENTIMPRA_LEFT_ADAPTER",
        "LENTIMPRA_PROMOTER",
        "LENTIMPRA_RIGHT_ADAPTER",
    }:
        from . import constructs

        return getattr(constructs, name)
    if name in {
        "DataConfig",
        "HeadConfig",
        "OptimConfig",
        "StageConfig",
        "CheckpointConfig",
        "LoggingConfig",
        "RuntimeConfig",
        "CachedEmbeddingsConfig",
        "TrainConfig",
        "build_head",
        "load_train_config",
        "merge_train_config",
        "parse_hidden_sizes",
    }:
        from . import config

        return getattr(config, name)
    if name in {"AlphaGenomeEncoderModel", "load_config_from_checkpoint"}:
        from . import model

        return getattr(model, name)
    if name in {"MPRAHead", "JoresMPRAHead"}:
        from . import heads

        return getattr(heads, name)
    if name in {
        "train_epoch",
        "evaluate",
        "run_training_stage",
        "run_two_stage_training",
        "save_checkpoint",
        "load_checkpoint",
        "set_encoder_trainable",
        "create_optimizer",
        "create_scheduler",
        "scheduler_stepper",
        "add_metrics_to_history",
        "stable_run_id",
        "training_run_id",
    }:
        from . import train

        return getattr(train, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
