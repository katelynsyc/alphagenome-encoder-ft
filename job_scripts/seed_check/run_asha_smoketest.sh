#!/bin/bash
#SBATCH --job-name=ag_asha_smoketest
#SBATCH --output=/grid/koo/home/kachu/projects/alphagenome-encoder-ft/job_scripts/seed_check/%x_%j.log
#SBATCH --error=/grid/koo/home/kachu/projects/alphagenome-encoder-ft/job_scripts/seed_check/%x_%j.log
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=64G
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=00:45:00

set -ex

REPO_ROOT="/grid/koo/home/kachu/projects/alphagenome-encoder-ft"
cd "$REPO_ROOT"

PYTHON="$REPO_ROOT/.venv/bin/python"
export PYTHONUNBUFFERED=1

# Small/fast config (subset_frac=0.03, stage.num_epochs=3, stage.second_stage_epochs=20) plus
# 2 GPUs (so 2 trials run concurrently, which is what actually exercises different trials
# reaching stage2 at different real times) -- exercises the real train_ag_tune.py ->
# train_ag.run() -> run_two_stage_training pipeline end to end (data loading, checkpointing,
# Ray Tune reporting, ASHA scheduling) in minutes instead of hours. Not meant to produce a
# good model.
PYTHONPATH=src "$PYTHON" scripts/1_finetune/train_ag_tune.py \
  --config configs/ag_jores_smoketest.json \
  --pretrained_weights /grid/koo/home/shared/models/alphagenome/torch/model_all_folds.safetensors \
  --input_tsv metadata/modelling_data_tamsACR.tsv \
  --experiment_name ag_smoketest_stage2_asha \
  --storage_path "$REPO_ROOT/results/ray_tune" \
  --num_samples 6 \
  --gpus_per_trial 1 \
  --cpus_per_trial 4
