#!/bin/bash
#SBATCH --job-name=repro_seed_check
#SBATCH --output=/grid/koo/home/kachu/projects/alphagenome-encoder-ft/job_scripts/seed_check/%x_%j.log
#SBATCH --error=/grid/koo/home/kachu/projects/alphagenome-encoder-ft/job_scripts/seed_check/%x_%j.log
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-gpu=10
#SBATCH --mem-per-gpu=128G
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=03:00:00

set -ex

REPO_ROOT="/grid/koo/home/kachu/projects/alphagenome-encoder-ft"
cd "$REPO_ROOT"

PYTHON="$REPO_ROOT/.venv/bin/python"
SEED="${1:?usage: sbatch run_seed_check.sh <seed>}"
CKPT_DIR="$REPO_ROOT/results/ray_tune/ag_hpsweep_1000_best_trial_30aecdfb/seed_check_seed${SEED}"
mkdir -p "$CKPT_DIR"

# Exact resolved config of the best trial (train_fn_30aecdfb), only runtime.seed differs.
# split_mode=jores partitions rows by a fixed 'set' column in the tsv, not by seed, so
# train/val/test membership is identical across seeds -- only weight init / batch order /
# augmentation draws differ.
PYTHONPATH=src "$PYTHON" scripts/1_finetune/train_ag.py \
  --config configs/ag_jores.json \
  --pretrained_weights /grid/koo/home/shared/models/alphagenome/torch/model_all_folds.safetensors \
  --input_tsv metadata/modelling_data_tamsACR.tsv \
  --hidden_sizes "4096,2048" \
  --dropout 0.05 \
  --learning_rate 0.008 \
  --weight_decay 1e-8 \
  --batch_size 1024 \
  --second_stage_lr 0.0008 \
  --second_stage_dropout 0.1 \
  --checkpoint_dir "$CKPT_DIR" \
  --save_mode minimal \
  --seed "$SEED" \
  --no-use_wandb

# Evaluate the resulting stage2 best checkpoint on the held-out test split, same as
# scripts/2_test/evaluate_jores.py was run for the original trial (test.ipynb cell 2).
PYTHONPATH=src "$PYTHON" scripts/2_test/evaluate_jores.py \
  --checkpoint_path "$CKPT_DIR/stage2/best.pt" \
  --input_tsv metadata/modelling_data_tamsACR.tsv
