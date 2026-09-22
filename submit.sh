#!/bin/bash -l
#SBATCH --job-name=crispr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

PYTHON=${PYTHON:-python}
MODEL=${MODEL:-google/flan-t5-base}
DATA=${DATA:-data/BBQ/SES.jsonl}
SEEDS=${SEEDS:-2026}
EVAL=${EVAL:-generation}
ATTR_BATCH_SIZE=${ATTR_BATCH_SIZE:-4}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-32}

echo "model: $MODEL | data: $DATA | seeds: $SEEDS | eval: $EVAL"

$PYTHON compute_attribution.py --model "$MODEL" --data "$DATA" --seeds $SEEDS \
    --batch_size "$ATTR_BATCH_SIZE"

$PYTHON run_crispr.py --model "$MODEL" --data "$DATA" --seeds $SEEDS --eval "$EVAL" \
    --batch_size "$EVAL_BATCH_SIZE"
