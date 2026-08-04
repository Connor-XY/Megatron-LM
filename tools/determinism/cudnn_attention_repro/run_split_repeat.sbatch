#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Run the QAD recipe N times and collect an all-rank activation digest trace
# from each repeat, so the module that first diverges can be identified.
#
# Fill in the SITE CONFIGURATION block, then:
#   REPEAT_COUNT=10 sbatch run_split_repeat.sbatch
#
# The divergence is intermittent. Read the "Sample size" section of README.md
# before drawing a conclusion from a small number of repeats: a five-repeat cell
# reading all-same is between 0.1% and 29% likely by luck depending on the JIT
# cache state, which the cell itself does not reveal.
#
#SBATCH --job-name=cudnn-attn-split
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/cudnn-attn-split-%j.out

set -euo pipefail

# ---------------------------------------------------------------- SITE CONFIG
ACCOUNT=${ACCOUNT:?set ACCOUNT}
PARTITION=${PARTITION:?set PARTITION}
ROOT=${ROOT:?set ROOT, a shared path visible to every node}
SOURCE=${SOURCE:-$ROOT/Megatron-LM}       # Megatron-LM checkout
IMAGE=${IMAGE:?set IMAGE, the container to run in}
DATA=${DATA:?set DATA, the training data path}
# model_config.yaml for hybrid_nemotron_v3_pico_7b_a1b_tp1_ep8_QAD_dgx_h100_1N8G,
# retrieved per README.md; that recipe is not on main.
CONFIG=${CONFIG:?set CONFIG, path to the QAD model_config.yaml}
# -----------------------------------------------------------------------------

REPEAT_COUNT=${REPEAT_COUNT:-10}
RUN_ROOT=$ROOT/runs/cudnn-attn-split-$SLURM_JOB_ID
mkdir -p "$RUN_ROOT" logs

echo "REPEAT_COUNT=$REPEAT_COUNT"
echo "RUN_ROOT=$RUN_ROOT"

srun --ntasks=1 \
  --container-image="$IMAGE" \
  --container-mounts="$ROOT":"$ROOT","$SOURCE":/opt/megatron-lm \
  --container-workdir=/opt/megatron-lm \
  bash -lc '
set -euo pipefail

RUN_ROOT='"$RUN_ROOT"'
REPEAT_COUNT='"$REPEAT_COUNT"'
CONFIG='"$CONFIG"'
DATA='"$DATA"'

for repeat in $(seq 1 "$REPEAT_COUNT"); do
  echo "REPEAT_BEGIN=$repeat"
  work="$RUN_ROOT/repeat-$repeat"
  mkdir -p "$work/output" "$work/digest-trace" "$work/cache" "$work/checkpoint"

  # Collect input and output digests for every module in decoder layers 0..5 on
  # every rank. Two calls per module is enough to localize the producer and
  # keeps the trace small.
  export MCORE_DIGEST_TRACE="$work/digest-trace/rank{rank}.jsonl"
  export MCORE_DIGEST_TRACE_MAX_LAYER=5
  export MCORE_DIGEST_TRACE_MAX_CALLS=2

  # A warm CUDA JIT cache roughly triples the incidence relative to a cold one
  # (0.75 against 0.22), so keep it consistent across every repeat and across
  # any control arm, or the arms are not comparable.
  export CUDA_CACHE_PATH="$work/cuda-cache"
  mkdir -p "$CUDA_CACHE_PATH"

  bash /opt/megatron-lm/tests/functional_tests/shell_test_utils/_run_training.sh \
    TRAINING_SCRIPT_PATH=/opt/megatron-lm/pretrain_hybrid.py \
    TRAINING_PARAMS_PATH="$CONFIG" \
    OUTPUT_PATH="$work/output" \
    CHECKPOINT_SAVE_PATH="$work/checkpoint" \
    DATA_PATH="$DATA" \
    DATA_CACHE_PATH="$work/cache" \
    RUN_NUMBER=1 \
    REPEAT="$repeat" \
    2>&1 | tee "$work/full.log"

  echo "REPEAT_SUCCESS=$repeat trace_files=$(ls -1 "$work/digest-trace" | wc -l)"
  rm -rf "$work/checkpoint" "$work/cache"
done

echo "REPEATS_COMPLETE=$REPEAT_COUNT"
'

echo "Now run:"
echo "  python tools/determinism/cudnn_attention_repro/analyze_split.py $RUN_ROOT"
