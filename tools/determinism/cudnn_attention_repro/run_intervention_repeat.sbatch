#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# One script, one variable. Runs the QAD recipe N times under a single named
# intervention so every arm is diff-identical to its control except for the
# INTERVENTION value. Companion to run_split_repeat.sbatch; written for the
# follow-up experiments requested in nvbug 6489647 (cuDNN 9.25, allocator
# caching off, compute-sanitizer).
#
#   INTERVENTION=control    sbatch run_intervention_repeat.sbatch   # matched control
#   INTERVENTION=nocache    sbatch run_intervention_repeat.sbatch   # PYTORCH_NO_CUDA_MEMORY_CACHING=1
#   INTERVENTION=cudnn925   sbatch run_intervention_repeat.sbatch   # swap in cuDNN 9.25 wheel
#   INTERVENTION=memcheck   sbatch run_intervention_repeat.sbatch   # compute-sanitizer, 1 repeat
#   INTERVENTION=racecheck  sbatch run_intervention_repeat.sbatch
#   INTERVENTION=initcheck  sbatch run_intervention_repeat.sbatch
#   INTERVENTION=synccheck  sbatch run_intervention_repeat.sbatch
#
# Rate arms (control / nocache / cudnn925) default to 12 repeats. Read the
# "Sample size" section of README.md before shrinking that: the per-repeat
# incidence varies 0.40-1.00 across nominally identical jobs, so a small cell
# is uninformative and each treatment arm must be judged against a control
# submitted in the same window, not against a remembered rate.
#
# Sanitizer arms default to 1 repeat with the calibration set cut to
# SANITIZER_CALIB_SIZE samples. The divergence is on the FIRST calibration
# batch, so a short sanitized run still covers the region of interest, and
# compute-sanitizer streams findings as they occur, so even a timed-out job
# leaves a usable report.
#
#SBATCH --job-name=cudnn-attn-intervention
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/cudnn-attn-intervention-%j.out

set -euo pipefail

# ---------------------------------------------------------------- SITE CONFIG
ACCOUNT=${ACCOUNT:?set ACCOUNT}
PARTITION=${PARTITION:?set PARTITION}
ROOT=${ROOT:?set ROOT, a shared path visible to every node}
SOURCE=${SOURCE:-$ROOT/Megatron-LM}       # Megatron-LM checkout
IMAGE=${IMAGE:?set IMAGE, the container to run in}
DATA=${DATA:?set DATA, the training data path}
CONFIG=${CONFIG:?set CONFIG, path to the QAD model_config.yaml}
# -----------------------------------------------------------------------------

INTERVENTION=${INTERVENTION:?set INTERVENTION: control|nocache|cudnn925|memcheck|racecheck|initcheck|synccheck}

case "$INTERVENTION" in
  control|nocache|cudnn925) REPEAT_COUNT=${REPEAT_COUNT:-12} ;;
  memcheck|racecheck|initcheck|synccheck) REPEAT_COUNT=${REPEAT_COUNT:-1} ;;
  *) echo "unknown INTERVENTION=$INTERVENTION" >&2; exit 2 ;;
esac

# cuDNN wheel to test. The container under test ships 9.23; the bug asks for
# 9.25. The wheel is installed once per job into the run root and takes effect
# through LD_LIBRARY_PATH + LD_PRELOAD; the job aborts if the loaded version
# does not match CUDNN_EXPECT_PREFIX, so a silent fallback to the container's
# cuDNN cannot masquerade as a clean result.
CUDNN_WHEEL_SPEC=${CUDNN_WHEEL_SPEC:-nvidia-cudnn-cu13==9.25.*}
CUDNN_EXPECT_PREFIX=${CUDNN_EXPECT_PREFIX:-925}

SANITIZER_CALIB_SIZE=${SANITIZER_CALIB_SIZE:-16}
SANITIZER_TRAIN_ITERS=${SANITIZER_TRAIN_ITERS:-2}

RUN_ROOT=$ROOT/runs/cudnn-attn-$INTERVENTION-$SLURM_JOB_ID
mkdir -p "$RUN_ROOT" logs

echo "INTERVENTION=$INTERVENTION"
echo "REPEAT_COUNT=$REPEAT_COUNT"
echo "RUN_ROOT=$RUN_ROOT"

srun --ntasks=1 \
  --container-image="$IMAGE" \
  --container-mounts="$ROOT":"$ROOT","$SOURCE":/opt/megatron-lm \
  --container-workdir=/opt/megatron-lm \
  bash -lc '
set -euo pipefail

RUN_ROOT='"$RUN_ROOT"'
ROOT='"$ROOT"'
REPEAT_COUNT='"$REPEAT_COUNT"'
CONFIG='"$CONFIG"'
DATA='"$DATA"'
INTERVENTION='"$INTERVENTION"'
CUDNN_WHEEL_SPEC="'"$CUDNN_WHEEL_SPEC"'"
CUDNN_EXPECT_PREFIX='"$CUDNN_EXPECT_PREFIX"'
SANITIZER_CALIB_SIZE='"$SANITIZER_CALIB_SIZE"'
SANITIZER_TRAIN_ITERS='"$SANITIZER_TRAIN_ITERS"'

# ------------------------------------------------------------- provenance
{
  echo "container cuDNN at job start:"
  python3 -c "import ctypes; print(ctypes.CDLL(\"libcudnn.so.9\").cudnnGetVersion())"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1
  compute-sanitizer --version 2>/dev/null | head -2 || true
} | tee "$RUN_ROOT/provenance.txt"

# ------------------------------------------------------------- arm setup
SANITIZER_TOOL=""
case "$INTERVENTION" in
  control)
    ;;
  nocache)
    # Every allocation becomes a raw cudaMalloc/cudaFree. Removes block reuse
    # and changes allocation timing wholesale; expect each repeat to run
    # several times slower than control. Verify it was in effect from the
    # iteration-time signature in full.log, exactly as was done for
    # CUDA_LAUNCH_BLOCKING.
    export PYTORCH_NO_CUDA_MEMORY_CACHING=1
    ;;
  cudnn925)
    CUDNN_DIR="$RUN_ROOT/cudnn-wheel"
    pip install --target "$CUDNN_DIR" "$CUDNN_WHEEL_SPEC" \
      || pip install --no-index --find-links "$ROOT/wheels" \
           --target "$CUDNN_DIR" "$CUDNN_WHEEL_SPEC" \
      || { echo "FATAL: could not install $CUDNN_WHEEL_SPEC (no PyPI egress?)." \
           "Pre-download it with: pip download --no-deps -d \$ROOT/wheels \"$CUDNN_WHEEL_SPEC\"" >&2; exit 1; }
    CUDNN_LIB="$CUDNN_DIR/nvidia/cudnn/lib"
    export LD_LIBRARY_PATH="$CUDNN_LIB:${LD_LIBRARY_PATH:-}"
    export LD_PRELOAD="$CUDNN_LIB/libcudnn.so.9${LD_PRELOAD:+:$LD_PRELOAD}"
    ver=$(python3 -c "import ctypes; print(ctypes.CDLL(\"libcudnn.so.9\").cudnnGetVersion())")
    tver=$(uv run --no-sync python -c "import torch; torch.zeros(1, device=\"cuda\"); print(torch.backends.cudnn.version())")
    echo "loaded cuDNN: ctypes=$ver torch=$tver" | tee "$RUN_ROOT/cudnn-version.txt"
    case "$ver" in "$CUDNN_EXPECT_PREFIX"*) ;; *) echo "FATAL: ctypes cuDNN $ver != ${CUDNN_EXPECT_PREFIX}xx" >&2; exit 1 ;; esac
    case "$tver" in "$CUDNN_EXPECT_PREFIX"*) ;; *) echo "FATAL: torch cuDNN $tver != ${CUDNN_EXPECT_PREFIX}xx" >&2; exit 1 ;; esac
    ;;
  memcheck|racecheck|initcheck|synccheck)
    SANITIZER_TOOL=$INTERVENTION
    # For bounds and uninitialized-read checking, the caching allocator hides
    # the crime scene: suballocations from one cached block are one allocation
    # to the tool, and reused blocks arrive pre-written. Disable it for those
    # two tools only.
    case "$SANITIZER_TOOL" in
      memcheck|initcheck) export PYTORCH_NO_CUDA_MEMORY_CACHING=1 ;;
    esac
    # Cut the calibration set and the stage-2 iteration count; the divergence
    # is on the first calibration batch, so coverage of the suspect region
    # survives the cut and each sanitized repeat stays inside the walltime.
    SANITIZED_CONFIG="$RUN_ROOT/model_config.sanitizer.yaml"
    sed -e "s/--calib-size 256/--calib-size $SANITIZER_CALIB_SIZE/" \
        -e "s/--train-iters: 10/--train-iters: $SANITIZER_TRAIN_ITERS/" \
        -e "s/--exit-interval: 10/--exit-interval: $SANITIZER_TRAIN_ITERS/" \
        "$CONFIG" > "$SANITIZED_CONFIG"
    CONFIG=$SANITIZED_CONFIG
    ;;
esac

for repeat in $(seq 1 "$REPEAT_COUNT"); do
  echo "REPEAT_BEGIN=$repeat"
  work="$RUN_ROOT/repeat-$repeat"
  mkdir -p "$work/output" "$work/digest-trace" "$work/cache" "$work/checkpoint"

  # A warm CUDA JIT cache roughly triples the incidence relative to a cold one
  # (0.75 against 0.22), so keep it consistent across every repeat and every
  # arm, or the arms are not comparable.
  export CUDA_CACHE_PATH="$work/cuda-cache"
  mkdir -p "$CUDA_CACHE_PATH"

  if [ -z "$SANITIZER_TOOL" ]; then
    # Rate arms carry the digest trace so analyze_split.py can localize the
    # first diverging module, same as run_split_repeat.sbatch.
    export MCORE_DIGEST_TRACE="$work/digest-trace/rank{rank}.jsonl"
    export MCORE_DIGEST_TRACE_MAX_LAYER=5
    export MCORE_DIGEST_TRACE_MAX_CALLS=2

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
  else
    # The digest-trace hooks add kernels the sanitizer would also instrument;
    # this arm hunts memory/race errors, not splits, so leave them off.
    unset MCORE_DIGEST_TRACE MCORE_DIGEST_TRACE_MAX_LAYER MCORE_DIGEST_TRACE_MAX_CALLS || true

    # --target-processes all follows torchrun into the 8 rank processes; %p
    # keeps their reports from clobbering each other. Expect known NCCL noise
    # under memcheck; findings inside TE/cuDNN/Megatron MoE kernels are the
    # signal. A nonzero exit here is a finding, not a launcher failure, so it
    # must not abort the job before the logs are flushed.
    compute-sanitizer --tool "$SANITIZER_TOOL" --target-processes all \
      --log-file "$work/compute-sanitizer.%p.log" \
      bash /opt/megatron-lm/tests/functional_tests/shell_test_utils/_run_training.sh \
        TRAINING_SCRIPT_PATH=/opt/megatron-lm/pretrain_hybrid.py \
        TRAINING_PARAMS_PATH="$CONFIG" \
        OUTPUT_PATH="$work/output" \
        CHECKPOINT_SAVE_PATH="$work/checkpoint" \
        DATA_PATH="$DATA" \
        DATA_CACHE_PATH="$work/cache" \
        RUN_NUMBER=1 \
        REPEAT="$repeat" \
        2>&1 | tee "$work/full.log" || echo "SANITIZER_WRAPPED_EXIT=$?"

    grep -lE "ERROR SUMMARY: [1-9]" "$work"/compute-sanitizer.*.log 2>/dev/null \
      | while read -r f; do echo "SANITIZER_ERRORS_IN=$f"; done || true
  fi

  echo "REPEAT_SUCCESS=$repeat"
  rm -rf "$work/checkpoint" "$work/cache"
done

echo "REPEATS_COMPLETE=$REPEAT_COUNT"
'

echo "Now run:"
echo "  python tools/determinism/cudnn_attention_repro/analyze_split.py $RUN_ROOT"
echo "For sanitizer arms, read \$work/compute-sanitizer.*.log instead."
