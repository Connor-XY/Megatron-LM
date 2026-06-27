#!/bin/bash
# Det-vs-nondet per-NVTX-range breakdown for a DSV3-style MoE + MLA config.
#
# Usage:
#   bash perf_breakdown_moe.sh LEADERBOARD_DIR LOG_DIR
#
# The defaults reproduce the original TP2xEP4 profiling baseline. Override the
# exported shape variables to exercise a guarded production path, for example:
#   HIDDEN_SIZE=2048 MICRO_BATCH_SIZE=1 GLOBAL_BATCH_SIZE=8 \
#   NUM_EXPERTS=16 MOE_ROUTER_TOPK=8 MOE_FFN_HIDDEN_SIZE=1024 \
#   bash perf_breakdown_moe.sh /tmp/leaderboard /tmp/logs
set -euo pipefail

OUT="${1:?usage: $0 LEADERBOARD_DIR LOG_DIR}"
LOG_DIR="${2:?usage: $0 LEADERBOARD_DIR LOG_DIR}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export HIDDEN_SIZE="${HIDDEN_SIZE:-1024}"
export NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-16}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
export NUM_EXPERTS="${NUM_EXPERTS:-8}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-2}"
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-512}"
export MOE_ROUTER_NUM_GROUPS="${MOE_ROUTER_NUM_GROUPS:-2}"
export MOE_ROUTER_GROUP_TOPK="${MOE_ROUTER_GROUP_TOPK:-1}"

for shape_var in \
  HIDDEN_SIZE NUM_ATTENTION_HEADS MICRO_BATCH_SIZE GLOBAL_BATCH_SIZE \
  NUM_EXPERTS MOE_ROUTER_TOPK MOE_FFN_HIDDEN_SIZE \
  MOE_ROUTER_NUM_GROUPS MOE_ROUTER_GROUP_TOPK; do
  if ! [[ ${!shape_var} =~ ^[1-9][0-9]*$ ]]; then
    echo "$shape_var must be a positive integer, got '${!shape_var}'" >&2
    exit 64
  fi
done

if (( HIDDEN_SIZE % NUM_ATTENTION_HEADS != 0 )); then
  echo "HIDDEN_SIZE must be divisible by NUM_ATTENTION_HEADS" >&2
  exit 64
fi
if (( NUM_EXPERTS % MOE_ROUTER_NUM_GROUPS != 0 )); then
  echo "NUM_EXPERTS must be divisible by MOE_ROUTER_NUM_GROUPS" >&2
  exit 64
fi
if (( MOE_ROUTER_TOPK > (NUM_EXPERTS / MOE_ROUTER_NUM_GROUPS) * MOE_ROUTER_GROUP_TOPK )); then
  echo "MOE_ROUTER_TOPK exceeds the experts available in selected groups" >&2
  exit 64
fi

export CUDA_DEVICE_MAX_CONNECTIONS=1
export LOG_DIR
rm -rf "$LOG_DIR/torchrun-det" "$LOG_DIR/torchrun-nondet"

bash "$SCRIPT_DIR/run_nsys_breakdown.sh" "$OUT" -- \
  bash -c '
    set -euo pipefail
    uv run --no-sync python -m torch.distributed.run \
      --log-dir "$LOG_DIR/torchrun-$DETERMINISM_PERF_MODE" \
      --tee "0:3,7:3" --redirects "3" --nproc_per_node 8 \
      pretrain_gpt.py \
        --num-layers 4 --hidden-size "$HIDDEN_SIZE" \
        --num-attention-heads "$NUM_ATTENTION_HEADS" \
        --seq-length 256 --max-position-embeddings 256 \
        --micro-batch-size "$MICRO_BATCH_SIZE" \
        --global-batch-size "$GLOBAL_BATCH_SIZE" --train-iters 8 \
        --lr 1e-4 --lr-decay-style constant --lr-decay-iters 100 --min-lr 1e-5 \
        --weight-decay 0 --clip-grad 1.0 \
        --tensor-model-parallel-size 2 --pipeline-model-parallel-size 1 \
        --expert-model-parallel-size 4 --expert-tensor-parallel-size 1 \
        --sequence-parallel \
        --distributed-backend nccl \
        --tokenizer-type NullTokenizer --vocab-size 256 --mock-data --split 1,0,0 \
        --transformer-impl transformer_engine --use-mcore-models \
        --no-gradient-accumulation-fusion --bf16 \
        --normalization RMSNorm --swiglu --disable-bias-linear \
        --position-embedding-type rope --rotary-base 10000 \
        --multi-latent-attention --q-lora-rank 512 --kv-lora-rank 256 \
        --qk-head-dim 128 --qk-pos-emb-head-dim 64 --v-head-dim 128 --qk-layernorm \
        --num-experts "$NUM_EXPERTS" --moe-router-topk "$MOE_ROUTER_TOPK" \
        --moe-grouped-gemm --moe-ffn-hidden-size "$MOE_FFN_HIDDEN_SIZE" \
        --moe-token-dispatcher-type alltoall \
        --moe-router-load-balancing-type seq_aux_loss --moe-aux-loss-coeff 1e-4 \
        --moe-router-score-function sigmoid --moe-router-enable-expert-bias \
        --moe-router-dtype fp32 --moe-router-num-groups "$MOE_ROUTER_NUM_GROUPS" \
        --moe-router-group-topk "$MOE_ROUTER_GROUP_TOPK" \
        --log-interval 1 --eval-iters 0 --eval-interval 10000 --no-load-optim --no-load-rng \
        $([ "$DETERMINISM_PERF_MODE" = det ] && echo --deterministic-mode) \
        --profile --nvtx-ranges --profile-step-start 5 --profile-step-end 7
  '
