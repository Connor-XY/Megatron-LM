#!/bin/bash

# Run two deterministic 32-GPU launches and strictly compare their semantic traces.

set -euo pipefail

MODEL=${1:-}
if [[ "$MODEL" != "dsv3" && "$MODEL" != "nemotron" && "$MODEL" != "dsv4" ]]; then
    echo "Usage: $0 {dsv3|nemotron|dsv4}" >&2
    exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
ROOT_DIR=$(realpath "$SCRIPT_DIR/../../../..")

NUM_NODES=${NUM_NODES:-${SLURM_NNODES:-1}}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
NODE_RANK=${NODE_RANK:-${SLURM_NODEID:-0}}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))

if ((WORLD_SIZE != 32)); then
    echo "Model certification requires exactly 32 ranks, got ${NUM_NODES}x${GPUS_PER_NODE}." >&2
    exit 2
fi

OUTPUT_PATH=${OUTPUT_PATH:-$ROOT_DIR/assets_dir}
TRACE_ROOT=${TRACE_ROOT:-$OUTPUT_PATH/determinism-certification/$MODEL}
CERTIFICATION_TIMEOUT=${CERTIFICATION_TIMEOUT:-900}
CERT_RUN_ID=${RUN_ID:-${SLURM_JOB_ID:-manual}}
STATUS_FILE=$TRACE_ROOT/certification.status

wait_for_file() {
    local path=$1
    local start_time
    start_time=$(date +%s)
    while [[ ! -f "$path" ]]; do
        if (($(date +%s) - start_time > CERTIFICATION_TIMEOUT)); then
            echo "Timed out waiting for $path" >&2
            return 1
        fi
        sleep 1
    done
}

if ((NODE_RANK == 0)); then
    rm -rf "$TRACE_ROOT"
    mkdir -p "$TRACE_ROOT"
    touch "$TRACE_ROOT/setup.ready"
else
    wait_for_file "$TRACE_ROOT/setup.ready"
fi

export PYTHONPATH=$ROOT_DIR
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
export NCCL_ALGO=${NCCL_ALGO:-Ring}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export MAMBA_DETERMINISTIC=1
export TRITON_CACHE_AUTOTUNING=0
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton-cache-$CERT_RUN_ID-$NODE_RANK}
export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-/tmp/cuda-cache-$CERT_RUN_ID-$NODE_RANK}
mkdir -p "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"

PYTHON_BIN=${PYTHON_BIN:-python}
COMMON_ARGS=(
    --seq-length 16
    --max-position-embeddings 16
    --micro-batch-size 1
    --global-batch-size 32
    --train-iters 2
    --lr 1e-4
    --lr-decay-style constant
    --lr-decay-iters 10
    --min-lr 1e-5
    --weight-decay 0
    --clip-grad 1.0
    --seed 1234
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 32
    --expert-tensor-parallel-size 1
    --distributed-backend nccl
    --tokenizer-type NullTokenizer
    --vocab-size 128
    --mock-data
    --split 1,0,0
    --transformer-impl transformer_engine
    --use-mcore-models
    --no-gradient-accumulation-fusion
    --bf16
    --moe-grouped-gemm
    --moe-ffn-hidden-size 2048
    --moe-token-dispatcher-type alltoall
    --moe-router-load-balancing-type seq_aux_loss
    --moe-aux-loss-coeff 1e-4
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-dtype fp32
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
    --ddp-reduce-scatter-hierarchical-group-size 4
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --deterministic-mode
    --determinism-trace-interval 1
    --determinism-trace-tensor-hashes
    --log-interval 1
    --eval-iters 0
    --eval-interval 10000
    --no-load-optim
    --no-load-rng
)

if [[ "$MODEL" == "dsv3" || "$MODEL" == "dsv4" ]]; then
    TRAINING_SCRIPT=pretrain_gpt.py
    MODEL_ARGS=(
        --num-layers 1
        --hidden-size 2048
        --num-attention-heads 16
        --normalization RMSNorm
        --swiglu
        --disable-bias-linear
        --position-embedding-type rope
        --rotary-base 10000
        --multi-latent-attention
        --q-lora-rank 32
        --kv-lora-rank 32
        --qk-head-dim 128
        --qk-pos-emb-head-dim 64
        --v-head-dim 128
        --qk-layernorm
        --num-experts 256
        --moe-router-topk 8
        --moe-router-num-groups 8
        --moe-router-group-topk 4
        --moe-router-pre-softmax
        --moe-router-topk-scaling-factor 2.5
    )
    if [[ "$MODEL" == "dsv4" ]]; then
        # DSV4-style = the DSV3 proxy plus DeepSeek Sparse Attention. Requires
        # the fast_hadamard_transform package in the image (the DSA indexer
        # asserts it). topk 8 stays below the 16-token proxy sequence; a
        # nonzero indexer-loss coefficient keeps the indexer KL loss — and its
        # tensor-parallel score reduction — on the certified training path.
        # indexer head dim must exceed qk-pos-emb-head-dim (64) for the RoPE
        # split; 128 matches the functional DSA config.
        MODEL_ARGS+=(
            --experimental-attention-variant dsa
            --dsa-indexer-n-heads 16
            --dsa-indexer-head-dim 128
            --dsa-indexer-topk 8
            --dsa-indexer-loss-coeff 0.01
            --no-rope-fusion
        )
    fi
else
    TRAINING_SCRIPT=pretrain_hybrid.py
    MODEL_ARGS=(
        --hidden-size 2048
        --ffn-hidden-size 2048
        --num-attention-heads 16
        --group-query-attention
        --num-query-groups 2
        --kv-channels 128
        --hybrid-layer-pattern "M*E"
        --position-embedding-type none
        --spec megatron.core.models.hybrid.hybrid_layer_specs hybrid_stack_spec
        --mamba-num-heads 64
        --mamba-head-dim 64
        --mamba-state-dim 128
        --mamba-num-groups 8
        --normalization RMSNorm
        --squared-relu
        --use-fused-weighted-squared-relu
        --disable-bias-linear
        --untie-embeddings-and-output-weights
        --attention-backend fused
        --attention-dropout 0
        --hidden-dropout 0
        --num-experts 512
        --moe-router-topk 22
        --moe-router-topk-scaling-factor 5.0
        --moe-shared-expert-intermediate-size 1024
        --moe-permute-fusion
    )
fi

run_one() {
    local run_name=$1
    local port=$2
    local trace_dir=$TRACE_ROOT/$run_name
    "$PYTHON_BIN" -m torch.distributed.run \
        --nnodes "$NUM_NODES" \
        --nproc-per-node "$GPUS_PER_NODE" \
        --node-rank "$NODE_RANK" \
        --master-addr "$MASTER_ADDR" \
        --master-port "$port" \
        --log-dir "$TRACE_ROOT/logs-$run_name" \
        --tee "0:3,$((GPUS_PER_NODE - 1)):3" \
        --redirects 3 \
        "$ROOT_DIR/$TRAINING_SCRIPT" \
        "${COMMON_ARGS[@]}" \
        "${MODEL_ARGS[@]}" \
        --determinism-trace-dir "$trace_dir"
}

run_one run-a "$MASTER_PORT"
run_one run-b "$((MASTER_PORT + 1))"

if ((NODE_RANK == 0)); then
    set +e
    "$PYTHON_BIN" "$ROOT_DIR/tools/determinism/certify_traces.py" \
        "$TRACE_ROOT/run-a" "$TRACE_ROOT/run-b" \
        --expected-ranks 32 \
        --expected-iterations 2 \
        --require-event-prefix te.attention.backend.selected \
        --require-collective-prefix moe.ep_ \
        --require-collective-prefix moe.router_expert_bias. \
        --require-collective-prefix data_parallel. \
        --require-dp-fp32-accumulation \
        --require-dp-hierarchical-fp32-accumulation \
        --json >"$TRACE_ROOT/certification.json.tmp"
    certification_exit_code=$?
    set -e
    if ((certification_exit_code == 0)); then
        mv "$TRACE_ROOT/certification.json.tmp" "$TRACE_ROOT/certification.json"
    else
        cat "$TRACE_ROOT/certification.json.tmp" >&2 || true
    fi
    printf '%s\n' "$certification_exit_code" >"$STATUS_FILE"
fi

wait_for_file "$STATUS_FILE"
certification_exit_code=$(cat "$STATUS_FILE")
if ((certification_exit_code != 0)); then
    echo "$MODEL certification failed with exit code $certification_exit_code" >&2
    exit "$certification_exit_code"
fi

chmod -R g+w "$TRACE_ROOT"
echo "MODEL_CERTIFICATION_OK model=$MODEL report=$TRACE_ROOT/certification.json"
