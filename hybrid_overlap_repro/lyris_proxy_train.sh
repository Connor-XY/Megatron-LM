#!/bin/bash

#SBATCH -A coreai_dlalgo_llm
#SBATCH -p gb200
#SBATCH -N 32
#SBATCH --segment=16
#SBATCH -t 00:40:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=4
#SBATCH --overcommit
#SBATCH --exclusive
#SBATCH --comment=metrics
#SBATCH --dependency=singleton
# NOTE: --job-name is supplied by submit_all.sh. The part after the first ':'
#       becomes ${NAME}, which drives both the output dir and the W&B run name.

# =========================================================================
#  Scenario knobs — override per submission with:
#     sbatch --export=ALL,MODEL=gpt,OVERLAP=1,FSDP=0 lyris_proxy_train.sh
# =========================================================================
MODEL="${MODEL:-hybrid}"        # hybrid | gpt
OVERLAP="${OVERLAP:-0}"         # 1 -> --overlap-moe-expert-parallel-comm (A2A / EP-comm overlap)
FSDP="${FSDP:-0}"               # 1 -> Megatron-FSDP (HSDP) config (TP1/GBS128); 0 -> DDP+distopt (TP4/GBS64)
NUM_LAYERS="${NUM_LAYERS:-54}"  # hybrid MUST stay 54 (matches the override pattern); GPT may differ

# ===== env (verbatim from xren's lyris perf config) =====
export NCCL_IB_SL=1
export NCCL_IB_TIMEOUT=19
export UB_TIMEOUT=720
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export TORCHINDUCTOR_WORKER_START=fork
export NCCL_P2P_NET_CHUNKSIZE=2097152
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.9}
export NVTE_CPU_OFFLOAD_V1=1
export TMPDIR=/dev/shm
export PYTHONDONTWRITEBYTECODE=1
export NVTE_USE_CUTLASS_GROUPED_GEMM=0
export NVTE_USE_FAST_MATH=1
export NCCL_SHM_DISABLE=1
export NCCL_PROTO=simple
# hybridEP NVLink-domain size override. On NVL72/MNNVL hardware deep_ep auto-detects
# the whole 64-GPU domain -> A2A stays intra-NVLink (cheap) -> overlap can't win, AND
# EP<64 is rejected ("ranks N not divisible by ranks-per-node 64"). Forcing a smaller
# domain makes A2A cross domains over RDMA (expensive) -> the regime where overlap pays
# off, and makes EP32 valid. Empty/unset -> auto-detect (original behavior).
if [ -n "${RANKS_PER_NVL_DOMAIN:-}" ]; then export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=${RANKS_PER_NVL_DOMAIN}; fi
export NCCL_NVLS_ENABLE=0
export NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=${RANKS_PER_NVL_DOMAIN:-64}
export USE_MNNVL=1

# CUDA_DEVICE_MAX_CONNECTIONS rules (asserted in arguments.py):
#   - TP+SP baseline (ddp, no overlap) requires exactly 1
#   - overlap wants many (32) to parallelize the A2A
#   - Megatron-FSDP requires > 1 (or unset)
# => only the ddp+baseline case may be 1.
if [ "${OVERLAP}" = "1" ] || [ "${FSDP}" = "1" ]; then
    export CUDA_DEVICE_MAX_CONNECTIONS=32
else
    export CUDA_DEVICE_MAX_CONNECTIONS=1
fi

# =========================================================================
#  Paths  — VERIFY/EDIT these on lyris before the first submit.
#  MEGATRON_LM_DIR must point at a checkout of branch yxu1/hybrid-ep-overlap-test.
#  PERF_OPT_DIR reuses xren's perf_optimization checkout for bindpcie + tokenizer.
# =========================================================================
MEGATRON_LM_DIR="${MEGATRON_LM_DIR:-/lustre/fsw/coreai_dlalgo_llm/yxu1/Megatron-LM-hybrid-overlap}"
PERF_OPT_DIR="${PERF_OPT_DIR:-/lustre/fsw/coreai_dlalgo_llm/xren/my_work/nemo_megatron/perf_optimization}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/lustre/fsw/coreai_dlalgo_llm/yxu1/results/hybrid_ep_overlap}"
IMAGE="${IMAGE:-gitlab-master.nvidia.com/xren/nemo_megatron_perf_optimization:mcore-moe-pytorch26.04-texren7b2742cd-hybridep42144303-nccl2.30u1719b2bbb-arm}"
TOKENIZER_MODEL_PATH="${TOKENIZER_MODEL_PATH:-${PERF_OPT_DIR}/megatron/nemotron/nemotron6/tokenizers/multiMixV8.gpt4o_nc_sd.500000.128k.vocab.json}"

# =========================================================================
#  Weights & Biases
#    - Set WANDB_API_KEY in your submitting shell (export it before submit_all.sh),
#      or drop a token in ~/.netrc on the lyris login node.
#    - Default mode is "offline": GB200 compute nodes typically have no internet
#      egress. After the runs, sync with:  wandb sync <RUN_DIR>/wandb/offline-run-*
#      (set WANDB_MODE=online before submit if these nodes do have egress.)
# =========================================================================
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
[ -n "${WANDB_ENTITY:-}" ] && export WANDB_ENTITY
WANDB_PROJECT="${WANDB_PROJECT:-hybrid-ep-overlap-test}"

########################################################
#### CHANGES SHOULD NOT BE NEEDED BEYOND THIS POINT ####
########################################################

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
IFS=':' read -r -a array <<< "${SLURM_JOB_NAME}"
NAME="${array[1]:-${MODEL}_$([ "$FSDP" = 1 ] && echo fsdp || echo ddp)_$([ "$OVERLAP" = 1 ] && echo overlap || echo baseline)}"
NAME="${NAME}${RUNTAG:+_${RUNTAG}}"

if [ -n "${SLURM_JOB_ID:-}" ] ; then
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
    ENV_LOG_FILENAME=${NAME}_${SLURM_JOB_ID}_${DATETIME}.env.log
else
    SCRIPT_PATH=$(realpath "$0")
    ENV_LOG_FILENAME=${NAME}_${DATETIME}.env.log
fi

RUN_DIR="${OUTPUT_ROOT}/${NAME}"
LOGS_DIR="${RUN_DIR}/logs"
DATACACHE_DIR="${RUN_DIR}/../data-cache"
export TRITON_CACHE_DIR="/tmp/triton_cache_${SLURM_NODEID}"

mkdir -p ${LOGS_DIR}
mkdir -p ${DATACACHE_DIR}
mkdir -p ${RUN_DIR}/wandb
mkdir -p ${RUN_DIR}/scripts

# ----- log environment -----
{
  echo "<< START PATHS >>"
  echo "NAME=${NAME}  MODEL=${MODEL}  OVERLAP=${OVERLAP}  FSDP=${FSDP}"
  echo "CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS}"
  echo "IMAGE=${IMAGE}"
  echo "MEGATRON_LM_DIR=${MEGATRON_LM_DIR}"
  echo "PERF_OPT_DIR=${PERF_OPT_DIR}"
  echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
  echo "WANDB_MODE=${WANDB_MODE}  WANDB_PROJECT=${WANDB_PROJECT}"
  echo "<< END PATHS >>"
  echo "<< GIT >>"
  git -C ${MEGATRON_LM_DIR} log --oneline -1
  git -C ${MEGATRON_LM_DIR} status --porcelain --branch
  echo "<< END GIT >>"
} 2>&1 | tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}

cp ${SCRIPT_PATH} ${RUN_DIR}/scripts 2>/dev/null || true

SEQ_LEN=${SEQ_LEN:-8192}
TRAIN_SAMPLES=$(( ${GBS:-128} * (${EXIT_INTERVAL:-40} + 3) ))
LR_WARMUP_SAMPLES=2000
LR_DECAY_SAMPLES=$((TRAIN_SAMPLES-LR_WARMUP_SAMPLES))
LR_WSD_DECAY_SAMPLES=40000

# ---------- per-config knobs (DDP vs FSDP) ----------
if [ "${FSDP}" = "1" ]; then
    # TP4 + sequence-parallel (same model-parallel layout as the ddp cells) so
    # activations shard across the TP group -- avoids the TP1 activation OOM.
    # Megatron-FSDP then shards params/grads/optim across the remaining DP=32.
    # DP=128/4=32 -> GBS128 gives 4 microbatches (matches xren ..._fsdp.sh).
    TP="${TP_OVERRIDE:-4}"
    GBS="${GBS:-128}"
    CKPT_FORMAT="--ckpt-format fsdp_dtensor"
    PERF_OPTIONS=""                    # FSDP config runs without CUDA graphs
    dp_options=" \
        --use-megatron-fsdp \
        --num-distributed-optimizer-instances 2 \
        --outer-dp-sharding-strategy optim \
        --data-parallel-sharding-strategy optim_grads_params \
        --no-gradient-accumulation-fusion \
        --megatron-fsdp-grad-comm-dtype bf16 \
        --megatron-fsdp-main-params-dtype fp32 \
        --megatron-fsdp-main-grads-dtype bf16"
else
    TP="${TP_OVERRIDE:-4}"
    GBS="${GBS:-64}"
    CKPT_FORMAT="--ckpt-format torch_dist"
    dp_options=""
fi

# Sequence-parallel only helps (and is only valid) when TP>1. At TP1 (the
# recommended layout for EP overlap, per Megatron's "don't combine TP with
# overlap_moe_expert_parallel_comm") there is no TP/SP comm to penalize, so
# CUDA_DEVICE_MAX_CONNECTIONS=32 helps the A2A overlap with zero TP cost.
if [ "${TP}" -gt 1 ]; then SP_OPTION=" --sequence-parallel"; else SP_OPTION=""; fi

# ---------- per-model knobs (hybrid vs gpt) ----------
if [ "${MODEL}" = "hybrid" ]; then
    PRETRAIN="pretrain_hybrid.py"
    # Grouped/bracketed pattern: EP/A2A overlap for hybrid is built around layer-group
    # stacks (each [...] bracket = one HybridStack FSDP/overlap unit; cf. PR test
    # patterns "[*E]","[M*E]"). Derived from xren's ungrouped MEMEMEM*E.../*E by
    # bracketing each MoE (E) with its preceding mixers -> [ME][ME][ME][M*E] per block.
    # Same pattern is used for baseline AND overlap so the loss comparison is valid.
    # Physical layers preserved: 6 blocks x (2+2+2+3) = 54.
    # Pattern depends on PP: pipe '|' separators mark pipeline-stage boundaries, and
    # must fall on bracket-group boundaries (a group can't be split across stages).
    # 24 groups total ([ME][ME][ME][M*E] x6). HYBRID_PATTERN env overrides if set.
    if [ -n "${HYBRID_PATTERN:-}" ]; then
        HPAT="${HYBRID_PATTERN}"
    elif [ "${PP_OVERRIDE:-1}" = "2" ]; then   # 2 stages x 12 groups (27 layers each)
        HPAT="[ME][ME][ME][M*E][ME][ME][ME][M*E][ME][ME][ME][M*E]|[ME][ME][ME][M*E][ME][ME][ME][M*E][ME][ME][ME][M*E]/*E"
    else                                        # PP1: single segment, 24 groups
        HPAT="[ME][ME][ME][M*E][ME][ME][ME][M*E][ME][ME][ME][M*E][ME][ME][ME][M*E][ME][ME][ME][M*E][ME][ME][ME][M*E]/*E"
    fi
    model_options=" \
        --is-hybrid-model \
        --position-embedding-type none \
        --hybrid-override-pattern '${HPAT}' \
        --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec"
    CUDA_GRAPH_SCOPE="mamba attn moe_router moe_preprocess"
else
    PRETRAIN="pretrain_gpt.py"
    model_options=" \
        --position-embedding-type rope \
        --rotary-percent 1.0"
    CUDA_GRAPH_SCOPE="attn moe_router moe_preprocess"
fi

# CUDA graph mode: CUDA_GRAPH = none (default) | full_iteration
# Without graphs these runs are CPU-launch-bound (the overlap can't show through).
# full_iteration captures the whole step, removing per-kernel launch overhead.
# It works with EP overlap only after disabling the assert in
# model_chunk_schedule_plan.py (done, per Pingtian). Requires
# --no-check-for-nan-in-loss-and-grad and empty cuda-graph-modules. Applied to
# BOTH baseline and overlap so each pair still differs only by the overlap flag.
RECOMPUTE_OPTIONS=" --recompute-granularity selective --recompute-modules ${RECOMPUTE_MODULES:-moe_act}"
case "${CUDA_GRAPH:-none}" in
    full_iteration)
        export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,graph_capture_record_stream_reuse:True"
        export NCCL_GRAPH_REGISTER=0
        PERF_OPTIONS=" --cuda-graph-impl full_iteration --no-check-for-nan-in-loss-and-grad"
        # Fine-grained activation offloading needs extra flags under full-iteration
        # graphs; drop it (Pingtian's config doesn't offload).
        offload_options=""
        # PAGED_STASH=1 -> pages routed-expert backward activations so the full-iter
        # graph fits (Pingtian's "paged stash + full-iter cg"). Requires a capacity
        # factor, and replaces moe_act recompute (paged stash covers those activations).
        if [ "${PAGED_STASH:-0}" = "1" ]; then
            # capacity-factor requires the TE op fuser
            PERF_OPTIONS="$PERF_OPTIONS --moe-paged-stash --moe-expert-rank-capacity-factor ${CAP_FACTOR:-1.0} --use-transformer-engine-op-fuser"
            RECOMPUTE_OPTIONS=""
        fi ;;
    local)
        # Per-module (local) CUDA graphs: capture mamba/attn/router/experts, leave the
        # A2A dispatch eager (the dispatch is what fails full-iter capture). This is
        # xren's proven baseline CG mode; works with offload + recompute. Removes most
        # CPU-launch overhead without trying to graph the un-capturable dispatch.
        PERF_OPTIONS=" --cuda-graph-impl local --cuda-graph-scope ${CUDA_GRAPH_SCOPE} --te-rng-tracker --cuda-graph-warmup-steps ${CG_WARMUP:-1}"
        offload_options=" --fine-grained-activation-offloading --offload-modules ${OFFLOAD_MODULES:-moe_act}" ;;
    te)
        # Transformer-Engine CUDA graphs, scope = attn + router + preprocess (leaves the
        # A2A dispatch/experts eager). --cuda-graph-warmup-steps primes the stateful
        # dispatch/combine queue eager BEFORE capture (fixes the local-CG 'Pop empty
        # queue' on replay). Mirrors Pingtian/Guihong's working CG config.
        # TE-CG + expandable_segments requires NCCL_GRAPH_REGISTER=0.
        export NCCL_GRAPH_REGISTER=0
        PERF_OPTIONS=" --cuda-graph-impl transformer_engine --cuda-graph-scope ${TE_CG_SCOPE:-${CUDA_GRAPH_SCOPE}} --te-rng-tracker --cuda-graph-warmup-steps ${CG_WARMUP:-1}"
        offload_options=" --fine-grained-activation-offloading --offload-modules ${OFFLOAD_MODULES:-moe_act}" ;;
    *)
        PERF_OPTIONS=""
        offload_options=" --fine-grained-activation-offloading --offload-modules ${OFFLOAD_MODULES:-moe_act}" ;;
esac

if [ "${OFFLOAD:-1}" = "0" ]; then offload_options=""; fi

# A2A / EP-comm overlap toggle. --delay-wgrad-compute is the companion that
# delays the weight-gradient GEMM so the expert A2A can overlap it (mirrors the
# validated DeepSeek-V3 overlap config). Only valid with overlap enabled.
if [ "${OVERLAP}" = "1" ]; then
    overlap_options=" --overlap-moe-expert-parallel-comm $([ "${DELAY_WGRAD:-1}" = "1" ] && echo --delay-wgrad-compute)"
else
    overlap_options=""
fi

options=" \
        --moe-router-score-function sigmoid \
        --moe-grouped-gemm \
        --num-experts 512 \
        --moe-router-topk 22 \
        --moe-aux-loss-coeff 1e-4 \
        --moe-router-topk-scaling-factor 2.5 \
        --moe-router-enable-expert-bias \
        --moe-router-dtype fp32 \
        --moe-router-load-balancing-type seq_aux_loss \
        --moe-shared-expert-intermediate-size 10240 \
        --moe-latent-size 2048 \
        --moe-permute-fusion \
        --moe-token-dispatcher-type flex \
        --moe-flex-dispatcher-backend ${DISPATCHER_BACKEND:-hybridep} \
        --moe-hybridep-num-sms ${NUM_SMS:-32} \
        --moe-router-fusion \
        --moe-router-force-load-balancing \
        \
        --num-workers 0 \
        --disable-gloo-process-groups \
        ${CKPT_FORMAT} \
        --ckpt-fully-parallel-save \
        --ckpt-fully-parallel-load \
        --ckpt-assume-constant-structure \
        \
        --squared-relu \
        --no-mmap-bin-files \
        --distributed-timeout-minutes 30 \
        --exit-duration-in-mins 1430 \
        --no-create-attention-mask-in-dataloader \
        \
        --overlap-grad-reduce \
        --overlap-param-gather \
        --tensor-model-parallel-size ${TP} \
        --expert-model-parallel-size ${EP_OVERRIDE:-64} \
        --expert-tensor-parallel-size 1 \
        --pipeline-model-parallel-size ${PP_OVERRIDE:-1} \
        ${NLPVPS:+--num-layers-per-virtual-pipeline-stage ${NLPVPS}} \
        --use-distributed-optimizer \
        --high-priority-stream-groups ep \
        --ddp-num-buckets 24 \
        --grad-reduce-in-bf16 \
        ${SP_OPTION} \
        \
        --mock-data \
        --untie-embeddings-and-output-weights \
        --init-method-std 0.0099 \
        --num-layers ${NUM_LAYERS} \
        --hidden-size 8192 \
        --num-attention-heads 64 \
        --group-query-attention \
        --num-query-groups 8 \
        --ffn-hidden-size 5120 \
        --kv-channels 128 \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${SEQ_LEN} \
        --train-samples ${TRAIN_SAMPLES} \
        --lr-decay-style WSD \
        --lr-warmup-samples ${LR_WARMUP_SAMPLES} \
        --lr-decay-samples ${LR_DECAY_SAMPLES} \
        --lr-wsd-decay-style minus_sqrt \
        --lr-wsd-decay-samples ${LR_WSD_DECAY_SAMPLES} \
        --data-cache-path ${DATACACHE_DIR} \
        --tiktoken-pattern v2 \
        --tokenizer-type TikTokenizer \
        --tokenizer-model ${TOKENIZER_MODEL_PATH} \
        --distributed-backend nccl \
        --micro-batch-size ${MBS:-1} \
        --global-batch-size ${GBS} \
        --lr 8.0e-4 \
        --min-lr 8.0e-6 \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --disable-bias-linear \
        --normalization RMSNorm \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --log-interval ${LOG_INTERVAL:-10} \
        --timing-log-level 2 \
        --log-params-norm \
        --log-num-zeros-in-grad \
        --log-throughput \
        --log-device-memory-used \
        --eval-interval 250 \
        --eval-iters 0 \
        --bf16 \
        --use-mcore-models \
        --transformer-impl transformer_engine \
        --enable-experimental \
        --manual-gc \
        --manual-gc-interval 100 \
        --use-fused-weighted-squared-relu \
        --cross-entropy-loss-fusion \
        --cross-entropy-fusion-impl native \
        ${RECOMPUTE_OPTIONS} \
        --rerun-mode disabled \
        ${NAN_OFF:+--no-check-for-nan-in-loss-and-grad} \
        --exit-interval ${EXIT_INTERVAL:-250}"

mxfp8_options=" \
    --moe-router-padding-for-quantization \
    --fp8-format e4m3 \
    --fp8-recipe mxfp8 \
    --reuse-grad-buf-for-mxfp8-param-ag \
    --fp8-param-gather"

# MTP=1: --overlap-moe-expert-parallel-comm asserts mtp_num_layers <= 1.
# Kept at 1 for ALL runs (incl. baseline) so each baseline/overlap pair is comparable.
mtp_options=" \
    --mtp-num-layers 1 \
    --mtp-use-repeated-layer \
    --calculate-per-token-loss \
    --mtp-loss-scaling-factor 0.3"

wandb_options=" \
    --wandb-project ${WANDB_PROJECT} \
    --wandb-exp-name ${NAME} \
    --wandb-save-dir ${RUN_DIR}/wandb"

# Optional nsys profiling of a few steps (PROFILE=1). Only rank 0 captures
# (--profile-ranks 0); other ranks' cudaProfilerApi range never opens so their
# traces are empty. \${SLURM_PROCID} is escaped to expand per-task in the inner sh.
if [ "${PROFILE:-0}" = "1" ]; then
    profile_options=" --profile --profile-step-start ${PROFILE_START:-30} --profile-step-end ${PROFILE_END:-32} --profile-ranks 0"
    nsys_prefix="nsys profile -s none -t nvtx,cuda --cuda-graph-trace=node --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop -o ${RUN_DIR}/${NAME}_rank\${SLURM_PROCID} "
else
    profile_options=""
    nsys_prefix=""
fi

# 'set -f' disables filename globbing in the inner shell so the bracketed/'*'
# hybrid-override-pattern (e.g. [M*E], /*E) is passed to python verbatim.
# deep_ep JIT (.cu/.so) scratch: per-job Lustre dir (ample quota). /dev/shm overflowed
# at 128-rank scale (3-arch fatbins x 4 ranks/node -> EDQUOT). cf job 525156.
DEEPEP_HOST_DIR="/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/yxu1/deepep_jit/job_${SLURM_JOB_ID:-manual}"
mkdir -p "${DEEPEP_HOST_DIR}"
run_cmd="set -f; export TMPDIR=/root/.deepep; ${nsys_prefix}python -u ${MEGATRON_LM_DIR}/${PRETRAIN} \
    ${options} ${model_options} ${PERF_OPTIONS} ${offload_options} ${dp_options} \
    ${mxfp8_options} ${mtp_options} ${overlap_options} ${profile_options} ${wandb_options}"

srun -l --mpi=pmix \
    --container-image "${IMAGE}" \
    --container-mounts "/lustre:/lustre,${DEEPEP_HOST_DIR}:/root/.deepep" \
    --output="${LOGS_DIR}/%x_%j_${DATETIME}.log" \
    ${PERF_OPT_DIR}/megatron/nemotron/nemotron6/job_launch/bindpcie \
    sh -c "${run_cmd}"
