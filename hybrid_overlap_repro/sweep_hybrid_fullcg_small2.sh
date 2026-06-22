#!/bin/bash
#SBATCH -A nemotron_sw_pre -p batch -q normal --nodes=2 --gres=gpu:4 --ntasks-per-node=4 -t 0:40:00 -J hybFCGsm2
# SMALL-SCALE full_iteration validation, hybridEP-valid: EP8==NVL_domain8 (hybridEP needs EP==detected domain;
# 2 nodes => domain 8 => EP8/PP1). 64 experts/EP8 = 8 experts/rank. PP1 => no pipeline (PP2 win is the 16-node
# job's role); validates the thing that matters: full-iter CAPTURE past the timing-log-level fix + mamba clean +
# overlap composes. Pattern = the PP2 pattern with pipes stripped (single PP stage).
set -e
BASE=/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/yxu1
cd "$BASE/hybrid-overlap-test" 2>/dev/null || cd "$BASE/Megatron-LM-hybrid-overlap/hybrid_overlap_repro"
export MEGATRON_LM_DIR=$BASE/Megatron-LM-hybrid-overlap PERF_OPT_DIR=$BASE/perf_opt OUTPUT_ROOT=$BASE/results/hybrid_ep_overlap
export IMAGE=gitlab-master.nvidia.com/xren/nemo_megatron_perf_optimization:mcore-moe-pytorch26.04-texren7b2742cd-hybridep42144303-nccl2.30u1719b2bbb-arm
export WANDB_CONSOLE=off WANDB_MODE=offline WANDB_PROJECT=hybrid-ep-overlap-test
export MODEL=hybrid FSDP=0 TP_OVERRIDE=1 SEQ_LEN=3072 GBS=64 EXIT_INTERVAL=15
export NUM_EXPERTS=64 TOPK=6 LR_WARMUP_SAMPLES=64
export CUDA_GRAPH=full_iteration PAGED_STASH=1 CAP_FACTOR=2
export EP_OVERRIDE=8 PP_OVERRIDE=1 RANKS_PER_NVL_DOMAIN=8
# 24-group pattern (== the PP2 sweep's) with PP pipes removed -> single PP stage
export HYBRID_PATTERN="$(echo '[ME][ME][ME][M*E][ME][ME]|[ME][M*E][ME][ME][ME][M*E]|[ME][ME][ME][M*E][ME][ME]|[ME][M*E][ME][ME][ME][M*E]/*E' | tr -d '|')"
L=$MEGATRON_LM_DIR/hybrid_overlap_repro/lyris_proxy_train.sh
echo "== pattern: $HYBRID_PATTERN =="
echo "== HYB FULLCG small2 BASELINE @ $(date) =="
RUNTAG=hyb_fcgsm2_base OVERLAP=0 bash "$L" || true
echo "== HYB FULLCG small2 OVERLAP  @ $(date) =="
RUNTAG=hyb_fcgsm2_ovl  OVERLAP=1 bash "$L" || true
echo "== DONE @ $(date) =="
