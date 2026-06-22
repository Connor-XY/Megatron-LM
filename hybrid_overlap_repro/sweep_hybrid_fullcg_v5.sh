#!/bin/bash
#SBATCH -A nemotron_sw_pre -p batch -q normal --nodes=16 --gres=gpu:4 --ntasks-per-node=4 -t 2:00:00 -J hybFCGv5
# CONFIRM the PP2 overlap win. 616989 showed steady-state (iter20, 10-iter avg) baseline 935.3 vs overlap 985.4
# TFLOP (+5.4%). Re-run with LOG_INTERVAL=5 + EXIT=35 -> steady points at iters 15/20/25/30/35 (each a 5-iter
# avg) for a robust median. -t 2:00 (each arm ~30min JIT/capture + iters; overlap hit 0:55 walltime before).
# Same-allocation back-to-back (cross-node noise controlled). full_iteration + paged-stash cap2, GBS256.
set -e
BASE=/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/yxu1
cd "$BASE/hybrid-overlap-test" 2>/dev/null || cd "$BASE/Megatron-LM-hybrid-overlap/hybrid_overlap_repro"
export MEGATRON_LM_DIR=$BASE/Megatron-LM-hybrid-overlap PERF_OPT_DIR=$BASE/perf_opt OUTPUT_ROOT=$BASE/results/hybrid_ep_overlap
export IMAGE=gitlab-master.nvidia.com/xren/nemo_megatron_perf_optimization:mcore-moe-pytorch26.04-texren7b2742cd-hybridep42144303-nccl2.30u1719b2bbb-arm
export WANDB_CONSOLE=off WANDB_MODE=offline WANDB_PROJECT=hybrid-ep-overlap-test
export MODEL=hybrid FSDP=0 TP_OVERRIDE=1 SEQ_LEN=3072 GBS=256 EXIT_INTERVAL=35 LOG_INTERVAL=5
export CUDA_GRAPH=full_iteration PAGED_STASH=1 CAP_FACTOR=2
export EP_OVERRIDE=32 PP_OVERRIDE=2 RANKS_PER_NVL_DOMAIN=32
export HYBRID_PATTERN='[ME][ME][ME][M*E][ME][ME]|[ME][M*E][ME][ME][ME][M*E]|[ME][ME][ME][M*E][ME][ME]|[ME][M*E][ME][ME][ME][M*E]/*E'
L=$MEGATRON_LM_DIR/hybrid_overlap_repro/lyris_proxy_train.sh
echo "== HYB FULLCG v5 BASELINE @ $(date) =="
RUNTAG=hyb_fcg5_base OVERLAP=0 bash "$L" || true
echo "== HYB FULLCG v5 OVERLAP  @ $(date) =="
RUNTAG=hyb_fcg5_ovl  OVERLAP=1 bash "$L" || true
echo "== DONE @ $(date) =="
