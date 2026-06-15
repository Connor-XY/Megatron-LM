#!/bin/bash
#SBATCH -A nemotron_sw_pre -p batch -q short --nodes=16 --gres=gpu:4 --ntasks-per-node=4 -t 1:50:00 -J hybpp2v2mfixRR
# TRACK 1 close-out: hybrid EP32/PP2 te-CG BROAD + VPP2 + recompute, nvl32. EP A2A overlap requires VPP when PP>1
# (transformer_config:2404). For hybrid, NLPVPS is FORBIDDEN with --hybrid-layer-pattern; VPP is set by making
# the pattern's |-segment count a multiple of PP (>1): 4 segments / PP2 = VPP2. 24 groups -> 6 groups/segment,
# split on bracket boundaries. Mirrors GPT 548668 (which used VPP). A/B baseline+overlap, one allocation.
set -e
BASE=/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/yxu1
cd $BASE/hybrid-overlap-test
export MEGATRON_LM_DIR=$BASE/Megatron-LM-hybrid-overlap PERF_OPT_DIR=$BASE/perf_opt OUTPUT_ROOT=$BASE/results/hybrid_ep_overlap
export IMAGE=gitlab-master.nvidia.com/xren/nemo_megatron_perf_optimization:mcore-moe-pytorch26.04-texren7b2742cd-hybridep42144303-nccl2.30u1719b2bbb-arm
export WANDB_CONSOLE=off WANDB_MODE=offline WANDB_PROJECT=hybrid-ep-overlap-test
export MODEL=hybrid FSDP=0 TP_OVERRIDE=1 CUDA_GRAPH=te GBS=512 EXIT_INTERVAL=80
export RECOMPUTE_MODULES='core_attn moe_act shared_experts' OFFLOAD=0 EP_OVERRIDE=32 PP_OVERRIDE=2 SEQ_LEN=3072
export HYBRID_PATTERN='[ME][ME][ME][M*E][ME][ME]|[ME][M*E][ME][ME][ME][M*E]|[ME][ME][ME][M*E][ME][ME]|[ME][M*E][ME][ME][ME][M*E]/*E'
export RANKS_PER_NVL_DOMAIN=32 RUNTAG=hybrid_ep32pp2vpp2_mambafix_rr
echo "== HYBRID PP2VPP2 teBROAD BASELINE @ \$(date) =="; OVERLAP=0 bash lyris_proxy_train.sh
echo "== HYBRID PP2VPP2 teBROAD OVERLAP  @ \$(date) =="; OVERLAP=1 bash lyris_proxy_train.sh
echo "== DONE @ \$(date) =="
