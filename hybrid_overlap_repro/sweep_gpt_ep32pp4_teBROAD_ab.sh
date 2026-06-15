#!/bin/bash
#SBATCH -A nemotron_sw_pre -p batch -q short --nodes=32 --gres=gpu:4 --ntasks-per-node=4 -t 1:50:00 -J gptep32pp4teB
# Corrected regime (Sangkug/gao_deng): 1F1B PIPELINE overlap at PP2 + 4 microbatches is where the
# 7-10% win comes from (NOT single-stage A2A). GPT (TE partial CG works), EP64 (MNNVL-domain, hybridep),
# PP2/VPP4, mbs2, GBS512->4 microbatches. On cmh (proper NVL fabric; aws-dfw TCP hangs hybridEP multi-node).
# w/ + w/o overlap same allocation. If GPT overlap WINS -> profile -> port to hybrid.
set -e
BASE=/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/yxu1
cd $BASE/hybrid-overlap-test
export MEGATRON_LM_DIR=$BASE/Megatron-LM-hybrid-overlap PERF_OPT_DIR=$BASE/perf_opt OUTPUT_ROOT=$BASE/results/hybrid_ep_overlap
export IMAGE=gitlab-master.nvidia.com/xren/nemo_megatron_perf_optimization:mcore-moe-pytorch26.04-texren7b2742cd-hybridep42144303-nccl2.30u1719b2bbb-arm
export WANDB_CONSOLE=off WANDB_MODE=offline WANDB_PROJECT=hybrid-ep-overlap-test
export MODEL=gpt FSDP=0 TP_OVERRIDE=1 CUDA_GRAPH=te GBS=512 MBS=1 EXIT_INTERVAL=80
export RECOMPUTE_MODULES='core_attn moe_act shared_experts' OFFLOAD=0 EP_OVERRIDE=32 PP_OVERRIDE=4 NLPVPS=4 NUM_LAYERS=48 SEQ_LEN=3072
export RANKS_PER_NVL_DOMAIN=32 RUNTAG=gpt_ep32pp4_teBROAD
echo "== GPT EP32PP4 BASELINE @ $(date) =="; OVERLAP=0 bash lyris_proxy_train.sh
echo "== GPT EP32PP4 OVERLAP  @ $(date) =="; OVERLAP=1 bash lyris_proxy_train.sh
echo "== DONE @ $(date) =="
