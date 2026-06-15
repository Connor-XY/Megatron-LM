#!/bin/bash
J="$1"
R=/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/yxu1/results/hybrid_ep_overlap
prev=""; ticks=0
logs() { ls -t $R/*/logs/*_${J}_*.log 2>/dev/null | grep -v env.log; }
while squeue -j "$J" -h -o '%i' 2>/dev/null | grep -q "$J"; do
  for f in $(logs); do
    e=$(grep -aoE 'Disk quota exceeded|Failed to compile|CUDA out of memory|CANCELLED AT|AttributeError|AssertionError|RuntimeError: [^ ]+' "$f" 2>/dev/null | tail -1)
    [ -n "$e" ] && [ "$e|$(basename $f)" != "$prev" ] && { echo "ERR [$(basename $f)] $e"; prev="$e|$(basename $f)"; }
  done
  ticks=$((ticks+1))
  if [ $((ticks % 6)) -eq 1 ]; then
    f=$(logs | head -1)
    echo "HB [${f##*/}] $(grep -aE 'elapsed time per iteration' "$f" 2>/dev/null | tail -1 | grep -oE 'iteration +[0-9]+/[0-9]+|throughput per GPU \(TFLOP/s/GPU\): *[0-9.]+' | tr '\n' ' ')${f:+ }$( [ -z "$(grep -aE 'elapsed time per iteration' "$f" 2>/dev/null)" ] && echo '(capturing/no-iter)')"
  fi
  sleep 120
done
echo "=== $J LEFT QUEUE @ $(date) ==="
for f in $(logs); do echo "== ${f##*/} =="; grep -aoE 'Disk quota exceeded|Failed to compile|CUDA out of memory|CANCELLED AT|DONE @|throughput per GPU \(TFLOP/s/GPU\): *[0-9.]+|elapsed time per iteration \(ms\): *[0-9.]+' "$f" 2>/dev/null | tail -8; done
