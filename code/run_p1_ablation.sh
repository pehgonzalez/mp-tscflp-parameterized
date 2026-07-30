#!/bin/bash
# Covering-pruning ablation on the boundary cells (n = 20 and n = 24). Each
# instance is run twice, with the full algorithm and with rule P1 disabled
# through MPTSCFL_NO_P1, so the two records differ only in that rule.
# Resumable, an instance already present in the output directory is skipped.
cd "$(dirname "$0")"
LINES=../results/p1_ablation_lines
mkdir -p "$LINES"
worker() {
  par=$1
  ls data/kstar/kstar_nI10_*.txt data/kstar_boundary/kstar_nI12_*.txt | sort | nl -ba |
  while read i f; do
    [ $((i % 2)) -ne $par ] && continue
    b=$(basename "$f" .txt)
    if [ ! -f "$LINES/$b.on" ]; then
      out=$(./build/xp "$f" 60 -1 2>/dev/null | grep "^instance=")
      [ -n "$out" ] && printf '%s\n' "$out" > "$LINES/$b.on"
    fi
    if [ ! -f "$LINES/$b.off" ]; then
      out=$(MPTSCFL_NO_P1=1 ./build/xp "$f" 60 -1 2>/dev/null | grep "^instance=")
      [ -n "$out" ] && printf '%s\n' "$out" > "$LINES/$b.off"
    fi
    echo "[$(date +%H:%M:%S)] w$par $b done"
  done
}
worker 0 & worker 1 & wait
echo "ABLATION COMPLETE: $(ls $LINES | wc -l) lines"
