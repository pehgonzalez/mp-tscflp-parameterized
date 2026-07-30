#!/bin/bash
# Rerun of the full Q1 campaign (460 runs, 60 s, k=-1) on this machine.
# Resumable, instances already present in the lines directory are skipped.
cd "$(dirname "$0")"
LINES=../results/q1_rerun_lines
mkdir -p "$LINES"
worker() {
  par=$1
  ls data/instances_kstar/kstar_*.txt data/kstar/kstar_*.txt data/kstar_boundary/kstar_*.txt | sort | nl -ba | while read i f; do
    [ $((i % 2)) -ne $par ] && continue
    b=$(basename "$f" .txt)
    [ -f "$LINES/$b.line" ] && continue
    line=$(./build/xp "$f" 60 -1 2>/dev/null | grep "^instance=")
    if [ -n "$line" ]; then
      echo "$line" > "$LINES/$b.line"
      echo "[$(date +%H:%M:%S)] w$par $b $(echo "$line" | grep -o 'status=[A-Z]*') $(echo "$line" | grep -o 'time=[0-9.]*')"
    fi
  done
}
worker 0 & worker 1 & wait
echo "RERUN COMPLETE: $(ls $LINES | wc -l) lines"
