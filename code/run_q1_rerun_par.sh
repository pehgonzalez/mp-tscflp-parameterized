#!/bin/bash
# Rerun da campanha Q1 completa (460 runs, 60 s, k=-1) nesta maquina.
# Retomavel: pula instancias ja presentes no arquivo de linhas.
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
echo "RERUN CONCLUIDO: $(ls $LINES | wc -l) linhas"
