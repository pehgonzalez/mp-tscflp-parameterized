#!/bin/bash
# Campanha da revisao: extensao de fronteira com orcamento de 600s e 10 seeds
# por celula (Required 2 do parecer). Resumivel: pula instancias ja no CSV.
cd "$(dirname "$0")"
OUT=../results/q1_boundary600.csv
mkdir -p ../results
if [ ! -f "$OUT" ]; then
  echo "instance,nI,nJ,nK,nL,tI,tJ,seed,status,obj,k_used,kstar,nodes,time,timeout,p1,p2i,p2b,p3" > "$OUT"
fi
run_one() {
  f="$1"; b=$(basename "$f" .txt)
  grep -q "^$b," "$OUT" && return
  line=$(./build/xp "$f" 600 -1 2>/dev/null | grep "^instance=")
  get(){ echo "$line" | tr ' ' '\n' | grep "^$1=" | cut -d= -f2; }
  # campos do nome: kstar_nI10_nJ10_nK30_nL3_tI2_tJ2_s1
  IFS=_ read -r _ nI nJ nK nL tI tJ sd <<< "$b"
  echo "$b,${nI#nI},${nJ#nJ},${nK#nK},${nL#nL},${tI#tI},${tJ#tJ},${sd#s},$(get status),$(get obj),$(get k_used),$(get kstar),$(get nodes),$(get time),$(get timeout),$(get p1),$(get p2i),$(get p2b),$(get p3)" >> "$OUT"
  echo "[$(date +%H:%M:%S)] $b $(get status) t=$(get time)"
}
export -f run_one OUT 2>/dev/null
ls data/boundary600/kstar_*.txt | sort | while read f; do run_one "$f"; done
echo "CAMPANHA CONCLUIDA $(date)"
