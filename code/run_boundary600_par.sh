#!/bin/bash
cd "$(dirname "$0")"
OUT=../results/q1_boundary600.csv
mkdir -p ../results
[ -f "$OUT" ] || echo "instance,nI,nJ,nK,nL,tI,tJ,seed,status,obj,k_used,kstar,nodes,time,timeout,p1,p2i,p2b,p3" > "$OUT"
worker() {
  par=$1
  ls data/boundary600/kstar_*.txt | sort | while read f; do
    b=$(basename "$f" .txt)
    sd=${b##*_s}
    [ $((sd % 2)) -ne $par ] && continue
    grep -q "^$b," "$OUT" && continue
    line=$(./build/xp "$f" 600 -1 2>/dev/null | grep "^instance=")
    get(){ echo "$line" | tr ' ' '\n' | grep "^$1=" | cut -d= -f2; }
    IFS=_ read -r _ nI nJ nK nL tI tJ s <<< "$b"
    flock "$OUT" -c "echo '$b,${nI#nI},${nJ#nJ},${nK#nK},${nL#nL},${tI#tI},${tJ#tJ},${s#s},$(get status),$(get obj),$(get k_used),$(get kstar),$(get nodes),$(get time),$(get timeout),$(get p1),$(get p2i),$(get p2b),$(get p3)' >> '$OUT'"
    echo "[$(date +%H:%M:%S)] w$par $b $(get status) t=$(get time)"
  done
}
worker 0 & worker 1 & wait
echo "CAMPAIGN FINISHED $(date)"
