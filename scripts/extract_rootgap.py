#!/usr/bin/env python3
"""
Extracts, from the Gurobi logs of the benchmark campaign, the root
relaxation objective and the number of explored nodes per run, and
writes results/mauri_rootgap.csv. No rerun is involved, only reading
the logs archived next to the raw results. The mediation coefficients
quoted in the paper are recomputable from this CSV.

Usage: python3 scripts/extract_rootgap.py <log-directory>
"""
import csv, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")

def main():
    logdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "logs_campanha")
    rows = list(csv.DictReader(open(os.path.join(RES, "mauri_mip.csv"))))
    out = []
    for r in rows:
        base = os.path.basename(r["gurobi_log"])
        path = os.path.join(logdir, base)
        txt = open(path, encoding="utf-8", errors="ignore").read()
        m_root = re.search(r"Root relaxation: objective ([0-9.e+]+)", txt)
        m_nodes = re.findall(r"Explored (\d+) nodes", txt)
        m_best = re.search(r"Best objective ([0-9.e+-]+), best bound ([0-9.e+-]+)", txt)
        out.append({
            "instance": r["instance"], "obj": r["obj"], "gap": r["gap"],
            "status": r["status"],
            "root_lp": m_root.group(1) if m_root else "",
            "nodes": m_nodes[-1] if m_nodes else "",
            "best_obj": m_best.group(1) if m_best else "",
            "best_bound": m_best.group(2) if m_best else "",
        })
    dest = os.path.join(RES, "mauri_rootgap.csv")
    with open(dest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        for o in out:
            w.writerow(o)
    print(dest, len(out), "rows")

if __name__ == "__main__":
    main()
