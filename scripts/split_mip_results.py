#!/usr/bin/env python3
"""Split the combined runner output into the three CSVs the tables read.

The Windows campaign runners append every run to one combined file,
results/mip_results.csv (900 rows, compact model and Benders, controlled
family and benchmark). scripts/make_paper_tables.py reads three split
files instead, controlled_mip.csv, controlled_bd.csv and mauri_mip.csv,
and this script produces them from the combined file, closing the gap in
the pipeline. Controlled-family instances are the generated kstar files
and benchmark instances are the PSC ones, method 0 is the compact model
and method 1 the branch-and-Benders-cut, matching the runner contract.

Usage: python3 scripts/split_mip_results.py [results/mip_results.csv]
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RES, "mip_results.csv")
    rows = list(csv.DictReader(open(src, newline="")))
    header = list(rows[0].keys())
    out = {
        "controlled_mip.csv": [r for r in rows
                               if r["instance"].startswith("kstar") and r["method"] == "0"],
        "controlled_bd.csv": [r for r in rows
                              if r["instance"].startswith("kstar") and r["method"] == "1"],
        "mauri_mip.csv": [r for r in rows
                          if r["instance"].startswith("PSC") and r["method"] == "0"],
    }
    total = sum(len(v) for v in out.values())
    leftover = len(rows) - total
    for name, sel in out.items():
        path = os.path.join(RES, name)
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=header)
            w.writeheader()
            w.writerows(sel)
        print("%s: %d rows" % (name, len(sel)))
    print("leftover rows not in any split (benchmark Benders etc.): %d" % leftover)


if __name__ == "__main__":
    main()
