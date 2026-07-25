#!/usr/bin/env python3
"""assemble_q1.py — join xp one-line records with MANIFEST.csv into q1_xp.csv
and print the Q1 group summary (solved counts, median time over solved,
censoring) used by results/q1_summary.md.

Usage: python3 assemble_q1.py <lines_dir> <manifest.csv> <out.csv> [<manifest2.csv> ...]
"""
import sys, os, glob, csv, statistics

FIELDS = ["instance", "nI", "nJ", "nK", "nL", "tI", "tJ", "seed", "kstar_target",
          "md5", "status", "obj", "k_used", "kstar", "nodes", "time", "agg",
          "timeout", "p1", "p2i", "p2b", "p3", "rneg", "cap"]


def main():
    lines_dir, out_csv = sys.argv[1], sys.argv[3]
    manifests = [sys.argv[2]] + sys.argv[4:]
    meta = {}
    for mpath in manifests:
        for r in csv.DictReader(open(mpath)):
            meta[r["file"]] = r

    rows = []
    for lf in sorted(glob.glob(os.path.join(lines_dir, "*.line"))):
        txt = open(lf).read().strip()
        if not txt.startswith("instance="):
            print(f"WARN unparsable {lf}: {txt[:80]!r}"); continue
        kv = dict(t.split("=", 1) for t in txt.split())
        m = meta.get(kv["instance"])
        if m is None:
            print(f"WARN no manifest entry for {kv['instance']}"); continue
        rows.append([kv["instance"], m["nI"], m["nJ"], m["nK"], m["nL"], m["tI"],
                     m["tJ"], m["seed"], m["kstar"], m["md5"], kv["status"],
                     kv["obj"], kv["k_used"], kv["kstar"], kv["nodes"], kv["time"],
                     kv["agg"], kv["timeout"], kv["p1"], kv["p2i"], kv["p2b"],
                     kv["p3"], kv["rneg"], kv["cap"]])
        assert kv["kstar"] == m["kstar"], f"kstar mismatch on {kv['instance']}"
    rows.sort(key=lambda r: r[0])
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(FIELDS); w.writerows(rows)
    print(f"{len(rows)} rows -> {out_csv}")

    groups = {}
    for r in rows:
        key = (int(r[13]) if r[13] != "inf" else -1, int(r[1]), int(r[2]),
               int(r[3]), int(r[4]))  # (kstar, nI, nJ, nK, nL)
        groups.setdefault(key, []).append(r)
    print("kstar nI nJ nK nL | n solved censored | med_time_solved med_nodes")
    for key in sorted(groups):
        g = groups[key]
        solved = [r for r in g if r[10] == "OPTIMAL"]
        cens = len(g) - len(solved)
        mt = f"{statistics.median(float(r[15]) for r in solved):.3f}" if solved else "-"
        mn = f"{statistics.median(int(r[14]) for r in solved):.0f}" if solved else "-"
        print(f"{key[0]:5d} {key[1]:3d} {key[2]:3d} {key[3]:4d} {key[4]:3d} | "
              f"{len(g):2d} {len(solved):2d} {cens:2d} | {mt:>10s} {mn:>10s}")


if __name__ == "__main__":
    main()
