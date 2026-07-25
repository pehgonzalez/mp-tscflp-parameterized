#!/usr/bin/env python3
"""kstar_check.py — independent covering-bound computation (the covering-count lemma, root case).

Parses PSC instances directly and computes, per product l,
    k*_I(l) = min{ s : sum of s largest b_{.l} >= D_l }   (inf if impossible)
    k*_J(l) analogously on p_{.l},
then kstar = max_l k*_I(l) + max_l k*_J(l).  Cross-checks against the kstar=
field printed by the xp binary (raw one-line records) and writes a CSV.

Usage: python3 kstar_check.py <data_dir> <xp_raw_lines.txt> <out.csv>
Exit 0 iff every instance parses (integers, non-negative) and every kstar
matches xp. Deterministic; no randomness involved.
"""
import sys, os, glob, csv


def kstar_prefix(vals, D):
    if D <= 0:
        return 0
    acc = 0
    for s, v in enumerate(sorted(vals, reverse=True), 1):
        acc += v
        if acc >= D:
            return s
    return float("inf")


def parse_psc(path):
    toks = open(path).read().split()
    it = iter(toks)

    def nx():
        t = next(it)
        v = int(t)          # raises if non-integer -> integrality assert
        assert v >= 0, f"negative value in {path}"
        return v

    nI, nJ, nK, nL = nx(), nx(), nx(), nx()
    q = [[nx() for _ in range(nL)] for _ in range(nK)]
    b, f = [], []
    for _ in range(nI):
        row = [nx() for _ in range(nL)]
        f.append(nx())
        b.append(row)
    for _ in range(nL):
        for _ in range(nI):
            for _ in range(nJ):
                nx()
    p, g = [], []
    for _ in range(nJ):
        row = [nx() for _ in range(nL)]
        g.append(nx())
        p.append(row)
    for _ in range(nL):
        for _ in range(nJ):
            for _ in range(nK):
                nx()
    try:
        next(it)
        raise AssertionError(f"trailing tokens in {path}")
    except StopIteration:
        pass
    return nI, nJ, nK, nL, q, b, p


def main():
    data_dir, raw_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    xp_kstar = {}
    for line in open(raw_path):
        kv = dict(t.split("=", 1) for t in line.split())
        xp_kstar[kv["instance"]] = kv["kstar"]

    rows, bad = [], 0
    for path in sorted(glob.glob(os.path.join(data_dir, "*.txt"))):
        name = os.path.basename(path)
        nI, nJ, nK, nL, q, b, p = parse_psc(path)
        D = [sum(q[k][l] for k in range(nK)) for l in range(nL)]
        kI = [kstar_prefix([b[i][l] for i in range(nI)], D[l]) for l in range(nL)]
        kJ = [kstar_prefix([p[j][l] for j in range(nJ)], D[l]) for l in range(nL)]
        ks = max(kI) + max(kJ)
        ks_str = "inf" if ks == float("inf") else str(ks)
        match = (name in xp_kstar) and (xp_kstar[name] == ks_str)
        if not match:
            bad += 1
            print(f"MISMATCH {name}: py={ks_str} xp={xp_kstar.get(name)}")
        rows.append([name, nI, nJ, nK, nL, ks_str, xp_kstar.get(name, ""),
                     max(kI), max(kJ),
                     ";".join(map(str, kI)), ";".join(map(str, kJ)),
                     ";".join(map(str, D)), int(match)])

    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["instance", "nI", "nJ", "nK", "nL", "kstar_py", "kstar_xp",
                    "kstar_I", "kstar_J", "kI_per_product", "kJ_per_product",
                    "D_per_product", "xp_match"])
        w.writerows(rows)
    print(f"{len(rows)} instances, {bad} mismatches -> {out_path}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
