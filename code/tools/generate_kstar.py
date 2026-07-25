#!/usr/bin/env python3
"""generate_kstar.py — deterministic MP-TSCFLP instance generator with a k* knob.

Produces PSC-format instances whose root covering lower bound (Lemma A4.1.1,
root case)

    k* = max_l k*_I(l) + max_l k*_J(l),
    k*_side(l) = min{ s : sum of the s largest capacities of that side for
                      product l >= D_l }

is controlled exactly: targets tI (factory side) and tJ (depot side) give
k* = tI + tJ.

Mechanism (per side, binding on product 0):
  * tI - 1 facilities share total capacity D_0 - 1 for product 0, split as
    evenly as possible (values base or base+1 with base = (D_0-1)//(tI-1)) —
    large capacities that still do NOT suffice, so k*_I(0) >= tI;
  * every remaining facility gets a moderate capacity in [1, base], so the
    tI-th sorted prefix reaches >= D_0, giving k*_I(0) = tI exactly.
    (Ties with `base` are harmless: the (tI-1)-prefix sum is D_0 - 1 either
    way.) Requires D_0 > (tI-1) and nI >= tI. For tI = 1 one facility gets
    capacity >= D_0.
  * for products l >= 1 every facility gets a capacity in
    [ceil(D_l / t), D_l], so k*_side(l) <= t and the max over l stays at
    product 0's value t (>= is automatic since max >= k*_side(0) = t).
Total capacity always covers demand on both sides, so instances are feasible
(Prop. A1.2, complete stages).

Determinism: every datum comes from random.Random seeded with the parameter
tuple + seed; the same (params, seed) always reproduces byte-identical files
(mirrors the generate_sparse.py discipline of the sister project).
The generator VERIFIES k* on the finished instance and refuses to emit a
file where the check fails. MANIFEST.csv records params + MD5 per file.

Usage:
  python3 generate_kstar.py [--outdir DIR]
      [--nIJ 10,15,20] [--nK 30] [--nL 3] [--targets 2,4,6,8] [--seeds 1,2,3]
"""

import argparse
import csv
import hashlib
import os
import random

COST_MAX = 100      # flow costs in [1, COST_MAX]
FIX_MIN, FIX_MAX = 100, 500   # fixed costs
Q_MIN, Q_MAX = 5, 20          # per-customer demands


def kstar_prefix(vals, D):
    """min s with sum of s largest >= D; float('inf') if impossible; 0 if D<=0."""
    if D <= 0:
        return 0
    acc = 0
    for s, v in enumerate(sorted(vals, reverse=True), start=1):
        acc += v
        if acc >= D:
            return s
    return float("inf")


def side_capacities(rng, m, t, D):
    """Capacities (product 0) for one side of m facilities with k* target t."""
    assert 1 <= t <= m, f"need t <= m (t={t}, m={m})"
    caps = [0] * m
    if t == 1:
        caps[0] = D + rng.randint(0, D)
        for i in range(1, m):
            caps[i] = rng.randint(1, max(1, D // 2))
        return caps
    assert D > t - 1, f"need D_0 > t-1 (D={D}, t={t})"
    base, rem = divmod(D - 1, t - 1)
    for i in range(t - 1):
        caps[i] = base + (1 if i < rem else 0)
    for i in range(t - 1, m):
        caps[i] = rng.randint(1, base)  # moderate: never exceeds a "big" one
    return caps


def gen_instance(nI, nJ, nK, nL, tI, tJ, seed):
    rng = random.Random(f"kstar|{nI}|{nJ}|{nK}|{nL}|{tI}|{tJ}|{seed}")
    q = [[rng.randint(Q_MIN, Q_MAX) for _ in range(nL)] for _ in range(nK)]
    D = [sum(q[k][l] for k in range(nK)) for l in range(nL)]

    b = [[0] * nL for _ in range(nI)]
    p = [[0] * nL for _ in range(nJ)]
    capsI0 = side_capacities(rng, nI, tI, D[0])
    capsJ0 = side_capacities(rng, nJ, tJ, D[0])
    for i in range(nI):
        b[i][0] = capsI0[i]
    for j in range(nJ):
        p[j][0] = capsJ0[j]
    for l in range(1, nL):
        loI = -(-D[l] // tI)  # ceil(D_l / tI)
        loJ = -(-D[l] // tJ)
        for i in range(nI):
            b[i][l] = rng.randint(loI, D[l])
        for j in range(nJ):
            p[j][l] = rng.randint(loJ, D[l])

    f = [rng.randint(FIX_MIN, FIX_MAX) for _ in range(nI)]
    g = [rng.randint(FIX_MIN, FIX_MAX) for _ in range(nJ)]
    c = [[[rng.randint(1, COST_MAX) for _ in range(nJ)] for _ in range(nI)]
         for _ in range(nL)]  # c[l][i][j]
    d = [[[rng.randint(1, COST_MAX) for _ in range(nK)] for _ in range(nJ)]
         for _ in range(nL)]  # d[l][j][k]

    # VERIFY the knob: k* of the finished instance must equal tI + tJ.
    kI = max(kstar_prefix([b[i][l] for i in range(nI)], D[l]) for l in range(nL))
    kJ = max(kstar_prefix([p[j][l] for j in range(nJ)], D[l]) for l in range(nL))
    assert kI == tI and kJ == tJ, (
        f"k* verification failed: got ({kI},{kJ}), target ({tI},{tJ}) "
        f"[nI={nI} nJ={nJ} nK={nK} nL={nL} seed={seed}]")

    return dict(nI=nI, nJ=nJ, nK=nK, nL=nL, q=q, b=b, p=p, f=f, g=g, c=c, d=d,
                kstar=kI + kJ)


def write_psc(inst, path):
    L = inst["nL"]
    lines = [f"{inst['nI']} {inst['nJ']} {inst['nK']} {inst['nL']}"]
    for k in range(inst["nK"]):
        lines.append(" ".join(str(inst["q"][k][l]) for l in range(L)))
    for i in range(inst["nI"]):
        lines.append(" ".join(str(inst["b"][i][l]) for l in range(L)) + f" {inst['f'][i]}")
    for l in range(L):
        for i in range(inst["nI"]):
            lines.append(" ".join(str(inst["c"][l][i][j]) for j in range(inst["nJ"])))
    for j in range(inst["nJ"]):
        lines.append(" ".join(str(inst["p"][j][l]) for l in range(L)) + f" {inst['g'][j]}")
    for l in range(L):
        for j in range(inst["nJ"]):
            lines.append(" ".join(str(inst["d"][l][j][k]) for k in range(inst["nK"])))
    data = ("\n".join(lines) + "\n").encode()
    with open(path, "wb") as fh:
        fh.write(data)
    return hashlib.md5(data).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "..", "data", "kstar"))
    ap.add_argument("--nIJ", default="10,15,20")
    ap.add_argument("--nK", default="30")       # comma list allowed (Phase C grid)
    ap.add_argument("--nL", default="3")        # comma list allowed
    ap.add_argument("--targets", default="2,4,6,8")
    ap.add_argument("--seeds", default="1,2,3")
    args = ap.parse_args()

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    # each --nIJ entry is "m" (nI = nJ = m) or "a:b" (nI = a, nJ = b)
    sizes = [tuple(int(y) for y in x.split(":")) if ":" in x else (int(x), int(x))
             for x in args.nIJ.split(",")]
    nKs = [int(x) for x in args.nK.split(",")]
    nLs = [int(x) for x in args.nL.split(",")]
    targets = [int(x) for x in args.targets.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    rows = []
    for (mI, mJ) in sizes:       # n = mI + mJ
      for nK in nKs:
       for nL in nLs:
        for t in targets:        # tI = tJ = t  => k* = 2t
            if t > min(mI, mJ):
                continue
            for s in seeds:
                inst = gen_instance(mI, mJ, nK, nL, t, t, s)
                fname = f"kstar_nI{mI}_nJ{mJ}_nK{nK}_nL{nL}_tI{t}_tJ{t}_s{s}.txt"
                md5 = write_psc(inst, os.path.join(outdir, fname))
                rows.append([fname, mI, mJ, nK, nL, t, t, s, inst["kstar"], md5])
                print(f"{fname}  kstar={inst['kstar']}  md5={md5}")

    with open(os.path.join(outdir, "MANIFEST.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "nI", "nJ", "nK", "nL", "tI", "tJ", "seed", "kstar", "md5"])
        w.writerows(rows)
    print(f"{len(rows)} instances -> {outdir}/MANIFEST.csv")


if __name__ == "__main__":
    main()
