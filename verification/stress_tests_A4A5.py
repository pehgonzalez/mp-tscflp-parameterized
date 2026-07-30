"""
Independent stress tests complementing verify_A4_* and verify_A5_*.

TEST 1 (parking/saturation in the numeric-cell reduction, via an
  INDEPENDENT LP): (C1)-(C4) in inequality form ALLOW flow x entering a
  closed depot (waste), whereas the MCMF oracle forbids this
  structurally (arc Din->Dout with cap 0); battery [A] of
  verify_A5_R8ii does not exercise this case directly. Here:
  brute force over ALL designs of the image instances of the
  |K| = |L| = 1 cell reduction using scipy.linprog on the ORIGINAL LP in
  inequality form (which allows parking), checking:
    - OPT_LP == t* (min cover) when coverable; OPT_LP >= Q+1 otherwise;
    - "<=>" for every t in {1..m};
    - per feasible design: v_LP == Q * u(Z) (the lemma's closed form).
  Families: small exhaustive + adversarial (empty set, uncovered
  element, duplicate sets, cover only with all sets).

TEST 2 (protected witness of the B&B under massive ties):
  B&B with P1+P2+P3 vs brute force on instances with maximal ties:
  all values in {0,1} (or {0,1,2}), ZERO capacities allowed
  (open facility that never carries flow -- maximum stress for P3),
  f=g=0, c=d=0, and combinations. All cardinalities k.

TEST 3 (validity of the Benders cut on ALL designs + NUMERIC
  rays):
  On small instances: optimal dual u* obtained at a feasible generator
  design; checks the cut on ALL 2^n designs (feasible: v >= rhs;
  infeasible: unbounded dual -- verified via status). And the NON
  tautological test of rays r1/r2: u* + theta*r_i must be dual-feasible
  (all |I||J|+|J||K| constraints verified numerically) and the
  objective must grow by exactly theta*(D_l - cap_side).
"""

import itertools
import random
import os, sys

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_mp_tscfl import gen_instance, demand_total, routing_value
from verify_A5_R8ii import build_reduction, min_cover, uncovered
from verify_A4_xp_bb import DesignCache, brute_force, bnb

INF = float("inf")
FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("FAILURE:", msg)


# ---------------------------------------------------------------------------
# TEST 1
# ---------------------------------------------------------------------------

def lp_cell(inst, y, z):
    """Residual LP |K|=|L|=1 in INEQUALITY form (allows parking)."""
    nI, nJ = inst["nI"], inst["nJ"]
    D = inst["q"][0][0]
    nx = nI * nJ
    nvar = nx + nJ
    cost = [inst["c"][i][j][0] for i in range(nI) for j in range(nJ)] \
        + [inst["d"][j][0][0] for j in range(nJ)]
    A_ub, b_ub = [], []
    row = [0.0] * nvar                    # C1
    for j in range(nJ):
        row[nx + j] = -1.0
    A_ub.append(row); b_ub.append(-float(D))
    for j in range(nJ):                   # C2
        row = [0.0] * nvar
        row[nx + j] = 1.0
        for i in range(nI):
            row[i * nJ + j] = -1.0
        A_ub.append(row); b_ub.append(0.0)
    for i in range(nI):                   # C3
        row = [0.0] * nvar
        for j in range(nJ):
            row[i * nJ + j] = 1.0
        A_ub.append(row); b_ub.append(float(inst["b"][i][0] * y[i]))
    for j in range(nJ):                   # C4
        row = [0.0] * nvar
        row[nx + j] = 1.0
        A_ub.append(row); b_ub.append(float(inst["p"][j][0] * z[j]))
    res = linprog(cost, A_ub=A_ub, b_ub=b_ub,
                  bounds=[(0, None)] * nvar, method="highs")
    if res.status == 2:
        return False, None
    return True, res.fun


def attack1_family(nU, fam, counters):
    inst, Q = build_reduction(nU, fam)
    m = len(fam)
    nI, nJ = inst["nI"], inst["nJ"]
    full_I = (1 << nI) - 1
    opt = INF
    for my in range(1 << nI):
        y = [(my >> i) & 1 for i in range(nI)]
        for mz in range(1 << nJ):
            z = [(mz >> j) & 1 for j in range(nJ)]
            feas, v = lp_cell(inst, y, z)
            pred_feas = (my == full_I) and (mz != 0)
            check(feas == pred_feas,
                  f"A1 LP feas fam={fam} nU={nU} my={my} mz={mz}: "
                  f"lp={feas} predicted={pred_feas}")
            counters["desenhos"] += 1
            if feas:
                pred_v = Q * uncovered(nU, fam, mz)
                check(abs(v - pred_v) <= 1e-6,
                      f"A1 LP lemma fam={fam} nU={nU} mz={mz}: "
                      f"v_lp={v} predicted={pred_v} (parking?)")
                total = bin(mz).count("1") + v
                opt = min(opt, total)
    tstar = min_cover(nU, fam)
    if tstar is not None:
        check(abs(opt - tstar) <= 1e-6,
              f"A1 OPT fam={fam} nU={nU}: {opt} != t*={tstar}")
    else:
        check(opt >= Q + 1 - 1e-6, f"A1 OPT fam={fam}: {opt} < Q+1")
    for t in range(1, m + 1):
        counters["iff"] += 1
        check((tstar is not None and tstar <= t) == (opt <= t + 1e-6),
              f"A1 <=> fam={fam} t={t}")


def attack1():
    counters = {"desenhos": 0, "iff": 0, "familias": 0}
    # small exhaustive (|U| <= 3, m <= 3)
    for nU in range(1, 4):
        subsets = list(range(1 << nU))
        for m in range(1, 4):
            for fam in itertools.combinations(subsets, m):
                counters["familias"] += 1
                attack1_family(nU, fam, counters)
    # targeted adversarial (nU=4)
    adv = [
        (4, (0, 0b1111, 0b0001)),          # empty set + universe
        (4, (0b0011, 0b0101, 0b0110)),     # element 3 uncovered
        (4, (0b0111, 0b0111, 0b1000)),     # duplicates
        (4, (0b1000, 0b0100, 0b0010, 0b0001)),  # covers only with all (t*=4=m)
        (4, (0b1111,)),                    # m=1, t*=1
    ]
    for nU, fam in adv:
        counters["familias"] += 1
        attack1_family(nU, fam, counters)
    return counters


# ---------------------------------------------------------------------------
# TEST 2
# ---------------------------------------------------------------------------

def attack2(n=240, seed0=555000):
    rng = random.Random(seed0)
    n_checks = 0
    for t in range(n):
        nI = rng.randint(1, 4)
        nJ = rng.randint(1, 4)
        nK = rng.randint(1, 3)
        nL = rng.randint(1, 2)
        mode = t % 4
        vmax = 1 if mode < 2 else 2
        inst = {
            "nI": nI, "nJ": nJ, "nK": nK, "nL": nL,
            "f": [rng.randint(0, vmax) for _ in range(nI)],
            "g": [rng.randint(0, vmax) for _ in range(nJ)],
            "c": [[[rng.randint(0, vmax) for _ in range(nL)]
                   for _ in range(nJ)] for _ in range(nI)],
            "d": [[[rng.randint(0, vmax) for _ in range(nL)]
                   for _ in range(nK)] for _ in range(nJ)],
            # ZERO capacities allowed (useless open facilities)
            "b": [[rng.randint(0, 3) for _ in range(nL)] for _ in range(nI)],
            "p": [[rng.randint(0, 3) for _ in range(nL)] for _ in range(nJ)],
            "q": [[rng.randint(0, 2) for _ in range(nL)] for _ in range(nK)],
        }
        if mode == 1:       # f=g=0 (P3 stress, optimum ties)
            inst["f"] = [0] * nI
            inst["g"] = [0] * nJ
        if mode == 2:       # c=d=0 (P1 stress)
            inst["c"] = [[[0] * nL for _ in range(nJ)] for _ in range(nI)]
            inst["d"] = [[[0] * nL for _ in range(nK)] for _ in range(nJ)]
        if mode == 3:       # all zero except capacities (total tie)
            inst["f"] = [0] * nI
            inst["g"] = [0] * nJ
            inst["c"] = [[[0] * nL for _ in range(nJ)] for _ in range(nI)]
            inst["d"] = [[[0] * nL for _ in range(nK)] for _ in range(nJ)]
        cache = DesignCache(inst)
        opt = brute_force(cache)
        nfac = nI + nJ
        for k in range(nfac + 1):
            v_p, _ = bnb(inst, cache, k, use_prunings=True)
            n_checks += 1
            check(v_p == opt[k],
                  f"A2 seed-idx={t} mode={mode} k={k}: bnb={v_p} "
                  f"brute={opt[k]}")
    return n_checks


# ---------------------------------------------------------------------------
# TEST 3
# ---------------------------------------------------------------------------

def dual_lp_full(inst, l, y, z):
    nI, nJ, nK = inst["nI"], inst["nJ"], inst["nK"]
    aid = lambda k: k
    bid = lambda j: nK + j
    gid = lambda i: nK + nJ + i
    did = lambda j: nK + nJ + nI + j
    nv = nK + nJ + nI + nJ
    obj = np.zeros(nv)
    for k in range(nK):
        obj[aid(k)] = -inst["q"][k][l]
    for i in range(nI):
        obj[gid(i)] = inst["b"][i][l] * y[i]
    for j in range(nJ):
        obj[did(j)] = inst["p"][j][l] * z[j]
    A, rhs = [], []
    for i in range(nI):
        for j in range(nJ):
            row = np.zeros(nv)
            row[bid(j)] = 1.0
            row[gid(i)] = -1.0
            A.append(row); rhs.append(inst["c"][i][j][l])
    for j in range(nJ):
        for k in range(nK):
            row = np.zeros(nv)
            row[aid(k)] = 1.0
            row[bid(j)] = -1.0
            row[did(j)] = -1.0
            A.append(row); rhs.append(inst["d"][j][k][l])
    res = linprog(obj, A_ub=np.array(A), b_ub=np.array(rhs),
                  bounds=(0, None), method="highs")
    if res.status != 0:
        return res.status, None, None, (np.array(A), np.array(rhs))
    return 0, -res.fun, res.x, (np.array(A), np.array(rhs))


def attack3(n_inst=25, seed0=777000):
    n_cut = n_ray = n_unb = 0
    for t in range(n_inst):
        inst = gen_instance(seed0 + t, max_i=3, max_j=3, max_k=3, max_l=2)
        # capacities x2 to get feasible generators with moderate slack
        if t % 2:
            inst["b"] = [[v * 2 for v in r] for r in inst["b"]]
            inst["p"] = [[v * 2 for v in r] for r in inst["p"]]
        nI, nJ, nK = inst["nI"], inst["nJ"], inst["nK"]
        for l in range(inst["nL"]):
            D = demand_total(inst, l)
            if D == 0:
                continue
            y1, z1 = [1] * nI, [1] * nJ
            feas, v1 = routing_value(inst, l, y1, z1)
            if not feas:
                continue
            st, vd, u, (A, rhs) = dual_lp_full(inst, l, y1, z1)
            check(st == 0 and abs(vd - v1) <= 1e-6,
                  f"A3 duality inst={t} l={l}")
            if st != 0:
                continue
            alpha = u[:nK]
            gamma = u[nK + nJ: nK + nJ + nI]
            delta = u[nK + nJ + nI:]
            # cut on ALL designs
            for my in range(1 << nI):
                for mz in range(1 << nJ):
                    y2 = [(my >> i) & 1 for i in range(nI)]
                    z2 = [(mz >> j) & 1 for j in range(nJ)]
                    rhs_cut = sum(inst["q"][k][l] * alpha[k]
                                  for k in range(nK)) \
                        - sum(inst["b"][i][l] * gamma[i] * y2[i]
                              for i in range(nI)) \
                        - sum(inst["p"][j][l] * delta[j] * z2[j]
                              for j in range(nJ))
                    feas2, v2 = routing_value(inst, l, y2, z2)
                    if feas2:
                        n_cut += 1
                        check(v2 >= rhs_cut - 1e-5,
                              f"A3 cut inst={t} l={l} my={my} mz={mz}: "
                              f"v={v2} < rhs={rhs_cut}")
                    else:
                        st2, _, _, _ = dual_lp_full(inst, l, y2, z2)
                        n_unb += 1
                        check(st2 == 3,
                              f"A3 dual not unbounded inst={t} l={l} "
                              f"my={my} mz={mz}: status={st2}")
            # NUMERIC rays: u* + theta r_i dual-feasible; linear objective
            nv = nK + nJ + nI + nJ
            r1 = np.zeros(nv); r2 = np.zeros(nv)
            r1[:nK] = 1.0                                # alpha
            r1[nK:nK + nJ] = 1.0                         # beta
            r1[nK + nJ:nK + nJ + nI] = 1.0               # gamma
            r2[:nK] = 1.0                                # alpha
            r2[nK + nJ + nI:] = 1.0                      # delta
            capB = sum(inst["b"][i][l] * y1[i] for i in range(nI))
            capP = sum(inst["p"][j][l] * z1[j] for j in range(nJ))
            for ray, slope in [(r1, D - capB), (r2, D - capP)]:
                for theta in (1.0, 10.0, 1000.0):
                    pt = u + theta * ray
                    n_ray += 1
                    check(np.all(A @ pt <= rhs + 1e-7),
                          f"A3 ray outside the polyhedron inst={t} l={l} "
                          f"theta={theta}")
                    # dual objective at pt (max) = vd + theta*slope
                    objv = sum(inst["q"][k][l] * pt[k] for k in range(nK)) \
                        - sum(inst["b"][i][l] * y1[i] *
                              pt[nK + nJ + i] for i in range(nI)) \
                        - sum(inst["p"][j][l] * z1[j] *
                              pt[nK + nJ + nI + j] for j in range(nJ))
                    check(abs(objv - (vd + theta * slope)) <= 1e-5,
                          f"A3 slope inst={t} l={l} theta={theta}: "
                          f"{objv} != {vd + theta * slope}")
    return n_cut, n_ray, n_unb


def main():
    print("== TEST 1: numeric-cell reduction via LP in inequality form (parking) ==")
    c1 = attack1()
    print(f"  families {c1['familias']}  LP-designs {c1['desenhos']}  "
          f"iff {c1['iff']}")
    print("== TEST 2: B&B under massive ties / capacity 0 ==")
    c2 = attack2()
    print(f"  comparisons {c2}")
    print("== TEST 3: exhaustive cuts + numeric rays ==")
    ncut, nray, nunb = attack3()
    print(f"  cuts {ncut}  rays {nray}  unbounded {nunb}")
    if FAILS:
        print(f"\nRESULT: {len(FAILS)} FAILURES")
        sys.exit(1)
    print("\nRESULT: PASS (0 failures)")


if __name__ == "__main__":
    main()
