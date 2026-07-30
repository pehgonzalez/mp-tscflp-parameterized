#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_A3a_partition.py -- Computational verification of the paper's PARTITION
reduction: WEAK NP-hardness of the MP-TSCFLP with |J| = |L| = |K| = 1 (via PARTITION)
and pseudo-polynomial solvability of the same cell (knapsack-cover DP).

Reduction being verified (item (i) of the theorem), from PARTITION (a_1..a_m, A = sum a_i,
WLOG A even, target D = A/2):
  * plants = items: I = {1..m}, b_i = a_i, f_i = a_i;
  * 1 depot: g_1 = 0, p_1 = D;
  * 1 customer, 1 product: q = D;
  * all transport costs zero (c = d = 0);
  * budget B = D = A/2.
Claim: there exists S with sum_{i in S} a_i = A/2  <=>  OPT_MP <= A/2.
(Structural: OPT_MP = min { sum_{i in S} a_i : S subset, sum >= D },
since every feasible solution costs exactly the sum of the open f_i -- the
feasibility condition F1 requires sum b_i y_i >= D and transport is
free.)

The optimum of the reduced MP-TSCFLP is computed by brute force INDEPENDENT of
the formula: enumeration of ALL designs (y,z) via common_mp_tscfl.all_designs
with exact routing by the integer MCMF of common_mp_tscfl.routing_value
(the paper's routing oracle -- no closed form from the proof is
used here).

DP being verified (item (ii) of the theorem), cell |J| = |K| = |L| = 1 with
GENERAL data (integer f, g, c, d, b, p, q >= 0):
  OPT = g_1 + d_111 * D + DP(D), where DP(t) = minimum cost of opening a
  subset of plants and sending exactly t units (u_i <= b_i,
  cost f_i + c_i * u_i per used plant), with the cases D = 0 (OPT = 0)
  and infeasibility (p_1 < D or sum b_i < D) handled separately.

Also verified (SYMMETRIC cell |I| = |K| = |L| = 1 of item (iii)
of the same theorem): the mirrored PARTITION reduction (items = DEPOTS:
p_j = g_j = a_j; single free plant with b_1 = D) and the symmetric DP
(plant forced open; unit cost via depot j = c_1j1 + d_j11).

Batteries:
  (A) EXHAUSTIVE: all multisets with 1 <= m <= 5, values in 0..8
      (deduplicated via combinations_with_replacement). For even A: tests
      the double implication with EXACT budget B = A/2 + the structural
      identity OPT_MP = min{sum_S : sum_S >= D}. For odd A: confirms
      that PARTITION answers "no" (validation of the WLOG convention).
  (B) RANDOM: >= 100 seeded instances, m <= 10, values <= 40;
      if the sum A comes out odd, parity is forced with a[0] += 1 (hence
      values <= 41), same tests as (A).
  (C) DP: >= 120 seeded instances of the general cell |J|=|K|=|L|=1
      (NONZERO transport costs), DP + offsets == brute force
      (all_designs + MCMF), including D = 0 and infeasible cases.
  (D) MIRROR: symmetric reduction (items = depots) on the SAME exhaustive
      instances of (A) with even A + the 100 of (B).
  (E) SYMMETRIC DP: >= 120 seeded instances of the cell |I|=|K|=|L|=1.
Output: counts and PASS/FAIL; exit code != 0 on failure.
"""

import itertools
import random
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_mp_tscfl import all_designs, routing_value

# ---------------------------------------------------------------------------
# PARTITION reduction (item (i))
# ---------------------------------------------------------------------------

def build_partition_instance(a):
    """MP-TSCFLP instance of the reduction (even A required). Returns (inst, B)."""
    A = sum(a)
    assert A % 2 == 0, "reduction defined only for even A (theorem's WLOG)"
    D = A // 2
    m = len(a)
    inst = {
        "nI": m, "nJ": 1, "nK": 1, "nL": 1,
        "f": list(a),
        "g": [0],
        "c": [[[0]] for _ in range(m)],        # c[i][0][0] = 0
        "d": [[[0]]],                          # d[0][0][0] = 0
        "b": [[ai] for ai in a],               # b[i][0] = a_i
        "p": [[D]],                            # p[0][0] = D
        "q": [[D]],                            # q[0][0] = D
    }
    return inst, D


def brute_force_mp_opt(inst):
    """OPT of the MP-TSCFLP by brute force: all (y,z), exact MCMF routing
    (the routing oracle). Returns None if no design is feasible."""
    best = None
    for y, z in all_designs(inst["nI"], inst["nJ"]):
        total = sum(inst["f"][i] * y[i] for i in range(inst["nI"])) \
              + sum(inst["g"][j] * z[j] for j in range(inst["nJ"]))
        ok = True
        for l in range(inst["nL"]):
            feas, val = routing_value(inst, l, y, z)
            if not feas:
                ok = False
                break
            total += val
        if ok and (best is None or total < best):
            best = total
    return best


def brute_force_partition(a):
    """Is there a subset with sum exactly sum(a)/2? (brute force)."""
    A = sum(a)
    if A % 2 == 1:
        return False
    target = A // 2
    sums = {0}
    for ai in a:
        sums |= {s + ai for s in sums}
    return target in sums


def structural_opt(a):
    """min { sum_{i in S} a_i : sum_{i in S} a_i >= D } (the lemma's identity)."""
    D = sum(a) // 2
    best = None
    m = len(a)
    for mask in range(1 << m):
        s = sum(a[i] for i in range(m) if (mask >> i) & 1)
        if s >= D and (best is None or s < best):
            best = s
    return best


def check_partition_instance(a, failures):
    """Tests the reduction on an instance with even A. Returns #checks."""
    A = sum(a)
    D = A // 2
    inst, B = build_partition_instance(a)
    opt = brute_force_mp_opt(inst)
    part_yes = brute_force_partition(a)
    checks = 0

    # (1) instance always feasible (sum b = A >= D)
    checks += 1
    if opt is None:
        failures.append((a, "reduction infeasible (expected feasible)"))
        return checks
    # (2) structural identity of the lemma: OPT = min{sum_S : sum_S >= D}
    checks += 1
    if opt != structural_opt(a):
        failures.append((a, "OPT=%s != structural=%s" % (opt, structural_opt(a))))
    # (3) double implication with exact budget B = D
    checks += 1
    if (opt <= B) != part_yes:
        failures.append((a, "OPT=%s<=B=%s is %s, but PARTITION is %s"
                         % (opt, B, opt <= B, part_yes)))
    # (4) lower bound: OPT >= D always
    checks += 1
    if opt < D:
        failures.append((a, "OPT=%s < D=%s" % (opt, D)))
    return checks


def build_partition_mirror_instance(a):
    """Symmetric reduction (item (iii) of the theorem): items = depots."""
    A = sum(a)
    assert A % 2 == 0
    D = A // 2
    m = len(a)
    inst = {
        "nI": 1, "nJ": m, "nK": 1, "nL": 1,
        "f": [0],
        "g": list(a),
        "c": [[[0] for _ in range(m)]],        # c[0][j][0] = 0
        "d": [[[0]] for _ in range(m)],        # d[j][0][0] = 0
        "b": [[D]],                            # b[0][0] = D
        "p": [[aj] for aj in a],               # p[j][0] = a_j
        "q": [[D]],
    }
    return inst, D


def check_partition_mirror(a, failures):
    """Tests the mirrored reduction on an instance with even A. Returns #checks."""
    inst, B = build_partition_mirror_instance(a)
    opt = brute_force_mp_opt(inst)
    part_yes = brute_force_partition(a)
    checks = 0
    checks += 1
    if opt is None:
        failures.append((a, "mirror: reduction infeasible (expected feasible)"))
        return checks
    checks += 1
    if opt != structural_opt(a):
        failures.append((a, "mirror: OPT=%s != structural=%s"
                         % (opt, structural_opt(a))))
    checks += 1
    if (opt <= B) != part_yes:
        failures.append((a, "mirror: OPT=%s<=B=%s is %s, but PARTITION is %s"
                         % (opt, B, opt <= B, part_yes)))
    return checks


# ---------------------------------------------------------------------------
# Pseudo-polynomial DP (item (ii) of the theorem)
# ---------------------------------------------------------------------------

def knapsack_cover_dp(f, c, b, D):
    """DP(t): minimum cost of sending exactly t units, t = 0..D.
    Plant i used with u in 1..b_i units costs f_i + c_i*u.
    Returns DP(D) or None (infeasible: sum b < D)."""
    dp = [None] * (D + 1)
    dp[0] = 0
    for i in range(len(f)):
        ndp = dp[:]
        for t in range(1, D + 1):
            best = ndp[t]
            for u in range(1, min(b[i], t) + 1):
                if dp[t - u] is not None:
                    cand = dp[t - u] + f[i] + c[i] * u
                    if best is None or cand < best:
                        best = cand
            ndp[t] = best
        dp = ndp
    return dp[D]


def dp_cell_opt(inst):
    """OPT of the cell |J|=|K|=|L|=1 via the formula of item (ii) of the theorem."""
    D = inst["q"][0][0]
    if D == 0:
        return 0
    if inst["p"][0][0] < D:
        return None  # feasibility condition F2 fails for every design
    core = knapsack_cover_dp(inst["f"],
                             [inst["c"][i][0][0] for i in range(inst["nI"])],
                             [inst["b"][i][0] for i in range(inst["nI"])],
                             D)
    if core is None:
        return None  # feasibility condition F1 fails
    return inst["g"][0] + inst["d"][0][0][0] * D + core


def gen_cell_instance(seed):
    """Random instance of the general cell |J|=|K|=|L|=1."""
    rng = random.Random(seed)
    m = rng.randint(1, 6)
    D = rng.randint(0, 12)
    return {
        "nI": m, "nJ": 1, "nK": 1, "nL": 1,
        "f": [rng.randint(0, 9) for _ in range(m)],
        "g": [rng.randint(0, 9)],
        "c": [[[rng.randint(0, 9)]] for _ in range(m)],
        "d": [[[rng.randint(0, 9)]]],
        "b": [[rng.randint(0, 6)] for _ in range(m)],
        "p": [[rng.randint(0, 14)]],
        "q": [[D]],
    }


def dp_cell_opt_mirror(inst):
    """OPT of the symmetric cell |I|=|K|=|L|=1 (item (iii) of the theorem):
    OPT = f_1 + DP(D) over depots, unit cost via j = c_1j1 + d_j11,
    capacity p_j, fixed cost g_j; infeasible if b_1 < D or sum p < D."""
    D = inst["q"][0][0]
    if D == 0:
        return 0
    if inst["b"][0][0] < D:
        return None  # feasibility condition F1 fails for every design
    m = inst["nJ"]
    core = knapsack_cover_dp(inst["g"],
                             [inst["c"][0][j][0] + inst["d"][j][0][0]
                              for j in range(m)],
                             [inst["p"][j][0] for j in range(m)],
                             D)
    if core is None:
        return None  # feasibility condition F2 fails
    return inst["f"][0] + core


def gen_cell_instance_mirror(seed):
    """Random instance of the symmetric cell |I|=|K|=|L|=1."""
    rng = random.Random(seed)
    m = rng.randint(1, 6)
    D = rng.randint(0, 12)
    return {
        "nI": 1, "nJ": m, "nK": 1, "nL": 1,
        "f": [rng.randint(0, 9)],
        "g": [rng.randint(0, 9) for _ in range(m)],
        "c": [[[rng.randint(0, 9)] for _ in range(m)]],
        "d": [[[rng.randint(0, 9)]] for _ in range(m)],
        "b": [[rng.randint(0, 14)]],
        "p": [[rng.randint(0, 6)] for _ in range(m)],
        "q": [[D]],
    }


def main():
    failures = []

    # (A) exhaustive: multisets m <= 5, values 0..8
    n_even = n_odd = checks_A = 0
    for m in range(1, 6):
        for a in itertools.combinations_with_replacement(range(9), m):
            a = list(a)
            if sum(a) % 2 == 1:
                n_odd += 1
                checks_A += 1
                if brute_force_partition(a):  # WLOG convention: odd A => no
                    failures.append((a, "odd A but PARTITION answered yes"))
            else:
                n_even += 1
                checks_A += check_partition_instance(a, failures)
    print("[A] exhaustive: %d multisets (m<=5, values<=8): %d with even A "
          "(reduction tested), %d with odd A (triviality confirmed); "
          "%d checks" % (n_even + n_odd, n_even, n_odd, checks_A))

    # (B) seeded random: >= 100 instances, m <= 10, even A
    rng = random.Random(20260710)
    n_B = 0
    checks_B = 0
    while n_B < 100:
        m = rng.randint(2, 10)
        a = [rng.randint(0, 40) for _ in range(m)]
        if sum(a) % 2 == 1:
            a[0] += 1  # forces even A without biasing the relevant distribution
        checks_B += check_partition_instance(a, failures)
        n_B += 1
    print("[B] random: %d instances (seed 20260710, m<=10, values<=41), "
          "%d checks" % (n_B, checks_B))

    # (C) pseudo-polynomial DP of the general cell |J|=|K|=|L|=1
    n_C = 120
    stats = {"feas": 0, "infeas": 0, "D0": 0}
    for s in range(n_C):
        inst = gen_cell_instance(300 + s)
        opt_bf = brute_force_mp_opt(inst)
        opt_dp = dp_cell_opt(inst)
        if inst["q"][0][0] == 0:
            stats["D0"] += 1
        elif opt_bf is None:
            stats["infeas"] += 1
        else:
            stats["feas"] += 1
        if opt_bf != opt_dp:
            failures.append((inst, "DP=%s != brute=%s" % (opt_dp, opt_bf)))
    print("[C] DP: %d instances of the cell |J|=|K|=|L|=1 with general transport "
          "(seeds 300..%d): %d feasible, %d infeasible, %d with D=0"
          % (n_C, 300 + n_C - 1, stats["feas"], stats["infeas"], stats["D0"]))

    # (D) mirrored reduction (items = depots) on the same instances
    n_D = checks_D = 0
    for m in range(1, 6):
        for a in itertools.combinations_with_replacement(range(9), m):
            a = list(a)
            if sum(a) % 2 == 0:
                n_D += 1
                checks_D += check_partition_mirror(a, failures)
    rng = random.Random(20260710)
    n_Drand = 0
    while n_Drand < 100:
        m = rng.randint(2, 10)
        a = [rng.randint(0, 40) for _ in range(m)]
        if sum(a) % 2 == 1:
            a[0] += 1
        checks_D += check_partition_mirror(a, failures)
        n_Drand += 1
    print("[D] mirror (items=depots): %d exhaustive (even A) + %d "
          "random, %d checks" % (n_D, n_Drand, checks_D))

    # (E) symmetric DP of the cell |I|=|K|=|L|=1
    n_E = 120
    stats_E = {"feas": 0, "infeas": 0, "D0": 0}
    for s in range(n_E):
        inst = gen_cell_instance_mirror(800 + s)
        opt_bf = brute_force_mp_opt(inst)
        opt_dp = dp_cell_opt_mirror(inst)
        if inst["q"][0][0] == 0:
            stats_E["D0"] += 1
        elif opt_bf is None:
            stats_E["infeas"] += 1
        else:
            stats_E["feas"] += 1
        if opt_bf != opt_dp:
            failures.append((inst, "mirror DP=%s != brute=%s"
                             % (opt_dp, opt_bf)))
    print("[E] symmetric DP: %d instances of the cell |I|=|K|=|L|=1 "
          "(seeds 800..%d): %d feasible, %d infeasible, %d with D=0"
          % (n_E, 800 + n_E - 1, stats_E["feas"], stats_E["infeas"],
             stats_E["D0"]))

    if failures:
        print("\nFAILURES (%d):" % len(failures))
        for f in failures[:20]:
            print("  %s : %s" % (f[0], f[1]))
        sys.exit(1)
    print("\nALL TESTS PASSED.")


if __name__ == "__main__":
    main()
