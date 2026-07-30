#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_A2_setcover.py -- Computational verification of the paper's covering
theorem and of the gap arithmetic of the inapproximability theorem.

Reduction being verified (covering theorem), from SET COVER (U, S = {S_1..S_m}, t):
  * 1 product (|L| = 1), 1 plant (|I| = 1), f_1 = 0, c_{1j} = 0 for all j;
  * depots = sets S_j, fixed cost g_j = 1, capacity p_j = |U|*Q;
  * plant capacity b_1 = |U|*Q;
  * customers = elements u of U, demand q_u = Q := |S| + 1 (amplifier);
  * d_{ju} = 0 if u belongs to S_j, else 1;
  * budget B = t (WLOG 1 <= t <= m).
Claim: a cover of size <= t exists  <=>  OPT_MP <= t.

EXACT SOLUTION OF THE RESTRICTED MP-TSCFLP (justification of the "greedy"):
With (y, z) fixed (y_1 = 1, D = set of open depots, D nonempty),
optimal routing is trivial in this subclass:
  (i)   stage 1 has zero cost (c = 0) and the plant capacity
        b_1 = |U|*Q covers the total demand |U|*Q, so it never binds;
  (ii)  each depot capacity p_j = |U|*Q likewise covers the entire total
        demand on its own, so no depot capacity is active;
  (iii) transport cost is separable per unit of flow: each unit
        of customer u's demand sent through open depot j costs
        exactly d_{ju} (stage 2) + 0 (stage 1).
Hence every unit of u's demand costs >= min_{j in D} d_{ju}, and this
per-unit cost is attainable by routing all of u's demand Q through an open
depot minimizing d_{ju} (feasible by (i)-(ii)). Thus the optimal routing
-- including over FRACTIONAL/SPLIT flows -- equals exactly
      sum_u Q * min_{j in D} d_{ju},
and since d in {0,1}: transport cost = Q * #{u : u not covered by D}.
(The script verify_A2_splittable.py confirms this against an independent LP.)

The global optimum is obtained by brute force over ALL subsets D
(nonempty; empty D is infeasible since sum_j w_{ju} >= Q > 0 requires an
open depot with capacity): OPT_MP = min_D [ |D| + Q * #{u not covered by D} ].

SET COVER is also solved by independent brute force (all
subsets of the family).

Batteries:
  (A) ALL Set Cover instances with |U| <= 4, |S| <= 4
      (families = combinations of distinct subsets of U, already
      deduplicated by construction; empty sets allowed),
      for every t in {1..m}: tests the covering theorem's double implication.
  (B) >= 50 random instances with fixed seed, |U| <= 6, |S| <= 6.
  (C) Gap arithmetic of the inapproximability theorem with Q' := |U|*|S| + |S| + 1:
      - if coverable: OPT_MP(Q') = t* (size of the minimum cover);
      - Q' > (1 + ln|U|) * |S|  (>= max{1, alpha} * t* for all
        alpha <= ln|U| and t* <= |S|; the theorem statement uses the factor
        max{1, (1-eps) ln|U|} -- confirmed by independent cross-check);
      - for every D: cost(D; Q') < Q'  =>  D covers U.
Output: counts and PASS/FAIL per battery; exit code != 0 on failure.
"""

import itertools
import math
import random
import sys


# ---------------------------------------------------------------------------
# Exact solvers
# ---------------------------------------------------------------------------

def mp_cost_given_depots(n, sets, Q, depots):
    """Exact cost of the reduced MP-TSCFLP solution with D = depots open.

    Exact even over fractional flows -- see justification (i)-(iii)
    in the header. Returns |D| (fixed costs g=1; f=0) + Q per element of U
    not covered by any open depot.
    """
    cost = len(depots)
    for u in range(n):
        if not any(u in sets[j] for j in depots):
            cost += Q
    return cost


def solve_mp_bruteforce(n, sets, Q):
    """OPT of the reduced MP-TSCFLP: brute force over all (y,z).

    y_1 = 1 always (f=0, no cost; needed to carry flow).
    Empty D is infeasible for n >= 1 (positive demand with no open depot).
    Returns None if m = 0 (infeasible).
    """
    m = len(sets)
    best = None
    for mask in range(1, 1 << m):
        depots = [j for j in range(m) if (mask >> j) & 1]
        c = mp_cost_given_depots(n, sets, Q, depots)
        if best is None or c < best:
            best = c
    return best


def solve_setcover_bruteforce(n, sets):
    """Minimum cover size, or None if the family does not cover U."""
    m = len(sets)
    universe = set(range(n))
    best = None
    for r in range(1, m + 1):
        if best is not None:
            break  # combinations in increasing order of size
        for combo in itertools.combinations(range(m), r):
            if set().union(*(sets[j] for j in combo)) >= universe:
                best = r
                break
    return best


# ---------------------------------------------------------------------------
# Test batteries
# ---------------------------------------------------------------------------

def check_instance_A21(n, sets, failures):
    """Tests the covering theorem's double implication for every t in {1..m};
    returns #tests."""
    m = len(sets)
    Q = m + 1
    opt_mp = solve_mp_bruteforce(n, sets, Q)
    tstar = solve_setcover_bruteforce(n, sets)

    # Structural sanity (used in the covering theorem's proof):
    if tstar is not None:
        # coverable instance: OPT_MP = t* exactly
        if opt_mp != tstar:
            failures.append((n, sets, "OPT_MP=%s != t*=%s (Q=%d)" % (opt_mp, tstar, Q)))
    else:
        # not coverable: every D leaves an element uncovered => cost >= Q + 1
        if opt_mp is not None and opt_mp <= m:  # in particular opt_mp < Q
            failures.append((n, sets, "not coverable but OPT_MP=%s <= m=%d" % (opt_mp, m)))

    tests = 0
    for t in range(1, m + 1):  # WLOG 1 <= t <= m
        cover_exists = (tstar is not None and tstar <= t)
        mp_yes = (opt_mp is not None and opt_mp <= t)
        tests += 1
        if cover_exists != mp_yes:
            failures.append((n, sets, "t=%d: cover<=t is %s but OPT_MP<=t is %s"
                             % (t, cover_exists, mp_yes)))
    return tests


def check_instance_A23_gap(n, sets, failures):
    """Verifies the gap arithmetic of the inapproximability theorem with
    Q' = n*m + m + 1."""
    m = len(sets)
    Qp = n * m + m + 1
    tstar = solve_setcover_bruteforce(n, sets)

    # (c2) Q' dominates (1 + ln|U|) * m: hence max{1, alpha} * t* <= (1+ln n)*m
    #      < Q' for all alpha <= ln n and t* <= m (chain from the proof of the
    #      inapproximability theorem with the factor max{1, (1-eps) ln|U|}).
    if not (Qp > (1 + math.log(n)) * m):
        failures.append((n, sets, "Q'=%d <= (1+ln(n))*m=%.4f"
                         % (Qp, (1 + math.log(n)) * m)))

    checks = 1
    # (c3) every solution of cost < Q' covers U  (complete enumeration of D)
    for mask in range(1, 1 << m):
        depots = [j for j in range(m) if (mask >> j) & 1]
        c = mp_cost_given_depots(n, sets, Qp, depots)
        covers = all(any(u in sets[j] for j in depots) for u in range(n))
        checks += 1
        if c < Qp and not covers:
            failures.append((n, sets, "D=%s cost=%d < Q'=%d but does not cover" % (depots, c, Qp)))
        # (c3') if it covers, cost = |D| >= t*
        if covers and c != len(depots):
            failures.append((n, sets, "D covers but cost=%d != |D|=%d" % (c, len(depots))))

    # (c1) if coverable, OPT_MP(Q') = t*
    if tstar is not None:
        opt_mp = solve_mp_bruteforce(n, sets, Qp)
        checks += 1
        if opt_mp != tstar:
            failures.append((n, sets, "gap: OPT_MP(Q')=%s != t*=%s" % (opt_mp, tstar)))
    return checks


def enumerate_all_families(max_n, max_m):
    """All (deduplicated) families of up to max_m distinct subsets
    of a universe of size n, for n = 1..max_n. Combinations of DISTINCT
    sets => no duplicate families and no repeated sets."""
    for n in range(1, max_n + 1):
        candidates = [frozenset(c)
                      for r in range(n + 1)
                      for c in itertools.combinations(range(n), r)]
        for msize in range(1, max_m + 1):
            for family in itertools.combinations(candidates, msize):
                yield n, list(family)


def random_instances(count, max_n, max_m, seed):
    rng = random.Random(seed)
    out = []
    while len(out) < count:
        n = rng.randint(3, max_n)
        m = rng.randint(3, max_m)
        fam = []
        for _ in range(m):
            fam.append(frozenset(u for u in range(n) if rng.random() < 0.5))
        out.append((n, fam))
    return out


def main():
    failures = []

    # (A) exhaustive |U| <= 4, |S| <= 4
    n_inst_A = 0
    n_tests_A = 0
    for n, fam in enumerate_all_families(4, 4):
        n_inst_A += 1
        n_tests_A += check_instance_A21(n, fam, failures)
    print("[A] exhaustive: %d instances (|U|<=4, |S|<=4), %d equivalence "
          "tests (all t in 1..m)" % (n_inst_A, n_tests_A))

    # (B) seeded random |U| <= 6, |S| <= 6
    rand = random_instances(60, 6, 6, seed=20260710)
    n_tests_B = 0
    for n, fam in rand:
        n_tests_B += check_instance_A21(n, fam, failures)
    print("[B] random: %d instances (seed 20260710, |U|<=6, |S|<=6), "
          "%d equivalence tests" % (len(rand), n_tests_B))

    # (C) gap arithmetic of the inapproximability theorem on the same instances
    n_checks_C = 0
    for n, fam in itertools.chain(enumerate_all_families(4, 4), rand):
        n_checks_C += check_instance_A23_gap(n, fam, failures)
    print("[C] inapproximability gap: %d checks (Q'=nm+m+1; OPT=t*; cost<Q' => covers; "
          "Q'>(1+ln(n))*m)" % n_checks_C)

    if failures:
        print("\nFAILURES (%d):" % len(failures))
        for f in failures[:20]:
            print("  n=%d fam=%s : %s" % (f[0], [sorted(s) for s in f[1]], f[2]))
        sys.exit(1)
    print("\nALL TESTS PASSED.")


if __name__ == "__main__":
    main()
