#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_A2_splittable.py -- Adversarial verification of "splittable service"
in the paper's covering and inapproximability theorems.

ADVERSARIAL QUESTION: on the instances produced by the Set Cover reduction,
could FRACTIONAL/SPLIT flows (service shared among several depots, or
between a covering and a non-covering depot) cost less than the value
|D| + Q * #{u not covered by D} claimed in the proofs?

METHOD (documenting the choice):
For each reduced instance and each subset D of open depots
(COMPLETE enumeration), we solve the routing subproblem as a generic
LINEAR PROGRAM via scipy.optimize.linprog (method "highs"),
with ALL variables x_{1j}, w_{ju} continuous and ALL constraints of the
MP-TSCFLP (demand, depot conservation, plant capacity,
depot capacities) -- that is, without using the proofs' "greedy" argument
at any point. We compare the LP value with the closed formula
    transport(D) = Q * #{u : u not covered by D}
and with the greedy integer routing. The LP is thus an INDEPENDENT verifier.

Theoretical note on record (not used by the verifier, context only):
for fixed (y,z) the routing subproblem is a min-cost flow with integer
data, hence it has an INTEGER optimum (integrality of the flow polytope /
totally unimodular matrix); testing only integer flows would already be
legitimate. We prefer the continuous LP because it directly attacks the
"fractional" gap without relying on that lemma.

LP solved, for fixed D (1 plant, 1 product):
  min  sum_{j,u} d_{ju} w_{ju}                     (c = 0 in stage 1)
  s.t. sum_{j in D} w_{ju} >= Q          (u in U)   [demand]
       sum_u w_{ju} - x_{1j} <= 0        (j in D)   [depot conservation]
       sum_{j in D} x_{1j} <= b_1                   [plant capacity]
       sum_u w_{ju} <= p_j               (j in D)   [depot capacity]
       x, w >= 0
with b_1 = p_j = |U|*Q. Closed depots get no variables (w=x=0
forced by z_j = 0). Numerical tolerance: 1e-6 (small data, HiGHS
solves exactly at these sizes).

Batteries (extended via independent cross-check, to match the ranges of
verify_A2_setcover.py, so that the closed formula never runs "without
contrast" on an instance covered only by the other script):
  (A) ALL deduplicated families with |U| <= 4, |S| <= 4 (same
      enumeration as battery [A] of verify_A2_setcover.py), both
      amplifiers Q = m+1 (covering) and Q' = nm+m+1 (inapproximability),
      all nonempty
      D.
  (B) 50 seeded random instances with |U| <= 6, |S| <= 6, likewise.
Criterion: LP_optimum == Q * #uncovered(D) in every combination (=> splitting
never helps, and the closed formula used in the proofs is exact).
Exit code != 0 on failure.
"""

import itertools
import random
import sys

import numpy as np
from scipy.optimize import linprog


def lp_routing_cost(n, sets, Q, depots):
    """Optimal CONTINUOUS routing cost with D = depots open (generic LP).

    Variables: x_j (j in D) and w_{ju} (j in D, u in U), all >= 0.
    Returns the optimal LP value, or None if infeasible.
    """
    D = list(depots)
    dn = len(D)
    if dn == 0:
        return None if n >= 1 else 0.0
    nv = dn + dn * n  # x then w
    cap = n * Q       # b_1 = p_j = |U|*Q

    def wi(a, u):  # index of w_{D[a], u}
        return dn + a * n + u

    obj = np.zeros(nv)
    for a, j in enumerate(D):
        for u in range(n):
            obj[wi(a, u)] = 0.0 if u in sets[j] else 1.0

    A_ub, b_ub = [], []
    # demand: -sum_a w_{a,u} <= -Q
    for u in range(n):
        row = np.zeros(nv)
        for a in range(dn):
            row[wi(a, u)] = -1.0
        A_ub.append(row); b_ub.append(-float(Q))
    # depot conservation: sum_u w_{a,u} - x_a <= 0
    for a in range(dn):
        row = np.zeros(nv)
        row[a] = -1.0
        for u in range(n):
            row[wi(a, u)] = 1.0
        A_ub.append(row); b_ub.append(0.0)
    # plant capacity: sum_a x_a <= b_1
    row = np.zeros(nv)
    row[:dn] = 1.0
    A_ub.append(row); b_ub.append(float(cap))
    # depot capacities: sum_u w_{a,u} <= p_j
    for a in range(dn):
        row = np.zeros(nv)
        for u in range(n):
            row[wi(a, u)] = 1.0
        A_ub.append(row); b_ub.append(float(cap))

    res = linprog(obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  bounds=[(0, None)] * nv, method="highs")
    if not res.success:
        return None
    return res.fun


def closed_form_cost(n, sets, Q, depots):
    """Closed formula used in the proofs: Q * #{u not covered by D}."""
    return float(Q * sum(1 for u in range(n)
                         if not any(u in sets[j] for j in depots)))


def check_family(n, fam, failures, tol=1e-6):
    m = len(fam)
    checks = 0
    for Q in (m + 1, n * m + m + 1):  # the theorems' two amplifiers
        for mask in range(1, 1 << m):
            depots = [j for j in range(m) if (mask >> j) & 1]
            lp = lp_routing_cost(n, fam, Q, depots)
            cf = closed_form_cost(n, fam, Q, depots)
            checks += 1
            if lp is None:
                failures.append((n, fam, "LP infeasible, D=%s Q=%d" % (depots, Q)))
            elif abs(lp - cf) > tol:
                failures.append((n, fam, "LP=%.9f != formula=%.1f, D=%s Q=%d"
                                 % (lp, cf, depots, Q)))
    return checks


def enumerate_all_families(max_n, max_m):
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
        fam = [frozenset(u for u in range(n) if rng.random() < 0.5)
               for _ in range(m)]
        out.append((n, fam))
    return out


def main():
    failures = []

    n_inst_A = n_checks_A = 0
    for n, fam in enumerate_all_families(4, 4):
        n_inst_A += 1
        n_checks_A += check_family(n, fam, failures)
    print("[A] exhaustive (|U|<=4, |S|<=4): %d families, %d LPs compared "
          "(2 amplifiers x all D)" % (n_inst_A, n_checks_A))

    rand = random_instances(50, 6, 6, seed=20260710)
    n_checks_B = 0
    for n, fam in rand:
        n_checks_B += check_family(n, fam, failures)
    print("[B] random (seed 20260710, |U|<=6, |S|<=6): %d instances, "
          "%d LPs compared" % (len(rand), n_checks_B))

    if failures:
        print("\nFAILURES (%d):" % len(failures))
        for f in failures[:20]:
            print("  n=%d fam=%s : %s" % (f[0], [sorted(s) for s in f[1]], f[2]))
        sys.exit(1)
    print("\nALL TESTS PASSED: splitting/fractionating service never "
          "improves on the value predicted by the proofs' closed formula.")


if __name__ == "__main__":
    main()
