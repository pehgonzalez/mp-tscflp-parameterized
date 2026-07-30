#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_A3a_aggregation.py -- Computational verification of the paper's
customer aggregation lemma: with |J| = 1, the customers collapse.

Lemma (statement being verified): let P be an instance with |J| = 1 and
Delta := sum_{k,l} d_{1kl} * q_{kl}. Let R be the instance with K' = {1},
q'_{1l} := D_l = sum_k q_{kl}, d'_{11l} := 0, everything else unchanged.
Then P is feasible <=> R is feasible, and OPT(P) = OPT(R) + Delta.

Verification by FULL brute force on both sides: enumeration of all
designs (y,z) (common_mp_tscfl.all_designs) with exact routing via the
integer MCMF (common_mp_tscfl.routing_value, the routing oracle) -- no
closed form.

Batteries:
  (A) >= 40 seeded instances with |J| = |L| = 1, |K| <= 4, |I| <= 4,
      general data (random f, g, c, d, b, p, q) -- the lemma's cell;
      includes infeasible cases and cases with zero demand.
  (B) >= 20 instances with |J| = 1, |L| = 2 (the lemma holds for general
      |L|; the aggregation is per product).
Output: counts (feasible / infeasible / D=0) and PASS/FAIL.
"""

import random
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_mp_tscfl import all_designs, routing_value


def brute_force_mp_opt(inst):
    """OPT by brute force (all designs, exact MCMF per product)."""
    best = None
    for y, z in all_designs(inst["nI"], inst["nJ"]):
        total = sum(inst["f"][i] * y[i] for i in range(inst["nI"])) \
              + sum(inst["g"][j] * z[j] for j in range(inst["nJ"]))
        feas = True
        for l in range(inst["nL"]):
            ok, val = routing_value(inst, l, y, z)
            if not ok:
                feas = False
                break
            total += val
        if feas and (best is None or total < best):
            best = total
    return best


def gen_singleJ_instance(seed, n_l):
    """Random instance with |J| = 1, |L| = n_l, |K| <= 4, |I| <= 4."""
    rng = random.Random(seed)
    nI = rng.randint(1, 4)
    nK = rng.randint(1, 4)
    inst = {
        "nI": nI, "nJ": 1, "nK": nK, "nL": n_l,
        "f": [rng.randint(0, 9) for _ in range(nI)],
        "g": [rng.randint(0, 9)],
        "c": [[[rng.randint(0, 9) for _ in range(n_l)]] for _ in range(nI)],
        "d": [[[rng.randint(0, 9) for _ in range(n_l)] for _ in range(nK)]],
        "b": [[rng.randint(0, 12) for _ in range(n_l)] for _ in range(nI)],
        "p": [[rng.randint(0, 20) for _ in range(n_l)]],
        "q": [[rng.randint(0, 3) for _ in range(n_l)] for _ in range(nK)],
    }
    for l in range(n_l):
        if rng.random() < 0.15:  # product with no demand (edge case)
            for k in range(nK):
                inst["q"][k][l] = 0
    return inst


def aggregate(inst):
    """Builds (R, Delta) from the lemma."""
    nK, nL = inst["nK"], inst["nL"]
    delta = sum(inst["d"][0][k][l] * inst["q"][k][l]
                for k in range(nK) for l in range(nL))
    D = [sum(inst["q"][k][l] for k in range(nK)) for l in range(nL)]
    red = dict(inst)
    red["nK"] = 1
    red["q"] = [D]
    red["d"] = [[[0] * nL]]
    return red, delta


def run_battery(label, count, seed0, n_l, failures):
    stats = {"feas": 0, "infeas": 0, "D0": 0}
    for s in range(count):
        inst = gen_singleJ_instance(seed0 + s, n_l)
        red, delta = aggregate(inst)
        opt_p = brute_force_mp_opt(inst)
        opt_r = brute_force_mp_opt(red)
        if all(q == 0 for row in inst["q"] for q in row):
            stats["D0"] += 1
        elif opt_p is None:
            stats["infeas"] += 1
        else:
            stats["feas"] += 1
        if (opt_p is None) != (opt_r is None):
            failures.append((inst, "feasibility mismatch: P=%s R=%s"
                             % (opt_p, opt_r)))
        elif opt_p is not None and opt_p != opt_r + delta:
            failures.append((inst, "OPT(P)=%s != OPT(R)+Delta=%s+%s"
                             % (opt_p, opt_r, delta)))
    print("[%s] %d instances (seeds %d..%d, |L|=%d): %d feasible, "
          "%d infeasible, %d with zero demand"
          % (label, count, seed0, seed0 + count - 1, n_l,
             stats["feas"], stats["infeas"], stats["D0"]))


def main():
    failures = []
    run_battery("A", 40, 500, 1, failures)   # cell of the aggregation lemma
    run_battery("B", 20, 700, 2, failures)   # lemma for general |L|
    if failures:
        print("\nFAILURES (%d):" % len(failures))
        for f in failures[:10]:
            print("  %s : %s" % (f[0], f[1]))
        sys.exit(1)
    print("\nALL TESTS PASSED.")


if __name__ == "__main__":
    main()
