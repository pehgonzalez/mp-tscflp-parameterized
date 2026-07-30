#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_A3b_w2.py -- Computational verification of the paper's W[2] results
(W[2]-hardness and ETH lower bound).

COMPOSITE reduction being verified:

  DOMINATING SET (G = (V,E), target t)
    -> SET COVER via closed neighborhoods:
         U = V, family = { N[v] : v in V }  (same parameter t;
         |U| = |family| = n = |V|)
    -> MP-TSCFLP via the paper's covering construction Phi (reused
       verbatim; see verify_A2_setcover.py):
         |L| = 1, |I| = 1, f_1 = 0, c = 0, amplifier Q = m + 1 = n + 1,
         depots = sets (g = 1, p_j = n_U * Q), customers = elements
         (q_u = Q), d_{ju} = 0 iff u in N[v_j], else 1;
       budget B = t and cardinality k = t + 1 (the +1 is the single plant,
       which must open and counts in sum(y) + sum(z) of the
       cardinality-parameterized version).

The MP-TSCFLP side is solved by BRUTE FORCE over ALL designs
(y, z) -- including y_1 = 0 -- with EXACT ROUTING via the integer min-cost
flow oracle of the common module (`routing_value` from common_mp_tscfl.py),
without using the proofs' closed formula. gamma(G) is computed by independent
brute force over subsets of V.

Checks, per graph (t := gamma(G)):
  (1) [iff]    for EVERY t' in {1..n}:
                 gamma(G) <= t'  <=>  MP-TSCFLP(B = t', k = t'+1) is YES;
  (2) [exact]  MP(B=t,   k=t+1) = YES;
               MP(B=t-1, k=t+1) = NO;   MP(B=t, k=t) = NO
               (the accounting B = t, k = t+1 is tight on both axes);
  (3) [slack]  in every feasible design: y_1 = 1 and total cost >= sum(z);
               hence budget B already forces <= B open depots and the
               cardinality constraint k = B + 1 is SLACK -- also checked
               directly: for every t' in {0..n+1},
               MP(B = t', k = t'+1) = MP(B = t', no cardinality constraint);
  (4) [sanity] a design with empty D or y_1 = 0 is infeasible (n >= 1).

Batteries:
  (A) ALL 64 labeled graphs with |V| = 4;
  (B) >= 200 seeded random graphs with |V| in {5, 6, 7}
      (seed 20260710, varied densities).

Output: counts and PASS/FAIL per battery; exit code != 0 on failure.
"""

import itertools
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_mp_tscfl import routing_value, all_designs  # noqa: E402


# ---------------------------------------------------------------------------
# Dominating Set -> Set Cover (closed neighborhoods)
# ---------------------------------------------------------------------------

def closed_neighborhoods(n, edges):
    """Family { N[v] : v in V } over U = V = {0..n-1}."""
    nb = [set([v]) for v in range(n)]
    for (u, v) in edges:
        nb[u].add(v)
        nb[v].add(u)
    return [frozenset(nb[v]) for v in range(n)]


def domination_number(n, edges):
    """gamma(G) by brute force (always exists: V dominates, via v in N[v])."""
    fam = closed_neighborhoods(n, edges)
    universe = set(range(n))
    for r in range(1, n + 1):
        for combo in itertools.combinations(range(n), r):
            if set().union(*(fam[j] for j in combo)) >= universe:
                return r
    raise AssertionError("V always dominates -- unreachable")


# ---------------------------------------------------------------------------
# Set Cover -> MP-TSCFLP (covering construction Phi, verbatim)
# ---------------------------------------------------------------------------

def build_mp_instance(n_U, sets):
    """MP-TSCFLP instance Phi(U, S) in common_mp_tscfl format.

    Q = m + 1; b_1 = p_j = n_U * Q; q_u = Q; f = 0, g = 1, c = 0,
    d_{ju} = [u not in S_j]."""
    m = len(sets)
    Q = m + 1
    cap = n_U * Q
    inst = {
        "nI": 1, "nJ": m, "nK": n_U, "nL": 1,
        "f": [0],
        "g": [1] * m,
        "c": [[[0] for _ in range(m)]],                      # c[i][j][l]
        "d": [[[0 if u in sets[j] else 1] for u in range(n_U)]
              for j in range(m)],                            # d[j][k][l]
        "b": [[cap]],                                        # b[i][l]
        "p": [[cap] for _ in range(m)],                      # p[j][l]
        "q": [[Q] for _ in range(n_U)],                      # q[k][l]
    }
    return inst, Q


def enumerate_solutions(inst):
    """Brute force over ALL (y, z), exact routing via integer MCMF.

    Returns list of feasible designs as tuples
    (total_cost, cardinality sum(y)+sum(z), sum(z), y_1)."""
    sols = []
    for y, z in all_designs(inst["nI"], inst["nJ"]):
        total_route = 0
        feasible = True
        for l in range(inst["nL"]):
            ok, val = routing_value(inst, l, y, z)
            if not ok:
                feasible = False
                break
            total_route += val
        if not feasible:
            continue
        fixed = sum(inst["f"][i] * y[i] for i in range(inst["nI"])) \
              + sum(inst["g"][j] * z[j] for j in range(inst["nJ"]))
        sols.append((fixed + total_route, sum(y) + sum(z), sum(z), y[0]))
    return sols


def mp_yes(sols, B, k):
    """Is MP-TSCFLP(B, k) YES? (k = None: no cardinality constraint)."""
    return any(cost <= B and (k is None or card <= k)
               for cost, card, _sz, _y1 in sols)


# ---------------------------------------------------------------------------
# Per-graph checks
# ---------------------------------------------------------------------------

def check_graph(n, edges, failures, counts):
    fam = closed_neighborhoods(n, edges)
    t = domination_number(n, edges)          # 1 <= t <= n always
    inst, Q = build_mp_instance(n, fam)
    sols = enumerate_solutions(inst)

    tag = "n=%d edges=%s" % (n, sorted(edges))

    # (1) iff for every t' in 1..n (parameters t' <= m = n, A2's WLOG valid)
    for tp in range(1, n + 1):
        counts["iff"] += 1
        lhs = (t <= tp)
        rhs = mp_yes(sols, B=tp, k=tp + 1)
        if lhs != rhs:
            failures.append((tag, "iff t'=%d: gamma<=t' is %s, MP(t',t'+1) is %s"
                             % (tp, lhs, rhs)))

    # (2) tightness of the accounting at t = gamma(G)
    counts["exact"] += 3
    if not mp_yes(sols, B=t, k=t + 1):
        failures.append((tag, "MP(B=%d,k=%d) should be YES" % (t, t + 1)))
    if mp_yes(sols, B=t - 1, k=t + 1):
        failures.append((tag, "MP(B=%d,k=%d) should be NO" % (t - 1, t + 1)))
    if mp_yes(sols, B=t, k=t):
        failures.append((tag, "MP(B=%d,k=%d) should be NO" % (t, t)))

    # (3) cardinality slack: cost >= sum(z) and y_1 = 1 in every feasible;
    #     and MP(B,k=B+1) coincides with MP(B, no cardinality) for every B
    for cost, card, sz, y1 in sols:
        counts["slack"] += 1
        if y1 != 1:
            failures.append((tag, "feasible design with y_1=0"))
        if cost < sz:
            failures.append((tag, "cost %d < sum(z) %d" % (cost, sz)))
        if cost <= n and card > cost + 1:
            failures.append((tag, "cost %d but cardinality %d > cost+1"
                             % (cost, card)))
    for B in range(0, n + 2):
        counts["slack"] += 1
        if mp_yes(sols, B, B + 1) != mp_yes(sols, B, None):
            failures.append((tag, "B=%d: k=B+1 differs from k=infinity" % B))

    # (4) sanity: no feasible design with empty D
    counts["sanity"] += 1
    if any(sz == 0 for _c, _k, sz, _y in sols):
        failures.append((tag, "feasible design with empty D"))


# ---------------------------------------------------------------------------
# Batteries
# ---------------------------------------------------------------------------

def all_labeled_graphs(n):
    pairs = list(itertools.combinations(range(n), 2))
    for mask in range(1 << len(pairs)):
        yield [pairs[i] for i in range(len(pairs)) if (mask >> i) & 1]


def random_graphs(count, seed):
    rng = random.Random(seed)
    out = []
    densities = [0.15, 0.3, 0.5, 0.7, 0.9]
    while len(out) < count:
        n = rng.choice([5, 6, 7])
        p = rng.choice(densities)
        edges = [(u, v) for (u, v) in itertools.combinations(range(n), 2)
                 if rng.random() < p]
        out.append((n, edges))
    return out


def main():
    failures = []
    counts = {"iff": 0, "exact": 0, "slack": 0, "sanity": 0}

    # (A) all labeled graphs with |V| = 4
    n_graphs_A = 0
    for edges in all_labeled_graphs(4):
        check_graph(4, edges, failures, counts)
        n_graphs_A += 1
    print("[A] exhaustive: %d labeled graphs with |V| = 4" % n_graphs_A)

    # (B) seeded random graphs, |V| in {5,6,7}
    rg = random_graphs(210, seed=20260710)
    for n, edges in rg:
        check_graph(n, edges, failures, counts)
    print("[B] random: %d graphs (seed 20260710, |V| in {5,6,7})" % len(rg))

    print("\nChecks: iff=%d, tightness(B,k)=%d, cardinality-slack=%d, "
          "sanity=%d  (total %d)"
          % (counts["iff"], counts["exact"], counts["slack"],
             counts["sanity"], sum(counts.values())))

    if failures:
        print("\nFAILURES (%d):" % len(failures))
        for tag, msg in failures[:20]:
            print("  %s : %s" % (tag, msg))
        sys.exit(1)
    print("ALL TESTS PASSED.")


if __name__ == "__main__":
    main()
