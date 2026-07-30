#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_A3a_products.py -- Computational verification of the paper's
products construction and its mirrored corollary.

Primary construction, from SET COVER (U, S = {S_1..S_m}, t), PRODUCTS as
elements:
  * products = elements of U (|L| = n_U), 1 depot, 1 customer;
  * plants = sets: f_i = 1, b_il = Q for EVERY l,
    c_i1l = 0 if l in S_i else 1;
  * depot: g_1 = 0, p_1l = Q; customer: q_1l = Q; d_11l = 0;
  * amplifier Q := m + 1; budget B = t (WLOG 1 <= t <= m).
Claim: a cover of size <= t  <=>  OPT_MP <= t.
Closed form of the structural lemma (TO BE verified, not used as a source):
  cost(Y open, z=1) = |Y| + Q * #{l not covered by Y};  Y empty or z=0
  infeasible.

Mirrored construction (|I| = |K| = 1, depots x products):
  * 1 plant: f_1 = 0, b_1l = Q, c_1jl = 0;
  * depots = sets: g_j = 1, p_jl = Q, d_j1l = 0 if l in S_j else 1;
  * customer: q_1l = Q; identical amplifier and budget.
Closed form: cost(y=1, Z open) = |Z| + Q * #{l not covered by Z}.

INDEPENDENCE: the brute-force optimum enumerates ALL designs (y,z)
(common_mp_tscfl.all_designs) and routes each product via the exact integer
MCMF (common_mp_tscfl.routing_value, the routing oracle) -- no closed form.
The closed form is tested AGAINST this routing on all designs, and
additionally against a full continuous LP (scipy.linprog/HiGHS) on an
adversarial subsample (splittable-service gap), including infeasible
designs (the LP must report infeasibility).

Batteries:
  (A) EXHAUSTIVE: all deduplicated families with |U| <= 4, |S| <= 4
      (same enumeration as battery [A] of verify_A2_setcover.py), for
      BOTH constructions: OPT == t* (coverable) / OPT > m (not coverable);
      double implication for every t in 1..m; closed form == MCMF on all
      designs.
  (B) RANDOM: >= 50 seeded instances, |U| <= 6, |S| <= 6, likewise.
  (C) adversarial LP: >= 500 LPs (scipy.linprog) confirming the closed
      form (and infeasibility) on sampled designs of the two
      constructions.
Output: counts and PASS/FAIL; exit code != 0 on failure.
"""

import itertools
import random
import os, sys

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_mp_tscfl import all_designs, routing_value


# ---------------------------------------------------------------------------
# Constructions
# ---------------------------------------------------------------------------

def build_primary(n_u, sets):
    """Primary construction: plants = sets, products = elements."""
    m = len(sets)
    Q = m + 1
    return {
        "nI": m, "nJ": 1, "nK": 1, "nL": n_u,
        "f": [1] * m,
        "g": [0],
        "c": [[[0 if l in sets[i] else 1 for l in range(n_u)]]
              for i in range(m)],                       # c[i][0][l]
        "d": [[[0] * n_u]],                             # d[0][0][l]
        "b": [[Q] * n_u for _ in range(m)],
        "p": [[Q] * n_u],
        "q": [[Q] * n_u],
    }, Q


def build_mirror(n_u, sets):
    """Mirrored construction: depots = sets, products = elements."""
    m = len(sets)
    Q = m + 1
    return {
        "nI": 1, "nJ": m, "nK": 1, "nL": n_u,
        "f": [0],
        "g": [1] * m,
        "c": [[[0] * n_u for _ in range(m)]],           # c[0][j][l]
        "d": [[[0 if l in sets[j] else 1 for l in range(n_u)]]
              for j in range(m)],                       # d[j][0][l]
        "b": [[Q] * n_u],
        "p": [[Q] * n_u for _ in range(m)],
        "q": [[Q] * n_u],
    }, Q


def closed_form(kind, n_u, sets, Q, y, z):
    """Closed form of the structural lemma. Returns total cost or None
    (infeasible). kind = 'primary' (open = plants) or 'mirror'."""
    if kind == "primary":
        opens = [i for i in range(len(sets)) if y[i] == 1]
        if not opens or z[0] == 0:
            return None
        fixed = len(opens)          # f = 1 per open plant; g = 0
    else:
        opens = [j for j in range(len(sets)) if z[j] == 1]
        if not opens or y[0] == 0:
            return None
        fixed = len(opens)          # g = 1 per open depot; f = 0
    uncovered = sum(1 for l in range(n_u)
                    if not any(l in sets[o] for o in opens))
    return fixed + Q * uncovered


# ---------------------------------------------------------------------------
# Brute force (independent: MCMF per product on all designs)
# ---------------------------------------------------------------------------

def brute_force_designs(inst):
    """Yields (y, z, total_cost | None) for all designs, with
    exact MCMF routing per product (the routing oracle)."""
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
        yield y, z, (total if feas else None)


def solve_setcover_bruteforce(n_u, sets):
    m = len(sets)
    universe = set(range(n_u))
    for r in range(1, m + 1):
        for combo in itertools.combinations(range(m), r):
            if set().union(*(sets[j] for j in combo)) >= universe:
                return r
    return None


def check_construction(kind, n_u, sets, failures):
    """Tests one construction on a family. Returns (#iff, #closed-form)."""
    m = len(sets)
    builder = build_primary if kind == "primary" else build_mirror
    inst, Q = builder(n_u, sets)
    tstar = solve_setcover_bruteforce(n_u, sets)

    opt = None
    n_cf = 0
    for y, z, cost in brute_force_designs(inst):
        cf = closed_form(kind, n_u, sets, Q, y, z)
        n_cf += 1
        if cf != cost:  # includes feasibility: None == None
            failures.append((kind, n_u, sets,
                             "y=%s z=%s: MCMF=%s != closed form=%s"
                             % (y, z, cost, cf)))
        if cost is not None and (opt is None or cost < opt):
            opt = cost

    if tstar is not None:
        if opt != tstar:
            failures.append((kind, n_u, sets, "OPT=%s != t*=%s" % (opt, tstar)))
    else:
        if opt is not None and opt <= m:
            failures.append((kind, n_u, sets,
                             "not coverable but OPT=%s <= m=%d" % (opt, m)))

    n_iff = 0
    for t in range(1, m + 1):  # WLOG 1 <= t <= m
        cover_exists = (tstar is not None and tstar <= t)
        mp_yes = (opt is not None and opt <= t)
        n_iff += 1
        if cover_exists != mp_yes:
            failures.append((kind, n_u, sets,
                             "t=%d: cover<=t %s but OPT<=t %s"
                             % (t, cover_exists, mp_yes)))
    return n_iff, n_cf


# ---------------------------------------------------------------------------
# Adversarial LP (splittable-service gap)
# ---------------------------------------------------------------------------

def lp_primary(n_u, sets, Q, y, z):
    """Full continuous LP of the primary construction for design (y,z).
    Vars: x[i][l] (m*n_u), w[l] (n_u). Returns transport value or None."""
    m = len(sets)
    nx = m * n_u
    nv = nx + n_u
    cobj = np.zeros(nv)
    for i in range(m):
        for l in range(n_u):
            cobj[i * n_u + l] = 0 if l in sets[i] else 1
    A_ub, b_ub = [], []
    for l in range(n_u):            # (C1): -w_l <= -Q
        row = np.zeros(nv)
        row[nx + l] = -1
        A_ub.append(row); b_ub.append(-Q)
    for l in range(n_u):            # (C2): w_l - sum_i x_il <= 0
        row = np.zeros(nv)
        row[nx + l] = 1
        for i in range(m):
            row[i * n_u + l] = -1
        A_ub.append(row); b_ub.append(0)
    bounds = []
    for i in range(m):              # (C3): x_il <= Q*y_i (|J|=1)
        for l in range(n_u):
            bounds.append((0, Q * y[i]))
    for l in range(n_u):            # (C4): w_l <= Q*z (|K|=1)
        bounds.append((0, Q * z[0]))
    res = linprog(cobj, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  bounds=bounds, method="highs")
    return res.fun if res.status == 0 else None


def lp_mirror(n_u, sets, Q, y, z):
    """Full continuous LP of the mirror. Vars: x[j][l], w[j][l].
    (C3) is one coupling row per product: sum_j x_jl <= Q*y_1."""
    m = len(sets)
    nx = m * n_u
    nv = 2 * nx
    cobj = np.zeros(nv)
    for j in range(m):
        for l in range(n_u):
            cobj[nx + j * n_u + l] = 0 if l in sets[j] else 1
    A_ub, b_ub = [], []
    for l in range(n_u):            # (C1): -sum_j w_jl <= -Q
        row = np.zeros(nv)
        for j in range(m):
            row[nx + j * n_u + l] = -1
        A_ub.append(row); b_ub.append(-Q)
    for j in range(m):              # (C2): w_jl - x_jl <= 0
        for l in range(n_u):
            row = np.zeros(nv)
            row[nx + j * n_u + l] = 1
            row[j * n_u + l] = -1
            A_ub.append(row); b_ub.append(0)
    for l in range(n_u):            # (C3): sum_j x_jl <= Q*y_1 (coupled)
        row = np.zeros(nv)
        for j in range(m):
            row[j * n_u + l] = 1
        A_ub.append(row); b_ub.append(Q * y[0])
    bounds = []
    for j in range(m):
        for l in range(n_u):
            bounds.append((0, None))
    for j in range(m):              # (C4): w_jl <= Q*z_j
        for l in range(n_u):
            bounds.append((0, Q * z[j]))
    res = linprog(cobj, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  bounds=bounds, method="highs")
    return res.fun if res.status == 0 else None


def check_lp(kind, n_u, sets, y, z, failures):
    """Compares LP with the closed form (value and feasibility)."""
    Q = len(sets) + 1
    lp = (lp_primary if kind == "primary" else lp_mirror)(n_u, sets, Q, y, z)
    cf = closed_form(kind, n_u, sets, Q, y, z)
    if cf is None:
        if lp is not None:
            failures.append((kind, n_u, sets,
                             "y=%s z=%s: closed form infeasible, LP=%s"
                             % (y, z, lp)))
    else:
        fixed = sum(y) if kind == "primary" else sum(z)
        if lp is None or abs((fixed + lp) - cf) > 1e-6:
            failures.append((kind, n_u, sets,
                             "y=%s z=%s: LP total=%s != closed form=%s"
                             % (y, z, None if lp is None else fixed + lp, cf)))


# ---------------------------------------------------------------------------
# Enumeration / sampling
# ---------------------------------------------------------------------------

def enumerate_all_families(max_n, max_m):
    """Identical to the one in verify_A2_setcover.py (deduplicated families)."""
    for n in range(1, max_n + 1):
        candidates = [frozenset(cmb)
                      for r in range(n + 1)
                      for cmb in itertools.combinations(range(n), r)]
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

    # (A) exhaustive |U| <= 4, |S| <= 4, both constructions
    n_fam = iff_A = cf_A = 0
    families = list(enumerate_all_families(4, 4))
    for n_u, fam in families:
        n_fam += 1
        for kind in ("primary", "mirror"):
            a, b = check_construction(kind, n_u, fam, failures)
            iff_A += a
            cf_A += b
    print("[A] exhaustive: %d families (|U|<=4, |S|<=4) x 2 constructions: "
          "%d equivalence tests (all t), %d designs with "
          "closed form == MCMF" % (n_fam, iff_A, cf_A))

    # (B) seeded random |U| <= 6, |S| <= 6
    rand = random_instances(50, 6, 6, seed=20260710)
    iff_B = cf_B = 0
    for n_u, fam in rand:
        for kind in ("primary", "mirror"):
            a, b = check_construction(kind, n_u, fam, failures)
            iff_B += a
            cf_B += b
    print("[B] random: %d instances (seed 20260710, |U|<=6, |S|<=6) "
          "x 2 constructions: %d equivalence tests, %d designs "
          "closed form == MCMF" % (len(rand), iff_B, cf_B))

    # (C) adversarial LP: >= 500 LPs on sampled designs
    rng = random.Random(99)
    pool = families + rand
    n_lp = 0
    while n_lp < 600:
        n_u, fam = pool[rng.randrange(len(pool))]
        m = len(fam)
        kind = "primary" if n_lp % 2 == 0 else "mirror"
        # sample arbitrary design (including infeasible ones)
        if kind == "primary":
            y = [rng.randint(0, 1) for _ in range(m)]
            z = [rng.randint(0, 1)]
        else:
            y = [rng.randint(0, 1)]
            z = [rng.randint(0, 1) for _ in range(m)]
        check_lp(kind, n_u, fam, y, z, failures)
        n_lp += 1
    print("[C] adversarial LP: %d LPs (seed 99; random designs, "
          "including infeasible ones) compared with the closed form" % n_lp)

    if failures:
        print("\nFAILURES (%d):" % len(failures))
        for f in failures[:20]:
            print("  [%s] n_U=%d fam=%s : %s"
                  % (f[0], f[1], [sorted(s) for s in f[2]], f[3]))
        sys.exit(1)
    print("\nALL TESTS PASSED.")


if __name__ == "__main__":
    main()
