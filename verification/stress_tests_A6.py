#!/usr/bin/env python3
"""
Independent stress tests complementing verify_A6_*.

ATK1: cross-composition (clients and products) outside the ranges of
      verify_A6_crosscomp.py: n_U=1, t_hat=m, t0=3 (odd padding), adversarial
      sources (one near-YES: minimum cover = t_hat+1; others NO;
      one trivially near-YES with large sets). OR check of both
      sides by brute force + structural check on EVERY design of cost <=B.
ATK2: guard sanity - composition D1 (without guards) must BREAK the OR
      (opening both selectors of a pair bypasses everything). Confirms that
      the guards are load-bearing and that the test would detect the hole.
ATK3: customer aggregation with extreme/asymmetric demands
      and verification via an independent LP (scipy.linprog) besides the MCMF.
ATK4: capping with b,p >> D_l and the (B,k) answer for every k.
ATK5: probe n_U = 0 (outside the paper's convention n_U>=1): documents that
      the PRODUCTS composition would fail without it (guards without a plant).
"""
import itertools
import random
import copy
from common_mp_tscfl import all_designs

try:
    from scipy.optimize import linprog
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

from verify_A6_crosscomp import (compose_clients, compose_products,
                                 sc_yes, pad_to_pow2)
from verify_A6_aggregation import total_cost, cost_map, merge_customers


# ---------------------------------------------------------------------------
# Independent LP: original residual LP in inequality form (per product)
# ---------------------------------------------------------------------------

def lp_block_value(inst, l, y, z):
    """Value of block l at (y,z) via scipy.linprog (None if infeasible)."""
    nI, nJ, nK = inst["nI"], inst["nJ"], inst["nK"]
    nx, nw = nI * nJ, nJ * nK
    cvec = ([inst["c"][i][j][l] for i in range(nI) for j in range(nJ)]
            + [inst["d"][j][k][l] for j in range(nJ) for k in range(nK)])
    A_ub, b_ub = [], []
    # (C1) -sum_j w_jk <= -q_kl
    for k in range(nK):
        row = [0.0] * (nx + nw)
        for j in range(nJ):
            row[nx + j * nK + k] = -1.0
        A_ub.append(row); b_ub.append(-inst["q"][k][l])
    # (C2) sum_k w_jk - sum_i x_ij <= 0
    for j in range(nJ):
        row = [0.0] * (nx + nw)
        for i in range(nI):
            row[i * nJ + j] = -1.0
        for k in range(nK):
            row[nx + j * nK + k] = 1.0
        A_ub.append(row); b_ub.append(0.0)
    # (C3) sum_j x_ij <= b_il y_i
    for i in range(nI):
        row = [0.0] * (nx + nw)
        for j in range(nJ):
            row[i * nJ + j] = 1.0
        A_ub.append(row); b_ub.append(inst["b"][i][l] * y[i])
    # (C4) sum_k w_jk <= p_jl z_j
    for j in range(nJ):
        row = [0.0] * (nx + nw)
        for k in range(nK):
            row[nx + j * nK + k] = 1.0
        A_ub.append(row); b_ub.append(inst["p"][j][l] * z[j])
    res = linprog(cvec, A_ub=A_ub, b_ub=b_ub, bounds=(0, None),
                  method="highs")
    if not res.success:
        return None
    return res.fun


def lp_total_cost(inst, y, z):
    cost = sum(inst["f"][i] * y[i] for i in range(inst["nI"]))
    cost += sum(inst["g"][j] * z[j] for j in range(inst["nJ"]))
    for l in range(inst["nL"]):
        v = lp_block_value(inst, l, y, z)
        if v is None:
            return None
        cost += v
    return cost


# ---------------------------------------------------------------------------
# ATK1: cross-composition outside the ranges
# ---------------------------------------------------------------------------

def full_check(compose, sets_list, nU, t_hat, label, expect_yes=None):
    inst, B, meta = compose(sets_list, nU, t_hat)
    tau, m, insts = meta["tau"], meta["m"], meta["insts"]
    or_src = any(sc_yes(nU, s, t_hat) for s in sets_list)
    if expect_yes is not None:
        assert or_src == expect_yes, f"[{label}] source: expected {expect_yes}"
    yes = False
    n_low = 0
    for y, z in all_designs(inst["nI"], inst["nJ"]):
        c = total_cost(inst, y, z)
        if c is None or c > B:
            continue
        yes = True
        n_low += 1
        # structural check on EVERY cheap design
        if compose is compose_products:
            assert all(y), f"[{label}] closed plant in a cheap design"
        for beta in range(tau):
            opens = [z[m + 2 * beta + v] for v in (0, 1)]
            assert sum(opens) == 1, \
                f"[{label}] CHEAT: pair {beta} with {sum(opens)} selectors"
        vpat = [0 if z[m + 2 * beta] else 1 for beta in range(tau)]
        istar = sum(v << beta for beta, v in enumerate(vpat))
        zsets = [j for j in range(m) if z[j]]
        assert len(zsets) <= t_hat, f"[{label}] CHEAT: |Z| > t_hat"
        for e in range(nU):
            assert any(e in insts[istar][j] for j in zsets), \
                f"[{label}] CHEAT: element {e} of i*={istar} uncovered"
    assert yes == or_src, f"[{label}] OR FAILED: composed={yes}, " \
        f"sources={or_src}"
    return yes, n_low


def atk1():
    total = 0
    # (a) n_U = 1 (outside the scripts' range nU>=2), t_hat = m, t0 = 3
    subsets1 = [frozenset(), frozenset([0])]
    for combo in itertools.product(subsets1, repeat=2):
        for combo2 in itertools.product(subsets1, repeat=2):
            for combo3 in itertools.product(subsets1, repeat=2):
                srcs = [list(combo), list(combo2), list(combo3)]
                for t_hat in (1, 2):
                    full_check(compose_clients, srcs, 1, t_hat,
                               f"ATK1a-cli nU=1 t^={t_hat}")
                    full_check(compose_products, srcs, 1, t_hat,
                               f"ATK1a-prod nU=1 t^={t_hat}")
                    total += 2
    print(f"[ATK1a] n_U=1, t0=3, t^ in {{1,2}}: {total} compositions: PASS")

    # (b) adversarial sources: near-YES (minimum cover = t_hat+1)
    #     nU=3, m=3, t_hat=1; each source covers U only with 2 sets.
    near = [frozenset([0, 1]), frozenset([1, 2]), frozenset([0, 2])]
    no1 = [frozenset([0]), frozenset([1]), frozenset()]      # does not cover 2
    no2 = [frozenset([2]), frozenset([2]), frozenset([2])]   # does not cover 0,1
    n = 0
    for srcs in ([near, near], [near, no1, no2], [no1, no2, near, near],
                 [no1, no1], [no2, no1, no1]):
        for s in srcs:
            assert not sc_yes(3, s, 1)
        y1, _ = full_check(compose_clients, srcs, 3, 1,
                           "ATK1b-cli near-YES", expect_yes=False)
        y2, _ = full_check(compose_products, srcs, 3, 1,
                           "ATK1b-prod near-YES", expect_yes=False)
        n += 2
    # and the YES control version: one source gets the full set
    yes_src = [frozenset([0, 1, 2]), frozenset(), frozenset()]
    for srcs in ([near, yes_src], [no1, no2, yes_src, near]):
        full_check(compose_clients, srcs, 3, 1, "ATK1b-cli control-YES",
                   expect_yes=True)
        full_check(compose_products, srcs, 3, 1, "ATK1b-prod control-YES",
                   expect_yes=True)
        n += 2
    print(f"[ATK1b] near-YES sources (min cover = t^+1) and controls: "
          f"{n} compositions: PASS")

    # (c) t_hat = m (edge of the WLOG convention), nU=2, m=2, t0=2
    subsets = [frozenset(), frozenset([0]), frozenset([1]), frozenset([0, 1])]
    n = 0
    for s1 in itertools.product(subsets, repeat=2):
        for s2 in itertools.product(subsets, repeat=2):
            full_check(compose_clients, [list(s1), list(s2)], 2, 2,
                       "ATK1c-cli t^=m")
            n += 1
    for k in range(0, 256, 5):
        a, b = divmod(k, 16)
        s1 = list(itertools.product(subsets, repeat=2))[a]
        s2 = list(itertools.product(subsets, repeat=2))[b]
        full_check(compose_products, [list(s1), list(s2)], 2, 2,
                   "ATK1c-prod t^=m")
        n += 1
    print(f"[ATK1c] t^ = m = 2 (WLOG edge): {n} compositions: PASS")


# ---------------------------------------------------------------------------
# ATK2: D1 without guards must break (guard sanity)
# ---------------------------------------------------------------------------

def compose_clients_noguards(sets_list, nU, t_hat):
    """D1: same clients composition, WITHOUT guards."""
    m = len(sets_list[0])
    insts, tau, tp = pad_to_pow2(sets_list)
    B = tau * (t_hat + 1) + t_hat
    W = B + 1
    nJ = m + 2 * tau
    D = tp * nU
    nK = tp * nU
    g = [1] * m + [t_hat + 1] * (2 * tau)
    d = [[[0] for _ in range(nK)] for _ in range(nJ)]
    for j in range(nJ):
        for k in range(nK):
            i, e = divmod(k, nU)
            if j < m:
                d[j][k][0] = 0 if e in insts[i][j] else W
            else:
                beta, v = divmod(j - m, 2)
                d[j][k][0] = 0 if ((i >> beta) & 1) != v else W
    inst = {"nI": 1, "nJ": nJ, "nK": nK, "nL": 1,
            "f": [0], "g": g,
            "c": [[[0] for _ in range(nJ)]],
            "d": d, "b": [[D]],
            "p": [[D] for _ in range(nJ)],
            "q": [[1] for _ in range(nK)]}
    return inst, B


def atk2():
    # sources all NO, t0 = 4 => tau = 2: two selectors of ONE pair must
    # bypass all instances in D1 (cost 2(t^+1) <= B) -> false YES.
    no1 = [frozenset([0]), frozenset([1])]
    srcs = [no1, no1, no1, no1]
    assert not any(sc_yes(2, s, 1) for s in srcs)
    inst, B = compose_clients_noguards(srcs, 2, 1)
    yes = any(c is not None and c <= B
              for c in (total_cost(inst, y, z)
                        for y, z in all_designs(inst["nI"], inst["nJ"])))
    assert yes, "ATK2: D1 without guards did NOT break (unexpected)"
    # and the final version with guards, on the same sources, answers NO:
    y2, _ = full_check(compose_clients, srcs, 2, 1, "ATK2-final",
                       expect_yes=False)
    print("[ATK2] D1 (no guards) gives a false YES with all-NO sources; "
          "D2 (with guards) answers NO: PASS (hole confirmed and closed)")


# ---------------------------------------------------------------------------
# ATK3: aggregation with extreme demands + independent LP
# ---------------------------------------------------------------------------

def atk3():
    assert HAVE_SCIPY, "scipy unavailable"
    rng = random.Random(20260710)
    n_inst = n_cmp = n_lp = 0
    for trial in range(25):
        nI = rng.randint(1, 3); nJ = rng.randint(1, 3)
        nK = rng.randint(2, 4); nL = rng.randint(1, 2)
        inst = {
            "nI": nI, "nJ": nJ, "nK": nK, "nL": nL,
            "f": [rng.choice([0, 1, 50]) for _ in range(nI)],
            "g": [rng.choice([0, 2, 40]) for _ in range(nJ)],
            "c": [[[rng.choice([0, 1, 7, 30]) for _ in range(nL)]
                   for _ in range(nJ)] for _ in range(nI)],
            "d": [[[rng.choice([0, 3, 25]) for _ in range(nL)]
                   for _ in range(nK)] for _ in range(nJ)],
            "b": [[rng.choice([0, 5, 60, 200]) for _ in range(nL)]
                  for _ in range(nI)],
            "p": [[rng.choice([0, 7, 55, 200]) for _ in range(nL)]
                  for _ in range(nJ)],
            "q": [[rng.choice([0, 1, 50]) for _ in range(nL)]
                  for _ in range(nK)],
        }
        # customers 0 and 1: identical columns, EXTREME demands and in
        # different products (k0 heavy in product 0, k1 heavy in the last product)
        for j in range(nJ):
            for l in range(nL):
                inst["d"][j][1][l] = inst["d"][j][0][l]
        inst["q"][0] = [60] + [0] * (nL - 1)
        inst["q"][1] = [0] * (nL - 1) + [47]
        # tight capacity: guarantees a zero-slack scenario in half the trials
        if trial % 2 == 0:
            for l in range(nL):
                D = sum(inst["q"][k][l] for k in range(nK))
                for i in range(nI):
                    inst["b"][i][l] = max(inst["b"][i][l], 1)
                # adjust so that sum b == D exactly (tight capacity)
                tot = sum(inst["b"][i][l] for i in range(nI))
                if tot > D:
                    exc = tot - D
                    for i in range(nI):
                        red = min(exc, inst["b"][i][l])
                        inst["b"][i][l] -= red; exc -= red
                elif tot < D:
                    inst["b"][0][l] += D - tot
        merged = merge_customers(inst, 0, 1)
        cmo, cmm = cost_map(inst), cost_map(merged)
        assert set(cmo) == set(cmm)
        for key in cmo:
            assert cmo[key] == cmm[key], \
                f"ATK3 trial={trial} design {key}: {cmo[key]} != {cmm[key]}"
            n_cmp += 1
        # independent LP contrast on up to 6 designs per instance
        keys = sorted(cmo)[:: max(1, len(cmo) // 6)]
        for (yy, zz) in keys:
            v_o = lp_total_cost(inst, list(yy), list(zz))
            v_m = lp_total_cost(merged, list(yy), list(zz))
            for v_lp, v_bf in ((v_o, cmo[(yy, zz)]), (v_m, cmm[(yy, zz)])):
                if v_bf is None:
                    assert v_lp is None
                else:
                    assert v_lp is not None and abs(v_lp - v_bf) < 1e-6
                n_lp += 1
        n_inst += 1
    print(f"[ATK3] extreme aggregation (demands 60/47 in distinct products, "
          f"tight capacities): {n_inst} instances, {n_cmp} comparisons, "
          f"{n_lp} LP contrasts: PASS")


# ---------------------------------------------------------------------------
# ATK4: capping with b,p >> D and the (B,k) version
# ---------------------------------------------------------------------------

def atk4():
    rng = random.Random(777)
    n_inst = n_k = 0
    for trial in range(20):
        nI = rng.randint(1, 3); nJ = rng.randint(1, 3)
        nK = rng.randint(1, 3); nL = rng.randint(1, 2)
        inst = {
            "nI": nI, "nJ": nJ, "nK": nK, "nL": nL,
            "f": [rng.randint(0, 9) for _ in range(nI)],
            "g": [rng.randint(0, 9) for _ in range(nJ)],
            "c": [[[rng.randint(0, 5) for _ in range(nL)]
                   for _ in range(nJ)] for _ in range(nI)],
            "d": [[[rng.randint(0, 5) for _ in range(nL)]
                   for _ in range(nK)] for _ in range(nJ)],
            "b": [[rng.choice([0, 3, 10 ** 6]) for _ in range(nL)]
                  for _ in range(nI)],
            "p": [[rng.choice([0, 4, 10 ** 6]) for _ in range(nL)]
                  for _ in range(nJ)],
            "q": [[rng.randint(0, 8) for _ in range(nL)]
                  for _ in range(nK)],
        }
        capped = copy.deepcopy(inst)
        for l in range(nL):
            D = sum(inst["q"][k][l] for k in range(nK))
            for i in range(nI):
                capped["b"][i][l] = min(capped["b"][i][l], D)
            for j in range(nJ):
                capped["p"][j][l] = min(capped["p"][j][l], D)
        cmo, cmc = cost_map(inst), cost_map(capped)
        assert cmo == cmc, f"ATK4 trial={trial}: capping altered some design"
        # (B,k) answer for every k and B around the optimum
        for k in range(nI + nJ + 1):
            def best_k(cm):
                vals = [v for (yy, zz), v in cm.items()
                        if v is not None and sum(yy) + sum(zz) <= k]
                return min(vals) if vals else None
            assert best_k(cmo) == best_k(cmc)
            n_k += 1
        n_inst += 1
    print(f"[ATK4] capping with b,p up to 10^6: {n_inst} instances, "
          f"(B,k) answers identical for every k ({n_k} checks): PASS")


# ---------------------------------------------------------------------------
# ATK5: probe n_U = 0 (outside the convention) - documents the dependency
# ---------------------------------------------------------------------------

def atk5():
    srcs = [[frozenset(), frozenset()], [frozenset(), frozenset()]]
    or_src = any(sc_yes(0, s, 1) for s in srcs)  # empty cover: YES
    assert or_src
    # clients: still works (guards only)
    inst, B, meta = compose_clients(srcs, 0, 1)
    yes_cli = any(c is not None and c <= B
                  for c in (total_cost(inst, y, z)
                            for y, z in all_designs(inst["nI"], inst["nJ"])))
    # products: the guard requires a carrier plant (b[0][tp+beta] = 1);
    # with n_U = 0 the CONSTRUCTION itself fails -- no composed instance exists.
    try:
        inst2, B2, _ = compose_products(srcs, 0, 1)
        yes_prod = any(c is not None and c <= B2
                       for c in (total_cost(inst2, y, z)
                                 for y, z in all_designs(inst2["nI"],
                                                         inst2["nJ"])))
        prod_msg = "YES" if yes_prod else "NO (OR would fail)"
    except IndexError:
        prod_msg = "CONSTRUCTION FAILS (guard without a carrier plant)"
    print(f"[ATK5] probe n_U=0 (outside convention A2 par.0): sources YES; "
          f"clients -> {'YES' if yes_cli else 'NO'}; "
          f"products -> {prod_msg} "
          f"(the products composition DEPENDS on n_U>=1; cf. O1(b))")


if __name__ == "__main__":
    atk1()
    atk2()
    atk3()
    atk4()
    atk5()
    print("stress_tests_A6.py: completed")
