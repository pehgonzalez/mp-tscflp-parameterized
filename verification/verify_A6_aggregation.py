#!/usr/bin/env python3
"""
Computational verification of the paper's aggregation and compression results.

Batteries:
  [A] Aggregation   - aggregation of customers with identical d columns is exact
                    PER DESIGN (identical cost and feasibility on every (y,z)),
                    hence it preserves OPT and the set of optimal designs.
                    40 seeded random instances + 4 adversarial ones
                    (zero demand, total ties, global infeasibility,
                    three-way merge with order independence).
  [B] Counterexample - merging PRODUCTS with identical cost columns
                    is NOT exact: a value counterexample (OPT 10 -> 0) and
                    a feasibility counterexample, verified numerically.
  [C] Subadditivity - merging products never INCREASES the cost of a design
                    (v_merged <= v_l + v_l'), with strict cases observed.
  [D] Proportional  - merging PROPORTIONAL products ((q,b,p) scaled by
                    lambda, equal costs) is exact per design.
  [E] Capping       - capping b_il := min(b_il, D_l), p_jl := min(p_jl, D_l)
                    is exact per design.

Brute force: enumeration of all designs (y,z) + exact integer MCMF
routing oracle (common_mp_tscfl.routing_value). No closed
form from the proofs is used as a source.
"""
import copy
from common_mp_tscfl import gen_instance, routing_value, all_designs


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def total_cost(inst, y, z):
    """Total cost of design (y,z): fixed + routing; None if infeasible."""
    cost = sum(inst["f"][i] * y[i] for i in range(inst["nI"]))
    cost += sum(inst["g"][j] * z[j] for j in range(inst["nJ"]))
    for l in range(inst["nL"]):
        feas, val = routing_value(inst, l, y, z)
        if not feas:
            return None
        cost += val
    return cost


def cost_map(inst):
    """Map {(y,z) -> cost | None} over all designs."""
    return {(tuple(y), tuple(z)): total_cost(inst, y, z)
            for y, z in all_designs(inst["nI"], inst["nJ"])}


def opt_and_argmin(cm):
    finite = {k: v for k, v in cm.items() if v is not None}
    if not finite:
        return None, set()
    opt = min(finite.values())
    return opt, {k for k, v in finite.items() if v == opt}


def merge_customers(inst, k1, k2):
    """Merges k2 into k1 (identical d columns required), adding demands."""
    assert k1 != k2
    for j in range(inst["nJ"]):
        for l in range(inst["nL"]):
            assert inst["d"][j][k1][l] == inst["d"][j][k2][l], \
                "d columns not identical"
    out = copy.deepcopy(inst)
    for l in range(inst["nL"]):
        out["q"][k1][l] += out["q"][k2][l]
    del out["q"][k2]
    for j in range(inst["nJ"]):
        del out["d"][j][k2]
    out["nK"] -= 1
    return out


def merge_products(inst, l1, l2):
    """Merges l2 into l1 (identical c,d columns required), adding q, b, p."""
    nI, nJ, nK = inst["nI"], inst["nJ"], inst["nK"]
    for i in range(nI):
        for j in range(nJ):
            assert inst["c"][i][j][l1] == inst["c"][i][j][l2]
    for j in range(nJ):
        for k in range(nK):
            assert inst["d"][j][k][l1] == inst["d"][j][k][l2]
    out = copy.deepcopy(inst)
    for i in range(nI):
        out["b"][i][l1] += out["b"][i][l2]
        del out["b"][i][l2]
    for j in range(nJ):
        out["p"][j][l1] += out["p"][j][l2]
        del out["p"][j][l2]
    for k in range(nK):
        out["q"][k][l1] += out["q"][k][l2]
        del out["q"][k][l2]
    for i in range(nI):
        for j in range(nJ):
            del out["c"][i][j][l2]
    for j in range(nJ):
        for k in range(nK):
            del out["d"][j][k][l2]
    out["nL"] -= 1
    return out


def check_design_equality(inst_a, inst_b, label):
    """Identical costs (including None) on EVERY design; OPT and argmin."""
    cma, cmb = cost_map(inst_a), cost_map(inst_b)
    assert set(cma) == set(cmb)
    n = 0
    for key in cma:
        assert cma[key] == cmb[key], f"[{label}] design {key}: " \
            f"{cma[key]} != {cmb[key]}"
        n += 1
    oa, aa = opt_and_argmin(cma)
    ob, ab = opt_and_argmin(cmb)
    assert oa == ob and aa == ab, f"[{label}] OPT/argmin diverge"
    return n


# ---------------------------------------------------------------------------
# [A] customer aggregation
# ---------------------------------------------------------------------------

def battery_A():
    n_inst = n_checks = n_zero_demand = n_threeway = 0
    for seed in range(6000, 6040):
        s = seed
        inst = gen_instance(s, max_i=3, max_j=3, max_k=4, max_l=2, vmax=6)
        while inst["nK"] < 2:
            s += 977
            inst = gen_instance(s, max_i=3, max_j=3, max_k=4, max_l=2, vmax=6)
        # column of customer 1 := column of customer 0 (forced duplicate)
        for j in range(inst["nJ"]):
            for l in range(inst["nL"]):
                inst["d"][j][1][l] = inst["d"][j][0][l]
        if seed % 5 == 0:  # embedded adversarial: duplicate with zero demand
            for l in range(inst["nL"]):
                inst["q"][1][l] = 0
            n_zero_demand += 1
        merged = merge_customers(inst, 0, 1)
        n_checks += check_design_equality(inst, merged, f"A seed={seed}")
        # three-way merge with order independence
        if seed % 7 == 0 and inst["nK"] >= 3:
            inst3 = copy.deepcopy(inst)
            for j in range(inst3["nJ"]):
                for l in range(inst3["nL"]):
                    inst3["d"][j][2][l] = inst3["d"][j][0][l]
            m_ab_c = merge_customers(merge_customers(inst3, 0, 1), 0, 1)
            m_ac_b = merge_customers(merge_customers(inst3, 0, 2), 0, 1)
            n_checks += check_design_equality(inst3, m_ab_c,
                                              f"A3 seed={seed} (0,1)+2")
            n_checks += check_design_equality(m_ab_c, m_ac_b,
                                              f"A3 seed={seed} order")
            n_threeway += 1
        n_inst += 1

    # 4 hand-built adversarial instances
    adv = []
    # adv1: all-zero (ties on all designs)
    adv.append({"nI": 1, "nJ": 1, "nK": 2, "nL": 1,
                "f": [0], "g": [0], "c": [[[0]]],
                "d": [[[0], [0]]], "b": [[5]], "p": [[5]],
                "q": [[2], [3]]})
    # adv2: tight capacity after the merge (p = q1+q2)
    adv.append({"nI": 1, "nJ": 2, "nK": 2, "nL": 1,
                "f": [1], "g": [2, 3], "c": [[[1], [4]]],
                "d": [[[2], [2]], [[0], [0]]], "b": [[7]], "p": [[7], [7]],
                "q": [[0], [7]]})
    # adv3: infeasible on every design (b < D)
    adv.append({"nI": 1, "nJ": 1, "nK": 2, "nL": 1,
                "f": [0], "g": [0], "c": [[[1]]],
                "d": [[[1], [1]]], "b": [[2]], "p": [[9]],
                "q": [[2], [2]]})
    # adv4: two products, identical d columns in BOTH products
    adv.append({"nI": 2, "nJ": 2, "nK": 2, "nL": 2,
                "f": [3, 1], "g": [2, 2],
                "c": [[[1, 0], [2, 5]], [[0, 3], [1, 1]]],
                "d": [[[4, 1], [4, 1]], [[0, 2], [0, 2]]],
                "b": [[5, 3], [4, 4]], "p": [[6, 4], [5, 5]],
                "q": [[3, 2], [2, 1]]})
    for idx, inst in enumerate(adv):
        merged = merge_customers(inst, 0, 1)
        n_checks += check_design_equality(inst, merged, f"A adv{idx + 1}")
        n_inst += 1

    print(f"[A] customer aggregation: {n_inst} instances "
          f"({n_zero_demand} with zero demand, {n_threeway} three-way merges), "
          f"{n_checks} per-design comparisons: PASS")


# ---------------------------------------------------------------------------
# [B] product counterexamples
# ---------------------------------------------------------------------------

def battery_B():
    # B1: value. Identical costs across products; crossed capacities.
    orig = {"nI": 2, "nJ": 1, "nK": 1, "nL": 2,
            "f": [0, 0], "g": [0],
            "c": [[[0, 0]], [[10, 10]]],
            "d": [[[0, 0]]],
            "b": [[2, 0], [0, 2]], "p": [[2, 2]],
            "q": [[1, 1]]}
    merged = merge_products(orig, 0, 1)
    opt_o, arg_o = opt_and_argmin(cost_map(orig))
    opt_m, arg_m = opt_and_argmin(cost_map(merged))
    assert opt_o == 10, f"B1: original OPT = {opt_o}, expected 10"
    assert opt_m == 0, f"B1: merged OPT = {opt_m}, expected 0"
    assert arg_o != arg_m, "B1: sets of optimal designs should differ"
    # design (1,0),(1): infeasible in the original (product 2 without capacity),
    # feasible and optimal in the merged instance
    key = ((1, 0), (1,))
    assert total_cost(orig, [1, 0], [1]) is None
    assert total_cost(merged, [1, 0], [1]) == 0
    print(f"[B1] value counterexample: OPT 10 -> 0, argmin changes, "
          f"design {key} infeasible->optimal: PASS")

    # B2: feasibility. Original infeasible on every design; merged feasible.
    orig2 = {"nI": 1, "nJ": 1, "nK": 1, "nL": 2,
             "f": [0], "g": [0],
             "c": [[[0, 0]]], "d": [[[0, 0]]],
             "b": [[2, 0]], "p": [[2, 2]],
             "q": [[1, 1]]}
    merged2 = merge_products(orig2, 0, 1)
    cm_o = cost_map(orig2)
    assert all(v is None for v in cm_o.values()), "B2: original has a feasible design"
    opt2, _ = opt_and_argmin(cost_map(merged2))
    assert opt2 == 0, f"B2: merged OPT = {opt2}, expected 0"
    print("[B2] feasibility counterexample (infeasible -> feasible): PASS")


# ---------------------------------------------------------------------------
# [C] subadditivity of product merging
# ---------------------------------------------------------------------------

def battery_C():
    n_inst = n_checks = n_strict = 0
    for seed in range(6200, 6230):
        s = seed
        inst = gen_instance(s, max_i=3, max_j=3, max_k=3, max_l=2, vmax=6)
        while inst["nL"] < 2:
            s += 977
            inst = gen_instance(s, max_i=3, max_j=3, max_k=3, max_l=2, vmax=6)
        # equalize cost columns of products 0 and 1 (b,p,q independent)
        for i in range(inst["nI"]):
            for j in range(inst["nJ"]):
                inst["c"][i][j][1] = inst["c"][i][j][0]
        for j in range(inst["nJ"]):
            for k in range(inst["nK"]):
                inst["d"][j][k][1] = inst["d"][j][k][0]
        merged = merge_products(inst, 0, 1)
        cmo, cmm = cost_map(inst), cost_map(merged)
        for key in cmo:
            if cmo[key] is not None:
                assert cmm[key] is not None, \
                    f"[C seed={seed}] feasible became infeasible after merge"
                assert cmm[key] <= cmo[key], \
                    f"[C seed={seed}] merge increased cost at {key}"
                if cmm[key] < cmo[key]:
                    n_strict += 1
                n_checks += 1
        n_inst += 1
    assert n_strict > 0, "[C] no strict case observed (weak battery)"
    print(f"[C] subadditivity: {n_inst} instances, {n_checks} feasible "
          f"designs, {n_strict} strictly smaller: PASS")


# ---------------------------------------------------------------------------
# [D] proportional products
# ---------------------------------------------------------------------------

def battery_D():
    n_inst = n_checks = 0
    for seed in range(6300, 6330):
        s = seed
        base = gen_instance(s, max_i=3, max_j=3, max_k=3, max_l=1, vmax=6)
        lam = 1 + seed % 3
        nI, nJ, nK = base["nI"], base["nJ"], base["nK"]
        two = {"nI": nI, "nJ": nJ, "nK": nK, "nL": 2,
               "f": base["f"][:], "g": base["g"][:],
               "c": [[[base["c"][i][j][0]] * 2 for j in range(nJ)]
                     for i in range(nI)],
               "d": [[[base["d"][j][k][0]] * 2 for k in range(nK)]
                     for j in range(nJ)],
               "b": [[base["b"][i][0], lam * base["b"][i][0]]
                     for i in range(nI)],
               "p": [[base["p"][j][0], lam * base["p"][j][0]]
                     for j in range(nJ)],
               "q": [[base["q"][k][0], lam * base["q"][k][0]]
                     for k in range(nK)]}
        merged = merge_products(two, 0, 1)
        n_checks += check_design_equality(two, merged,
                                          f"D seed={seed} lam={lam}")
        n_inst += 1
    print(f"[D] proportional products: {n_inst} instances "
          f"(lambda in 1..3), {n_checks} per-design comparisons: PASS")


# ---------------------------------------------------------------------------
# [E] capacity capping
# ---------------------------------------------------------------------------

def battery_E():
    n_inst = n_checks = n_capped = 0
    for seed in range(6100, 6130):
        inst = gen_instance(seed, max_i=3, max_j=3, max_k=3, max_l=2, vmax=9)
        capped = copy.deepcopy(inst)
        changed = False
        for l in range(inst["nL"]):
            D = sum(inst["q"][k][l] for k in range(inst["nK"]))
            for i in range(inst["nI"]):
                if capped["b"][i][l] > D:
                    capped["b"][i][l] = D
                    changed = True
            for j in range(inst["nJ"]):
                if capped["p"][j][l] > D:
                    capped["p"][j][l] = D
                    changed = True
        if changed:
            n_capped += 1
        n_checks += check_design_equality(inst, capped, f"E seed={seed}")
        n_inst += 1
    print(f"[E] capping b,p <= D_l: {n_inst} instances "
          f"({n_capped} effectively altered), {n_checks} comparisons: PASS")


if __name__ == "__main__":
    battery_A()
    battery_B()
    battery_C()
    battery_D()
    battery_E()
    print("verify_A6_aggregation.py: ALL BATTERIES PASSED")
