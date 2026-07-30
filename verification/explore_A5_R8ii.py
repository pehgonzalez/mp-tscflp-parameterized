"""
Computational exploration of the paper's open problem on the cell
|K| = |L| = 1 with general transport costs.

Cell: |K| = |L| = 1, GENERAL transport costs (c_ij, d_j), |I| and |J|
free. We write c^(Y,Z) := f(Y) + g(Z) + v(Y,Z) for the total cost of
design (Y,Z), with c^ = +infinity if the design is infeasible, and v(Y,Z)
the value of the residual LP (MCMF routing oracle, exact integer
implementation from common_mp_tscfl.py).

Batteries (all with fixed seeds; counts printed at the end):

  [S]   Sub/supermodularity of c^(Y,Z) and of v(Y,Z) on the product
        lattice 2^I x 2^J (join = union, meet = intersection).
        MOTIVATION: if c^ were submodular, submodular function
        minimization (poly, Groetschel-Lovasz-Schrijver) would settle
        the open problem POSITIVELY. Tested first because it is decisive.
        Structural obstruction recorded separately: with A and B feasible
        the meet can be infeasible (c^(meet) = +inf), which already
        violates the submodular inequality; we also count violations with
        all FOUR points finite, which are the clean evidence.

  [P]   Prefix invariant: is there an optimal design whose set Y
        (resp. Z) is a prefix under natural orderings? Orderings on
        side I: (P1) f_i increasing; (P2) f_i/b_i increasing;
        (P3) (f_i + b_i * mincost_i)/b_i increasing, where
        mincost_i = min_j (c_ij + d_j). Side J: symmetric.
        Tie-tolerant prefix test: Y passes iff
        max_{i in Y} key_i <= min_{i not in Y} key_i.

  [G]   Decoupled heuristic: choose Y minimizing f(Y) subject to
        sum b_i >= D and Z minimizing g(Z) subject to sum p_j >= D
        (ignoring transport), then route optimally. Compare
        with OPT (brute force): we count strict failures.

  [SEP] Sanity of the separable-costs proposition: on instances with
        SEPARABLE costs
        c_ij = gamma_i + delta_j, the pair of decoupled DPs (side I with
        (f_i, gamma_i, b_i); side J with (g_j, delta_j + d_j, p_j)) must
        match the brute force EXACTLY. (The formal verification
        of the proposition is in verify_A5_R8ii.py; here it is the
        exploration's beacon.)

Brute force: enumeration of ALL designs (y,z) + integer MCMF
oracle (routing_value). No closed form is used as a source.
"""

import random
from fractions import Fraction

from common_mp_tscfl import routing_value

INF = float("inf")


# ---------------------------------------------------------------------------
# Instances of the cell |K| = |L| = 1
# ---------------------------------------------------------------------------

def make_inst(f, g, b, p, c, d, D):
    nI, nJ = len(f), len(g)
    return {
        "nI": nI, "nJ": nJ, "nK": 1, "nL": 1,
        "f": list(f), "g": list(g),
        "c": [[[c[i][j]] for j in range(nJ)] for i in range(nI)],
        "d": [[[d[j]]] for j in range(nJ)],
        "b": [[b[i]] for i in range(nI)],
        "p": [[p[j]] for j in range(nJ)],
        "q": [[D]],
    }


def gen_cell(seed, max_i=4, max_j=4, vmax=8, capmax=10, dcap=25):
    """Random instance of the cell, always feasible with everything open."""
    rng = random.Random(seed)
    nI = rng.randint(1, max_i)
    nJ = rng.randint(1, max_j)
    f = [rng.randint(0, vmax) for _ in range(nI)]
    g = [rng.randint(0, vmax) for _ in range(nJ)]
    b = [rng.randint(1, capmax) for _ in range(nI)]
    p = [rng.randint(1, capmax) for _ in range(nJ)]
    c = [[rng.randint(0, vmax) for _ in range(nJ)] for _ in range(nI)]
    d = [rng.randint(0, vmax) for _ in range(nJ)]
    D = rng.randint(1, min(sum(b), sum(p), dcap))
    return make_inst(f, g, b, p, c, d, D)


def value_table(inst):
    """(total cost, routing cost) of all designs; INF if
    infeasible. Source: MCMF oracle, design by design."""
    nI, nJ = inst["nI"], inst["nJ"]
    tab = {}
    for my in range(1 << nI):
        y = [(my >> i) & 1 for i in range(nI)]
        fy = sum(inst["f"][i] for i in range(nI) if y[i])
        for mz in range(1 << nJ):
            z = [(mz >> j) & 1 for j in range(nJ)]
            feas, v = routing_value(inst, 0, y, z)
            if feas:
                gz = sum(inst["g"][j] for j in range(nJ) if z[j])
                tab[(my, mz)] = (fy + gz + v, v)
            else:
                tab[(my, mz)] = (INF, INF)
    return tab


# ---------------------------------------------------------------------------
# [S] sub/supermodularity on the product lattice
# ---------------------------------------------------------------------------

def battery_S(inst_list):
    cnt = {
        "quadruplas_AB_finitas": 0,
        "meet_inviavel": 0,          # c^(A), c^(B) finite, meet = +inf
        "quatro_finitos": 0,
        "viol_submod_chat": 0,       # c^(join)+c^(meet) > c^(A)+c^(B)
        "viol_supermod_chat": 0,     # c^(join)+c^(meet) < c^(A)+c^(B)
        "viol_submod_v": 0,
        "viol_supermod_v": 0,
        "quatro_finitos_v": 0,
    }
    first_sub = first_super = None
    for seed, inst in inst_list:
        tab = value_table(inst)
        keys = sorted(tab)
        n = len(keys)
        for a in range(n):
            A = keys[a]
            cA, vA = tab[A]
            if cA == INF:
                continue
            for bidx in range(a + 1, n):
                B = keys[bidx]
                cB, vB = tab[B]
                if cB == INF:
                    continue
                cnt["quadruplas_AB_finitas"] += 1
                J = (A[0] | B[0], A[1] | B[1])
                M = (A[0] & B[0], A[1] & B[1])
                cJ, vJ = tab[J]
                cM, vM = tab[M]
                if cM == INF:
                    cnt["meet_inviavel"] += 1
                    continue
                cnt["quatro_finitos"] += 1
                lhs, rhs = cJ + cM, cA + cB
                if lhs > rhs:
                    cnt["viol_submod_chat"] += 1
                    if first_sub is None:
                        first_sub = (seed, A, B, J, M, cA, cB, cJ, cM)
                elif lhs < rhs:
                    cnt["viol_supermod_chat"] += 1
                    if first_super is None:
                        first_super = (seed, A, B, J, M, cA, cB, cJ, cM)
                # pure routing
                cnt["quatro_finitos_v"] += 1
                lv, rv = vJ + vM, vA + vB
                if lv > rv:
                    cnt["viol_submod_v"] += 1
                elif lv < rv:
                    cnt["viol_supermod_v"] += 1
    return cnt, first_sub, first_super


# ---------------------------------------------------------------------------
# [P] prefix invariant
# ---------------------------------------------------------------------------

def prefix_ok(keys, members, universe):
    """Y is a prefix (tie-tolerant) under the ordering by keys."""
    inside = [keys[i] for i in universe if i in members]
    outside = [keys[i] for i in universe if i not in members]
    if not inside or not outside:
        return True
    return max(inside) <= min(outside)


def battery_P(inst_list):
    lados = ["I_f", "I_f/b", "I_efetivo", "J_g", "J_g/p", "J_efetivo"]
    fail = {nome: 0 for nome in lados}
    exemplos = {nome: None for nome in lados}
    n_inst = 0
    for seed, inst in inst_list:
        tab = value_table(inst)
        opt = min(v for v, _ in tab.values())
        if opt == INF:
            continue
        n_inst += 1
        opts = [k for k, (v, _) in tab.items() if v == opt]
        nI, nJ = inst["nI"], inst["nJ"]
        f, g = inst["f"], inst["g"]
        b = [inst["b"][i][0] for i in range(nI)]
        p = [inst["p"][j][0] for j in range(nJ)]
        cd = [[inst["c"][i][j][0] + inst["d"][j][0][0] for j in range(nJ)]
              for i in range(nI)]
        minc_i = [min(cd[i]) for i in range(nI)]
        minc_j = [min(cd[i][j] for i in range(nI)) for j in range(nJ)]
        keysI = {
            "I_f": [Fraction(f[i]) for i in range(nI)],
            "I_f/b": [Fraction(f[i], b[i]) for i in range(nI)],
            "I_efetivo": [Fraction(f[i] + b[i] * minc_i[i], b[i])
                          for i in range(nI)],
        }
        keysJ = {
            "J_g": [Fraction(g[j]) for j in range(nJ)],
            "J_g/p": [Fraction(g[j], p[j]) for j in range(nJ)],
            "J_efetivo": [Fraction(g[j] + p[j] * minc_j[j], p[j])
                          for j in range(nJ)],
        }
        for nome, keys in keysI.items():
            ok = any(prefix_ok(keys, {i for i in range(nI) if my >> i & 1},
                               range(nI)) for my, _ in opts)
            if not ok:
                fail[nome] += 1
                if exemplos[nome] is None:
                    exemplos[nome] = seed
        for nome, keys in keysJ.items():
            ok = any(prefix_ok(keys, {j for j in range(nJ) if mz >> j & 1},
                               range(nJ)) for _, mz in opts)
            if not ok:
                fail[nome] += 1
                if exemplos[nome] is None:
                    exemplos[nome] = seed
    return n_inst, fail, exemplos


# ---------------------------------------------------------------------------
# [G] decoupled heuristic vs OPT
# ---------------------------------------------------------------------------

def battery_G(inst_list):
    n_inst = 0
    n_fail = 0
    max_gap = 0
    exemplo = None
    for seed, inst in inst_list:
        tab = value_table(inst)
        opt = min(v for v, _ in tab.values())
        if opt == INF:
            continue
        n_inst += 1
        nI, nJ = inst["nI"], inst["nJ"]
        D = inst["q"][0][0]
        b = [inst["b"][i][0] for i in range(nI)]
        p = [inst["p"][j][0] for j in range(nJ)]
        # side I: min f(Y) s.t. sum_{i in Y} b_i >= D (brute force)
        bestY = min((my for my in range(1 << nI)
                     if sum(b[i] for i in range(nI) if my >> i & 1) >= D),
                    key=lambda my: (sum(inst["f"][i] for i in range(nI)
                                        if my >> i & 1), bin(my).count("1"),
                                    my))
        bestZ = min((mz for mz in range(1 << nJ)
                     if sum(p[j] for j in range(nJ) if mz >> j & 1) >= D),
                    key=lambda mz: (sum(inst["g"][j] for j in range(nJ)
                                        if mz >> j & 1), bin(mz).count("1"),
                                    mz))
        heur = tab[(bestY, bestZ)][0]
        assert heur < INF  # aggregate condition => feasible (feasibility characterization)
        assert heur >= opt
        if heur > opt:
            n_fail += 1
            if heur - opt > max_gap:
                max_gap = heur - opt
                exemplo = (seed, heur, opt)
    return n_inst, n_fail, max_gap, exemplo


# ---------------------------------------------------------------------------
# [SEP] separable costs: decoupled DPs == brute force
# ---------------------------------------------------------------------------

def side_dp(items, D):
    """min sum_{i in S} (F_i + gam_i * u_i) with u_i in [1..cap_i] for i in S,
    sum u_i = D. items = list of (F, gam, cap)."""
    T = [0] + [INF] * D
    for (F, gam, cap) in items:
        newT = list(T)
        for t in range(1, D + 1):
            up = min(cap, t)
            for u in range(1, up + 1):
                if T[t - u] < INF:
                    cand = T[t - u] + F + gam * u
                    if cand < newT[t]:
                        newT[t] = cand
        T = newT
    return T[D]


def side_dp_deque(items, D):
    """Same problem as side_dp in O(len(items) * D) via sliding-window
    minimum with a monotone deque (substitution t = s - u, key
    T[t] - gam * t), valid for gam of either sign. INF entries do not
    enter the deque, never attain a finite minimum."""
    from collections import deque
    T = [0] + [INF] * D
    for (F, gam, cap) in items:
        newT = list(T)
        dq = deque()
        for s in range(1, D + 1):
            t_new = s - 1
            if T[t_new] < INF:
                key = T[t_new] - gam * t_new
                while dq and (T[dq[-1]] - gam * dq[-1]) >= key:
                    dq.pop()
                dq.append(t_new)
            lo = s - cap
            while dq and dq[0] < lo:
                dq.popleft()
            if dq:
                cand = F + gam * s + (T[dq[0]] - gam * dq[0])
                if cand < newT[s]:
                    newT[s] = cand
        T = newT
    return T[D]


def battery_DEQ(n=600, seed0=8000):
    """side_dp_deque == side_dp on random lists, gam of both
    signs, varied capacities and D, including infeasible cases."""
    n_ok = n_bad = 0
    for s in range(seed0, seed0 + n):
        rng = random.Random(s)
        m = rng.randint(1, 6)
        items = [(rng.randint(0, 10), rng.randint(-5, 5), rng.randint(1, 9))
                 for _ in range(m)]
        D = rng.randint(0, 40)
        a, b = side_dp(items, D), side_dp_deque(items, D)
        if a == b:
            n_ok += 1
        else:
            n_bad += 1
            print(f"  [DEQ] MISMATCH seed={s}: naive={a} deque={b}")
    return n_ok, n_bad


def sep_opt(inst, gamma, delta):
    D = inst["q"][0][0]
    if D == 0:
        return 0
    nI, nJ = inst["nI"], inst["nJ"]
    a = side_dp([(inst["f"][i], gamma[i], inst["b"][i][0])
                 for i in range(nI)], D)
    bb = side_dp([(inst["g"][j], delta[j] + inst["d"][j][0][0],
                   inst["p"][j][0]) for j in range(nJ)], D)
    return a + bb if a < INF and bb < INF else INF


def battery_SEP(n=60, seed0=7000):
    n_ok = n_bad = 0
    for s in range(seed0, seed0 + n):
        rng = random.Random(s)
        nI = rng.randint(1, 4)
        nJ = rng.randint(1, 4)
        f = [rng.randint(0, 8) for _ in range(nI)]
        g = [rng.randint(0, 8) for _ in range(nJ)]
        b = [rng.randint(1, 8) for _ in range(nI)]
        p = [rng.randint(1, 8) for _ in range(nJ)]
        gamma = [rng.randint(0, 5) for _ in range(nI)]
        delta = [rng.randint(0, 5) for _ in range(nJ)]
        c = [[gamma[i] + delta[j] for j in range(nJ)] for i in range(nI)]
        d = [rng.randint(0, 4) for _ in range(nJ)]
        r = rng.random()
        if r < 0.1:
            D = 0
        elif r < 0.25:
            D = max(sum(b), sum(p)) + rng.randint(1, 5)   # infeasible
        else:
            D = rng.randint(1, min(sum(b), sum(p)))
        inst = make_inst(f, g, b, p, c, d, D)
        tab = value_table(inst)
        opt = min(v for v, _ in tab.values())
        dpval = sep_opt(inst, gamma, delta)
        if opt == dpval:
            n_ok += 1
        else:
            n_bad += 1
            print(f"  [SEP] MISMATCH seed={s}: brute={opt} dp={dpval}")
    return n_ok, n_bad


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    okd, badd = battery_DEQ()
    print(f"== [DEQ] sliding window vs naive DP: {okd} ok, {badd} mismatches ==")
    # 150 4x4 instances + 40 5x5 instances (fixed seeds)
    small = [(s, gen_cell(s, 4, 4)) for s in range(1000, 1150)]
    big = [(s, gen_cell(s, 5, 5)) for s in range(2000, 2040)]
    todas = small + big

    print("== [S] sub/supermodularity of c^ and v on the lattice 2^I x 2^J ==")
    cnt, first_sub, first_super = battery_S(todas)
    for k, v in cnt.items():
        print(f"  {k}: {v}")
    if first_sub:
        s, A, B, J, M, cA, cB, cJ, cM = first_sub
        print(f"  first SUBmodularity counterexample (4 finite points):")
        print(f"    seed={s} A={A} B={B} join={J} meet={M}")
        print(f"    c^(A)={cA} c^(B)={cB} c^(join)={cJ} c^(meet)={cM} "
              f"(lhs={cJ+cM} > rhs={cA+cB})")
    if first_super:
        s, A, B, J, M, cA, cB, cJ, cM = first_super
        print(f"  first SUPERmodularity counterexample:")
        print(f"    seed={s} A={A} B={B} join={J} meet={M}")
        print(f"    c^(A)={cA} c^(B)={cB} c^(join)={cJ} c^(meet)={cM} "
              f"(lhs={cJ+cM} < rhs={cA+cB})")

    print("== [P] prefix invariant on optimal designs ==")
    n_inst, fail, exemplos = battery_P(todas)
    print(f"  instances with finite OPT: {n_inst}")
    for nome in fail:
        print(f"  ordering {nome}: {fail[nome]} instances without ANY "
              f"prefix-optimum (first seed: {exemplos[nome]})")

    print("== [G] decoupled heuristic (ignores transport) vs OPT ==")
    n_inst, n_fail, max_gap, exemplo = battery_G(todas)
    print(f"  instances: {n_inst}; strict failures: {n_fail}; "
          f"maximum gap: {max_gap}; example: {exemplo}")

    print("== [SEP] separable costs c_ij = gamma_i + delta_j ==")
    n_ok, n_bad = battery_SEP()
    print(f"  decoupled DP == brute force agreement: {n_ok} ok, "
          f"{n_bad} mismatches")

    print("== exploration completed ==")


if __name__ == "__main__":
    main()
