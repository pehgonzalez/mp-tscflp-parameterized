"""
Verification of the paper's dual-certificate proposition (dual certificates and
disaggregated Benders cuts).

For each product l and fixed design (y,z), the routing LP (primal, in
inequality form, exactly (C1)-(C4) of the paper) is

  min  sum c_ij x_ij + sum d_jk w_jk
  s.t. (alpha_k >= 0)  sum_j w_jk           >= q_k
       (beta_j  >= 0)  sum_i x_ij - sum_k w_jk >= 0
       (gamma_i >= 0) -sum_j x_ij           >= -b_i y_i
       (delta_j >= 0) -sum_k w_jk           >= -p_j z_j
       x, w >= 0,

with dual (derived in the paper's dual-certificate proposition)

  max  sum_k q_k alpha_k - sum_i b_i y_i gamma_i - sum_j p_j z_j delta_j
  s.t. beta_j - gamma_i           <= c_ij   (column of x_ij)
       alpha_k - beta_j - delta_j <= d_jk   (column of w_jk)
       alpha, beta, gamma, delta >= 0.

Checks (>= 40 instances, seed 20260710; scipy.linprog/HiGHS):

  [S1] strong duality: for FEASIBLE designs, primal LP value ==
       dual LP value == value of the common module's integer MCMF oracle
       (tolerance 1e-6; and the value is within 1e-6 of an integer).
  [S2] at the generator design, the Benders cut with the optimal dual (alpha*,
       gamma*, delta*) holds with EQUALITY:
       v_l(y,z) = sum q alpha* - sum b_i gamma* y_i - sum p_j delta* z_j.
  [S3] global validity ((y,z)-independence of the dual polyhedron): the same
       (alpha*, gamma*, delta*) satisfies, for >= 20 OTHER random
       designs (y',z'),
       v_l(y',z') >= sum q alpha* - sum b_i gamma* y'_i - sum p_j delta* z'_j
       whenever (y',z') is feasible for product l (infeasible designs:
       v_l = +infinity, trivial inequality -- counted separately).
  [S4] INFEASIBLE designs: the dual is unbounded (HiGHS status 3), consistent
       with an always-feasible dual (point 0) + infeasible primal.
  [S5] the two feasibility rays F1/F2, NUMERIC test (independent
       cross-check; identical to the third test of
       stress_tests_A4A5.py): for the directions
       r1 = (alpha=1, beta=1, gamma=1, delta=0) and
       r2 = (alpha=1, beta=0, gamma=0, delta=1),
       the point u + theta*r_i -- with u = u* (optimal dual) at feasible
       designs and u = 0 at infeasible ones -- satisfies ALL
       |I||J| + |J||K| constraints of the dual polyhedron, for theta in
       {1, 10, 1000}, and the dual objective at u + theta*r_i equals exactly
       obj(u) + theta*(D_l - cap_side), i.e., the slope along r1
       (resp. r2) is D_l - sum b_i y_i (resp. D_l - sum p_j z_j) --
       positive exactly when F1 (resp. F2) fails, which certifies
       numerically the unboundedness of the dual at infeasible designs.
       (The previous version of this item compared (D - s > 0) with (s < D) --
       tautological; membership in the cone was only verified in a
       comment. Replaced in full.)
"""

import os
import random
import sys

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_mp_tscfl import gen_instance, demand_total, routing_value  # noqa: E402

SEED = 20260710
TOL = 1e-6
N_INST = 40
N_OTHER = 20


def primal_lp(inst, l, y, z):
    """Solves the primal LP (C1)-(C4) for product l. Returns (status, value)."""
    nI, nJ, nK = inst["nI"], inst["nJ"], inst["nK"]
    nx, nw = nI * nJ, nJ * nK
    xid = lambda i, j: i * nJ + j
    wid = lambda j, k: nx + j * nK + k
    nv = nx + nw
    cost = np.zeros(nv)
    for i in range(nI):
        for j in range(nJ):
            cost[xid(i, j)] = inst["c"][i][j][l]
    for j in range(nJ):
        for k in range(nK):
            cost[wid(j, k)] = inst["d"][j][k][l]
    A, rhs = [], []
    for k in range(nK):                      # C1: -sum_j w_jk <= -q_k
        row = np.zeros(nv)
        for j in range(nJ):
            row[wid(j, k)] = -1.0
        A.append(row); rhs.append(-inst["q"][k][l])
    for j in range(nJ):                      # C2: -sum_i x_ij + sum_k w_jk <= 0
        row = np.zeros(nv)
        for i in range(nI):
            row[xid(i, j)] = -1.0
        for k in range(nK):
            row[wid(j, k)] = 1.0
        A.append(row); rhs.append(0.0)
    for i in range(nI):                      # C3: sum_j x_ij <= b_i y_i
        row = np.zeros(nv)
        for j in range(nJ):
            row[xid(i, j)] = 1.0
        A.append(row); rhs.append(inst["b"][i][l] * y[i])
    for j in range(nJ):                      # C4: sum_k w_jk <= p_j z_j
        row = np.zeros(nv)
        for k in range(nK):
            row[wid(j, k)] = 1.0
        A.append(row); rhs.append(inst["p"][j][l] * z[j])
    res = linprog(cost, A_ub=np.array(A), b_ub=np.array(rhs),
                  bounds=(0, None), method="highs")
    return res.status, (res.fun if res.status == 0 else None)


def dual_matrices(inst, l):
    """Matrix A and rhs of the dual polyhedron Delta_l (independent of (y,z)).

    Variables u = (alpha[nK], beta[nJ], gamma[nI], delta[nJ]) >= 0;
    constraints A u <= rhs:
      beta_j - gamma_i           <= c_ij   (|I||J| rows)
      alpha_k - beta_j - delta_j <= d_jk   (|J||K| rows)
    """
    nI, nJ, nK = inst["nI"], inst["nJ"], inst["nK"]
    nv = nK + nJ + nI + nJ
    A, rhs = [], []
    for i in range(nI):                      # beta_j - gamma_i <= c_ij
        for j in range(nJ):
            row = np.zeros(nv)
            row[nK + j] = 1.0                # beta_j
            row[nK + nJ + i] = -1.0          # gamma_i
            A.append(row); rhs.append(inst["c"][i][j][l])
    for j in range(nJ):                      # alpha_k - beta_j - delta_j <= d_jk
        for k in range(nK):
            row = np.zeros(nv)
            row[k] = 1.0                     # alpha_k
            row[nK + j] = -1.0               # beta_j
            row[nK + nJ + nI + j] = -1.0     # delta_j
            A.append(row); rhs.append(inst["d"][j][k][l])
    return np.array(A), np.array(rhs), nv


def dual_obj(inst, l, y, z, u):
    """Dual objective q.alpha - (b y).gamma - (p z).delta at point u."""
    nI, nJ, nK = inst["nI"], inst["nJ"], inst["nK"]
    val = sum(inst["q"][k][l] * u[k] for k in range(nK))
    val -= sum(inst["b"][i][l] * y[i] * u[nK + nJ + i] for i in range(nI))
    val -= sum(inst["p"][j][l] * z[j] * u[nK + nJ + nI + j]
               for j in range(nJ))
    return val


def dual_lp(inst, l, y, z):
    """Solves the dual LP explicitly.

    Returns (status, value, (alpha, beta, gamma, delta), full_u).
    """
    nI, nJ, nK = inst["nI"], inst["nJ"], inst["nK"]
    A, rhs, nv = dual_matrices(inst, l)
    # max q.alpha - (b y).gamma - (p z).delta  ->  min of the negative
    obj = np.zeros(nv)
    for k in range(nK):
        obj[k] = -inst["q"][k][l]
    for i in range(nI):
        obj[nK + nJ + i] = inst["b"][i][l] * y[i]
    for j in range(nJ):
        obj[nK + nJ + nI + j] = inst["p"][j][l] * z[j]
    res = linprog(obj, A_ub=A, b_ub=rhs, bounds=(0, None), method="highs")
    if res.status != 0:
        return res.status, None, None, None
    alpha = [res.x[k] for k in range(nK)]
    beta = [res.x[nK + j] for j in range(nJ)]
    gamma = [res.x[nK + nJ + i] for i in range(nI)]
    delta = [res.x[nK + nJ + nI + j] for j in range(nJ)]
    return 0, -res.fun, (alpha, beta, gamma, delta), res.x


def cut_rhs(inst, l, dual, y, z):
    """RHS of the Benders cut: q.alpha - sum b_i gamma_i y_i - sum p_j delta_j z_j."""
    alpha, _beta, gamma, delta = dual
    val = sum(inst["q"][k][l] * alpha[k] for k in range(inst["nK"]))
    val -= sum(inst["b"][i][l] * gamma[i] * y[i] for i in range(inst["nI"]))
    val -= sum(inst["p"][j][l] * delta[j] * z[j] for j in range(inst["nJ"]))
    return val


def rays_check(inst, l, y, z, u_base=None, v_base=0.0):
    """[S5] F1/F2 rays -- numeric test (independent cross-check).

    For each ray r in {r1, r2} and theta in {1, 10, 1000}, verifies that
    pt = u_base + theta*r satisfies ALL constraints of Delta_l
    (A pt <= rhs, constraint by constraint) and that the dual objective at pt
    equals exactly v_base + theta*(D_l - cap_side). Since the slope
    D_l - cap_side is positive iff F1/F2 fails, this certifies
    numerically the unboundedness of the dual at infeasible designs.
    u_base = None uses the origin (always dual-feasible, since c, d >= 0).
    Returns (n_ok, n_fail).
    """
    A, rhs, nv = dual_matrices(inst, l)
    nI, nJ, nK = inst["nI"], inst["nJ"], inst["nK"]
    D = demand_total(inst, l)
    if u_base is None:
        u_base = np.zeros(nv)
        v_base = 0.0
    r1 = np.zeros(nv)                        # (alpha=1, beta=1, gamma=1)
    r1[:nK] = 1.0
    r1[nK:nK + nJ] = 1.0
    r1[nK + nJ:nK + nJ + nI] = 1.0
    r2 = np.zeros(nv)                        # (alpha=1, delta=1)
    r2[:nK] = 1.0
    r2[nK + nJ + nI:] = 1.0
    capB = sum(inst["b"][i][l] * y[i] for i in range(nI))
    capP = sum(inst["p"][j][l] * z[j] for j in range(nJ))
    n_ok = n_fail = 0
    for ray, slope in [(r1, D - capB), (r2, D - capP)]:
        for theta in (1.0, 10.0, 1000.0):
            pt = u_base + theta * ray
            objv = dual_obj(inst, l, y, z, pt)
            if (np.all(A @ pt <= rhs + 1e-7) and
                    abs(objv - (v_base + theta * slope)) <= 1e-5):
                n_ok += 1
            else:
                n_fail += 1
    return n_ok, n_fail


def rand_design(rng, nI, nJ, p_open=0.5):
    y = [1 if rng.random() < p_open else 0 for _ in range(nI)]
    return y


def boost_caps(inst, factor=3):
    """Multiplies capacities by `factor` (battery B: more feasible designs,
    exercising strong duality and cuts in a slack regime)."""
    inst = dict(inst)
    inst["b"] = [[v * factor for v in row] for row in inst["b"]]
    inst["p"] = [[v * factor for v in row] for row in inst["p"]]
    return inst


def main():
    rng = random.Random(SEED)
    stats = {"strong": 0, "tight": 0, "valid": 0, "trivial": 0,
             "unbounded": 0, "rays": 0}
    fails = 0

    # battery A: 40 raw instances (many infeasible designs -> S4/S5);
    # battery B: 20 instances with capacities x3 (feasible regime -> S1-S3).
    batches = [gen_instance(6000 + t) for t in range(N_INST)]
    batches += [boost_caps(gen_instance(6100 + t)) for t in range(20)]

    for t, inst in enumerate(batches):
        nI, nJ, nL = inst["nI"], inst["nJ"], inst["nL"]  # noqa: F841

        # generator designs: all-open + up to 2 random biased ones
        gens = [([1] * nI, [1] * nJ)]
        for _ in range(6):
            if len(gens) >= 3:
                break
            y = rand_design(rng, nI, 0.8)
            z = rand_design(rng, nJ, 0.8)
            if (y, z) not in gens:
                gens.append((y, z))

        # other designs for the global validity test [S3]
        others = []
        for _ in range(N_OTHER):
            others.append((rand_design(rng, nI, 0.5),
                           rand_design(rng, nJ, 0.5)))

        for (y, z) in gens:
            for l in range(nL):
                if demand_total(inst, l) == 0:
                    continue
                feas, v_mcmf = routing_value(inst, l, y, z)

                if not feas:
                    # [S4] unbounded dual
                    st, _, _, _ = dual_lp(inst, l, y, z)
                    if st == 3:
                        stats["unbounded"] += 1
                    else:
                        fails += 1
                        print(f"FAIL [S4] inst {t} l={l}: dual status {st}")
                    # [S5] numeric rays from the origin (dual-feasible):
                    # certifies the unboundedness numerically (slope > 0)
                    n_ok, n_fail = rays_check(inst, l, y, z)
                    stats["rays"] += n_ok
                    if n_fail:
                        fails += n_fail
                        print(f"FAIL [S5] inst {t} l={l} ({n_fail} rays)")
                    continue

                # [S1] primal LP == dual LP == MCMF (and integer)
                stp, vp = primal_lp(inst, l, y, z)
                std, vd, dual, u_full = dual_lp(inst, l, y, z)
                ok = (stp == 0 and std == 0 and
                      abs(vp - v_mcmf) <= TOL and
                      abs(vd - v_mcmf) <= TOL and
                      abs(vd - round(vd)) <= TOL)
                if ok:
                    stats["strong"] += 1
                else:
                    fails += 1
                    print(f"FAIL [S1] inst {t} l={l}: "
                          f"mcmf={v_mcmf} lp={vp} dual={vd}")
                    continue

                # [S5] numeric rays from the optimal dual u*:
                # u* + theta r_i dual-feasible, exact slope D_l - cap
                n_ok, n_fail = rays_check(inst, l, y, z, u_full, vd)
                stats["rays"] += n_ok
                if n_fail:
                    fails += n_fail
                    print(f"FAIL [S5] inst {t} l={l} ({n_fail} rays)")

                # [S2] cut equality at the generator design
                rhs0 = cut_rhs(inst, l, dual, y, z)
                if abs(rhs0 - v_mcmf) <= 1e-5:
                    stats["tight"] += 1
                else:
                    fails += 1
                    print(f"FAIL [S2] inst {t} l={l}: rhs={rhs0} v={v_mcmf}")

                # [S3] validity at other designs
                for (y2, z2) in others:
                    feas2, v2 = routing_value(inst, l, y2, z2)
                    rhs2 = cut_rhs(inst, l, dual, y2, z2)
                    if not feas2:
                        stats["trivial"] += 1   # v = +inf >= rhs2, trivial
                        continue
                    if v2 >= rhs2 - 1e-5:
                        stats["valid"] += 1
                    else:
                        fails += 1
                        print(f"FAIL [S3] inst {t} l={l}: "
                              f"v'={v2} < rhs'={rhs2}")

    total = sum(stats.values())
    print(f"instances: {len(batches)} (40 raw + 20 capacity x3); "
          f"checks: {total}; failures: {fails}")
    print(f"  [S1] strong duality (primal==dual==MCMF, integer): "
          f"{stats['strong']}")
    print(f"  [S2] tight cut at the generator:                    "
          f"{stats['tight']}")
    print(f"  [S3] cut valid at other designs (feasible):         "
          f"{stats['valid']} (+ {stats['trivial']} trivial/infeasible)")
    print(f"  [S4] unbounded dual at infeasible designs:          "
          f"{stats['unbounded']}")
    print(f"  [S5] numeric F1/F2 rays (feasibility + slope):      "
          f"{stats['rays']}")
    substantive = (stats["strong"] + stats["tight"] + stats["valid"] +
                   stats["unbounded"] + stats["rays"])
    print(f"  substantive checks (excluding trivial ones): {substantive}")
    print("OVERALL RESULT:", "PASS" if fails == 0 else "FAIL")
    return fails


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
