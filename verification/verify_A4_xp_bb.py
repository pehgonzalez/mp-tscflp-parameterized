"""
Verification of the paper's XP algorithm (XP algorithm in k as B&B with the
three prunings P1/P2/P3).

Reference implementation (Python) of the paper's algorithm on the binary
open/close decision tree (plants first, then depots),
in the cardinality OPTIMIZATION version: for each k in {0..n} it computes
OPT_k = min { cost(S) : S feasible, |S| <= k }. This subsumes ALL
budgets B at once (MP-TSCFLP(B,k) is YES iff OPT_k <= B) -- it is the
complete sweep (all cardinalities x all budgets).

Prunings (exactly those of the paper):
  P1 (covering, the covering counting lemma): at node (O,C), with
     r = k - |O| remaining
     openings, if max_l s_I(l;O,C) + max_l s_J(l;O,C) > r, no
     completion is feasible within the cardinality -> discard. s_side(l)
     is the minimum number of free facilities on that side, by capacities
     sorted decreasingly (greedy prefix), to close F1/F2.
  P2 (admissible bound, via monotonicity): LB(O,C) = fixed cost
     of O + v(open O and all free); discard if v = +inf (no feasible
     completion, by monotonicity) or LB >= incumbent.
  P3 (CNUF dominance): at the leaf, if the optimal flow returned
     by the oracle leaves some open facility unused (throughput 0
     in all products), the leaf is discarded without updating the
     incumbent (the protected witness -- a minimum-cardinality
     optimum -- is never discarded).

Batteries:
  [A] 60 seeded random instances (|I|,|J| <= 5, |K| <= 4,
      |L| <= 2, values <= 9);
  [B] 12 adversarial: 6 with zero fixed costs (f=g=0; optima with
      zero-marginal-cost facilities -- maximum stress for P3) and
      6 with zero transport (c=d=0; the tradeoff is 100%% fixed cost x
      covering -- stress for P1).

For EACH instance and EACH k in {0..n}:
  (1) independent brute force: OPT_k by enumeration of ALL
      designs (MCMF oracle of the common module, all_designs);
  (2) B&B with the three prunings == OPT_k  (the prunings never discard the optimum);
  (3) B&B without prunings (same tree, only the structural cut |O| <= k)
      == OPT_k (tree sanity);
  (4) node counts with/without prunings (record of the reduction).

Design costs memoized by bitmask (brute force is the source; the
B&B consumes the same oracle, but the EQUALITY tested is against the minimum
of the complete enumeration, which goes through no pruning).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_mp_tscfl import (MinCostFlow, gen_instance,  # noqa: E402
                             demand_total)

INF = float("inf")
SEED0 = 9000


# ---------------------------------------------------------------------------
# Oracle with usage extraction (throughput per facility)
# ---------------------------------------------------------------------------

def eval_design(inst, y, z):
    """Routes all products on design (y,z).

    Returns (feasible, value, usedI, usedJ):
      usedI[i] = True iff plant i carries flow > 0 in SOME
      product (arc S->F_i), in the optimal flow returned; likewise usedJ[j]
      (arc Din_j->Dout_j).
    """
    nI, nJ, nK = inst["nI"], inst["nJ"], inst["nK"]
    usedI = [False] * nI
    usedJ = [False] * nJ
    total = 0
    for l in range(inst["nL"]):
        D = demand_total(inst, l)
        if D == 0:
            continue
        # network identical to build_network of the common module, with arc handles
        S = 0
        F = lambda i: 1 + i
        Din = lambda j: 1 + nI + j
        Dout = lambda j: 1 + nI + nJ + j
        C = lambda k: 1 + nI + 2 * nJ + k
        T = 1 + nI + 2 * nJ + nK
        BIG = 1 + sum(sum(r) for r in inst["b"]) \
                + sum(sum(r) for r in inst["p"]) \
                + sum(sum(r) for r in inst["q"])
        mc = MinCostFlow(T + 1)
        hI, hJ = [], []
        for i in range(nI):
            mc.add_edge(S, F(i), inst["b"][i][l] * y[i], 0)
            hI.append((mc.graph[S][-1], inst["b"][i][l] * y[i]))
        for i in range(nI):
            for j in range(nJ):
                mc.add_edge(F(i), Din(j), BIG, inst["c"][i][j][l])
        for j in range(nJ):
            mc.add_edge(Din(j), Dout(j), inst["p"][j][l] * z[j], 0)
            hJ.append((mc.graph[Din(j)][-1], inst["p"][j][l] * z[j]))
        for j in range(nJ):
            for k in range(nK):
                mc.add_edge(Dout(j), C(k), BIG, inst["d"][j][k][l])
        for k in range(nK):
            mc.add_edge(C(k), T, inst["q"][k][l], 0)
        sent, cost = mc.flow(S, T, D)
        if sent < D:
            return False, None, None, None
        total += cost
        for i in range(nI):
            if hI[i][1] - hI[i][0][1] > 0:      # original cap - residual
                usedI[i] = True
        for j in range(nJ):
            if hJ[j][1] - hJ[j][0][1] > 0:
                usedJ[j] = True
    return True, total, usedI, usedJ


class DesignCache:
    """Memoization by bitmask (yI | zJ << nI) of eval_design + total cost."""

    def __init__(self, inst):
        self.inst = inst
        self.nI, self.nJ = inst["nI"], inst["nJ"]
        self.memo = {}

    def get(self, ymask, zmask):
        key = (ymask, zmask)
        if key not in self.memo:
            y = [(ymask >> i) & 1 for i in range(self.nI)]
            z = [(zmask >> j) & 1 for j in range(self.nJ)]
            feas, rout, uI, uJ = eval_design(self.inst, y, z)
            if not feas:
                self.memo[key] = (False, None, 0, 0)
            else:
                fixed = sum(self.inst["f"][i] for i in range(self.nI)
                            if (ymask >> i) & 1)
                fixed += sum(self.inst["g"][j] for j in range(self.nJ)
                             if (zmask >> j) & 1)
                umI = sum(1 << i for i in range(self.nI) if uI[i])
                umJ = sum(1 << j for j in range(self.nJ) if uJ[j])
                self.memo[key] = (True, fixed + rout, umI, umJ)
        return self.memo[key]


# ---------------------------------------------------------------------------
# Brute force (source of truth)
# ---------------------------------------------------------------------------

def brute_force(cache):
    """OPT_k for all k, by complete enumeration of designs."""
    nI, nJ = cache.nI, cache.nJ
    n = nI + nJ
    opt = [INF] * (n + 1)
    for ymask in range(1 << nI):
        for zmask in range(1 << nJ):
            feas, cost, _, _ = cache.get(ymask, zmask)
            if not feas:
                continue
            card = bin(ymask).count("1") + bin(zmask).count("1")
            for k in range(card, n + 1):
                if cost < opt[k]:
                    opt[k] = cost
    return opt


# ---------------------------------------------------------------------------
# B&B of the paper's algorithm
# ---------------------------------------------------------------------------

def covering_extra(caps_open, caps_free_sorted, D):
    """s = minimum number of free facilities (capacities already sorted desc)
    so that caps_open + prefix >= D; INF if impossible (covering counting
    lemma)."""
    if caps_open >= D:
        return 0
    need = D - caps_open
    acc = 0
    for s, cap in enumerate(caps_free_sorted, start=1):
        acc += cap
        if acc >= need:
            return s
    return INF


def bnb(inst, cache, k, use_prunings):
    """Returns (OPT_k, number of visited nodes)."""
    nI, nJ, nL = inst["nI"], inst["nJ"], inst["nL"]
    n = nI + nJ
    best = [INF]
    nodes = [0]

    def rec(idx, ymask, zmask, nopen):
        nodes[0] += 1
        r = k - nopen
        if r < 0:
            return
        if idx == n:                                   # leaf
            feas, cost, umI, umJ = cache.get(ymask, zmask)
            if not feas:
                return
            if use_prunings:
                # P3: open facility without use -> dominated leaf
                if (ymask & ~umI) or (zmask & ~umJ):
                    return
            if cost < best[0]:
                best[0] = cost
            return
        if use_prunings:
            # decided: 0..idx-1; free: idx..n-1
            # P1 -- covering pruning
            sI_max, sJ_max = 0, 0
            for l in range(nL):
                D = demand_total(inst, l)
                if D == 0:
                    continue
                capI = sum(inst["b"][i][l] for i in range(nI)
                           if (ymask >> i) & 1)
                freeI = sorted((inst["b"][i][l] for i in range(nI)
                                if i >= idx), reverse=True)
                sI = covering_extra(capI, freeI, D)
                capJ = sum(inst["p"][j][l] for j in range(nJ)
                           if (zmask >> j) & 1)
                freeJ = sorted((inst["p"][j][l] for j in range(nJ)
                                if nI + j >= idx), reverse=True)
                sJ = covering_extra(capJ, freeJ, D)
                sI_max = max(sI_max, sI)
                sJ_max = max(sJ_max, sJ)
            if sI_max + sJ_max > r:
                return
            # P2 -- admissible bound (open O and all free)
            upY = ymask | sum(1 << i for i in range(nI) if i >= idx)
            upZ = zmask | sum(1 << j for j in range(nJ) if nI + j >= idx)
            feas, _, _, _ = cache.get(upY, upZ)
            if not feas:
                return                                  # infeasible by monotonicity
            # LB = accumulated fixed cost of O + v(all-free-open)
            fixedO = sum(inst["f"][i] for i in range(nI) if (ymask >> i) & 1)
            fixedO += sum(inst["g"][j] for j in range(nJ) if (zmask >> j) & 1)
            up_cost = cache.get(upY, upZ)[1]            # fixed(up) + v(up)
            fixedUp = sum(inst["f"][i] for i in range(nI) if (upY >> i) & 1)
            fixedUp += sum(inst["g"][j] for j in range(nJ) if (upZ >> j) & 1)
            LB = fixedO + (up_cost - fixedUp)           # v(up) alone
            if LB >= best[0]:
                return
        # branch: open and close facility idx
        if idx < nI:
            rec(idx + 1, ymask | (1 << idx), zmask, nopen + 1)
            rec(idx + 1, ymask, zmask, nopen)
        else:
            j = idx - nI
            rec(idx + 1, ymask, zmask | (1 << j), nopen + 1)
            rec(idx + 1, ymask, zmask, nopen)

    rec(0, 0, 0, 0)
    return best[0], nodes[0]


# ---------------------------------------------------------------------------
# Batteries
# ---------------------------------------------------------------------------

def make_instances():
    insts = []
    for t in range(60):                       # [A] random
        insts.append(("rand%02d" % t,
                      gen_instance(SEED0 + t, max_i=5, max_j=5,
                                   max_k=4, max_l=2, vmax=9)))
    for t in range(6):                        # [B] f = g = 0
        inst = gen_instance(SEED0 + 100 + t, max_i=5, max_j=5,
                            max_k=4, max_l=2, vmax=9)
        inst["f"] = [0] * inst["nI"]
        inst["g"] = [0] * inst["nJ"]
        insts.append(("zfix%02d" % t, inst))
    for t in range(6):                        # [B] c = d = 0
        inst = gen_instance(SEED0 + 200 + t, max_i=5, max_j=5,
                            max_k=4, max_l=2, vmax=9)
        inst["c"] = [[[0] * inst["nL"] for _ in range(inst["nJ"])]
                     for _ in range(inst["nI"])]
        inst["d"] = [[[0] * inst["nL"] for _ in range(inst["nK"])]
                     for _ in range(inst["nJ"])]
        insts.append(("ztrn%02d" % t, inst))
    return insts


def main():
    insts = make_instances()
    fails = 0
    checks = 0
    tot_nodes_plain = 0
    tot_nodes_pruned = 0

    for name, inst in insts:
        n = inst["nI"] + inst["nJ"]
        cache = DesignCache(inst)
        opt = brute_force(cache)
        for k in range(n + 1):
            v_pruned, nd_p = bnb(inst, cache, k, use_prunings=True)
            v_plain, nd_0 = bnb(inst, cache, k, use_prunings=False)
            tot_nodes_pruned += nd_p
            tot_nodes_plain += nd_0
            checks += 2
            if v_pruned != opt[k]:
                fails += 1
                print(f"FAIL {name} k={k}: pruned B&B {v_pruned} "
                      f"!= brute {opt[k]}")
            if v_plain != opt[k]:
                fails += 1
                print(f"FAIL {name} k={k}: B&B without prunings {v_plain} "
                      f"!= brute {opt[k]}")

    red = 100.0 * (1.0 - tot_nodes_pruned / tot_nodes_plain)
    print(f"instances: {len(insts)} (60 random + 6 f=g=0 + 6 c=d=0)")
    print(f"comparisons B&B == brute force (all cardinalities, "
          f"hence all budgets): {checks}; failures: {fails}")
    print(f"visited nodes: without prunings {tot_nodes_plain}, "
          f"with prunings P1+P2+P3 {tot_nodes_pruned} "
          f"(reduction {red:.1f}%)")
    print("OVERALL RESULT:", "PASS" if fails == 0 else "FAIL")
    return fails


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
