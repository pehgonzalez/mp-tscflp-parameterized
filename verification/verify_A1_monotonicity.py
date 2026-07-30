"""
Computational verification of the monotonicity proposition from the paper.

For >= 20 small random instances with fixed seed
(|I|,|J| <= 3 to keep the pair enumeration tractable), computes
v_l(y,z) for ALL designs (convention: +infinity if infeasible) and
checks exhaustively, for every componentwise comparable pair
(y,z) <= (y',z') and every product l:

    v_l(y', z') <= v_l(y, z)

which includes, in particular: (y,z) feasible => (y',z') feasible.
Values via a self-contained MCMF implementation with integer arithmetic.

Independent cross-check: reflexive pairs (y,z) = (y',z')
make the inequality tautological; they are EXCLUDED from the check
count (counted separately, for coverage bookkeeping only).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common_mp_tscfl import gen_instance, routing_value, all_designs

INF = float("inf")


def main():
    n_instances = 20
    checks = fails = reflexive_skipped = 0

    for seed in range(201, 201 + n_instances):
        inst = gen_instance(seed, max_i=3, max_j=3)
        nI, nJ, nL = inst["nI"], inst["nJ"], inst["nL"]

        designs = list(all_designs(nI, nJ))
        values = {}
        for (y, z) in designs:
            key = (tuple(y), tuple(z))
            values[key] = []
            for l in range(nL):
                feas, v = routing_value(inst, l, y, z)
                values[key].append(v if feas else INF)

        for (y, z) in designs:
            for (y2, z2) in designs:
                if all(a <= b for a, b in zip(y, y2)) and \
                   all(a <= b for a, b in zip(z, z2)):
                    if (tuple(y), tuple(z)) == (tuple(y2), tuple(z2)):
                        # C1: reflexive pair -- tautological inequality,
                        # does not count as an effective check.
                        reflexive_skipped += nL
                        continue
                    va = values[(tuple(y), tuple(z))]
                    vb = values[(tuple(y2), tuple(z2))]
                    for l in range(nL):
                        checks += 1
                        if not (vb[l] <= va[l]):
                            fails += 1
                            print(f"[FAIL] seed={seed} l={l} "
                                  f"(y,z)={y},{z} v={va[l]} <= "
                                  f"(y',z')={y2},{z2} v'={vb[l]} VIOLATED")

    print(f"\nverify_A1_monotonicity: {n_instances} instances, "
          f"{checks} inequalities (STRICT comparable pair x product): "
          f"{checks - fails} PASS, {fails} FAIL "
          f"({reflexive_skipped} tautological reflexive checks skipped)")
    if fails:
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
