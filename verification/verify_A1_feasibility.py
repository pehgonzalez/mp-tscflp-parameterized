"""
Computational verification of the feasibility characterization from the paper.

For >= 30 small random instances with fixed seed, EXHAUSTIVELY enumerates
all designs (y,z) in {0,1}^{|I|} x {0,1}^{|J|}
(up to 256 per instance) and checks the equivalence

   [ for all l: sum_i b_il y_i >= D_l  AND  sum_j p_jl z_j >= D_l ]
                        <=>
   [ for all l: maximum S-T flow in N_l(y,z) >= D_l ]

where the maximum flow is computed via augmenting paths with
exact integer arithmetic (self-contained implementation, no LP).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common_mp_tscfl import (gen_instance, aggregate_condition,
                             flow_feasible, all_designs)


def main():
    n_instances = 30
    checks = fails = 0
    n_true = n_false = 0

    for seed in range(101, 101 + n_instances):
        inst = gen_instance(seed)
        for (y, z) in all_designs(inst["nI"], inst["nJ"]):
            checks += 1
            cond = aggregate_condition(inst, y, z)
            real = flow_feasible(inst, y, z)
            if cond != real:
                fails += 1
                print(f"[FAIL] seed={seed} y={y} z={z}: "
                      f"aggregate_condition={cond} flow_feasibility={real}")
            if real:
                n_true += 1
            else:
                n_false += 1

    print(f"\nverify_A1_feasibility: {n_instances} instances, "
          f"{checks} designs (y,z) checked exhaustively: "
          f"{checks - fails} PASS, {fails} FAIL "
          f"({n_true} feasible, {n_false} infeasible)")
    if fails:
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
