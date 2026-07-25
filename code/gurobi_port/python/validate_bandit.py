"""Validation of C1: bandit-controlled subgradient vs classical fixed-Polyak baseline.

Checks (per seed, identical instances):
  1. VALIDITY: bandit best_lb <= v* (certified bound never violated);
  2. UB certified by independent re-evaluation;
  3. paired comparison bandit vs baseline (same iteration budget): reports both
     wins and losses, per project rules.
Usage: python validate_bandit.py [nseeds] [iters]
"""
import sys

import gurobipy as gp
from gurobipy import GRB

from bandit_subgradient import bandit_lagrangian_dual
from lagrangian_gurobipy import lagrangian_dual, routing_value
from model_gurobipy import build_model
from validate_bruteforce import gen_instance


def main():
    nseeds = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    env = gp.Env(params={"OutputFlag": 0})
    wins = losses = ties = 0
    for seed in range(nseeds):
        inst = gen_instance(seed, I=3, J=4, K=5, L=2)
        m, *_ = build_model(inst, env=env)
        m.Params.OutputFlag = 0
        m.optimize()
        vstar = m.ObjVal
        tol = 1e-6 * max(1.0, abs(vstar))

        base = lagrangian_dual(inst, env, iters=iters)
        band = bandit_lagrangian_dual(inst, env, iters=iters, time_limit=60.0)

        assert band["best_lb"] <= vstar + tol, f"seed {seed}: INVALID bandit bound"
        assert base["best_lb"] <= vstar + tol, f"seed {seed}: INVALID baseline bound"
        if band["best_sol"] is not None:
            y, z = band["best_sol"]
            rc = routing_value(inst, y, z, env)
            total = rc + sum(f * v for f, v in zip(inst["f"], y)) \
                       + sum(g * v for g, v in zip(inst["g"], z))
            assert abs(total - band["best_ub"]) < tol, f"seed {seed}: UB mismatch"

        d = band["best_lb"] - base["best_lb"]
        mark = "bandit" if d > tol else ("baseline" if d < -tol else "empate")
        wins += mark == "bandit"
        losses += mark == "baseline"
        ties += mark == "empate"
        nmip = sum(1 for h in band["history"] if h[2] == "mip")
        print(f"seed {seed}: base_LD={base['best_lb']:10.2f}  band_LD={band['best_lb']:10.2f}"
              f"  v*={vstar:10.2f}  [{mark}; {nmip}/{len(band['history'])} its MIP]")
    print(f"pareado: bandit {wins} x {losses} baseline ({ties} empates)")
    print("BANDIT VALIDATION PASSED (validade dos bounds nunca violada)")


if __name__ == "__main__":
    main()
