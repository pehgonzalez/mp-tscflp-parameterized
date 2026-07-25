"""C1 (docs/LEARNING.md): UCB1 bandit controlling the subgradient loop.

Arms = (Polyak step factor, B-mode) with B-mode in {LP, MIP}. Both modes yield valid
lower bounds (A + B_LP <= A + B_MIP = L(lmb) <= v*), so the bandit trades cost for
bound quality but can never invalidate a reported bound.
Reward = best-bound improvement per second of wall time (normalized online).
"""
import math
import time

import gurobipy as gp
from gurobipy import GRB

from lagrangian_gurobipy import SubproblemB, repair, solve_A

ARMS = [(mu, mode) for mu in (0.5, 1.0, 2.0) for mode in ("lp", "mip")]


class SubproblemBLP(SubproblemB):
    """B with z relaxed to [0,1]: cheaper, weaker (still valid) bound."""

    def __init__(self, inst, env):
        super().__init__(inst, env)
        for j in range(self.J):
            self.z[j].VType = GRB.CONTINUOUS
            self.z[j].UB = 1.0


class UCB1:
    def __init__(self, narms):
        self.n = [0] * narms
        self.mean = [0.0] * narms
        self.t = 0

    def pick(self):
        self.t += 1
        for a in range(len(self.n)):  # play each arm once first
            if self.n[a] == 0:
                return a
        return max(range(len(self.n)),
                   key=lambda a: self.mean[a] + math.sqrt(2 * math.log(self.t) / self.n[a]))

    def update(self, a, reward):
        self.n[a] += 1
        self.mean[a] += (reward - self.mean[a]) / self.n[a]


def bandit_lagrangian_dual(inst, env, iters=300, time_limit=60.0, verbose=False):
    """Same contract as lagrangian_dual(); adds 'history' (arm choices) for analysis."""
    J, L = inst["J"], inst["L"]
    lmb = [[0.0] * L for _ in range(J)]
    spB = {"mip": SubproblemB(inst, env), "lp": SubproblemBLP(inst, env)}
    bandit = UCB1(len(ARMS))
    best_lb, best_ub, best_sol, best_lmb = -math.inf, math.inf, None, None
    history = []
    t0 = time.time()
    reward_scale = None
    for it in range(iters):
        if time.time() - t0 > time_limit:
            break
        arm = bandit.pick()
        mu, mode = ARMS[arm]
        tic = time.time()

        vA, ybar, xin = solve_A(inst, lmb)
        vB, zbar, wout = spB[mode].solve(lmb)
        lb = vA + vB  # valid for both modes (LEARNING.md, C1)
        elapsed = max(time.time() - tic, 1e-4)

        gain = max(0.0, lb - best_lb) if best_lb > -math.inf else abs(lb)
        if lb > best_lb:
            best_lb, best_lmb = lb, [row[:] for row in lmb]
        sol, ub = repair(inst, ybar, zbar, env)
        if ub < best_ub:
            best_ub, best_sol = ub, sol

        raw = gain / elapsed
        if reward_scale is None and raw > 0:
            reward_scale = raw  # first positive reward defines the scale
        bandit.update(arm, min(1.0, raw / reward_scale) if reward_scale else 0.0)
        history.append((it, mu, mode, lb, best_lb, elapsed))

        s = [[wout[j][l] - xin[j][l] for l in range(L)] for j in range(J)]
        norm2 = sum(s[j][l] ** 2 for j in range(J) for l in range(L))
        if norm2 < 1e-12:
            break
        t = mu * max(best_ub - lb, 1e-6) / norm2
        for j in range(J):
            for l in range(L):
                lmb[j][l] = max(0.0, lmb[j][l] + t * s[j][l])
        if verbose and it % 20 == 0:
            print(f"  it={it:4d} arm=({mu},{mode}) L={lb:12.2f} best={best_lb:12.2f}")
    return dict(best_lb=best_lb, best_ub=best_ub, best_sol=best_sol, lmb=best_lmb,
                history=history)
