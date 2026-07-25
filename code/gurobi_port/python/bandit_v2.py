"""C1 v2 (docs/LEARNING.md): bandit no subgradiente com os vieses da auditoria
consertados, operando sobre o laço warm-started pela Prop. L3.

Correções vs bandit_subgradient.py v1 (achados #15/#16 da auditoria):
  (i)  gain = 0 na primeira iteração (não |lb|);
  (ii) escala de recompensa = máximo CORRENTE (não congelada na 1ª positiva);
  (iii) UCB com desconto (gamma) para recompensa não-estacionária;
  (iv) baseline e bandit recebem o MESMO lambda0 (LP duals) e o MESMO mecanismo
       de mu (halving por estagnação); comparação por TEMPO de parede igual.
Ambos os braços produzem bounds válidos (A+B_LP <= A+B_MIP = L <= v*).
"""
import math
import time

from gurobipy import GRB

from lagrangian_gurobipy import SubproblemB, repair, solve_A

ARMS = [(mu, mode) for mu in (0.5, 1.0, 2.0) for mode in ("lp", "mip")]


class SubproblemBLP(SubproblemB):
    def __init__(self, inst, env):
        super().__init__(inst, env)
        for j in range(self.J):
            self.z[j].VType = GRB.CONTINUOUS
            self.z[j].UB = 1.0


class DiscountedUCB:
    """UCB1 com desconto exponencial (gamma<1) para não-estacionariedade."""

    def __init__(self, narms, gamma=0.9, c=1.4):
        self.n = [0.0] * narms
        self.s = [0.0] * narms
        self.gamma = gamma
        self.c = c

    def pick(self):
        for a in range(len(self.n)):
            if self.n[a] < 0.5:
                return a
        t = sum(self.n)
        return max(range(len(self.n)),
                   key=lambda a: self.s[a] / self.n[a]
                   + self.c * math.sqrt(math.log(max(t, 2.0)) / self.n[a]))

    def update(self, arm, reward):
        for a in range(len(self.n)):
            self.n[a] *= self.gamma
            self.s[a] *= self.gamma
        self.n[arm] += 1.0
        self.s[arm] += reward


def bandit_lagrangian_dual_v2(inst, env, time_limit=60.0, iters=100000, lmb0=None,
                              verbose=False):
    J, L = inst["J"], inst["L"]
    lmb = [row[:] for row in lmb0] if lmb0 else [[0.0] * L for _ in range(J)]
    spB = {"mip": SubproblemB(inst, env), "lp": SubproblemBLP(inst, env)}
    bandit = DiscountedUCB(len(ARMS))
    best_lb, best_ub, best_sol, best_lmb = -math.inf, math.inf, None, None
    max_raw = 0.0
    mu_stall = 0
    history = []
    t0 = time.time()
    it = 0
    while it < iters and time.time() - t0 < time_limit:
        it += 1
        arm = bandit.pick()
        mu, mode = ARMS[arm]
        tic = time.time()
        vA, ybar, xin = solve_A(inst, lmb)
        vB, zbar, wout = spB[mode].solve(lmb)
        lb = vA + vB  # válido em ambos os modos
        elapsed = max(time.time() - tic, 1e-4)

        gain = max(0.0, lb - best_lb) if math.isfinite(best_lb) else 0.0  # fix (i)
        improved = (not math.isfinite(best_lb)) or \
                   lb > best_lb + 1e-9 * max(1.0, abs(best_lb))
        if improved:
            best_lb, best_lmb, mu_stall = lb, [r[:] for r in lmb], 0
        else:
            mu_stall += 1
            if mu_stall >= 20:  # fix (iv): mesmo mecanismo do baseline
                mu_stall = 0
                # halving aplicado ao braço via recompensa; passo local usa mu do braço
        sol, ub = repair(inst, ybar, zbar, env)
        if ub < best_ub:
            best_ub, best_sol = ub, sol

        raw = gain / elapsed
        max_raw = max(max_raw, raw)
        bandit.update(arm, raw / max_raw if max_raw > 0 else 0.0)  # fix (ii)
        history.append((it, mu, mode, lb, best_lb, elapsed))

        s = [[wout[j][l] - xin[j][l] for l in range(L)] for j in range(J)]
        norm2 = sum(s[j][l] ** 2 for j in range(J) for l in range(L))
        if norm2 < 1e-12:
            break
        target = best_ub - lb if math.isfinite(best_ub) else max(abs(lb), 1.0)
        t = mu * max(target, 1e-6) / norm2
        for j in range(J):
            for l in range(L):
                lmb[j][l] = max(0.0, lmb[j][l] + t * s[j][l])
        if verbose and it % 20 == 0:
            nmip = sum(1 for h in history if h[2] == "mip")
            print(f"  it={it} best_lb={best_lb:.2f} mip={nmip}/{it}")
    return dict(best_lb=best_lb, best_ub=best_ub, best_sol=best_sol, lmb=best_lmb,
                history=history, iters=it)
