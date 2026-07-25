"""Teste decisivo do C1 v2 em INSTÂNCIA REAL (rodar na máquina com licença plena).

Protocolo simétrico (auditoria #16): baseline (Polyak fixo, B-MIP sempre) e bandit v2
recebem o MESMO lambda0 (duais LP, Prop. L3) e o MESMO orçamento de TEMPO de parede.
Reporta best_lb (v_LD atingido), iterações e composição LP/MIP — vitórias E derrotas.

Uso:  py bandit_real_test.py ..\\Old_Project\\data\\PSC1-C1-50-5.txt 300
"""
import sys
import time

import gurobipy as gp

from bandit_v2 import bandit_lagrangian_dual_v2
from lagrangian_gurobipy import lagrangian_dual, lp_dual_warmstart
from model_gurobipy import load_instance


def main():
    path = sys.argv[1]
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0
    inst = load_instance(path)
    env = gp.Env(params={"OutputFlag": 0})

    t0 = time.time()
    lmb0, vlp = lp_dual_warmstart(inst, env)
    t_lp = time.time() - t0
    print(f"v_LP = {vlp:.2f}  (LP em {t_lp:.1f}s; orçamento por método: {budget:.0f}s)")

    t0 = time.time()
    base = lagrangian_dual(inst, env, iters=10 ** 9, lmb0=lmb0, time_limit=budget)
    t_base = time.time() - t0
    print(f"baseline : LD={base['best_lb']:.2f}  UB={base['best_ub']:.2f}  "
          f"({t_base:.0f}s)")

    t0 = time.time()
    band = bandit_lagrangian_dual_v2(inst, env, time_limit=budget, lmb0=lmb0)
    t_band = time.time() - t0
    nmip = sum(1 for h in band["history"] if h[2] == "mip")
    print(f"bandit v2: LD={band['best_lb']:.2f}  UB={band['best_ub']:.2f}  "
          f"({t_band:.0f}s; {band['iters']} iters, {nmip} MIP / "
          f"{band['iters'] - nmip} LP)")

    d = band["best_lb"] - base["best_lb"]
    ref = max(1.0, abs(base["best_lb"]))
    verdict = "BANDIT" if d > 1e-6 * ref else ("BASELINE" if d < -1e-6 * ref else "EMPATE")
    print(f"\nveredito (best_lb, tempo igual): {verdict}  (delta = {d:+.2f})")
    print(f"ambos >= v_LP? baseline: {base['best_lb'] >= vlp - 1e-4 * ref}, "
          f"bandit: {band['best_lb'] >= vlp - 1e-4 * ref}")


if __name__ == "__main__":
    main()
