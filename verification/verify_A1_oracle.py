"""
Verificacao computacional da Prop. A1.1 (oraculo de roteamento).

Para >= 30 instancias aleatorias pequenas com semente fixa
(|I|,|J|,|K| <= 4, |L| <= 3, valores <= 9) e varios desenhos (y,z)
(tudo-aberto + desenhos aleatorios), compara, por produto l:

  (1) valor do MCMF na rede em camadas N_l(y,z), calculado pela
      implementacao propria (SSP, aritmetica inteira exata);
  (2) valor do MCMF calculado independentemente por
      networkx.network_simplex (simplex de redes, exato p/ inteiros);
  (3) otimo do PL residual ORIGINAL em forma de desigualdades
      (min c.x + d.w  s.a  sum_j w >= q, sum_i x >= sum_k w,
       sum_j x <= b*y, sum_k w <= p*z, x,w >= 0),
      resolvido por scipy.linprog (HiGHS).

Checagens:
  A) status de viabilidade coincide nas tres abordagens;
  B) (1) == (2) exatamente (inteiros);
  C) |(3) - (1)| <= tolerancia  (a reducao PL -> fluxo esta correta);
  D) (3) esta a distancia <= tolerancia de um inteiro
     (integralidade do valor otimo do PL residual).

(1) e (2) sao modelos de fluxo; (3) e o PL original com desigualdades,
logo C valida a reducao provada na Prop. A1.1 e D valida a
integralidade do valor.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common_mp_tscfl import (gen_instance, demand_total, build_network,
                             routing_value)

import random
import networkx as nx
import numpy as np
from scipy.optimize import linprog

TOL = 1e-6


def nx_mcmf(inst, l, y, z):
    """MCMF via networkx.network_simplex. Retorna (viavel, valor)."""
    nI, nJ, nK = inst["nI"], inst["nJ"], inst["nK"]
    D = demand_total(inst, l)
    if D == 0:
        return True, 0
    BIG = 1 + sum(sum(r) for r in inst["b"]) + sum(sum(r) for r in inst["p"]) \
            + sum(sum(r) for r in inst["q"])
    G = nx.DiGraph()
    G.add_node("S", demand=-D)
    G.add_node("T", demand=D)
    for i in range(nI):
        G.add_edge("S", f"F{i}", capacity=inst["b"][i][l] * y[i], weight=0)
        for j in range(nJ):
            G.add_edge(f"F{i}", f"Din{j}", capacity=BIG,
                       weight=inst["c"][i][j][l])
    for j in range(nJ):
        G.add_edge(f"Din{j}", f"Dout{j}", capacity=inst["p"][j][l] * z[j],
                   weight=0)
        for k in range(nK):
            G.add_edge(f"Dout{j}", f"C{k}", capacity=BIG,
                       weight=inst["d"][j][k][l])
    for k in range(nK):
        G.add_edge(f"C{k}", "T", capacity=inst["q"][k][l], weight=0)
    try:
        cost, _ = nx.network_simplex(G)
    except nx.NetworkXUnfeasible:
        return False, None
    return True, cost


def lp_residual(inst, l, y, z):
    """PL residual do produto l em forma original (desigualdades).

    Variaveis: x_ij (nI*nJ) depois w_jk (nJ*nK). Retorna
    (viavel, valor float) via HiGHS.
    """
    nI, nJ, nK = inst["nI"], inst["nJ"], inst["nK"]
    nx_var = nI * nJ
    nw_var = nJ * nK
    nvar = nx_var + nw_var

    def xi(i, j):
        return i * nJ + j

    def wi(j, k):
        return nx_var + j * nK + k

    cvec = np.zeros(nvar)
    for i in range(nI):
        for j in range(nJ):
            cvec[xi(i, j)] = inst["c"][i][j][l]
    for j in range(nJ):
        for k in range(nK):
            cvec[wi(j, k)] = inst["d"][j][k][l]

    A_ub, b_ub = [], []
    # demanda: -sum_j w_jk <= -q_kl
    for k in range(nK):
        row = np.zeros(nvar)
        for j in range(nJ):
            row[wi(j, k)] = -1.0
        A_ub.append(row)
        b_ub.append(-float(inst["q"][k][l]))
    # balanco no deposito: sum_k w_jk - sum_i x_ij <= 0
    for j in range(nJ):
        row = np.zeros(nvar)
        for k in range(nK):
            row[wi(j, k)] = 1.0
        for i in range(nI):
            row[xi(i, j)] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)
    # capacidade da fabrica: sum_j x_ij <= b_il y_i
    for i in range(nI):
        row = np.zeros(nvar)
        for j in range(nJ):
            row[xi(i, j)] = 1.0
        A_ub.append(row)
        b_ub.append(float(inst["b"][i][l] * y[i]))
    # capacidade do deposito: sum_k w_jk <= p_jl z_j
    for j in range(nJ):
        row = np.zeros(nvar)
        for k in range(nK):
            row[wi(j, k)] = 1.0
        A_ub.append(row)
        b_ub.append(float(inst["p"][j][l] * z[j]))

    res = linprog(cvec, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  bounds=[(0, None)] * nvar, method="highs")
    if res.status == 2:
        return False, None
    assert res.status == 0, f"linprog status inesperado: {res.status}"
    return True, float(res.fun)


def main():
    n_instances = 30
    checks = fails = 0
    n_feas_cases = n_infeas_cases = 0

    for seed in range(1, n_instances + 1):
        inst = gen_instance(seed)
        nI, nJ = inst["nI"], inst["nJ"]
        rng = random.Random(10_000 + seed)
        designs = [([1] * nI, [1] * nJ)]
        for _ in range(2):  # desenhos uniformes (exercitam inviabilidade)
            designs.append(([rng.randint(0, 1) for _ in range(nI)],
                            [rng.randint(0, 1) for _ in range(nJ)]))
        for _ in range(2):  # desenhos enviesados a abrir (exercitam valores)
            designs.append(([1 if rng.random() < 0.8 else 0
                             for _ in range(nI)],
                            [1 if rng.random() < 0.8 else 0
                             for _ in range(nJ)]))
        for (y, z) in designs:
            for l in range(inst["nL"]):
                f1, v1 = routing_value(inst, l, y, z)   # SSP proprio
                f2, v2 = nx_mcmf(inst, l, y, z)         # network simplex
                f3, v3 = lp_residual(inst, l, y, z)     # PL original

                checks += 1
                ok = True
                if not (f1 == f2 == f3):                        # (A)
                    ok = False
                    print(f"[FAIL A] seed={seed} l={l} y={y} z={z}: "
                          f"viab SSP={f1} nx={f2} LP={f3}")
                elif f1:
                    if v1 != v2:                                # (B)
                        ok = False
                        print(f"[FAIL B] seed={seed} l={l}: SSP={v1} nx={v2}")
                    if abs(v3 - v1) > TOL * max(1.0, abs(v1)):  # (C)
                        ok = False
                        print(f"[FAIL C] seed={seed} l={l}: LP={v3} MCMF={v1}")
                    if abs(v3 - round(v3)) > TOL:               # (D)
                        ok = False
                        print(f"[FAIL D] seed={seed} l={l}: LP={v3} nao-int")
                    n_feas_cases += 1
                else:
                    n_infeas_cases += 1
                if not ok:
                    fails += 1

    print(f"\nverify_A1_oracle: {n_instances} instancias, "
          f"{checks} casos (produto x desenho): "
          f"{checks - fails} PASS, {fails} FAIL "
          f"({n_feas_cases} viaveis, {n_infeas_cases} inviaveis)")
    if fails:
        sys.exit(1)
    print("TODOS OS TESTES PASSARAM")


if __name__ == "__main__":
    main()
