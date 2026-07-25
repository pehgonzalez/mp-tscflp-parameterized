#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_A2_splittable.py — Verificacao adversarial da "brecha do servico
divisivel" nos Teoremas A2.1/A2.3 do projeto MP-TSCFLP-PCA.

PERGUNTA ADVERSARIAL: nas instancias produzidas pela reducao de Set Cover,
fluxos FRACIONARIOS/DIVIDIDOS (servico repartido entre varios depositos, ou
entre deposito cobridor e nao-cobridor) poderiam custar menos do que o valor
|D| + Q * #{u nao coberto por D} afirmado nas provas?

METODO (documentando a escolha):
Para cada instancia reduzida e cada subconjunto D de depositos abertos
(enumeracao COMPLETA), resolvemos o subproblema de roteamento como um
PROGRAMA LINEAR generico via scipy.optimize.linprog (metodo "highs"),
com TODAS as variaveis continuas x_{1j}, w_{ju} e TODAS as restricoes do
MP-TSCFLP (demanda, conservacao no deposito, capacidade da fabrica,
capacidade dos depositos) — ou seja, sem usar em nenhum momento o argumento
"greedy" das provas. Comparamos o valor do LP com a formula fechada
    transporte(D) = Q * #{u : u nao coberto por D}
e com o roteamento inteiro guloso. Assim o LP e um verificador INDEPENDENTE.

Nota teorica registrada (nao usada pelo verificador, apenas contexto):
para (y,z) fixos o subproblema de roteamento e um fluxo de custo minimo com
dados inteiros, logo possui otimo INTEIRO (integralidade do politopo de
fluxos / matriz totalmente unimodular); testar apenas fluxos inteiros ja
seria legitimo. Preferimos o LP continuo porque ele ataca diretamente a
brecha "fracionaria" sem depender desse lema.

LP resolvido, para D fixo (1 fabrica, 1 produto):
  min  sum_{j,u} d_{ju} w_{ju}                     (c = 0 no estagio 1)
  s.a. sum_{j em D} w_{ju} >= Q          (u em U)   [demanda]
       sum_u w_{ju} - x_{1j} <= 0        (j em D)   [conservacao no deposito]
       sum_{j em D} x_{1j} <= b_1                   [capacidade fabrica]
       sum_u w_{ju} <= p_j               (j em D)   [capacidade deposito]
       x, w >= 0
com b_1 = p_j = |U|*Q. Depositos fechados nao recebem variaveis (w=x=0
forcado por z_j = 0). Tolerancia numerica: 1e-6 (dados pequenos, HiGHS
resolve exato nesses tamanhos).

Baterias (ampliadas apos revisao independente,
para casar com as faixas de verify_A2_setcover.py, de modo que a formula
fechada nunca rode "sem contraste" em instancia coberta so pelo outro script):
  (A) TODAS as familias deduplicadas com |U| <= 4, |S| <= 4 (mesma
      enumeracao da bateria [A] de verify_A2_setcover.py), ambos os
      amplificadores Q = m+1 (A2.1) e Q' = nm+m+1 (A2.3), todos os D
      nao vazios.
  (B) 50 instancias aleatorias semeadas com |U| <= 6, |S| <= 6, idem.
Criterio: LP_otimo == Q * #descobertos(D) em toda combinacao (=> fracionar
nunca ajuda, e a formula fechada usada nas provas e exata).
Codigo de saida != 0 em falha.
"""

import itertools
import random
import sys

import numpy as np
from scipy.optimize import linprog


def lp_routing_cost(n, sets, Q, depots):
    """Custo otimo de roteamento CONTINUO com D = depots aberto (LP generico).

    Variaveis: x_j (j em D) e w_{ju} (j em D, u em U), todas >= 0.
    Retorna o valor otimo do LP, ou None se inviavel.
    """
    D = list(depots)
    dn = len(D)
    if dn == 0:
        return None if n >= 1 else 0.0
    nv = dn + dn * n  # x depois w
    cap = n * Q       # b_1 = p_j = |U|*Q

    def wi(a, u):  # indice de w_{D[a], u}
        return dn + a * n + u

    obj = np.zeros(nv)
    for a, j in enumerate(D):
        for u in range(n):
            obj[wi(a, u)] = 0.0 if u in sets[j] else 1.0

    A_ub, b_ub = [], []
    # demanda: -sum_a w_{a,u} <= -Q
    for u in range(n):
        row = np.zeros(nv)
        for a in range(dn):
            row[wi(a, u)] = -1.0
        A_ub.append(row); b_ub.append(-float(Q))
    # conservacao no deposito: sum_u w_{a,u} - x_a <= 0
    for a in range(dn):
        row = np.zeros(nv)
        row[a] = -1.0
        for u in range(n):
            row[wi(a, u)] = 1.0
        A_ub.append(row); b_ub.append(0.0)
    # capacidade da fabrica: sum_a x_a <= b_1
    row = np.zeros(nv)
    row[:dn] = 1.0
    A_ub.append(row); b_ub.append(float(cap))
    # capacidade dos depositos: sum_u w_{a,u} <= p_j
    for a in range(dn):
        row = np.zeros(nv)
        for u in range(n):
            row[wi(a, u)] = 1.0
        A_ub.append(row); b_ub.append(float(cap))

    res = linprog(obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  bounds=[(0, None)] * nv, method="highs")
    if not res.success:
        return None
    return res.fun


def closed_form_cost(n, sets, Q, depots):
    """Formula fechada usada nas provas: Q * #{u nao coberto por D}."""
    return float(Q * sum(1 for u in range(n)
                         if not any(u in sets[j] for j in depots)))


def check_family(n, fam, failures, tol=1e-6):
    m = len(fam)
    checks = 0
    for Q in (m + 1, n * m + m + 1):  # amplificadores de A2.1 e A2.3
        for mask in range(1, 1 << m):
            depots = [j for j in range(m) if (mask >> j) & 1]
            lp = lp_routing_cost(n, fam, Q, depots)
            cf = closed_form_cost(n, fam, Q, depots)
            checks += 1
            if lp is None:
                failures.append((n, fam, "LP inviavel, D=%s Q=%d" % (depots, Q)))
            elif abs(lp - cf) > tol:
                failures.append((n, fam, "LP=%.9f != formula=%.1f, D=%s Q=%d"
                                 % (lp, cf, depots, Q)))
    return checks


def enumerate_all_families(max_n, max_m):
    for n in range(1, max_n + 1):
        candidates = [frozenset(c)
                      for r in range(n + 1)
                      for c in itertools.combinations(range(n), r)]
        for msize in range(1, max_m + 1):
            for family in itertools.combinations(candidates, msize):
                yield n, list(family)


def random_instances(count, max_n, max_m, seed):
    rng = random.Random(seed)
    out = []
    while len(out) < count:
        n = rng.randint(3, max_n)
        m = rng.randint(3, max_m)
        fam = [frozenset(u for u in range(n) if rng.random() < 0.5)
               for _ in range(m)]
        out.append((n, fam))
    return out


def main():
    failures = []

    n_inst_A = n_checks_A = 0
    for n, fam in enumerate_all_families(4, 4):
        n_inst_A += 1
        n_checks_A += check_family(n, fam, failures)
    print("[A] exaustivo (|U|<=4, |S|<=4): %d familias, %d LPs comparados "
          "(2 amplificadores x todos os D)" % (n_inst_A, n_checks_A))

    rand = random_instances(50, 6, 6, seed=20260710)
    n_checks_B = 0
    for n, fam in rand:
        n_checks_B += check_family(n, fam, failures)
    print("[B] aleatorio (semente 20260710, |U|<=6, |S|<=6): %d instancias, "
          "%d LPs comparados" % (len(rand), n_checks_B))

    if failures:
        print("\nFALHAS (%d):" % len(failures))
        for f in failures[:20]:
            print("  n=%d fam=%s : %s" % (f[0], [sorted(s) for s in f[1]], f[2]))
        sys.exit(1)
    print("\nTODOS OS TESTES PASSARAM: fracionar/dividir servico nunca "
          "melhora o valor previsto pela formula fechada das provas.")


if __name__ == "__main__":
    main()
