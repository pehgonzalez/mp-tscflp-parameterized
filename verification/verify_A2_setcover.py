#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_A2_setcover.py — Verificacao computacional do Teorema A2.1 (R2) e da
aritmetica do gap do Teorema A2.3 (R3) do projeto MP-TSCFLP-PCA.

Reducao verificada (Teorema A2.1), de SET COVER (U, S = {S_1..S_m}, t):
  * 1 produto (|L| = 1), 1 fabrica (|I| = 1), f_1 = 0, c_{1j} = 0 p/ todo j;
  * depositos = conjuntos S_j, custo fixo g_j = 1, capacidade p_j = |U|*Q;
  * capacidade da fabrica b_1 = |U|*Q;
  * clientes = elementos u de U, demanda q_u = Q := |S| + 1 (amplificador);
  * d_{ju} = 0 se u pertence a S_j, senao 1;
  * orcamento B = t (WLOG 1 <= t <= m).
Afirmacao: existe cobertura de tamanho <= t  <=>  OPT_MP <= t.

RESOLUCAO EXATA DO MP-TSCFLP RESTRITO (justificativa do "greedy"):
Com (y, z) fixados (y_1 = 1, D = conjunto de depositos abertos, D nao vazio),
o roteamento otimo e trivial nesta subclasse:
  (i)   estagio 1 tem custo zero (c = 0) e a capacidade da fabrica
        b_1 = |U|*Q cobre a demanda total |U|*Q, logo nunca restringe;
  (ii)  cada capacidade de deposito p_j = |U|*Q tambem cobre sozinha toda a
        demanda total, logo nenhuma capacidade de deposito e ativa;
  (iii) o custo de transporte e separavel por unidade de fluxo: cada unidade
        da demanda do cliente u enviada pelo deposito aberto j custa
        exatamente d_{ju} (estagio 2) + 0 (estagio 1).
Portanto toda unidade da demanda de u custa >= min_{j em D} d_{ju}, e esse
custo por unidade e atingivel roteando toda a demanda Q de u por um deposito
aberto que minimize d_{ju} (viavel por (i)-(ii)). Logo o roteamento otimo
— inclusive sobre fluxos FRACIONARIOS/DIVIDIDOS — vale exatamente
      sum_u Q * min_{j em D} d_{ju},
e como d in {0,1}: custo de transporte = Q * #{u : u nao coberto por D}.
(O script verify_A2_splittable.py confirma isso contra um LP independente.)

O otimo global e obtido por forca bruta sobre TODOS os subconjuntos D
(nao vazios; D vazio e inviavel pois sum_j w_{ju} >= Q > 0 exige deposito
aberto com capacidade): OPT_MP = min_D [ |D| + Q * #{u nao coberto por D} ].

SET COVER tambem e resolvido por forca bruta independente (todos os
subconjuntos da familia).

Baterias:
  (A) TODAS as instancias de Set Cover com |U| <= 4, |S| <= 4
      (familias = combinacoes de subconjuntos distintos de U, ja
      deduplicadas por construcao; conjuntos vazios permitidos),
      para todo t em {1..m}: testa a dupla implicacao de A2.1.
  (B) >= 50 instancias aleatorias com semente fixa, |U| <= 6, |S| <= 6.
  (C) Aritmetica do gap de A2.3 com Q' := |U|*|S| + |S| + 1:
      - se coberto: OPT_MP(Q') = t* (tamanho da cobertura minima);
      - Q' > (1 + ln|U|) * |S|  (>= max{1, alpha} * t* para todo
        alpha <= ln|U| e t* <= |S|; e o enunciado de A2.3 usa o fator
        max{1, (1-eps) ln|U|} — correcao O1 da revisao independente);
      - para todo D: custo(D; Q') < Q'  =>  D cobre U.
Saida: contagens e PASS/FAIL por bateria; codigo de saida != 0 em falha.
"""

import itertools
import math
import random
import sys


# ---------------------------------------------------------------------------
# Solucionadores exatos
# ---------------------------------------------------------------------------

def mp_cost_given_depots(n, sets, Q, depots):
    """Custo exato da solucao do MP-TSCFLP reduzido com D = depots aberto.

    Exato inclusive sobre fluxos fracionarios — ver justificativa (i)-(iii)
    no cabecalho. Retorna |D| (custos fixos g=1; f=0) + Q por elemento de U
    nao coberto por nenhum deposito aberto.
    """
    cost = len(depots)
    for u in range(n):
        if not any(u in sets[j] for j in depots):
            cost += Q
    return cost


def solve_mp_bruteforce(n, sets, Q):
    """OPT do MP-TSCFLP reduzido: forca bruta sobre todos os (y,z).

    y_1 = 1 sempre (f=0, sem custo; necessario para escoar fluxo).
    D vazio e inviavel para n >= 1 (demanda positiva sem deposito aberto).
    Retorna None se m = 0 (inviavel).
    """
    m = len(sets)
    best = None
    for mask in range(1, 1 << m):
        depots = [j for j in range(m) if (mask >> j) & 1]
        c = mp_cost_given_depots(n, sets, Q, depots)
        if best is None or c < best:
            best = c
    return best


def solve_setcover_bruteforce(n, sets):
    """Tamanho minimo de cobertura, ou None se a familia nao cobre U."""
    m = len(sets)
    universe = set(range(n))
    best = None
    for r in range(1, m + 1):
        if best is not None:
            break  # combinacoes em ordem crescente de tamanho
        for combo in itertools.combinations(range(m), r):
            if set().union(*(sets[j] for j in combo)) >= universe:
                best = r
                break
    return best


# ---------------------------------------------------------------------------
# Baterias de teste
# ---------------------------------------------------------------------------

def check_instance_A21(n, sets, failures):
    """Testa a dupla implicacao de A2.1 para todo t em {1..m}; retorna #testes."""
    m = len(sets)
    Q = m + 1
    opt_mp = solve_mp_bruteforce(n, sets, Q)
    tstar = solve_setcover_bruteforce(n, sets)

    # Sanidade estrutural (usada na prova de A2.1):
    if tstar is not None:
        # instancia cobrivel: OPT_MP = t* exatamente
        if opt_mp != tstar:
            failures.append((n, sets, "OPT_MP=%s != t*=%s (Q=%d)" % (opt_mp, tstar, Q)))
    else:
        # nao cobrivel: todo D deixa elemento descoberto => custo >= Q + 1
        if opt_mp is not None and opt_mp <= m:  # em particular opt_mp < Q
            failures.append((n, sets, "nao-cobrivel mas OPT_MP=%s <= m=%d" % (opt_mp, m)))

    tests = 0
    for t in range(1, m + 1):  # WLOG 1 <= t <= m
        cover_exists = (tstar is not None and tstar <= t)
        mp_yes = (opt_mp is not None and opt_mp <= t)
        tests += 1
        if cover_exists != mp_yes:
            failures.append((n, sets, "t=%d: cobre<=t e %s mas OPT_MP<=t e %s"
                             % (t, cover_exists, mp_yes)))
    return tests


def check_instance_A23_gap(n, sets, failures):
    """Verifica a aritmetica do gap de A2.3 com Q' = n*m + m + 1."""
    m = len(sets)
    Qp = n * m + m + 1
    tstar = solve_setcover_bruteforce(n, sets)

    # (c2) Q' domina (1 + ln|U|) * m: assim max{1, alpha} * t* <= (1+ln n)*m
    #      < Q' para todo alpha <= ln n e t* <= m (cadeia do Passo 2 de A2.3
    #      com o fator max{1, (1-eps) ln|U|} da correcao O1).
    if not (Qp > (1 + math.log(n)) * m):
        failures.append((n, sets, "Q'=%d <= (1+ln(n))*m=%.4f"
                         % (Qp, (1 + math.log(n)) * m)))

    checks = 1
    # (c3) toda solucao de custo < Q' cobre U  (enumeracao completa de D)
    for mask in range(1, 1 << m):
        depots = [j for j in range(m) if (mask >> j) & 1]
        c = mp_cost_given_depots(n, sets, Qp, depots)
        covers = all(any(u in sets[j] for j in depots) for u in range(n))
        checks += 1
        if c < Qp and not covers:
            failures.append((n, sets, "D=%s custo=%d < Q'=%d mas nao cobre" % (depots, c, Qp)))
        # (c3') se cobre, custo = |D| >= t*
        if covers and c != len(depots):
            failures.append((n, sets, "D cobre mas custo=%d != |D|=%d" % (c, len(depots))))

    # (c1) se cobrivel, OPT_MP(Q') = t*
    if tstar is not None:
        opt_mp = solve_mp_bruteforce(n, sets, Qp)
        checks += 1
        if opt_mp != tstar:
            failures.append((n, sets, "gap: OPT_MP(Q')=%s != t*=%s" % (opt_mp, tstar)))
    return checks


def enumerate_all_families(max_n, max_m):
    """Todas as familias (deduplicadas) de ate max_m subconjuntos distintos
    de um universo de tamanho n, para n = 1..max_n. Combinacoes de conjuntos
    DISTINTOS => sem familias duplicadas nem conjuntos repetidos."""
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
        fam = []
        for _ in range(m):
            fam.append(frozenset(u for u in range(n) if rng.random() < 0.5))
        out.append((n, fam))
    return out


def main():
    failures = []

    # (A) exaustivo |U| <= 4, |S| <= 4
    n_inst_A = 0
    n_tests_A = 0
    for n, fam in enumerate_all_families(4, 4):
        n_inst_A += 1
        n_tests_A += check_instance_A21(n, fam, failures)
    print("[A] exaustivo: %d instancias (|U|<=4, |S|<=4), %d testes de "
          "equivalencia (todos os t em 1..m)" % (n_inst_A, n_tests_A))

    # (B) aleatorio semeado |U| <= 6, |S| <= 6
    rand = random_instances(60, 6, 6, seed=20260710)
    n_tests_B = 0
    for n, fam in rand:
        n_tests_B += check_instance_A21(n, fam, failures)
    print("[B] aleatorio: %d instancias (semente 20260710, |U|<=6, |S|<=6), "
          "%d testes de equivalencia" % (len(rand), n_tests_B))

    # (C) aritmetica do gap de A2.3 nas mesmas instancias
    n_checks_C = 0
    for n, fam in itertools.chain(enumerate_all_families(4, 4), rand):
        n_checks_C += check_instance_A23_gap(n, fam, failures)
    print("[C] gap A2.3: %d verificacoes (Q'=nm+m+1; OPT=t*; custo<Q' => cobre; "
          "Q'>(1+ln(n))*m)" % n_checks_C)

    if failures:
        print("\nFALHAS (%d):" % len(failures))
        for f in failures[:20]:
            print("  n=%d fam=%s : %s" % (f[0], [sorted(s) for s in f[1]], f[2]))
        sys.exit(1)
    print("\nTODOS OS TESTES PASSARAM.")


if __name__ == "__main__":
    main()
