#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_A3a_partition.py — Verificacao computacional do Teorema A3a.1 (R4b):
NP-dificuldade FRACA do MP-TSCFLP com |J| = |L| = |K| = 1 (via PARTITION)
e resolubilidade pseudo-polinomial da mesma celula (DP knapsack-cover).

Reducao verificada (Teorema A3a.1(i)), de PARTITION (a_1..a_m, A = sum a_i,
WLOG A par, alvo D = A/2):
  * fabricas = itens: I = {1..m}, b_i = a_i, f_i = a_i;
  * 1 deposito: g_1 = 0, p_1 = D;
  * 1 cliente, 1 produto: q = D;
  * todos os custos de transporte nulos (c = d = 0);
  * orcamento B = D = A/2.
Afirmacao: existe S com sum_{i in S} a_i = A/2  <=>  OPT_MP <= A/2.
(Estrutural: OPT_MP = min { sum_{i in S} a_i : S subconjunto, sum >= D },
pois toda solucao viavel custa exatamente a soma dos f_i abertos — F1 de
A1.2 exige sum b_i y_i >= D e o transporte e gratuito.)

O otimo do MP-TSCFLP reduzido e computado por forca bruta INDEPENDENTE da
formula: enumeracao de TODOS os desenhos (y,z) via common_mp_tscfl.all_designs
com roteamento exato pelo MCMF inteiro de common_mp_tscfl.routing_value
(oraculo da Prop. A1.1 — nenhuma forma fechada da prova e usada aqui).

DP verificado (Teorema A3a.1(ii)), celula |J| = |K| = |L| = 1 com dados
GERAIS (f, g, c, d, b, p, q inteiros >= 0):
  OPT = g_1 + d_111 * D + DP(D), onde DP(t) = custo minimo de abrir um
  subconjunto de fabricas e enviar exatamente t unidades (u_i <= b_i,
  custo f_i + c_i * u_i por fabrica usada), com os casos D = 0 (OPT = 0)
  e inviabilidade (p_1 < D ou sum b_i < D) a parte.

Tambem verificados (celula SIMETRICA |I| = |K| = |L| = 1 do item (iii)
do Teorema A3a.1): a reducao espelhada de PARTITION (itens = DEPOSITOS:
p_j = g_j = a_j; fabrica unica gratuita com b_1 = D) e o DP simetrico
(fabrica forcada aberta; custo unitario via deposito j = c_1j1 + d_j11).

Baterias:
  (A) EXAUSTIVO: todos os multiconjuntos com 1 <= m <= 5, valores em 0..8
      (deduplicados por combinations_with_replacement). Para A par: testa
      a dupla implicacao com orcamento EXATO B = A/2 + a identidade
      estrutural OPT_MP = min{sum_S : sum_S >= D}. Para A impar: confirma
      que PARTITION responde "nao" (validacao da convencao WLOG).
  (B) ALEATORIO: >= 100 instancias semeadas, m <= 10, valores <= 40;
      se a soma A sai impar, forca-se paridade com a[0] += 1 (por isso
      valores <= 41), mesmos testes de (A).
  (C) DP: >= 120 instancias semeadas da celula geral |J|=|K|=|L|=1
      (custos de transporte NAO nulos), DP + offsets == forca bruta
      (all_designs + MCMF), incluindo casos D = 0 e inviaveis.
  (D) ESPELHO: reducao simetrica (itens = depositos) nas MESMAS instancias
      exaustivas de (A) com A par + as 100 de (B).
  (E) DP SIMETRICO: >= 120 instancias semeadas da celula |I|=|K|=|L|=1.
Saida: contagens e PASS/FAIL; codigo de saida != 0 em falha.
"""

import itertools
import random
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_mp_tscfl import all_designs, routing_value

# ---------------------------------------------------------------------------
# Reducao do Teorema A3a.1(i)
# ---------------------------------------------------------------------------

def build_partition_instance(a):
    """Instancia MP-TSCFLP da reducao (A par obrigatorio). Retorna (inst, B)."""
    A = sum(a)
    assert A % 2 == 0, "reducao definida apenas para A par (WLOG do teorema)"
    D = A // 2
    m = len(a)
    inst = {
        "nI": m, "nJ": 1, "nK": 1, "nL": 1,
        "f": list(a),
        "g": [0],
        "c": [[[0]] for _ in range(m)],        # c[i][0][0] = 0
        "d": [[[0]]],                          # d[0][0][0] = 0
        "b": [[ai] for ai in a],               # b[i][0] = a_i
        "p": [[D]],                            # p[0][0] = D
        "q": [[D]],                            # q[0][0] = D
    }
    return inst, D


def brute_force_mp_opt(inst):
    """OPT do MP-TSCFLP por forca bruta: todos os (y,z), roteamento MCMF
    exato (Prop. A1.1). Retorna None se nenhum desenho e viavel."""
    best = None
    for y, z in all_designs(inst["nI"], inst["nJ"]):
        total = sum(inst["f"][i] * y[i] for i in range(inst["nI"])) \
              + sum(inst["g"][j] * z[j] for j in range(inst["nJ"]))
        ok = True
        for l in range(inst["nL"]):
            feas, val = routing_value(inst, l, y, z)
            if not feas:
                ok = False
                break
            total += val
        if ok and (best is None or total < best):
            best = total
    return best


def brute_force_partition(a):
    """Existe subconjunto com soma exatamente sum(a)/2? (forca bruta)."""
    A = sum(a)
    if A % 2 == 1:
        return False
    target = A // 2
    sums = {0}
    for ai in a:
        sums |= {s + ai for s in sums}
    return target in sums


def structural_opt(a):
    """min { sum_{i in S} a_i : sum_{i in S} a_i >= D } (identidade do lema)."""
    D = sum(a) // 2
    best = None
    m = len(a)
    for mask in range(1 << m):
        s = sum(a[i] for i in range(m) if (mask >> i) & 1)
        if s >= D and (best is None or s < best):
            best = s
    return best


def check_partition_instance(a, failures):
    """Testa a reducao numa instancia com A par. Retorna #checks."""
    A = sum(a)
    D = A // 2
    inst, B = build_partition_instance(a)
    opt = brute_force_mp_opt(inst)
    part_yes = brute_force_partition(a)
    checks = 0

    # (1) instancia sempre viavel (sum b = A >= D)
    checks += 1
    if opt is None:
        failures.append((a, "reducao inviavel (esperado viavel)"))
        return checks
    # (2) identidade estrutural do lema: OPT = min{sum_S : sum_S >= D}
    checks += 1
    if opt != structural_opt(a):
        failures.append((a, "OPT=%s != estrutural=%s" % (opt, structural_opt(a))))
    # (3) dupla implicacao com orcamento exato B = D
    checks += 1
    if (opt <= B) != part_yes:
        failures.append((a, "OPT=%s<=B=%s e %s, mas PARTITION e %s"
                         % (opt, B, opt <= B, part_yes)))
    # (4) minorante: OPT >= D sempre
    checks += 1
    if opt < D:
        failures.append((a, "OPT=%s < D=%s" % (opt, D)))
    return checks


def build_partition_mirror_instance(a):
    """Reducao simetrica (item (iii) do Teorema A3a.1): itens = depositos."""
    A = sum(a)
    assert A % 2 == 0
    D = A // 2
    m = len(a)
    inst = {
        "nI": 1, "nJ": m, "nK": 1, "nL": 1,
        "f": [0],
        "g": list(a),
        "c": [[[0] for _ in range(m)]],        # c[0][j][0] = 0
        "d": [[[0]] for _ in range(m)],        # d[j][0][0] = 0
        "b": [[D]],                            # b[0][0] = D
        "p": [[aj] for aj in a],               # p[j][0] = a_j
        "q": [[D]],
    }
    return inst, D


def check_partition_mirror(a, failures):
    """Testa a reducao espelhada numa instancia com A par. Retorna #checks."""
    A = sum(a)
    D = A // 2
    inst, B = build_partition_mirror_instance(a)
    opt = brute_force_mp_opt(inst)
    part_yes = brute_force_partition(a)
    checks = 0
    checks += 1
    if opt is None:
        failures.append((a, "espelho: reducao inviavel (esperado viavel)"))
        return checks
    checks += 1
    if opt != structural_opt(a):
        failures.append((a, "espelho: OPT=%s != estrutural=%s"
                         % (opt, structural_opt(a))))
    checks += 1
    if (opt <= B) != part_yes:
        failures.append((a, "espelho: OPT=%s<=B=%s e %s, mas PARTITION e %s"
                         % (opt, B, opt <= B, part_yes)))
    return checks


# ---------------------------------------------------------------------------
# DP do Teorema A3a.1(ii)
# ---------------------------------------------------------------------------

def knapsack_cover_dp(f, c, b, D):
    """DP(t): custo minimo de enviar exatamente t unidades, t = 0..D.
    Fabrica i usada com u em 1..b_i unidades custa f_i + c_i*u.
    Retorna DP(D) ou None (inviavel: sum b < D)."""
    dp = [None] * (D + 1)
    dp[0] = 0
    for i in range(len(f)):
        ndp = dp[:]
        for t in range(1, D + 1):
            best = ndp[t]
            for u in range(1, min(b[i], t) + 1):
                if dp[t - u] is not None:
                    cand = dp[t - u] + f[i] + c[i] * u
                    if best is None or cand < best:
                        best = cand
            ndp[t] = best
        dp = ndp
    return dp[D]


def dp_cell_opt(inst):
    """OPT da celula |J|=|K|=|L|=1 pela formula do Teorema A3a.1(ii)."""
    D = inst["q"][0][0]
    if D == 0:
        return 0
    if inst["p"][0][0] < D:
        return None  # F2 de A1.2 falha para todo desenho
    core = knapsack_cover_dp(inst["f"],
                             [inst["c"][i][0][0] for i in range(inst["nI"])],
                             [inst["b"][i][0] for i in range(inst["nI"])],
                             D)
    if core is None:
        return None  # F1 de A1.2 falha
    return inst["g"][0] + inst["d"][0][0][0] * D + core


def gen_cell_instance(seed):
    """Instancia aleatoria da celula geral |J|=|K|=|L|=1."""
    rng = random.Random(seed)
    m = rng.randint(1, 6)
    D = rng.randint(0, 12)
    return {
        "nI": m, "nJ": 1, "nK": 1, "nL": 1,
        "f": [rng.randint(0, 9) for _ in range(m)],
        "g": [rng.randint(0, 9)],
        "c": [[[rng.randint(0, 9)]] for _ in range(m)],
        "d": [[[rng.randint(0, 9)]]],
        "b": [[rng.randint(0, 6)] for _ in range(m)],
        "p": [[rng.randint(0, 14)]],
        "q": [[D]],
    }


def dp_cell_opt_mirror(inst):
    """OPT da celula simetrica |I|=|K|=|L|=1 (item (iii) do Teorema A3a.1):
    OPT = f_1 + DP(D) sobre depositos, custo unitario via j = c_1j1 + d_j11,
    capacidade p_j, custo fixo g_j; inviavel se b_1 < D ou sum p < D."""
    D = inst["q"][0][0]
    if D == 0:
        return 0
    if inst["b"][0][0] < D:
        return None  # F1 de A1.2 falha para todo desenho
    m = inst["nJ"]
    core = knapsack_cover_dp(inst["g"],
                             [inst["c"][0][j][0] + inst["d"][j][0][0]
                              for j in range(m)],
                             [inst["p"][j][0] for j in range(m)],
                             D)
    if core is None:
        return None  # F2 de A1.2 falha
    return inst["f"][0] + core


def gen_cell_instance_mirror(seed):
    """Instancia aleatoria da celula simetrica |I|=|K|=|L|=1."""
    rng = random.Random(seed)
    m = rng.randint(1, 6)
    D = rng.randint(0, 12)
    return {
        "nI": 1, "nJ": m, "nK": 1, "nL": 1,
        "f": [rng.randint(0, 9)],
        "g": [rng.randint(0, 9) for _ in range(m)],
        "c": [[[rng.randint(0, 9)] for _ in range(m)]],
        "d": [[[rng.randint(0, 9)]] for _ in range(m)],
        "b": [[rng.randint(0, 14)]],
        "p": [[rng.randint(0, 6)] for _ in range(m)],
        "q": [[D]],
    }


def main():
    failures = []

    # (A) exaustivo: multiconjuntos m <= 5, valores 0..8
    n_even = n_odd = checks_A = 0
    for m in range(1, 6):
        for a in itertools.combinations_with_replacement(range(9), m):
            a = list(a)
            if sum(a) % 2 == 1:
                n_odd += 1
                checks_A += 1
                if brute_force_partition(a):  # convencao WLOG: A impar => nao
                    failures.append((a, "A impar mas PARTITION respondeu sim"))
            else:
                n_even += 1
                checks_A += check_partition_instance(a, failures)
    print("[A] exaustivo: %d multiconjuntos (m<=5, valores<=8): %d com A par "
          "(reducao testada), %d com A impar (trivialidade confirmada); "
          "%d checagens" % (n_even + n_odd, n_even, n_odd, checks_A))

    # (B) aleatorio semeado: >= 100 instancias, m <= 10, A par
    rng = random.Random(20260710)
    n_B = 0
    checks_B = 0
    while n_B < 100:
        m = rng.randint(2, 10)
        a = [rng.randint(0, 40) for _ in range(m)]
        if sum(a) % 2 == 1:
            a[0] += 1  # forca A par sem viciar a distribuicao relevante
        checks_B += check_partition_instance(a, failures)
        n_B += 1
    print("[B] aleatorio: %d instancias (semente 20260710, m<=10, valores<=41), "
          "%d checagens" % (n_B, checks_B))

    # (C) DP pseudo-polinomial da celula geral |J|=|K|=|L|=1
    n_C = 120
    stats = {"feas": 0, "infeas": 0, "D0": 0}
    for s in range(n_C):
        inst = gen_cell_instance(300 + s)
        opt_bf = brute_force_mp_opt(inst)
        opt_dp = dp_cell_opt(inst)
        if inst["q"][0][0] == 0:
            stats["D0"] += 1
        elif opt_bf is None:
            stats["infeas"] += 1
        else:
            stats["feas"] += 1
        if opt_bf != opt_dp:
            failures.append((inst, "DP=%s != brute=%s" % (opt_dp, opt_bf)))
    print("[C] DP: %d instancias da celula |J|=|K|=|L|=1 com transporte geral "
          "(sementes 300..%d): %d viaveis, %d inviaveis, %d com D=0"
          % (n_C, 300 + n_C - 1, stats["feas"], stats["infeas"], stats["D0"]))

    # (D) reducao espelhada (itens = depositos) nas mesmas instancias
    n_D = checks_D = 0
    for m in range(1, 6):
        for a in itertools.combinations_with_replacement(range(9), m):
            a = list(a)
            if sum(a) % 2 == 0:
                n_D += 1
                checks_D += check_partition_mirror(a, failures)
    rng = random.Random(20260710)
    n_Drand = 0
    while n_Drand < 100:
        m = rng.randint(2, 10)
        a = [rng.randint(0, 40) for _ in range(m)]
        if sum(a) % 2 == 1:
            a[0] += 1
        checks_D += check_partition_mirror(a, failures)
        n_Drand += 1
    print("[D] espelho (itens=depositos): %d exaustivas (A par) + %d "
          "aleatorias, %d checagens" % (n_D, n_Drand, checks_D))

    # (E) DP simetrico da celula |I|=|K|=|L|=1
    n_E = 120
    stats_E = {"feas": 0, "infeas": 0, "D0": 0}
    for s in range(n_E):
        inst = gen_cell_instance_mirror(800 + s)
        opt_bf = brute_force_mp_opt(inst)
        opt_dp = dp_cell_opt_mirror(inst)
        if inst["q"][0][0] == 0:
            stats_E["D0"] += 1
        elif opt_bf is None:
            stats_E["infeas"] += 1
        else:
            stats_E["feas"] += 1
        if opt_bf != opt_dp:
            failures.append((inst, "DP espelho=%s != brute=%s"
                             % (opt_dp, opt_bf)))
    print("[E] DP simetrico: %d instancias da celula |I|=|K|=|L|=1 "
          "(sementes 800..%d): %d viaveis, %d inviaveis, %d com D=0"
          % (n_E, 800 + n_E - 1, stats_E["feas"], stats_E["infeas"],
             stats_E["D0"]))

    if failures:
        print("\nFALHAS (%d):" % len(failures))
        for f in failures[:20]:
            print("  %s : %s" % (f[0], f[1]))
        sys.exit(1)
    print("\nTODOS OS TESTES PASSARAM.")


if __name__ == "__main__":
    main()
