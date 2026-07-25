#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_A3b_w2.py — Verificacao computacional dos resultados W[2] do artigo (W[2]-dureza;
R6: cota inferior ETH) do projeto MP-TSCFLP-PCA.

Reducao COMPOSTA verificada:

  DOMINATING SET (G = (V,E), alvo t)
    -> SET COVER por vizinhancas fechadas:
         U = V, familia = { N[v] : v em V }  (mesmo parametro t;
         |U| = |familia| = n = |V|)
    -> MP-TSCFLP via a construcao Phi do Teorema A2.1 (reutilizada
       verbatim; ver verify_A2_setcover.py):
         |L| = 1, |I| = 1, f_1 = 0, c = 0, amplificador Q = m + 1 = n + 1,
         depositos = conjuntos (g = 1, p_j = n_U * Q), clientes = elementos
         (q_u = Q), d_{ju} = 0 sse u em N[v_j], senao 1;
       orcamento B = t e cardinalidade k = t + 1 (o +1 e a fabrica unica,
       que precisa abrir e conta em sum(y) + sum(z) da Def. A1.D2).

O lado MP-TSCFLP e resolvido por FORCA BRUTA sobre TODOS os desenhos
(y, z) — inclusive y_1 = 0 — com ROTEAMENTO EXATO pelo oraculo inteiro de
fluxo de custo minimo do modulo comum (`routing_value` de common_mp_tscfl.py),
sem usar a formula fechada das provas. gamma(G) e computado por forca bruta
independente sobre subconjuntos de V.

Checagens, por grafo (t := gamma(G)):
  (1) [iff]    para TODO t' em {1..n}:
                 gamma(G) <= t'  <=>  MP-TSCFLP(B = t', k = t'+1) e SIM;
  (2) [exato]  MP(B=t,   k=t+1) = SIM;
               MP(B=t-1, k=t+1) = NAO;   MP(B=t, k=t) = NAO
               (a contabilidade B = t, k = t+1 e justa nos dois eixos);
  (3) [folga]  em todo desenho viavel: y_1 = 1 e custo total >= sum(z);
               logo orcamento B ja forca <= B depositos abertos e a
               restricao de cardinalidade k = B + 1 e FOLGADA — conferido
               tambem diretamente: para todo t' em {0..n+1},
               MP(B = t', k = t'+1) = MP(B = t', sem restricao de card.);
  (4) [sanidade] desenho com D vazio ou y_1 = 0 e inviavel (n >= 1).

Baterias:
  (A) TODOS os 64 grafos rotulados com |V| = 4;
  (B) >= 200 grafos aleatorios semeados com |V| em {5, 6, 7}
      (semente 20260710, densidades variadas).

Saida: contagens e PASS/FAIL por bateria; codigo de saida != 0 em falha.
"""

import itertools
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_mp_tscfl import routing_value, all_designs  # noqa: E402


# ---------------------------------------------------------------------------
# Dominating Set -> Set Cover (vizinhancas fechadas)
# ---------------------------------------------------------------------------

def closed_neighborhoods(n, edges):
    """Familia { N[v] : v em V } sobre U = V = {0..n-1}."""
    nb = [set([v]) for v in range(n)]
    for (u, v) in edges:
        nb[u].add(v)
        nb[v].add(u)
    return [frozenset(nb[v]) for v in range(n)]


def domination_number(n, edges):
    """gamma(G) por forca bruta (existe sempre: V domina, via N[v] ∋ v)."""
    fam = closed_neighborhoods(n, edges)
    universe = set(range(n))
    for r in range(1, n + 1):
        for combo in itertools.combinations(range(n), r):
            if set().union(*(fam[j] for j in combo)) >= universe:
                return r
    raise AssertionError("V sempre domina — inatingivel")


# ---------------------------------------------------------------------------
# Set Cover -> MP-TSCFLP (construcao Phi de A2.1, verbatim)
# ---------------------------------------------------------------------------

def build_mp_instance(n_U, sets):
    """Instancia MP-TSCFLP Phi(U, S) no formato de common_mp_tscfl.

    Q = m + 1; b_1 = p_j = n_U * Q; q_u = Q; f = 0, g = 1, c = 0,
    d_{ju} = [u nao em S_j]."""
    m = len(sets)
    Q = m + 1
    cap = n_U * Q
    inst = {
        "nI": 1, "nJ": m, "nK": n_U, "nL": 1,
        "f": [0],
        "g": [1] * m,
        "c": [[[0] for _ in range(m)]],                      # c[i][j][l]
        "d": [[[0 if u in sets[j] else 1] for u in range(n_U)]
              for j in range(m)],                            # d[j][k][l]
        "b": [[cap]],                                        # b[i][l]
        "p": [[cap] for _ in range(m)],                      # p[j][l]
        "q": [[Q] for _ in range(n_U)],                      # q[k][l]
    }
    return inst, Q


def enumerate_solutions(inst):
    """Forca bruta sobre TODOS os (y, z), roteamento exato via MCMF inteiro.

    Retorna lista de desenhos viaveis como tuplas
    (custo_total, cardinalidade sum(y)+sum(z), sum(z), y_1)."""
    sols = []
    for y, z in all_designs(inst["nI"], inst["nJ"]):
        total_route = 0
        feasible = True
        for l in range(inst["nL"]):
            ok, val = routing_value(inst, l, y, z)
            if not ok:
                feasible = False
                break
            total_route += val
        if not feasible:
            continue
        fixed = sum(inst["f"][i] * y[i] for i in range(inst["nI"])) \
              + sum(inst["g"][j] * z[j] for j in range(inst["nJ"]))
        sols.append((fixed + total_route, sum(y) + sum(z), sum(z), y[0]))
    return sols


def mp_yes(sols, B, k):
    """MP-TSCFLP(B, k) e SIM? (k = None: sem restricao de cardinalidade)."""
    return any(cost <= B and (k is None or card <= k)
               for cost, card, _sz, _y1 in sols)


# ---------------------------------------------------------------------------
# Checagens por grafo
# ---------------------------------------------------------------------------

def check_graph(n, edges, failures, counts):
    fam = closed_neighborhoods(n, edges)
    t = domination_number(n, edges)          # 1 <= t <= n sempre
    inst, Q = build_mp_instance(n, fam)
    sols = enumerate_solutions(inst)

    tag = "n=%d edges=%s" % (n, sorted(edges))

    # (1) iff para todo t' em 1..n (parametros t' <= m = n, WLOG de A2 valido)
    for tp in range(1, n + 1):
        counts["iff"] += 1
        lhs = (t <= tp)
        rhs = mp_yes(sols, B=tp, k=tp + 1)
        if lhs != rhs:
            failures.append((tag, "iff t'=%d: gamma<=t' e %s, MP(t',t'+1) e %s"
                             % (tp, lhs, rhs)))

    # (2) exatidao da contabilidade em t = gamma(G)
    counts["exact"] += 3
    if not mp_yes(sols, B=t, k=t + 1):
        failures.append((tag, "MP(B=%d,k=%d) deveria ser SIM" % (t, t + 1)))
    if mp_yes(sols, B=t - 1, k=t + 1):
        failures.append((tag, "MP(B=%d,k=%d) deveria ser NAO" % (t - 1, t + 1)))
    if mp_yes(sols, B=t, k=t):
        failures.append((tag, "MP(B=%d,k=%d) deveria ser NAO" % (t, t)))

    # (3) folga da cardinalidade: custo >= sum(z) e y_1 = 1 em todo viavel;
    #     e MP(B,k=B+1) coincide com MP(B, sem cardinalidade) para todo B
    for cost, card, sz, y1 in sols:
        counts["slack"] += 1
        if y1 != 1:
            failures.append((tag, "desenho viavel com y_1=0"))
        if cost < sz:
            failures.append((tag, "custo %d < sum(z) %d" % (cost, sz)))
        if cost <= n and card > cost + 1:
            failures.append((tag, "custo %d mas cardinalidade %d > custo+1"
                             % (cost, card)))
    for B in range(0, n + 2):
        counts["slack"] += 1
        if mp_yes(sols, B, B + 1) != mp_yes(sols, B, None):
            failures.append((tag, "B=%d: k=B+1 difere de k=infinito" % B))

    # (4) sanidade: nenhum desenho viavel com D vazio
    counts["sanity"] += 1
    if any(sz == 0 for _c, _k, sz, _y in sols):
        failures.append((tag, "desenho viavel com D vazio"))


# ---------------------------------------------------------------------------
# Baterias
# ---------------------------------------------------------------------------

def all_labeled_graphs(n):
    pairs = list(itertools.combinations(range(n), 2))
    for mask in range(1 << len(pairs)):
        yield [pairs[i] for i in range(len(pairs)) if (mask >> i) & 1]


def random_graphs(count, seed):
    rng = random.Random(seed)
    out = []
    densities = [0.15, 0.3, 0.5, 0.7, 0.9]
    while len(out) < count:
        n = rng.choice([5, 6, 7])
        p = rng.choice(densities)
        edges = [(u, v) for (u, v) in itertools.combinations(range(n), 2)
                 if rng.random() < p]
        out.append((n, edges))
    return out


def main():
    failures = []
    counts = {"iff": 0, "exact": 0, "slack": 0, "sanity": 0}

    # (A) todos os grafos rotulados com |V| = 4
    n_graphs_A = 0
    for edges in all_labeled_graphs(4):
        check_graph(4, edges, failures, counts)
        n_graphs_A += 1
    print("[A] exaustivo: %d grafos rotulados com |V| = 4" % n_graphs_A)

    # (B) grafos aleatorios semeados, |V| em {5,6,7}
    rg = random_graphs(210, seed=20260710)
    for n, edges in rg:
        check_graph(n, edges, failures, counts)
    print("[B] aleatorio: %d grafos (semente 20260710, |V| em {5,6,7})" % len(rg))

    print("\nChecagens: iff=%d, exatidao(B,k)=%d, folga-cardinalidade=%d, "
          "sanidade=%d  (total %d)"
          % (counts["iff"], counts["exact"], counts["slack"],
             counts["sanity"], sum(counts.values())))

    if failures:
        print("\nFALHAS (%d):" % len(failures))
        for tag, msg in failures[:20]:
            print("  %s : %s" % (tag, msg))
        sys.exit(1)
    print("TODOS OS TESTES PASSARAM.")


if __name__ == "__main__":
    main()
