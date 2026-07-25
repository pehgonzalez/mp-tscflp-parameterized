"""
Verificacao da observacao de clique-width do artigo (clique-width do grafo de incidencia).

O grafo de incidencia de uma instancia MP-TSCFLP (estagios completos) e
    G(nI, nJ, nK) = (I ∪ J ∪ K,  K_{I,J} ∪ K_{J,K}),
isto e, todas as arestas fabrica-deposito e deposito-cliente, nenhuma outra.

Checagens (todas construtivas, sobre o conjunto EXATO de arestas):

  [1] A 3-expressao "por camadas" da Obs. A4.4(a)
          eta_{2,3}( eta_{1,2}( ⊕_I 1(v) ⊕ ⊕_J 2(v) ⊕ ⊕_K 3(v) ) )
      gera exatamente K_{I,J} ∪ K_{J,K} (nem uma aresta a mais, nem a menos),
      usando exatamente 3 rotulos. Isto vale tambem para a estrutura
      TIPADA (I, J, K distinguiveis pelos rotulos finais 1, 2, 3).

  [2] A 2-expressao
          eta_{1,2}( ⊕_J 2(v) ⊕ ⊕_{I∪K} 1(v) )
      gera o mesmo conjunto de arestas — pois, como grafo simples,
      G(nI,nJ,nK) = K_{|J|, |I|+|K|} (bipartido completo entre J e I∪K).
      Isso valida cwd(G) ≤ 2; como G tem >= 1 aresta quando nJ >= 1 e
      nI+nK >= 1, cwd(G) = 2 (grafos de cwd 1 sao sem arestas).

  [3] Igualdade estrutural: o conjunto-alvo K_{I,J} ∪ K_{J,K} coincide com
      o bipartido completo J vs I∪K (a identidade usada em [2]).

Tamanhos testados: todos (nI,nJ,nK) em {1..4}^3, mais casos maiores e
degenerados. As operacoes de clique-width (criacao rotulada, uniao
disjunta ⊕, juncao eta_{a,b}, renomeacao rho) sao implementadas
literalmente (Courcelle–Olariu 2000).
"""

import itertools


# ---------------------------------------------------------------------------
# Operacoes de clique-width sobre grafos rotulados
# ---------------------------------------------------------------------------

class LG:
    """Grafo rotulado: lab[v] = rotulo; edges = conjunto de frozensets."""

    def __init__(self):
        self.lab = {}
        self.edges = set()


def single(v, label):
    g = LG()
    g.lab[v] = label
    return g


def oplus(g, h):
    """Uniao disjunta (os nomes de vertices ja sao distintos por construcao)."""
    assert not (set(g.lab) & set(h.lab)), "uniao nao disjunta"
    r = LG()
    r.lab = dict(g.lab)
    r.lab.update(h.lab)
    r.edges = set(g.edges) | set(h.edges)
    return r


def eta(g, a, b):
    """Adiciona todas as arestas entre rotulo a e rotulo b (a != b)."""
    assert a != b
    r = LG()
    r.lab = dict(g.lab)
    r.edges = set(g.edges)
    va = [v for v, l in g.lab.items() if l == a]
    vb = [v for v, l in g.lab.items() if l == b]
    for u in va:
        for w in vb:
            r.edges.add(frozenset((u, w)))
    return r


def rho(g, a, b):
    """Renomeia rotulo a -> b (nao usada nas expressoes finais; incluida
    por completude da assinatura de operacoes)."""
    r = LG()
    r.lab = {v: (b if l == a else l) for v, l in g.lab.items()}
    r.edges = set(g.edges)
    return r


# ---------------------------------------------------------------------------
# Alvo e expressoes
# ---------------------------------------------------------------------------

def target_edges(nI, nJ, nK):
    """K_{I,J} ∪ K_{J,K} — o grafo de incidencia pretendido."""
    E = set()
    for i in range(nI):
        for j in range(nJ):
            E.add(frozenset((("I", i), ("J", j))))
    for j in range(nJ):
        for k in range(nK):
            E.add(frozenset((("J", j), ("K", k))))
    return E


def bipartite_J_vs_IK(nI, nJ, nK):
    """Bipartido completo entre J e I ∪ K (identidade da checagem [3])."""
    E = set()
    side = [("I", i) for i in range(nI)] + [("K", k) for k in range(nK)]
    for j in range(nJ):
        for v in side:
            E.add(frozenset((("J", j), v)))
    return E


def expr_3_labels(nI, nJ, nK):
    """eta_{2,3}(eta_{1,2}( ⊕ 1(I) ⊕ 2(J) ⊕ 3(K) ))."""
    g = LG()
    for i in range(nI):
        g = oplus(g, single(("I", i), 1))
    for j in range(nJ):
        g = oplus(g, single(("J", j), 2))
    for k in range(nK):
        g = oplus(g, single(("K", k), 3))
    g = eta(g, 1, 2)
    g = eta(g, 2, 3)
    return g


def expr_2_labels(nI, nJ, nK):
    """eta_{1,2}( ⊕ 2(J) ⊕ 1(I ∪ K) )."""
    g = LG()
    for j in range(nJ):
        g = oplus(g, single(("J", j), 2))
    for i in range(nI):
        g = oplus(g, single(("I", i), 1))
    for k in range(nK):
        g = oplus(g, single(("K", k), 1))
    g = eta(g, 1, 2)
    return g


# ---------------------------------------------------------------------------
# Bateria
# ---------------------------------------------------------------------------

def main():
    sizes = list(itertools.product(range(1, 5), repeat=3))
    sizes += [(6, 5, 7), (8, 2, 3), (1, 1, 1), (10, 1, 10), (2, 9, 2),
              (1, 7, 1), (5, 1, 1), (1, 1, 5)]

    n_checks = 0
    fails = 0
    for (nI, nJ, nK) in sizes:
        E_target = target_edges(nI, nJ, nK)

        # [1] 3-expressao por camadas
        g3 = expr_3_labels(nI, nJ, nK)
        ok1 = (g3.edges == E_target)
        labels3 = set(g3.lab.values())
        ok1b = (labels3 <= {1, 2, 3})
        # rotulos finais preservam a tipagem I/J/K
        ok1c = all(g3.lab[("I", i)] == 1 for i in range(nI)) and \
               all(g3.lab[("J", j)] == 2 for j in range(nJ)) and \
               all(g3.lab[("K", k)] == 3 for k in range(nK))

        # [2] 2-expressao (bipartido completo)
        g2 = expr_2_labels(nI, nJ, nK)
        ok2 = (g2.edges == E_target)
        ok2b = (set(g2.lab.values()) <= {1, 2})

        # [3] identidade estrutural
        ok3 = (E_target == bipartite_J_vs_IK(nI, nJ, nK))

        # sanidade: nenhuma aresta I-I, J-J, K-K, I-K nas expressoes
        def no_bad(E):
            for e in E:
                (t1, _), (t2, _) = tuple(e)
                if {t1, t2} in ({"I"}, {"J"}, {"K"}, {"I", "K"}):
                    return False
                if t1 == t2 or {t1, t2} == {"I", "K"}:
                    return False
            return True
        ok4 = no_bad(g3.edges) and no_bad(g2.edges)

        # contagem exata de arestas
        ok5 = (len(g3.edges) == nI * nJ + nJ * nK)

        for ok, name in [(ok1, "3-expr == alvo"), (ok1b, "3 rotulos"),
                         (ok1c, "tipagem preservada"),
                         (ok2, "2-expr == alvo"), (ok2b, "2 rotulos"),
                         (ok3, "alvo == K_{J, I+K}"),
                         (ok4, "sem arestas espurias"),
                         (ok5, "contagem |E|")]:
            n_checks += 1
            if not ok:
                fails += 1
                print(f"FAIL ({nI},{nJ},{nK}): {name}")

    print(f"Tamanhos testados: {len(sizes)}; checagens: {n_checks}; "
          f"falhas: {fails}")
    print("RESULTADO GLOBAL:", "PASS" if fails == 0 else "FAIL")
    return fails


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
