"""
Verification of the paper's clique-width observation (clique-width of the incidence graph).

The incidence graph of an MP-TSCFLP instance (complete stages) is
    G(nI, nJ, nK) = (I u J u K,  K_{I,J} u K_{J,K}),
that is, all plant-depot and depot-customer edges, no others.

Checks (all constructive, on the EXACT edge set):

  [1] The "layered" 3-expression from the paper's clique-width observation
          eta_{2,3}( eta_{1,2}( oplus_I 1(v) oplus oplus_J 2(v) oplus oplus_K 3(v) ) )
      generates exactly K_{I,J} u K_{J,K} (not one edge more, not one less),
      using exactly 3 labels. This also holds for the TYPED
      structure (I, J, K distinguishable by the final labels 1, 2, 3).

  [2] The 2-expression
          eta_{1,2}( oplus_J 2(v) oplus oplus_{I u K} 1(v) )
      generates the same edge set -- since, as a simple graph,
      G(nI,nJ,nK) = K_{|J|, |I|+|K|} (complete bipartite between J and I u K).
      This validates cwd(G) <= 2; since G has >= 1 edge when nJ >= 1 and
      nI+nK >= 1, cwd(G) = 2 (graphs of cwd 1 are edgeless).

  [3] Structural equality: the target set K_{I,J} u K_{J,K} coincides with
      the complete bipartite J vs I u K (the identity used in [2]).

Sizes tested: all (nI,nJ,nK) in {1..4}^3, plus larger and
degenerate cases. The clique-width operations (labeled creation, disjoint
union oplus, join eta_{a,b}, relabeling rho) are implemented
literally (Courcelle-Olariu 2000).
"""

import itertools


# ---------------------------------------------------------------------------
# Clique-width operations on labeled graphs
# ---------------------------------------------------------------------------

class LG:
    """Labeled graph: lab[v] = label; edges = set of frozensets."""

    def __init__(self):
        self.lab = {}
        self.edges = set()


def single(v, label):
    g = LG()
    g.lab[v] = label
    return g


def oplus(g, h):
    """Disjoint union (vertex names are already distinct by construction)."""
    assert not (set(g.lab) & set(h.lab)), "union not disjoint"
    r = LG()
    r.lab = dict(g.lab)
    r.lab.update(h.lab)
    r.edges = set(g.edges) | set(h.edges)
    return r


def eta(g, a, b):
    """Adds all edges between label a and label b (a != b)."""
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
    """Relabels a -> b (not used in the final expressions; included
    for completeness of the operation signature)."""
    r = LG()
    r.lab = {v: (b if l == a else l) for v, l in g.lab.items()}
    r.edges = set(g.edges)
    return r


# ---------------------------------------------------------------------------
# Target and expressions
# ---------------------------------------------------------------------------

def target_edges(nI, nJ, nK):
    """K_{I,J} u K_{J,K} -- the intended incidence graph."""
    E = set()
    for i in range(nI):
        for j in range(nJ):
            E.add(frozenset((("I", i), ("J", j))))
    for j in range(nJ):
        for k in range(nK):
            E.add(frozenset((("J", j), ("K", k))))
    return E


def bipartite_J_vs_IK(nI, nJ, nK):
    """Complete bipartite between J and I u K (identity of check [3])."""
    E = set()
    side = [("I", i) for i in range(nI)] + [("K", k) for k in range(nK)]
    for j in range(nJ):
        for v in side:
            E.add(frozenset((("J", j), v)))
    return E


def expr_3_labels(nI, nJ, nK):
    """eta_{2,3}(eta_{1,2}( oplus 1(I) oplus 2(J) oplus 3(K) ))."""
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
    """eta_{1,2}( oplus 2(J) oplus 1(I u K) )."""
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
# Battery
# ---------------------------------------------------------------------------

def main():
    sizes = list(itertools.product(range(1, 5), repeat=3))
    sizes += [(6, 5, 7), (8, 2, 3), (1, 1, 1), (10, 1, 10), (2, 9, 2),
              (1, 7, 1), (5, 1, 1), (1, 1, 5)]

    n_checks = 0
    fails = 0
    for (nI, nJ, nK) in sizes:
        E_target = target_edges(nI, nJ, nK)

        # [1] layered 3-expression
        g3 = expr_3_labels(nI, nJ, nK)
        ok1 = (g3.edges == E_target)
        labels3 = set(g3.lab.values())
        ok1b = (labels3 <= {1, 2, 3})
        # final labels preserve the I/J/K typing
        ok1c = all(g3.lab[("I", i)] == 1 for i in range(nI)) and \
               all(g3.lab[("J", j)] == 2 for j in range(nJ)) and \
               all(g3.lab[("K", k)] == 3 for k in range(nK))

        # [2] 2-expression (complete bipartite)
        g2 = expr_2_labels(nI, nJ, nK)
        ok2 = (g2.edges == E_target)
        ok2b = (set(g2.lab.values()) <= {1, 2})

        # [3] structural identity
        ok3 = (E_target == bipartite_J_vs_IK(nI, nJ, nK))

        # sanity: no I-I, J-J, K-K, I-K edges in the expressions
        def no_bad(E):
            for e in E:
                (t1, _), (t2, _) = tuple(e)
                if {t1, t2} in ({"I"}, {"J"}, {"K"}, {"I", "K"}):
                    return False
                if t1 == t2 or {t1, t2} == {"I", "K"}:
                    return False
            return True
        ok4 = no_bad(g3.edges) and no_bad(g2.edges)

        # exact edge count
        ok5 = (len(g3.edges) == nI * nJ + nJ * nK)

        for ok, name in [(ok1, "3-expr == target"), (ok1b, "3 labels"),
                         (ok1c, "typing preserved"),
                         (ok2, "2-expr == target"), (ok2b, "2 labels"),
                         (ok3, "target == K_{J, I+K}"),
                         (ok4, "no spurious edges"),
                         (ok5, "edge count |E|")]:
            n_checks += 1
            if not ok:
                fails += 1
                print(f"FAIL ({nI},{nJ},{nK}): {name}")

    print(f"Sizes tested: {len(sizes)}; checks: {n_checks}; "
          f"failures: {fails}")
    print("OVERALL RESULT:", "PASS" if fails == 0 else "FAIL")
    return fails


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
