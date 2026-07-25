"""
Modulo comum de verificacao para os resultados estruturais do artigo (MP-TSCFLP).

Conteudo:
  - gerador de instancias aleatorias com semente (inteiros pequenos);
  - min-cost flow exato por caminhos aumentantes mais curtos (SSP),
    aritmetica 100% inteira (Bellman-Ford/SPFA no grafo residual);
  - construtor da rede em camadas N_l(y,z) da Prop. A1.1;
  - oraculo de roteamento por produto (viabilidade + valor otimo).

Somente stdlib. networkx / scipy sao usados apenas nos scripts de
verificacao, como implementacoes independentes de contraste.
"""

import random

INF = float("inf")


# ---------------------------------------------------------------------------
# Geracao de instancias
# ---------------------------------------------------------------------------

def gen_instance(seed, max_i=4, max_j=4, max_k=4, max_l=3, vmax=9):
    """Instancia MP-TSCFLP aleatoria com dados inteiros em [0, vmax].

    Com probabilidade 0.2 um produto tem demanda identicamente nula
    (para exercitar o caso de borda D_l = 0). Capacidades e demandas
    nulas surgem naturalmente do intervalo [0, vmax].
    """
    rng = random.Random(seed)
    nI = rng.randint(1, max_i)
    nJ = rng.randint(1, max_j)
    nK = rng.randint(1, max_k)
    nL = rng.randint(1, max_l)

    f = [rng.randint(0, vmax) for _ in range(nI)]
    g = [rng.randint(0, vmax) for _ in range(nJ)]
    c = [[[rng.randint(0, vmax) for _ in range(nL)] for _ in range(nJ)]
         for _ in range(nI)]
    d = [[[rng.randint(0, vmax) for _ in range(nL)] for _ in range(nK)]
         for _ in range(nJ)]
    b = [[rng.randint(0, vmax) for _ in range(nL)] for _ in range(nI)]
    p = [[rng.randint(0, vmax) for _ in range(nL)] for _ in range(nJ)]
    q = [[rng.randint(0, vmax) for _ in range(nL)] for _ in range(nK)]

    for l in range(nL):
        if rng.random() < 0.2:  # produto sem demanda (caso de borda)
            for k in range(nK):
                q[k][l] = 0

    return {
        "nI": nI, "nJ": nJ, "nK": nK, "nL": nL,
        "f": f, "g": g, "c": c, "d": d, "b": b, "p": p, "q": q,
    }


def demand_total(inst, l):
    """D_l = sum_k q_kl."""
    return sum(inst["q"][k][l] for k in range(inst["nK"]))


# ---------------------------------------------------------------------------
# Min-cost flow exato (successive shortest paths, aritmetica inteira)
# ---------------------------------------------------------------------------

class MinCostFlow:
    """MCMF por caminhos aumentantes de custo minimo (SPFA no residual).

    Todos os dados sao inteiros; todos os calculos permanecem em Z.
    """

    def __init__(self, n):
        self.n = n
        self.graph = [[] for _ in range(n)]

    def add_edge(self, u, v, cap, cost):
        assert isinstance(cap, int) and isinstance(cost, int) and cap >= 0
        self.graph[u].append([v, cap, cost, len(self.graph[v])])
        self.graph[v].append([u, 0, -cost, len(self.graph[u]) - 1])

    def flow(self, s, t, target):
        """Envia ate `target` unidades de s a t com custo minimo.

        Retorna (fluxo_enviado, custo_total). fluxo_enviado < target
        significa que o fluxo maximo s-t e inferior a target
        (SSP so para quando nao ha caminho aumentante).
        """
        total_flow = 0
        total_cost = 0
        n = self.n
        while total_flow < target:
            dist = [None] * n           # None = +infinito (tudo inteiro)
            in_queue = [False] * n
            prev_v = [-1] * n
            prev_e = [-1] * n
            dist[s] = 0
            queue = [s]
            in_queue[s] = True
            while queue:
                u = queue.pop(0)
                in_queue[u] = False
                du = dist[u]
                for idx, (v, cap, cost, _rev) in enumerate(self.graph[u]):
                    if cap > 0 and (dist[v] is None or du + cost < dist[v]):
                        dist[v] = du + cost
                        prev_v[v] = u
                        prev_e[v] = idx
                        if not in_queue[v]:
                            queue.append(v)
                            in_queue[v] = True
            if dist[t] is None:
                break  # sem caminho aumentante: fluxo maximo atingido
            # gargalo
            push = target - total_flow
            v = t
            while v != s:
                e = self.graph[prev_v[v]][prev_e[v]]
                push = min(push, e[1])
                v = prev_v[v]
            # atualiza residual
            v = t
            while v != s:
                e = self.graph[prev_v[v]][prev_e[v]]
                e[1] -= push
                self.graph[v][e[3]][1] += push
                v = prev_v[v]
            total_flow += push
            total_cost += push * dist[t]
        return total_flow, total_cost


# ---------------------------------------------------------------------------
# Rede em camadas N_l(y,z) da Prop. A1.1
# ---------------------------------------------------------------------------

def build_network(inst, l, y, z):
    """Constroi N_l(y,z). Retorna (mcmf, S, T, D_l, BIG).

    Nos: S | F_i | Din_j | Dout_j | C_k | T.
    Arcos: S->F_i        cap b_il*y_i, custo 0
           F_i->Din_j    cap BIG,      custo c_ijl
           Din_j->Dout_j cap p_jl*z_j, custo 0
           Dout_j->C_k   cap BIG,      custo d_jkl
           C_k->T        cap q_kl,     custo 0
    """
    nI, nJ, nK = inst["nI"], inst["nJ"], inst["nK"]
    S = 0
    F = lambda i: 1 + i
    Din = lambda j: 1 + nI + j
    Dout = lambda j: 1 + nI + nJ + j
    C = lambda k: 1 + nI + 2 * nJ + k
    T = 1 + nI + 2 * nJ + nK
    n_nodes = T + 1

    BIG = 1 + sum(sum(row) for row in inst["b"]) \
            + sum(sum(row) for row in inst["p"]) \
            + sum(sum(row) for row in inst["q"])

    mc = MinCostFlow(n_nodes)
    for i in range(nI):
        mc.add_edge(S, F(i), inst["b"][i][l] * y[i], 0)
    for i in range(nI):
        for j in range(nJ):
            mc.add_edge(F(i), Din(j), BIG, inst["c"][i][j][l])
    for j in range(nJ):
        mc.add_edge(Din(j), Dout(j), inst["p"][j][l] * z[j], 0)
    for j in range(nJ):
        for k in range(nK):
            mc.add_edge(Dout(j), C(k), BIG, inst["d"][j][k][l])
    for k in range(nK):
        mc.add_edge(C(k), T, inst["q"][k][l], 0)

    return mc, S, T, demand_total(inst, l), BIG


def routing_value(inst, l, y, z):
    """Oraculo da Prop. A1.1 para o produto l e desenho (y,z).

    Retorna (viavel: bool, valor: int|None). Viavel sse o fluxo maximo
    S-T em N_l(y,z) atinge D_l; nesse caso, valor = custo minimo de um
    fluxo de valor D_l (inteiro).
    """
    mc, S, T, D, _ = build_network(inst, l, y, z)
    if D == 0:
        return True, 0
    sent, cost = mc.flow(S, T, D)
    if sent < D:
        return False, None
    return True, cost


def max_flow_value(inst, l, y, z):
    """Fluxo maximo S-T em N_l(y,z), truncado em D_l (basta para A1.2)."""
    mc, S, T, D, _ = build_network(inst, l, y, z)
    sent, _ = mc.flow(S, T, D)
    return sent


def aggregate_condition(inst, y, z):
    """Condicao da Prop. A1.2: para todo l,
    sum_i b_il y_i >= D_l  e  sum_j p_jl z_j >= D_l."""
    for l in range(inst["nL"]):
        D = demand_total(inst, l)
        cap_b = sum(inst["b"][i][l] * y[i] for i in range(inst["nI"]))
        cap_p = sum(inst["p"][j][l] * z[j] for j in range(inst["nJ"]))
        if cap_b < D or cap_p < D:
            return False
    return True


def flow_feasible(inst, y, z):
    """Viabilidade real do desenho (y,z): fluxo maximo >= D_l em todo l."""
    for l in range(inst["nL"]):
        D = demand_total(inst, l)
        if D == 0:
            continue
        if max_flow_value(inst, l, y, z) < D:
            return False
    return True


def all_designs(nI, nJ):
    """Enumera todos os (y,z) em {0,1}^nI x {0,1}^nJ."""
    for my in range(1 << nI):
        y = [(my >> i) & 1 for i in range(nI)]
        for mz in range(1 << nJ):
            z = [(mz >> j) & 1 for j in range(nJ)]
            yield y, z
