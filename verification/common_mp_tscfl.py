"""
Common verification module for the structural results of the paper (MP-TSCFLP).

Contents:
  - seeded random instance generator (small integers);
  - exact min-cost flow via successive shortest paths (SSP),
    100% integer arithmetic (Bellman-Ford/SPFA on the residual graph);
  - builder of the layered network N_l(y,z) of the paper's routing oracle;
  - per-product routing oracle (feasibility + optimal value).

Stdlib only. networkx / scipy are used only in the verification
scripts, as independent contrast implementations.
"""

import random

INF = float("inf")


# ---------------------------------------------------------------------------
# Instance generation
# ---------------------------------------------------------------------------

def gen_instance(seed, max_i=4, max_j=4, max_k=4, max_l=3, vmax=9):
    """Random MP-TSCFLP instance with integer data in [0, vmax].

    With probability 0.2 a product has identically zero demand
    (to exercise the edge case D_l = 0). Zero capacities and demands
    arise naturally from the range [0, vmax].
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
        if rng.random() < 0.2:  # product with no demand (edge case)
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
# Exact min-cost flow (successive shortest paths, integer arithmetic)
# ---------------------------------------------------------------------------

class MinCostFlow:
    """MCMF via minimum-cost augmenting paths (SPFA on the residual).

    All data are integers; all computations stay in Z.
    """

    def __init__(self, n):
        self.n = n
        self.graph = [[] for _ in range(n)]

    def add_edge(self, u, v, cap, cost):
        assert isinstance(cap, int) and isinstance(cost, int) and cap >= 0
        self.graph[u].append([v, cap, cost, len(self.graph[v])])
        self.graph[v].append([u, 0, -cost, len(self.graph[u]) - 1])

    def flow(self, s, t, target):
        """Sends up to `target` units from s to t at minimum cost.

        Returns (sent_flow, total_cost). sent_flow < target
        means the maximum s-t flow is below target
        (SSP only stops when there is no augmenting path).
        """
        total_flow = 0
        total_cost = 0
        n = self.n
        while total_flow < target:
            dist = [None] * n           # None = +infinity (all integer)
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
                break  # no augmenting path: maximum flow reached
            # bottleneck
            push = target - total_flow
            v = t
            while v != s:
                e = self.graph[prev_v[v]][prev_e[v]]
                push = min(push, e[1])
                v = prev_v[v]
            # update residual
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
# Layered network N_l(y,z) of the routing oracle
# ---------------------------------------------------------------------------

def build_network(inst, l, y, z):
    """Builds N_l(y,z). Returns (mcmf, S, T, D_l, BIG).

    Nodes: S | F_i | Din_j | Dout_j | C_k | T.
    Arcs:  S->F_i        cap b_il*y_i, cost 0
           F_i->Din_j    cap BIG,      cost c_ijl
           Din_j->Dout_j cap p_jl*z_j, cost 0
           Dout_j->C_k   cap BIG,      cost d_jkl
           C_k->T        cap q_kl,     cost 0
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
    """The paper's routing oracle for product l and design (y,z).

    Returns (feasible: bool, value: int|None). Feasible iff the maximum
    S-T flow in N_l(y,z) reaches D_l; in that case, value = minimum cost
    of a flow of value D_l (integer).
    """
    mc, S, T, D, _ = build_network(inst, l, y, z)
    if D == 0:
        return True, 0
    sent, cost = mc.flow(S, T, D)
    if sent < D:
        return False, None
    return True, cost


def max_flow_value(inst, l, y, z):
    """Maximum S-T flow in N_l(y,z), truncated at D_l (sufficient for the
    feasibility characterization)."""
    mc, S, T, D, _ = build_network(inst, l, y, z)
    sent, _ = mc.flow(S, T, D)
    return sent


def aggregate_condition(inst, y, z):
    """Condition of the paper's feasibility proposition: for all l,
    sum_i b_il y_i >= D_l  and  sum_j p_jl z_j >= D_l."""
    for l in range(inst["nL"]):
        D = demand_total(inst, l)
        cap_b = sum(inst["b"][i][l] * y[i] for i in range(inst["nI"]))
        cap_p = sum(inst["p"][j][l] * z[j] for j in range(inst["nJ"]))
        if cap_b < D or cap_p < D:
            return False
    return True


def flow_feasible(inst, y, z):
    """Actual feasibility of design (y,z): maximum flow >= D_l for all l."""
    for l in range(inst["nL"]):
        D = demand_total(inst, l)
        if D == 0:
            continue
        if max_flow_value(inst, l, y, z) < D:
            return False
    return True


def all_designs(nI, nJ):
    """Enumerates all (y,z) in {0,1}^nI x {0,1}^nJ."""
    for my in range(1 << nI):
        y = [(my >> i) & 1 for i in range(nI)]
        for mz in range(1 << nJ):
            z = [(mz >> j) & 1 for j in range(nJ)]
            yield y, z
