#!/usr/bin/env python3
"""
Verificacao computacional das cross-compositions OR do artigo.

Duas composicoes de t instancias de SET COVER (mesmos n_U, m, t_hat) em
UMA instancia do MP-TSCFLP com parametro |I|+|J| = O(n_U + m + log t):

  compose_clients : Teorema A6.3  - celula |I| = |L| = 1 (sabor A2.1);
                    clientes-elemento por instancia + seletores por bits
                    complementares + clientes-guarda (1 por par de bits).
  compose_products: Teorema A6.4  - celula |K| = 1 (sabor A5.2); produtos =
                    instancias, fabricas = elementos com capacidade justa,
                    produtos-guarda.

Checagens, para cada composicao:
  (i)  OR-correcao: forca bruta no MP-TSCFLP composto (todos os desenhos +
       oraculo MCMF da Prop. A1.1) da SIM  <=>  alguma instancia fonte e SIM
       (forca bruta no SET COVER);
  (ii) estrutural: TODO desenho de custo <= B abre exatamente um seletor por
       par de bits, define i*, cobre a instancia i* com <= t_hat conjuntos
       (e, nos produtos, abre todas as fabricas).

Baterias:
  [C1] clientes, exaustiva: n_U=2, m=2, t_hat=1, t=2 — todos os 256 pares
       ordenados de sistemas de conjuntos.
  [C2] clientes, aleatoria com sementes: n_U<=3, m<=3, t em {2,3,4}.
  [P1] produtos, subamostra determinista da grade exaustiva de [C1].
  [P2] produtos, aleatoria com sementes.
"""
import itertools
import random
from common_mp_tscfl import routing_value, all_designs, aggregate_condition


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def total_cost(inst, y, z):
    cost = sum(inst["f"][i] * y[i] for i in range(inst["nI"]))
    cost += sum(inst["g"][j] * z[j] for j in range(inst["nJ"]))
    for l in range(inst["nL"]):
        feas, val = routing_value(inst, l, y, z)
        if not feas:
            return None
        cost += val
    return cost


def sc_min_cover(nU, sets):
    """Tamanho minimo de cobertura (None se nao cobre); forca bruta."""
    m = len(sets)
    full = (1 << nU) - 1
    masks = [sum(1 << e for e in s) for s in sets]
    best = None
    for sel in range(1 << m):
        u = 0
        for j in range(m):
            if (sel >> j) & 1:
                u |= masks[j]
        if u == full:
            pc = bin(sel).count("1")
            if best is None or pc < best:
                best = pc
    return best


def sc_yes(nU, sets, t_hat):
    mc = sc_min_cover(nU, sets)
    return mc is not None and mc <= t_hat


def pad_to_pow2(sets_list):
    t0 = len(sets_list)
    tau = 1
    while (1 << tau) < t0:
        tau += 1
    tp = 1 << tau
    return sets_list + [sets_list[0]] * (tp - t0), tau, tp


# ---------------------------------------------------------------------------
# Composicao 1: clientes (Teorema A6.3; celula |I| = |L| = 1)
# ---------------------------------------------------------------------------

def compose_clients(sets_list, nU, t_hat):
    m = len(sets_list[0])
    insts, tau, tp = pad_to_pow2(sets_list)
    B = tau * (t_hat + 1) + t_hat
    W = B + 1
    nJ = m + 2 * tau                    # conjuntos + seletores S_{beta,v}
    D = tp * nU + tau                   # demanda total (elementos + guardas)
    # clientes: (i,e) para i<tp, e<nU; depois guardas g_beta
    nK = tp * nU + tau
    g = [1] * m + [t_hat + 1] * (2 * tau)
    d = [[[0] for _ in range(nK)] for _ in range(nJ)]
    for j in range(nJ):
        for k in range(nK):
            if k < tp * nU:
                i, e = divmod(k, nU)
                if j < m:
                    d[j][k][0] = 0 if e in insts[i][j] else W
                else:
                    beta, v = divmod(j - m, 2)
                    d[j][k][0] = 0 if ((i >> beta) & 1) != v else W
            else:
                beta_g = k - tp * nU
                if j >= m and (j - m) // 2 == beta_g:
                    d[j][k][0] = 0
                else:
                    d[j][k][0] = W
    inst = {"nI": 1, "nJ": nJ, "nK": nK, "nL": 1,
            "f": [0], "g": g,
            "c": [[[0] for _ in range(nJ)]],
            "d": d,
            "b": [[D]],
            "p": [[D] for _ in range(nJ)],
            "q": [[1] for _ in range(nK)]}
    return inst, B, {"tau": tau, "tp": tp, "m": m, "insts": insts}


def check_clients(sets_list, nU, t_hat):
    inst, B, meta = compose_clients(sets_list, nU, t_hat)
    tau, m, insts = meta["tau"], meta["m"], meta["insts"]
    or_src = any(sc_yes(nU, s, t_hat) for s in sets_list)
    yes = False
    n_struct = 0
    for y, z in all_designs(inst["nI"], inst["nJ"]):
        if not aggregate_condition(inst, y, z):
            continue
        c = total_cost(inst, y, z)
        if c is None or c > B:
            continue
        yes = True
        # estrutural: exatamente um seletor por par
        vpat = []
        for beta in range(tau):
            opens = [z[m + 2 * beta + v] for v in (0, 1)]
            assert sum(opens) == 1, "par de seletores nao-singleton"
            vpat.append(0 if opens[0] else 1)
        istar = sum(v << beta for beta, v in enumerate(vpat))
        zsets = [j for j in range(m) if z[j]]
        assert len(zsets) <= t_hat, "mais conjuntos que t_hat"
        for e in range(nU):
            assert any(e in insts[istar][j] for j in zsets), \
                "i* nao coberto por Z"
        n_struct += 1
    assert yes == or_src, \
        f"OR falhou (clientes): composto={yes}, fontes={or_src}"
    return yes, n_struct


# ---------------------------------------------------------------------------
# Composicao 2: produtos (Teorema A6.4; celula |K| = 1, sabor A5.2)
# ---------------------------------------------------------------------------

def compose_products(sets_list, nU, t_hat):
    m = len(sets_list[0])
    insts, tau, tp = pad_to_pow2(sets_list)
    B = tau * (t_hat + 1) + t_hat
    W = B + 1
    Q = B + 1
    nI = nU
    nJ = m + 2 * tau
    nL = tp + tau                       # produtos principais + guardas
    c = [[[0] * nL for _ in range(nJ)] for _ in range(nI)]
    for i in range(nI):
        for j in range(nJ):
            for l in range(nL):
                if l < tp:              # produto principal l = instancia l
                    if j < m:
                        c[i][j][l] = 0 if i in insts[l][j] else 1
                    else:
                        beta, v = divmod(j - m, 2)
                        c[i][j][l] = 0 if ((l >> beta) & 1) != v else W
                else:                   # produto-guarda do par beta
                    beta_g = l - tp
                    if j >= m and (j - m) // 2 == beta_g:
                        c[i][j][l] = 0
                    else:
                        c[i][j][l] = W
    b = [[Q] * tp + [0] * tau for _ in range(nI)]
    for beta in range(tau):
        b[0][tp + beta] = 1             # guarda: so a fabrica 0 a carrega
    q = [[nU * Q] * tp + [1] * tau]
    p = [[nU * Q] * tp + [1] * tau for _ in range(nJ)]
    inst = {"nI": nI, "nJ": nJ, "nK": 1, "nL": nL,
            "f": [0] * nI,
            "g": [1] * m + [t_hat + 1] * (2 * tau),
            "c": c,
            "d": [[[0] * nL] for _ in range(nJ)],
            "b": b, "p": p, "q": q}
    return inst, B, {"tau": tau, "tp": tp, "m": m, "insts": insts}


def check_products(sets_list, nU, t_hat):
    inst, B, meta = compose_products(sets_list, nU, t_hat)
    tau, m, insts = meta["tau"], meta["m"], meta["insts"]
    or_src = any(sc_yes(nU, s, t_hat) for s in sets_list)
    yes = False
    n_struct = 0
    for y, z in all_designs(inst["nI"], inst["nJ"]):
        if not aggregate_condition(inst, y, z):
            continue
        c = total_cost(inst, y, z)
        if c is None or c > B:
            continue
        yes = True
        assert all(y), "fabrica fechada em solucao de custo <= B"
        vpat = []
        for beta in range(tau):
            opens = [z[m + 2 * beta + v] for v in (0, 1)]
            assert sum(opens) == 1, "par de seletores nao-singleton"
            vpat.append(0 if opens[0] else 1)
        lstar = sum(v << beta for beta, v in enumerate(vpat))
        zsets = [j for j in range(m) if z[j]]
        assert len(zsets) <= t_hat, "mais conjuntos que t_hat"
        for i in range(nU):
            assert any(i in insts[lstar][j] for j in zsets), \
                "fabrica descoberta no produto l*"
        n_struct += 1
    assert yes == or_src, \
        f"OR falhou (produtos): composto={yes}, fontes={or_src}"
    return yes, n_struct


# ---------------------------------------------------------------------------
# geracao de fontes
# ---------------------------------------------------------------------------

def all_systems_nU2_m2():
    """Todos os sistemas de 2 subconjuntos de {0,1} (16 sistemas)."""
    subsets = [frozenset(), frozenset([0]), frozenset([1]), frozenset([0, 1])]
    return [list(pair) for pair in itertools.product(subsets, repeat=2)]


def random_system(rng, nU, m):
    return [frozenset(e for e in range(nU) if rng.random() < 0.5)
            for _ in range(m)]


# ---------------------------------------------------------------------------
# baterias
# ---------------------------------------------------------------------------

def battery_C1():
    systems = all_systems_nU2_m2()
    n = n_yes = 0
    for s1 in systems:
        for s2 in systems:
            yes, _ = check_clients([s1, s2], nU=2, t_hat=1)
            n += 1
            n_yes += yes
    print(f"[C1] clientes, exaustiva n_U=2 m=2 t=2 t^=1: {n} composicoes "
          f"({n_yes} SIM, {n - n_yes} NAO): PASS")


def battery_C2():
    n = n_yes = 0
    for seed in range(7000, 7040):
        rng = random.Random(seed)
        nU = rng.randint(2, 3)
        m = rng.randint(2, 3)
        t_hat = rng.randint(1, m)
        t0 = rng.randint(2, 4)
        srcs = [random_system(rng, nU, m) for _ in range(t0)]
        yes, _ = check_clients(srcs, nU, t_hat)
        n += 1
        n_yes += yes
    print(f"[C2] clientes, aleatoria (sementes 7000-7039): {n} composicoes "
          f"({n_yes} SIM, {n - n_yes} NAO): PASS")


def battery_P1():
    systems = all_systems_nU2_m2()
    pairs = list(itertools.product(range(16), repeat=2))
    sel = [pairs[k] for k in range(0, 256, 4)]      # 64 pares deterministas
    n = n_yes = 0
    for a, bb in sel:
        yes, _ = check_products([systems[a], systems[bb]], nU=2, t_hat=1)
        n += 1
        n_yes += yes
    print(f"[P1] produtos, grade n_U=2 m=2 t=2 t^=1 (64 pares): {n} "
          f"composicoes ({n_yes} SIM, {n - n_yes} NAO): PASS")


def battery_P2():
    n = n_yes = 0
    for seed in range(7100, 7124):
        rng = random.Random(seed)
        nU = rng.randint(2, 3)
        m = rng.randint(2, 3)
        t_hat = rng.randint(1, m)
        t0 = rng.randint(2, 4)
        srcs = [random_system(rng, nU, m) for _ in range(t0)]
        yes, _ = check_products(srcs, nU, t_hat)
        n += 1
        n_yes += yes
    print(f"[P2] produtos, aleatoria (sementes 7100-7123): {n} composicoes "
          f"({n_yes} SIM, {n - n_yes} NAO): PASS")


if __name__ == "__main__":
    battery_C1()
    battery_C2()
    battery_P1()
    battery_P2()
    print("verify_A6_crosscomp.py: TODAS AS BATERIAS PASSARAM")
