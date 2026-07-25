#!/usr/bin/env python3
"""
Ataques adversariais independentes complementares aos scripts verify_A6_*.

ATK1: cross-composition (clientes e produtos) fora das faixas do script do
      desenvolvedor: n_U=1, t_hat=m, t0=3 (padding impar), fontes
      adversariais (uma quase-SIM: cobertura minima = t_hat+1; outras NAO;
      uma trivialmente quase-YES com conjuntos grandes). Checagem OR dos
      dois lados por forca bruta + estrutural em TODO desenho de custo <=B.
ATK2: sanidade dos guardas - composicao D1 (sem guardas) deve QUEBRAR o OR
      (abrir os dois seletores de um par contorna tudo). Confirma que os
      guardas sao load-bearing e que o teste detectaria o furo.
ATK3: agregacao de clientes (Prop. A6.1) com demandas extremas/assimetricas
      e verificacao por LP independente (scipy.linprog) alem do MCMF.
ATK4: capping (Obs. A6.1.4) com b,p >> D_l e resposta (B,k) para todo k.
ATK5: probe n_U = 0 (fora da convencao n_U>=1 de A2 par.0): documenta que a
      composicao por PRODUTOS falharia sem a convencao (guardas sem fabrica).
"""
import itertools
import random
import copy
from common_mp_tscfl import routing_value, all_designs

try:
    import numpy as np
    from scipy.optimize import linprog
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

from verify_A6_crosscomp import (compose_clients, compose_products,
                                 sc_yes, sc_min_cover, pad_to_pow2)
from verify_A6_aggregation import (total_cost, cost_map, opt_and_argmin,
                                   merge_customers)


# ---------------------------------------------------------------------------
# LP independente: PL residual original em desigualdades (por produto)
# ---------------------------------------------------------------------------

def lp_block_value(inst, l, y, z):
    """Valor do bloco l em (y,z) via scipy.linprog (None se inviavel)."""
    nI, nJ, nK = inst["nI"], inst["nJ"], inst["nK"]
    nx, nw = nI * nJ, nJ * nK
    cvec = ([inst["c"][i][j][l] for i in range(nI) for j in range(nJ)]
            + [inst["d"][j][k][l] for j in range(nJ) for k in range(nK)])
    A_ub, b_ub = [], []
    # (C1) -sum_j w_jk <= -q_kl
    for k in range(nK):
        row = [0.0] * (nx + nw)
        for j in range(nJ):
            row[nx + j * nK + k] = -1.0
        A_ub.append(row); b_ub.append(-inst["q"][k][l])
    # (C2) sum_k w_jk - sum_i x_ij <= 0
    for j in range(nJ):
        row = [0.0] * (nx + nw)
        for i in range(nI):
            row[i * nJ + j] = -1.0
        for k in range(nK):
            row[nx + j * nK + k] = 1.0
        A_ub.append(row); b_ub.append(0.0)
    # (C3) sum_j x_ij <= b_il y_i
    for i in range(nI):
        row = [0.0] * (nx + nw)
        for j in range(nJ):
            row[i * nJ + j] = 1.0
        A_ub.append(row); b_ub.append(inst["b"][i][l] * y[i])
    # (C4) sum_k w_jk <= p_jl z_j
    for j in range(nJ):
        row = [0.0] * (nx + nw)
        for k in range(nK):
            row[nx + j * nK + k] = 1.0
        A_ub.append(row); b_ub.append(inst["p"][j][l] * z[j])
    res = linprog(cvec, A_ub=A_ub, b_ub=b_ub, bounds=(0, None),
                  method="highs")
    if not res.success:
        return None
    return res.fun


def lp_total_cost(inst, y, z):
    cost = sum(inst["f"][i] * y[i] for i in range(inst["nI"]))
    cost += sum(inst["g"][j] * z[j] for j in range(inst["nJ"]))
    for l in range(inst["nL"]):
        v = lp_block_value(inst, l, y, z)
        if v is None:
            return None
        cost += v
    return cost


# ---------------------------------------------------------------------------
# ATK1: cross-composition fora das faixas
# ---------------------------------------------------------------------------

def full_check(compose, sets_list, nU, t_hat, label, expect_yes=None):
    inst, B, meta = compose(sets_list, nU, t_hat)
    tau, m, insts = meta["tau"], meta["m"], meta["insts"]
    or_src = any(sc_yes(nU, s, t_hat) for s in sets_list)
    if expect_yes is not None:
        assert or_src == expect_yes, f"[{label}] fonte: esperado {expect_yes}"
    yes = False
    n_low = 0
    for y, z in all_designs(inst["nI"], inst["nJ"]):
        c = total_cost(inst, y, z)
        if c is None or c > B:
            continue
        yes = True
        n_low += 1
        # estrutural em TODO desenho barato
        if compose is compose_products:
            assert all(y), f"[{label}] fabrica fechada em desenho barato"
        for beta in range(tau):
            opens = [z[m + 2 * beta + v] for v in (0, 1)]
            assert sum(opens) == 1, \
                f"[{label}] CHEAT: par {beta} com {sum(opens)} seletores"
        vpat = [0 if z[m + 2 * beta] else 1 for beta in range(tau)]
        istar = sum(v << beta for beta, v in enumerate(vpat))
        zsets = [j for j in range(m) if z[j]]
        assert len(zsets) <= t_hat, f"[{label}] CHEAT: |Z| > t_hat"
        for e in range(nU):
            assert any(e in insts[istar][j] for j in zsets), \
                f"[{label}] CHEAT: elemento {e} de i*={istar} descoberto"
    assert yes == or_src, f"[{label}] OR FALHOU: composto={yes}, " \
        f"fontes={or_src}"
    return yes, n_low


def atk1():
    total = 0
    # (a) n_U = 1 (fora da faixa nU>=2 dos scripts), t_hat = m, t0 = 3
    subsets1 = [frozenset(), frozenset([0])]
    for combo in itertools.product(subsets1, repeat=2):
        for combo2 in itertools.product(subsets1, repeat=2):
            for combo3 in itertools.product(subsets1, repeat=2):
                srcs = [list(combo), list(combo2), list(combo3)]
                for t_hat in (1, 2):
                    full_check(compose_clients, srcs, 1, t_hat,
                               f"ATK1a-cli nU=1 t^={t_hat}")
                    full_check(compose_products, srcs, 1, t_hat,
                               f"ATK1a-prod nU=1 t^={t_hat}")
                    total += 2
    print(f"[ATK1a] n_U=1, t0=3, t^ em {{1,2}}: {total} composicoes: PASS")

    # (b) fontes adversariais: quase-SIM (cobertura minima = t_hat+1)
    #     nU=3, m=3, t_hat=1; cada fonte cobre U so com 2 conjuntos.
    near = [frozenset([0, 1]), frozenset([1, 2]), frozenset([0, 2])]
    no1 = [frozenset([0]), frozenset([1]), frozenset()]      # nao cobre 2
    no2 = [frozenset([2]), frozenset([2]), frozenset([2])]   # nao cobre 0,1
    n = 0
    for srcs in ([near, near], [near, no1, no2], [no1, no2, near, near],
                 [no1, no1], [no2, no1, no1]):
        for s in srcs:
            assert not sc_yes(3, s, 1)
        y1, _ = full_check(compose_clients, srcs, 3, 1,
                           "ATK1b-cli quase-SIM", expect_yes=False)
        y2, _ = full_check(compose_products, srcs, 3, 1,
                           "ATK1b-prod quase-SIM", expect_yes=False)
        n += 2
    # e a versao SIM de controle: uma fonte ganha o conjunto cheio
    yes_src = [frozenset([0, 1, 2]), frozenset(), frozenset()]
    for srcs in ([near, yes_src], [no1, no2, yes_src, near]):
        full_check(compose_clients, srcs, 3, 1, "ATK1b-cli controle-SIM",
                   expect_yes=True)
        full_check(compose_products, srcs, 3, 1, "ATK1b-prod controle-SIM",
                   expect_yes=True)
        n += 2
    print(f"[ATK1b] fontes quase-SIM (min cover = t^+1) e controles: "
          f"{n} composicoes: PASS")

    # (c) t_hat = m (borda da convencao WLOG), nU=2, m=2, t0=2
    subsets = [frozenset(), frozenset([0]), frozenset([1]), frozenset([0, 1])]
    n = 0
    for s1 in itertools.product(subsets, repeat=2):
        for s2 in itertools.product(subsets, repeat=2):
            full_check(compose_clients, [list(s1), list(s2)], 2, 2,
                       "ATK1c-cli t^=m")
            n += 1
    for k in range(0, 256, 5):
        a, b = divmod(k, 16)
        s1 = list(itertools.product(subsets, repeat=2))[a]
        s2 = list(itertools.product(subsets, repeat=2))[b]
        full_check(compose_products, [list(s1), list(s2)], 2, 2,
                   "ATK1c-prod t^=m")
        n += 1
    print(f"[ATK1c] t^ = m = 2 (borda WLOG): {n} composicoes: PASS")


# ---------------------------------------------------------------------------
# ATK2: D1 sem guardas deve quebrar (sanidade dos guardas)
# ---------------------------------------------------------------------------

def compose_clients_noguards(sets_list, nU, t_hat):
    """D1: mesma composicao por clientes, SEM guardas."""
    m = len(sets_list[0])
    insts, tau, tp = pad_to_pow2(sets_list)
    B = tau * (t_hat + 1) + t_hat
    W = B + 1
    nJ = m + 2 * tau
    D = tp * nU
    nK = tp * nU
    g = [1] * m + [t_hat + 1] * (2 * tau)
    d = [[[0] for _ in range(nK)] for _ in range(nJ)]
    for j in range(nJ):
        for k in range(nK):
            i, e = divmod(k, nU)
            if j < m:
                d[j][k][0] = 0 if e in insts[i][j] else W
            else:
                beta, v = divmod(j - m, 2)
                d[j][k][0] = 0 if ((i >> beta) & 1) != v else W
    inst = {"nI": 1, "nJ": nJ, "nK": nK, "nL": 1,
            "f": [0], "g": g,
            "c": [[[0] for _ in range(nJ)]],
            "d": d, "b": [[D]],
            "p": [[D] for _ in range(nJ)],
            "q": [[1] for _ in range(nK)]}
    return inst, B


def atk2():
    # fontes todas NAO, t0 = 4 => tau = 2: dois seletores de UM par devem
    # contornar todas as instancias em D1 (custo 2(t^+1) <= B) -> falso SIM.
    no1 = [frozenset([0]), frozenset([1])]
    srcs = [no1, no1, no1, no1]
    assert not any(sc_yes(2, s, 1) for s in srcs)
    inst, B = compose_clients_noguards(srcs, 2, 1)
    yes = any(c is not None and c <= B
              for c in (total_cost(inst, y, z)
                        for y, z in all_designs(inst["nI"], inst["nJ"])))
    assert yes, "ATK2: D1 sem guardas NAO quebrou (inesperado!)"
    # e a versao final com guardas, nas mesmas fontes, responde NAO:
    y2, _ = full_check(compose_clients, srcs, 2, 1, "ATK2-final",
                       expect_yes=False)
    print("[ATK2] D1 (sem guardas) da falso-SIM com fontes todas-NAO; "
          "D2 (com guardas) responde NAO: PASS (furo confirmado e fechado)")


# ---------------------------------------------------------------------------
# ATK3: agregacao com demandas extremas + LP independente
# ---------------------------------------------------------------------------

def atk3():
    assert HAVE_SCIPY, "scipy indisponivel"
    rng = random.Random(20260710)
    n_inst = n_cmp = n_lp = 0
    for trial in range(25):
        nI = rng.randint(1, 3); nJ = rng.randint(1, 3)
        nK = rng.randint(2, 4); nL = rng.randint(1, 2)
        inst = {
            "nI": nI, "nJ": nJ, "nK": nK, "nL": nL,
            "f": [rng.choice([0, 1, 50]) for _ in range(nI)],
            "g": [rng.choice([0, 2, 40]) for _ in range(nJ)],
            "c": [[[rng.choice([0, 1, 7, 30]) for _ in range(nL)]
                   for _ in range(nJ)] for _ in range(nI)],
            "d": [[[rng.choice([0, 3, 25]) for _ in range(nL)]
                   for _ in range(nK)] for _ in range(nJ)],
            "b": [[rng.choice([0, 5, 60, 200]) for _ in range(nL)]
                  for _ in range(nI)],
            "p": [[rng.choice([0, 7, 55, 200]) for _ in range(nL)]
                  for _ in range(nJ)],
            "q": [[rng.choice([0, 1, 50]) for _ in range(nL)]
                  for _ in range(nK)],
        }
        # clientes 0 e 1: colunas identicas, demandas EXTREMAS e em produtos
        # diferentes (k0 pesado no produto 0, k1 pesado no ultimo produto)
        for j in range(nJ):
            for l in range(nL):
                inst["d"][j][1][l] = inst["d"][j][0][l]
        inst["q"][0] = [60] + [0] * (nL - 1)
        inst["q"][1] = [0] * (nL - 1) + [47]
        # capacidade apertada: garante um cenario com folga zero em metade
        if trial % 2 == 0:
            for l in range(nL):
                D = sum(inst["q"][k][l] for k in range(nK))
                for i in range(nI):
                    inst["b"][i][l] = max(inst["b"][i][l], 1)
                # ajusta para sum b == D exatamente (capacidade justa)
                tot = sum(inst["b"][i][l] for i in range(nI))
                if tot > D:
                    exc = tot - D
                    for i in range(nI):
                        red = min(exc, inst["b"][i][l])
                        inst["b"][i][l] -= red; exc -= red
                elif tot < D:
                    inst["b"][0][l] += D - tot
        merged = merge_customers(inst, 0, 1)
        cmo, cmm = cost_map(inst), cost_map(merged)
        assert set(cmo) == set(cmm)
        for key in cmo:
            assert cmo[key] == cmm[key], \
                f"ATK3 trial={trial} desenho {key}: {cmo[key]} != {cmm[key]}"
            n_cmp += 1
        # contraste LP independente em ate 6 desenhos por instancia
        keys = sorted(cmo)[:: max(1, len(cmo) // 6)]
        for (yy, zz) in keys:
            v_o = lp_total_cost(inst, list(yy), list(zz))
            v_m = lp_total_cost(merged, list(yy), list(zz))
            for v_lp, v_bf in ((v_o, cmo[(yy, zz)]), (v_m, cmm[(yy, zz)])):
                if v_bf is None:
                    assert v_lp is None
                else:
                    assert v_lp is not None and abs(v_lp - v_bf) < 1e-6
                n_lp += 1
        n_inst += 1
    print(f"[ATK3] agregacao extrema (demandas 60/47 em produtos distintos, "
          f"capacidades justas): {n_inst} instancias, {n_cmp} comparacoes, "
          f"{n_lp} contrastes LP: PASS")


# ---------------------------------------------------------------------------
# ATK4: capping com b,p >> D e a versao (B,k)
# ---------------------------------------------------------------------------

def atk4():
    rng = random.Random(777)
    n_inst = n_k = 0
    for trial in range(20):
        nI = rng.randint(1, 3); nJ = rng.randint(1, 3)
        nK = rng.randint(1, 3); nL = rng.randint(1, 2)
        inst = {
            "nI": nI, "nJ": nJ, "nK": nK, "nL": nL,
            "f": [rng.randint(0, 9) for _ in range(nI)],
            "g": [rng.randint(0, 9) for _ in range(nJ)],
            "c": [[[rng.randint(0, 5) for _ in range(nL)]
                   for _ in range(nJ)] for _ in range(nI)],
            "d": [[[rng.randint(0, 5) for _ in range(nL)]
                   for _ in range(nK)] for _ in range(nJ)],
            "b": [[rng.choice([0, 3, 10 ** 6]) for _ in range(nL)]
                  for _ in range(nI)],
            "p": [[rng.choice([0, 4, 10 ** 6]) for _ in range(nL)]
                  for _ in range(nJ)],
            "q": [[rng.randint(0, 8) for _ in range(nL)]
                  for _ in range(nK)],
        }
        capped = copy.deepcopy(inst)
        for l in range(nL):
            D = sum(inst["q"][k][l] for k in range(nK))
            for i in range(nI):
                capped["b"][i][l] = min(capped["b"][i][l], D)
            for j in range(nJ):
                capped["p"][j][l] = min(capped["p"][j][l], D)
        cmo, cmc = cost_map(inst), cost_map(capped)
        assert cmo == cmc, f"ATK4 trial={trial}: capping alterou algum desenho"
        # resposta (B,k) para todo k e B em torno do otimo
        for k in range(nI + nJ + 1):
            def best_k(cm):
                vals = [v for (yy, zz), v in cm.items()
                        if v is not None and sum(yy) + sum(zz) <= k]
                return min(vals) if vals else None
            assert best_k(cmo) == best_k(cmc)
            n_k += 1
        n_inst += 1
    print(f"[ATK4] capping com b,p ate 10^6: {n_inst} instancias, respostas "
          f"(B,k) identicas para todo k ({n_k} checagens): PASS")


# ---------------------------------------------------------------------------
# ATK5: probe n_U = 0 (fora da convencao) - documenta a dependencia
# ---------------------------------------------------------------------------

def atk5():
    srcs = [[frozenset(), frozenset()], [frozenset(), frozenset()]]
    or_src = any(sc_yes(0, s, 1) for s in srcs)  # cobertura vazia: SIM
    assert or_src
    # clientes: ainda funciona (so guardas)
    inst, B, meta = compose_clients(srcs, 0, 1)
    yes_cli = any(c is not None and c <= B
                  for c in (total_cost(inst, y, z)
                            for y, z in all_designs(inst["nI"], inst["nJ"])))
    # produtos: o guarda exige uma fabrica portadora (b[0][tp+beta] = 1);
    # com n_U = 0 a CONSTRUCAO ja falha -- nao existe instancia composta.
    try:
        inst2, B2, _ = compose_products(srcs, 0, 1)
        yes_prod = any(c is not None and c <= B2
                       for c in (total_cost(inst2, y, z)
                                 for y, z in all_designs(inst2["nI"],
                                                         inst2["nJ"])))
        prod_msg = "SIM" if yes_prod else "NAO (OR falharia)"
    except IndexError:
        prod_msg = "CONSTRUCAO FALHA (guarda sem fabrica portadora)"
    print(f"[ATK5] probe n_U=0 (fora da convencao A2 par.0): fontes SIM; "
          f"clientes -> {'SIM' if yes_cli else 'NAO'}; "
          f"produtos -> {prod_msg} "
          f"(a composicao por produtos DEPENDE de n_U>=1; cf. O1(b))")


if __name__ == "__main__":
    atk1()
    atk2()
    atk3()
    atk4()
    atk5()
    print("review_attacks_A6.py: concluido")
