"""
Verificacao exaustiva de dois resultados da camada numerica do artigo:

  Reducao de cobertura da celula — SET COVER -> MP-TSCFLP restrito a
  |K| = |L| = 1
    (elementos = fabricas, conjuntos = depositos, acoplamento
    c_ij in {0,1}; f = 0, g = 1, d = 0, b_i = Q := m+1, p_j = n_U*Q,
    D = n_U*Q, orcamento B = t). Estabelece a dureza FORTE da celula.

  Proposicao dos custos separaveis — custos de estagio 1 c_ij = gamma_i +
    delta_j  =>  a celula |K| = |L| = 1 e' pseudo-polinomial (dois
    min-knapsack-covers desacoplados).

Baterias:
  [A] EXAUSTIVA: todas as familias de subconjuntos distintos de U com
      |U| <= 4 e 1 <= m <= 4 (mesma enumeracao das 2.696 familias de
      A2/A3a). Para cada familia: forca bruta sobre TODOS os desenhos
      (y,z) com o oraculo MCMF inteiro (routing_value) — nenhuma forma
      fechada como fonte —, e checagens:
        (a) por desenho: viabilidade == forma fechada prevista
            (Y = I e Z != vazio) e, se viavel, v == Q * #descobertos(Z);
        (b) OPT == t* (tamanho minimo de cobertura, forca bruta) quando
            cobrivel; OPT > m quando nao cobrivel;
        (c) "<=>" da reducao para TODO orcamento t in {1..m}.
  [B] aleatoria: familias com |U|, |S| <= 5 (sementes fixas), mesmas
      checagens (a)-(c).
  [C] custos separaveis: instancias aleatorias (incluindo D = 0,
      inviaveis e decomposicoes com gamma_i NEGATIVO — so os arcos
      c_ij = gamma_i + delta_j precisam ser >= 0): DP desacoplado ==
      forca bruta, com viabilidade implicita via OPT (INF == INF).
  [D] contraste por PL independente (scipy/HiGHS): em instancias GERAIS
      da celula, o PL residual em desigualdades coincide com o MCMF
      (viabilidade + valor, tolerancia 1e-6) em desenhos amostrados —
      valida o uso do oraculo nesta celula.

Saida: contagens por bateria; exit code != 0 em qualquer falha.
"""

import itertools
import random
import sys

from common_mp_tscfl import routing_value

INF = float("inf")

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("FALHA:", msg)


# ---------------------------------------------------------------------------
# construcao da reducao de cobertura da celula
# ---------------------------------------------------------------------------

def build_reduction(nU, fam):
    """fam = tupla de mascaras (bitmasks sobre U). Retorna (inst, Q)."""
    m = len(fam)
    Q = m + 1
    nI, nJ = nU, m
    c = [[[0 if (fam[j] >> i) & 1 else 1] for j in range(nJ)]
         for i in range(nI)]
    return {
        "nI": nI, "nJ": nJ, "nK": 1, "nL": 1,
        "f": [0] * nI, "g": [1] * nJ,
        "c": c,
        "d": [[[0]] for _ in range(nJ)],
        "b": [[Q] for _ in range(nI)],
        "p": [[nU * Q] for _ in range(nJ)],
        "q": [[nU * Q]],
    }, Q


def min_cover(nU, fam):
    """Tamanho minimo de cobertura (forca bruta); None se nao cobrivel."""
    full = (1 << nU) - 1
    best = None
    for msk in range(1, 1 << len(fam)):
        u = 0
        for j in range(len(fam)):
            if (msk >> j) & 1:
                u |= fam[j]
        if u == full:
            sz = bin(msk).count("1")
            if best is None or sz < best:
                best = sz
    return best


def uncovered(nU, fam, mz):
    """# elementos sem deposito aberto que os contenha."""
    u = 0
    for j in range(len(fam)):
        if (mz >> j) & 1:
            u |= fam[j]
    return nU - bin(u).count("1")


def check_family(nU, fam, counters):
    inst, Q = build_reduction(nU, fam)
    m = len(fam)
    nI, nJ = inst["nI"], inst["nJ"]
    full_I = (1 << nI) - 1
    opt = INF
    for my in range(1 << nI):
        y = [(my >> i) & 1 for i in range(nI)]
        for mz in range(1 << nJ):
            z = [(mz >> j) & 1 for j in range(nJ)]
            feas, v = routing_value(inst, 0, y, z)
            pred_feas = (my == full_I) and (mz != 0)
            check(feas == pred_feas,
                  f"viabilidade fam={fam} nU={nU} my={my} mz={mz}: "
                  f"mcmf={feas} prevista={pred_feas}")
            counters["desenhos"] += 1
            if feas:
                pred_v = Q * uncovered(nU, fam, mz)
                check(v == pred_v,
                      f"lema estrutural fam={fam} nU={nU} mz={mz}: "
                      f"v={v} previsto={pred_v}")
                total = bin(mz).count("1") + v   # f = 0, g = 1
                if total < opt:
                    opt = total
    tstar = min_cover(nU, fam)
    if tstar is not None:
        check(opt == tstar,
              f"OPT fam={fam} nU={nU}: opt={opt} t*={tstar}")
    else:
        check(opt > m, f"OPT fam={fam} nU={nU}: opt={opt} <= m={m} "
                       f"mas nao cobrivel")
        check(opt >= Q + 1, f"OPT fam={fam} nU={nU}: opt={opt} < Q+1")
    for t in range(1, m + 1):
        counters["iff"] += 1
        cover_yes = (tstar is not None and tstar <= t)
        mp_yes = (opt <= t)
        check(cover_yes == mp_yes,
              f"<=> fam={fam} nU={nU} t={t}: cover={cover_yes} mp={mp_yes}")


def battery_A():
    counters = {"desenhos": 0, "iff": 0, "familias": 0}
    for nU in range(1, 5):
        subsets = list(range(1 << nU))   # inclui o conjunto vazio
        for m in range(1, 5):
            for fam in itertools.combinations(subsets, m):
                counters["familias"] += 1
                check_family(nU, fam, counters)
    return counters


def battery_B(n=40, seed0=20260710):
    counters = {"desenhos": 0, "iff": 0, "familias": 0}
    for s in range(seed0, seed0 + n):
        rng = random.Random(s)
        nU = rng.randint(2, 5)
        m = rng.randint(2, 5)
        fam = tuple(rng.randint(0, (1 << nU) - 1) for _ in range(m))
        counters["familias"] += 1
        check_family(nU, fam, counters)
    return counters


# ---------------------------------------------------------------------------
# Custos separaveis — DP desacoplado
# ---------------------------------------------------------------------------

def side_dp(items, D):
    T = [0] + [INF] * D
    for (F, gam, cap) in items:
        newT = list(T)
        for t in range(1, D + 1):
            for u in range(1, min(cap, t) + 1):
                if T[t - u] < INF:
                    cand = T[t - u] + F + gam * u
                    if cand < newT[t]:
                        newT[t] = cand
        T = newT
    return T[D]


def battery_C(n=120, seed0=8000):
    counters = {"instancias": 0, "viaveis": 0, "inviaveis": 0, "D0": 0}
    for s in range(seed0, seed0 + n):
        rng = random.Random(s)
        nI = rng.randint(1, 4)
        nJ = rng.randint(1, 4)
        f = [rng.randint(0, 8) for _ in range(nI)]
        g = [rng.randint(0, 8) for _ in range(nJ)]
        b = [rng.randint(1, 8) for _ in range(nI)]
        p = [rng.randint(1, 8) for _ in range(nJ)]
        # gamma pode ser negativo (a proposicao nao exige sinal);
        # delta >= 3 garante arcos c_ij = gamma_i + delta_j >= 0
        gamma = [rng.randint(-3, 5) for _ in range(nI)]
        delta = [rng.randint(3, 8) for _ in range(nJ)]
        d = [rng.randint(0, 4) for _ in range(nJ)]
        r = rng.random()
        if r < 0.1:
            D = 0
        elif r < 0.3:
            D = max(sum(b), sum(p)) + rng.randint(1, 5)
        else:
            D = rng.randint(1, min(sum(b), sum(p)))
        inst = {
            "nI": nI, "nJ": nJ, "nK": 1, "nL": 1, "f": f, "g": g,
            "c": [[[gamma[i] + delta[j]] for j in range(nJ)]
                  for i in range(nI)],
            "d": [[[d[j]]] for j in range(nJ)],
            "b": [[b[i]] for i in range(nI)],
            "p": [[p[j]] for j in range(nJ)],
            "q": [[D]],
        }
        # forca bruta
        opt = INF
        for my in range(1 << nI):
            y = [(my >> i) & 1 for i in range(nI)]
            fy = sum(f[i] for i in range(nI) if y[i])
            for mz in range(1 << nJ):
                z = [(mz >> j) & 1 for j in range(nJ)]
                feas, v = routing_value(inst, 0, y, z)
                if feas:
                    gz = sum(g[j] for j in range(nJ) if z[j])
                    opt = min(opt, fy + gz + v)
        # DP desacoplado
        if D == 0:
            dpval = 0
            counters["D0"] += 1
        else:
            a = side_dp([(f[i], gamma[i], b[i]) for i in range(nI)], D)
            bb = side_dp([(g[j], delta[j] + d[j], p[j])
                          for j in range(nJ)], D)
            dpval = a + bb if a < INF and bb < INF else INF
        check(opt == dpval,
              f"custos separaveis seed={s}: bruta={opt} dp={dpval}")
        counters["instancias"] += 1
        if opt < INF:
            counters["viaveis"] += 1
        else:
            counters["inviaveis"] += 1
    return counters


# ---------------------------------------------------------------------------
# [D] contraste por PL independente (scipy) na celula geral
# ---------------------------------------------------------------------------

def lp_residual(inst, y, z):
    """PL residual da celula |K|=|L|=1 em forma de desigualdade.
    Retorna (viavel, valor) via scipy.linprog/HiGHS."""
    from scipy.optimize import linprog
    nI, nJ = inst["nI"], inst["nJ"]
    D = inst["q"][0][0]
    nx = nI * nJ
    nvar = nx + nJ                      # x_ij, w_j
    cost = [inst["c"][i][j][0] for i in range(nI) for j in range(nJ)] \
        + [inst["d"][j][0][0] for j in range(nJ)]
    A_ub, b_ub = [], []
    # -(sum_j w_j) <= -D                                   (C1)
    row = [0.0] * nvar
    for j in range(nJ):
        row[nx + j] = -1.0
    A_ub.append(row)
    b_ub.append(-float(D))
    # w_j - sum_i x_ij <= 0                                (C2)
    for j in range(nJ):
        row = [0.0] * nvar
        row[nx + j] = 1.0
        for i in range(nI):
            row[i * nJ + j] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)
    # sum_j x_ij <= b_i y_i                                (C3)
    for i in range(nI):
        row = [0.0] * nvar
        for j in range(nJ):
            row[i * nJ + j] = 1.0
        A_ub.append(row)
        b_ub.append(float(inst["b"][i][0] * y[i]))
    # w_j <= p_j z_j                                       (C4)
    for j in range(nJ):
        row = [0.0] * nvar
        row[nx + j] = 1.0
        A_ub.append(row)
        b_ub.append(float(inst["p"][j][0] * z[j]))
    res = linprog(cost, A_ub=A_ub, b_ub=b_ub,
                  bounds=[(0, None)] * nvar, method="highs")
    if res.status == 2:
        return False, None
    return True, res.fun


def battery_D(n=25, seed0=9000, designs_per=8):
    counters = {"lps": 0}
    for s in range(seed0, seed0 + n):
        rng = random.Random(s)
        nI = rng.randint(1, 4)
        nJ = rng.randint(1, 4)
        f = [rng.randint(0, 8) for _ in range(nI)]
        g = [rng.randint(0, 8) for _ in range(nJ)]
        b = [rng.randint(1, 9) for _ in range(nI)]
        p = [rng.randint(1, 9) for _ in range(nJ)]
        c = [[rng.randint(0, 8) for _ in range(nJ)] for _ in range(nI)]
        d = [rng.randint(0, 8) for _ in range(nJ)]
        D = rng.randint(0, max(sum(b), sum(p)))
        inst = {
            "nI": nI, "nJ": nJ, "nK": 1, "nL": 1, "f": f, "g": g,
            "c": [[[c[i][j]] for j in range(nJ)] for i in range(nI)],
            "d": [[[d[j]]] for j in range(nJ)],
            "b": [[b[i]] for i in range(nI)],
            "p": [[p[j]] for j in range(nJ)],
            "q": [[D]],
        }
        for _ in range(designs_per):
            y = [rng.randint(0, 1) for _ in range(nI)]
            z = [rng.randint(0, 1) for _ in range(nJ)]
            feas_f, v_f = routing_value(inst, 0, y, z)
            feas_l, v_l = lp_residual(inst, y, z)
            counters["lps"] += 1
            check(feas_f == feas_l,
                  f"[D] viabilidade seed={s} y={y} z={z}: "
                  f"mcmf={feas_f} lp={feas_l}")
            if feas_f and feas_l:
                check(abs(v_l - v_f) <= 1e-6,
                      f"[D] valor seed={s} y={y} z={z}: mcmf={v_f} lp={v_l}")
                check(abs(v_l - round(v_l)) <= 1e-6,
                      f"[D] integralidade seed={s}: lp={v_l}")
    return counters


def main():
    print("== [A] exaustiva: familias |U| <= 4, m <= 4 ==")
    ca = battery_A()
    print(f"  familias: {ca['familias']}  desenhos verificados: "
          f"{ca['desenhos']}  testes '<=>': {ca['iff']}")

    print("== [B] aleatoria: familias |U|, |S| <= 5 ==")
    cb = battery_B()
    print(f"  familias: {cb['familias']}  desenhos verificados: "
          f"{cb['desenhos']}  testes '<=>': {cb['iff']}")

    print("== [C] custos separaveis (DP vs forca bruta) ==")
    cc = battery_C()
    print(f"  instancias: {cc['instancias']} (viaveis {cc['viaveis']}, "
          f"inviaveis {cc['inviaveis']}, D=0 {cc['D0']})")

    print("== [D] contraste por PL independente (scipy/HiGHS) ==")
    cd = battery_D()
    print(f"  LPs comparados: {cd['lps']}")

    if FAILS:
        print(f"\n== RESULTADO: {len(FAILS)} FALHAS ==")
        sys.exit(1)
    print("\n== RESULTADO: PASS (0 falhas) ==")


if __name__ == "__main__":
    main()
