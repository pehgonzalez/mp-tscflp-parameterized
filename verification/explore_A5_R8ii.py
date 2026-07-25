"""
Exploracao computacional do problema aberto do artigo sobre a celula
|K| = |L| = 1 com custos de transporte gerais.

Celula: |K| = |L| = 1, custos de transporte GERAIS (c_ij, d_j), |I| e |J|
livres. Escrevemos c^(Y,Z) := f(Y) + g(Z) + v(Y,Z) para o custo total do
desenho (Y,Z), com c^ = +infinito se o desenho e' inviavel, e v(Y,Z) o
valor do PL residual (oraculo MCMF de roteamento, implementacao inteira
exata de common_mp_tscfl.py).

Baterias (todas com sementes fixas; contagens impressas ao final):

  [S]   Sub/supermodularidade de c^(Y,Z) e de v(Y,Z) no reticulado
        produto 2^I x 2^J (join = uniao, meet = intersecao).
        MOTIVACAO: se c^ fosse submodular, a minimizacao de funcao
        submodular (poli, Groetschel-Lovasz-Schrijver) resolveria
        o problema aberto POSITIVAMENTE. Testamos primeiro por ser decisivo.
        Obstrucao estrutural registrada a parte: com A e B viaveis o
        meet pode ser inviavel (c^(meet) = +inf), o que ja viola a
        desigualdade submodular; contamos tambem violacoes com os
        QUATRO pontos finitos, que sao a evidencia limpa.

  [P]   Invariante de prefixo: existe desenho otimo cujo conjunto Y
        (resp. Z) e' prefixo sob ordenacoes naturais? Ordenacoes do
        lado I: (P1) f_i crescente; (P2) f_i/b_i crescente;
        (P3) (f_i + b_i * mincusto_i)/b_i crescente, onde
        mincusto_i = min_j (c_ij + d_j). Lado J: simetricas.
        Teste de prefixo tolerante a empates: Y passa sse
        max_{i in Y} chave_i <= min_{i not in Y} chave_i.

  [G]   Heuristica desacoplada: escolher Y minimizando f(Y) sujeito a
        sum b_i >= D e Z minimizando g(Z) sujeito a sum p_j >= D
        (ignorando o transporte), depois rotear otimamente. Comparar
        com OPT (forca bruta): contamos falhas estritas.

  [SEP] Sanidade da proposicao dos custos separaveis: em instancias com
        custos SEPARAVEIS
        c_ij = gamma_i + delta_j, o par de DPs desacoplados (lado I com
        (f_i, gamma_i, b_i); lado J com (g_j, delta_j + d_j, p_j)) deve
        coincidir EXATAMENTE com a forca bruta. (A verificacao formal
        da proposicao esta em verify_A5_R8ii.py; aqui e' o farol da
        exploracao.)

Forca bruta: enumeracao de TODOS os desenhos (y,z) + oraculo MCMF
inteiro (routing_value). Nenhuma forma fechada e' usada como fonte.
"""

import random
from fractions import Fraction

from common_mp_tscfl import routing_value

INF = float("inf")


# ---------------------------------------------------------------------------
# Instancias da celula |K| = |L| = 1
# ---------------------------------------------------------------------------

def make_inst(f, g, b, p, c, d, D):
    nI, nJ = len(f), len(g)
    return {
        "nI": nI, "nJ": nJ, "nK": 1, "nL": 1,
        "f": list(f), "g": list(g),
        "c": [[[c[i][j]] for j in range(nJ)] for i in range(nI)],
        "d": [[[d[j]]] for j in range(nJ)],
        "b": [[b[i]] for i in range(nI)],
        "p": [[p[j]] for j in range(nJ)],
        "q": [[D]],
    }


def gen_cell(seed, max_i=4, max_j=4, vmax=8, capmax=10, dcap=25):
    """Instancia aleatoria da celula, sempre viavel com tudo aberto."""
    rng = random.Random(seed)
    nI = rng.randint(1, max_i)
    nJ = rng.randint(1, max_j)
    f = [rng.randint(0, vmax) for _ in range(nI)]
    g = [rng.randint(0, vmax) for _ in range(nJ)]
    b = [rng.randint(1, capmax) for _ in range(nI)]
    p = [rng.randint(1, capmax) for _ in range(nJ)]
    c = [[rng.randint(0, vmax) for _ in range(nJ)] for _ in range(nI)]
    d = [rng.randint(0, vmax) for _ in range(nJ)]
    D = rng.randint(1, min(sum(b), sum(p), dcap))
    return make_inst(f, g, b, p, c, d, D)


def value_table(inst):
    """(custo total, custo de roteamento) de todos os desenhos; INF se
    inviavel. Fonte: oraculo MCMF, desenho a desenho."""
    nI, nJ = inst["nI"], inst["nJ"]
    tab = {}
    for my in range(1 << nI):
        y = [(my >> i) & 1 for i in range(nI)]
        fy = sum(inst["f"][i] for i in range(nI) if y[i])
        for mz in range(1 << nJ):
            z = [(mz >> j) & 1 for j in range(nJ)]
            feas, v = routing_value(inst, 0, y, z)
            if feas:
                gz = sum(inst["g"][j] for j in range(nJ) if z[j])
                tab[(my, mz)] = (fy + gz + v, v)
            else:
                tab[(my, mz)] = (INF, INF)
    return tab


# ---------------------------------------------------------------------------
# [S] sub/supermodularidade no reticulado produto
# ---------------------------------------------------------------------------

def battery_S(inst_list):
    cnt = {
        "quadruplas_AB_finitas": 0,
        "meet_inviavel": 0,          # c^(A), c^(B) finitos, meet = +inf
        "quatro_finitos": 0,
        "viol_submod_chat": 0,       # c^(join)+c^(meet) > c^(A)+c^(B)
        "viol_supermod_chat": 0,     # c^(join)+c^(meet) < c^(A)+c^(B)
        "viol_submod_v": 0,
        "viol_supermod_v": 0,
        "quatro_finitos_v": 0,
    }
    first_sub = first_super = None
    for seed, inst in inst_list:
        tab = value_table(inst)
        keys = sorted(tab)
        n = len(keys)
        for a in range(n):
            A = keys[a]
            cA, vA = tab[A]
            if cA == INF:
                continue
            for bidx in range(a + 1, n):
                B = keys[bidx]
                cB, vB = tab[B]
                if cB == INF:
                    continue
                cnt["quadruplas_AB_finitas"] += 1
                J = (A[0] | B[0], A[1] | B[1])
                M = (A[0] & B[0], A[1] & B[1])
                cJ, vJ = tab[J]
                cM, vM = tab[M]
                if cM == INF:
                    cnt["meet_inviavel"] += 1
                    continue
                cnt["quatro_finitos"] += 1
                lhs, rhs = cJ + cM, cA + cB
                if lhs > rhs:
                    cnt["viol_submod_chat"] += 1
                    if first_sub is None:
                        first_sub = (seed, A, B, J, M, cA, cB, cJ, cM)
                elif lhs < rhs:
                    cnt["viol_supermod_chat"] += 1
                    if first_super is None:
                        first_super = (seed, A, B, J, M, cA, cB, cJ, cM)
                # roteamento puro
                cnt["quatro_finitos_v"] += 1
                lv, rv = vJ + vM, vA + vB
                if lv > rv:
                    cnt["viol_submod_v"] += 1
                elif lv < rv:
                    cnt["viol_supermod_v"] += 1
    return cnt, first_sub, first_super


# ---------------------------------------------------------------------------
# [P] invariante de prefixo
# ---------------------------------------------------------------------------

def prefix_ok(keys, members, universe):
    """Y e' prefixo (tolerante a empates) sob a ordenacao por chaves."""
    inside = [keys[i] for i in universe if i in members]
    outside = [keys[i] for i in universe if i not in members]
    if not inside or not outside:
        return True
    return max(inside) <= min(outside)


def battery_P(inst_list):
    lados = ["I_f", "I_f/b", "I_efetivo", "J_g", "J_g/p", "J_efetivo"]
    fail = {nome: 0 for nome in lados}
    exemplos = {nome: None for nome in lados}
    n_inst = 0
    for seed, inst in inst_list:
        tab = value_table(inst)
        opt = min(v for v, _ in tab.values())
        if opt == INF:
            continue
        n_inst += 1
        opts = [k for k, (v, _) in tab.items() if v == opt]
        nI, nJ = inst["nI"], inst["nJ"]
        f, g = inst["f"], inst["g"]
        b = [inst["b"][i][0] for i in range(nI)]
        p = [inst["p"][j][0] for j in range(nJ)]
        cd = [[inst["c"][i][j][0] + inst["d"][j][0][0] for j in range(nJ)]
              for i in range(nI)]
        minc_i = [min(cd[i]) for i in range(nI)]
        minc_j = [min(cd[i][j] for i in range(nI)) for j in range(nJ)]
        keysI = {
            "I_f": [Fraction(f[i]) for i in range(nI)],
            "I_f/b": [Fraction(f[i], b[i]) for i in range(nI)],
            "I_efetivo": [Fraction(f[i] + b[i] * minc_i[i], b[i])
                          for i in range(nI)],
        }
        keysJ = {
            "J_g": [Fraction(g[j]) for j in range(nJ)],
            "J_g/p": [Fraction(g[j], p[j]) for j in range(nJ)],
            "J_efetivo": [Fraction(g[j] + p[j] * minc_j[j], p[j])
                          for j in range(nJ)],
        }
        for nome, keys in keysI.items():
            ok = any(prefix_ok(keys, {i for i in range(nI) if my >> i & 1},
                               range(nI)) for my, _ in opts)
            if not ok:
                fail[nome] += 1
                if exemplos[nome] is None:
                    exemplos[nome] = seed
        for nome, keys in keysJ.items():
            ok = any(prefix_ok(keys, {j for j in range(nJ) if mz >> j & 1},
                               range(nJ)) for _, mz in opts)
            if not ok:
                fail[nome] += 1
                if exemplos[nome] is None:
                    exemplos[nome] = seed
    return n_inst, fail, exemplos


# ---------------------------------------------------------------------------
# [G] heuristica desacoplada vs OPT
# ---------------------------------------------------------------------------

def battery_G(inst_list):
    n_inst = 0
    n_fail = 0
    max_gap = 0
    exemplo = None
    for seed, inst in inst_list:
        tab = value_table(inst)
        opt = min(v for v, _ in tab.values())
        if opt == INF:
            continue
        n_inst += 1
        nI, nJ = inst["nI"], inst["nJ"]
        D = inst["q"][0][0]
        b = [inst["b"][i][0] for i in range(nI)]
        p = [inst["p"][j][0] for j in range(nJ)]
        # lado I: min f(Y) s.a. sum_{i in Y} b_i >= D (forca bruta)
        bestY = min((my for my in range(1 << nI)
                     if sum(b[i] for i in range(nI) if my >> i & 1) >= D),
                    key=lambda my: (sum(inst["f"][i] for i in range(nI)
                                        if my >> i & 1), bin(my).count("1"),
                                    my))
        bestZ = min((mz for mz in range(1 << nJ)
                     if sum(p[j] for j in range(nJ) if mz >> j & 1) >= D),
                    key=lambda mz: (sum(inst["g"][j] for j in range(nJ)
                                        if mz >> j & 1), bin(mz).count("1"),
                                    mz))
        heur = tab[(bestY, bestZ)][0]
        assert heur < INF  # condicao agregada => viavel (caracterizacao de viabilidade)
        assert heur >= opt
        if heur > opt:
            n_fail += 1
            if heur - opt > max_gap:
                max_gap = heur - opt
                exemplo = (seed, heur, opt)
    return n_inst, n_fail, max_gap, exemplo


# ---------------------------------------------------------------------------
# [SEP] custos separaveis: DPs desacoplados == forca bruta
# ---------------------------------------------------------------------------

def side_dp(items, D):
    """min sum_{i in S} (F_i + gam_i * u_i) com u_i in [1..cap_i] p/ i em S,
    sum u_i = D. items = lista de (F, gam, cap)."""
    T = [0] + [INF] * D
    for (F, gam, cap) in items:
        newT = list(T)
        for t in range(1, D + 1):
            up = min(cap, t)
            for u in range(1, up + 1):
                if T[t - u] < INF:
                    cand = T[t - u] + F + gam * u
                    if cand < newT[t]:
                        newT[t] = cand
        T = newT
    return T[D]


def sep_opt(inst, gamma, delta):
    D = inst["q"][0][0]
    if D == 0:
        return 0
    nI, nJ = inst["nI"], inst["nJ"]
    a = side_dp([(inst["f"][i], gamma[i], inst["b"][i][0])
                 for i in range(nI)], D)
    bb = side_dp([(inst["g"][j], delta[j] + inst["d"][j][0][0],
                   inst["p"][j][0]) for j in range(nJ)], D)
    return a + bb if a < INF and bb < INF else INF


def battery_SEP(n=60, seed0=7000):
    n_ok = n_bad = 0
    for s in range(seed0, seed0 + n):
        rng = random.Random(s)
        nI = rng.randint(1, 4)
        nJ = rng.randint(1, 4)
        f = [rng.randint(0, 8) for _ in range(nI)]
        g = [rng.randint(0, 8) for _ in range(nJ)]
        b = [rng.randint(1, 8) for _ in range(nI)]
        p = [rng.randint(1, 8) for _ in range(nJ)]
        gamma = [rng.randint(0, 5) for _ in range(nI)]
        delta = [rng.randint(0, 5) for _ in range(nJ)]
        c = [[gamma[i] + delta[j] for j in range(nJ)] for i in range(nI)]
        d = [rng.randint(0, 4) for _ in range(nJ)]
        r = rng.random()
        if r < 0.1:
            D = 0
        elif r < 0.25:
            D = max(sum(b), sum(p)) + rng.randint(1, 5)   # inviavel
        else:
            D = rng.randint(1, min(sum(b), sum(p)))
        inst = make_inst(f, g, b, p, c, d, D)
        tab = value_table(inst)
        opt = min(v for v, _ in tab.values())
        dpval = sep_opt(inst, gamma, delta)
        if opt == dpval:
            n_ok += 1
        else:
            n_bad += 1
            print(f"  [SEP] DIVERGENCIA seed={s}: bruta={opt} dp={dpval}")
    return n_ok, n_bad


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    # 150 instancias 4x4 + 40 instancias 5x5 (sementes fixas)
    small = [(s, gen_cell(s, 4, 4)) for s in range(1000, 1150)]
    big = [(s, gen_cell(s, 5, 5)) for s in range(2000, 2040)]
    todas = small + big

    print("== [S] sub/supermodularidade de c^ e v no reticulado 2^I x 2^J ==")
    cnt, first_sub, first_super = battery_S(todas)
    for k, v in cnt.items():
        print(f"  {k}: {v}")
    if first_sub:
        s, A, B, J, M, cA, cB, cJ, cM = first_sub
        print(f"  primeiro contraexemplo SUBmodularidade (4 pontos finitos):")
        print(f"    seed={s} A={A} B={B} join={J} meet={M}")
        print(f"    c^(A)={cA} c^(B)={cB} c^(join)={cJ} c^(meet)={cM} "
              f"(lhs={cJ+cM} > rhs={cA+cB})")
    if first_super:
        s, A, B, J, M, cA, cB, cJ, cM = first_super
        print(f"  primeiro contraexemplo SUPERmodularidade:")
        print(f"    seed={s} A={A} B={B} join={J} meet={M}")
        print(f"    c^(A)={cA} c^(B)={cB} c^(join)={cJ} c^(meet)={cM} "
              f"(lhs={cJ+cM} < rhs={cA+cB})")

    print("== [P] invariante de prefixo em desenhos otimos ==")
    n_inst, fail, exemplos = battery_P(todas)
    print(f"  instancias com OPT finito: {n_inst}")
    for nome in fail:
        print(f"  ordenacao {nome}: {fail[nome]} instancias sem NENHUM "
              f"otimo-prefixo (primeiro seed: {exemplos[nome]})")

    print("== [G] heuristica desacoplada (ignora transporte) vs OPT ==")
    n_inst, n_fail, max_gap, exemplo = battery_G(todas)
    print(f"  instancias: {n_inst}; falhas estritas: {n_fail}; "
          f"gap maximo: {max_gap}; exemplo: {exemplo}")

    print("== [SEP] custos separaveis c_ij = gamma_i + delta_j ==")
    n_ok, n_bad = battery_SEP()
    print(f"  coincidencia DP desacoplado == forca bruta: {n_ok} ok, "
          f"{n_bad} divergencias")

    print("== exploracao concluida ==")


if __name__ == "__main__":
    main()
