#!/usr/bin/env python3
"""
Verificacao computacional dos resultados de agregacao e compressao do artigo.

Baterias:
  [A] Agregacao     - agregacao de clientes com colunas d identicas e exata
                    POR DESENHO (custo e viabilidade identicos em todo (y,z)),
                    logo preserva OPT e o conjunto de desenhos otimos.
                    40 instancias aleatorias com sementes + 4 adversariais
                    (demanda nula, empates totais, inviabilidade global,
                    fusao tripla com independencia de ordem).
  [B] Contraexemplo - fusao de PRODUTOS com colunas de custo identicas
                    NAO e exata: contraexemplo de valor (OPT 10 -> 0) e
                    contraexemplo de viabilidade, verificados numericamente.
  [C] Subaditividade - fusao de produtos nunca AUMENTA o custo de um desenho
                    (v_fundido <= v_l + v_l'), com casos estritos observados.
  [D] Proporcionais - fusao de produtos PROPORCIONAIS ((q,b,p) escalados por
                    lambda, custos iguais) e exata por desenho.
  [E] Capping       - capping b_il := min(b_il, D_l), p_jl := min(p_jl, D_l)
                    e exato por desenho.

Forca bruta: enumeracao de todos os desenhos (y,z) + oraculo MCMF inteiro
exato do oraculo de roteamento (common_mp_tscfl.routing_value). Nenhuma
forma fechada das provas e usada como fonte.
"""
import copy
from common_mp_tscfl import gen_instance, routing_value, all_designs


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def total_cost(inst, y, z):
    """Custo total do desenho (y,z): fixos + roteamento; None se inviavel."""
    cost = sum(inst["f"][i] * y[i] for i in range(inst["nI"]))
    cost += sum(inst["g"][j] * z[j] for j in range(inst["nJ"]))
    for l in range(inst["nL"]):
        feas, val = routing_value(inst, l, y, z)
        if not feas:
            return None
        cost += val
    return cost


def cost_map(inst):
    """Mapa {(y,z) -> custo | None} sobre todos os desenhos."""
    return {(tuple(y), tuple(z)): total_cost(inst, y, z)
            for y, z in all_designs(inst["nI"], inst["nJ"])}


def opt_and_argmin(cm):
    finite = {k: v for k, v in cm.items() if v is not None}
    if not finite:
        return None, set()
    opt = min(finite.values())
    return opt, {k for k, v in finite.items() if v == opt}


def merge_customers(inst, k1, k2):
    """Funde k2 em k1 (colunas d identicas exigidas), somando demandas."""
    assert k1 != k2
    for j in range(inst["nJ"]):
        for l in range(inst["nL"]):
            assert inst["d"][j][k1][l] == inst["d"][j][k2][l], \
                "colunas d nao identicas"
    out = copy.deepcopy(inst)
    for l in range(inst["nL"]):
        out["q"][k1][l] += out["q"][k2][l]
    del out["q"][k2]
    for j in range(inst["nJ"]):
        del out["d"][j][k2]
    out["nK"] -= 1
    return out


def merge_products(inst, l1, l2):
    """Funde l2 em l1 (colunas c,d identicas exigidas), somando q, b, p."""
    nI, nJ, nK = inst["nI"], inst["nJ"], inst["nK"]
    for i in range(nI):
        for j in range(nJ):
            assert inst["c"][i][j][l1] == inst["c"][i][j][l2]
    for j in range(nJ):
        for k in range(nK):
            assert inst["d"][j][k][l1] == inst["d"][j][k][l2]
    out = copy.deepcopy(inst)
    for i in range(nI):
        out["b"][i][l1] += out["b"][i][l2]
        del out["b"][i][l2]
    for j in range(nJ):
        out["p"][j][l1] += out["p"][j][l2]
        del out["p"][j][l2]
    for k in range(nK):
        out["q"][k][l1] += out["q"][k][l2]
        del out["q"][k][l2]
    for i in range(nI):
        for j in range(nJ):
            del out["c"][i][j][l2]
    for j in range(nJ):
        for k in range(nK):
            del out["d"][j][k][l2]
    out["nL"] -= 1
    return out


def check_design_equality(inst_a, inst_b, label):
    """Custos identicos (inclusive None) em TODO desenho; OPT e argmin."""
    cma, cmb = cost_map(inst_a), cost_map(inst_b)
    assert set(cma) == set(cmb)
    n = 0
    for key in cma:
        assert cma[key] == cmb[key], f"[{label}] desenho {key}: " \
            f"{cma[key]} != {cmb[key]}"
        n += 1
    oa, aa = opt_and_argmin(cma)
    ob, ab = opt_and_argmin(cmb)
    assert oa == ob and aa == ab, f"[{label}] OPT/argmin divergem"
    return n


# ---------------------------------------------------------------------------
# [A] agregacao de clientes
# ---------------------------------------------------------------------------

def battery_A():
    n_inst = n_checks = n_zero_demand = n_threeway = 0
    for seed in range(6000, 6040):
        s = seed
        inst = gen_instance(s, max_i=3, max_j=3, max_k=4, max_l=2, vmax=6)
        while inst["nK"] < 2:
            s += 977
            inst = gen_instance(s, max_i=3, max_j=3, max_k=4, max_l=2, vmax=6)
        # coluna do cliente 1 := coluna do cliente 0 (duplicata forcada)
        for j in range(inst["nJ"]):
            for l in range(inst["nL"]):
                inst["d"][j][1][l] = inst["d"][j][0][l]
        if seed % 5 == 0:  # adversarial embutido: duplicata com demanda nula
            for l in range(inst["nL"]):
                inst["q"][1][l] = 0
            n_zero_demand += 1
        merged = merge_customers(inst, 0, 1)
        n_checks += check_design_equality(inst, merged, f"A seed={seed}")
        # fusao tripla com independencia de ordem
        if seed % 7 == 0 and inst["nK"] >= 3:
            inst3 = copy.deepcopy(inst)
            for j in range(inst3["nJ"]):
                for l in range(inst3["nL"]):
                    inst3["d"][j][2][l] = inst3["d"][j][0][l]
            m_ab_c = merge_customers(merge_customers(inst3, 0, 1), 0, 1)
            m_ac_b = merge_customers(merge_customers(inst3, 0, 2), 0, 1)
            n_checks += check_design_equality(inst3, m_ab_c,
                                              f"A3 seed={seed} (0,1)+2")
            n_checks += check_design_equality(m_ab_c, m_ac_b,
                                              f"A3 seed={seed} ordem")
            n_threeway += 1
        n_inst += 1

    # 4 adversariais construidos a mao
    adv = []
    # adv1: tudo-zero (empates em todos os desenhos)
    adv.append({"nI": 1, "nJ": 1, "nK": 2, "nL": 1,
                "f": [0], "g": [0], "c": [[[0]]],
                "d": [[[0], [0]]], "b": [[5]], "p": [[5]],
                "q": [[2], [3]]})
    # adv2: capacidade justa apos a fusao (p = q1+q2)
    adv.append({"nI": 1, "nJ": 2, "nK": 2, "nL": 1,
                "f": [1], "g": [2, 3], "c": [[[1], [4]]],
                "d": [[[2], [2]], [[0], [0]]], "b": [[7]], "p": [[7], [7]],
                "q": [[0], [7]]})
    # adv3: inviavel em todo desenho (b < D)
    adv.append({"nI": 1, "nJ": 1, "nK": 2, "nL": 1,
                "f": [0], "g": [0], "c": [[[1]]],
                "d": [[[1], [1]]], "b": [[2]], "p": [[9]],
                "q": [[2], [2]]})
    # adv4: dois produtos, colunas d identicas nos DOIS produtos
    adv.append({"nI": 2, "nJ": 2, "nK": 2, "nL": 2,
                "f": [3, 1], "g": [2, 2],
                "c": [[[1, 0], [2, 5]], [[0, 3], [1, 1]]],
                "d": [[[4, 1], [4, 1]], [[0, 2], [0, 2]]],
                "b": [[5, 3], [4, 4]], "p": [[6, 4], [5, 5]],
                "q": [[3, 2], [2, 1]]})
    for idx, inst in enumerate(adv):
        merged = merge_customers(inst, 0, 1)
        n_checks += check_design_equality(inst, merged, f"A adv{idx + 1}")
        n_inst += 1

    print(f"[A] agregacao de clientes: {n_inst} instancias "
          f"({n_zero_demand} c/ demanda nula, {n_threeway} fusoes triplas), "
          f"{n_checks} comparacoes por desenho: PASS")


# ---------------------------------------------------------------------------
# [B] contraexemplos de produto
# ---------------------------------------------------------------------------

def battery_B():
    # B1: valor. Custos identicos entre produtos; capacidades cruzadas.
    orig = {"nI": 2, "nJ": 1, "nK": 1, "nL": 2,
            "f": [0, 0], "g": [0],
            "c": [[[0, 0]], [[10, 10]]],
            "d": [[[0, 0]]],
            "b": [[2, 0], [0, 2]], "p": [[2, 2]],
            "q": [[1, 1]]}
    merged = merge_products(orig, 0, 1)
    opt_o, arg_o = opt_and_argmin(cost_map(orig))
    opt_m, arg_m = opt_and_argmin(cost_map(merged))
    assert opt_o == 10, f"B1: OPT original = {opt_o}, esperado 10"
    assert opt_m == 0, f"B1: OPT fundido = {opt_m}, esperado 0"
    assert arg_o != arg_m, "B1: conjuntos de desenhos otimos deveriam diferir"
    # desenho (1,0),(1): inviavel no original (produto 2 sem capacidade),
    # viavel e otimo no fundido
    key = ((1, 0), (1,))
    assert total_cost(orig, [1, 0], [1]) is None
    assert total_cost(merged, [1, 0], [1]) == 0
    print(f"[B1] contraexemplo de valor: OPT 10 -> 0, argmin muda, "
          f"desenho {key} inviavel->otimo: PASS")

    # B2: viabilidade. Original inviavel em todo desenho; fundido viavel.
    orig2 = {"nI": 1, "nJ": 1, "nK": 1, "nL": 2,
             "f": [0], "g": [0],
             "c": [[[0, 0]]], "d": [[[0, 0]]],
             "b": [[2, 0]], "p": [[2, 2]],
             "q": [[1, 1]]}
    merged2 = merge_products(orig2, 0, 1)
    cm_o = cost_map(orig2)
    assert all(v is None for v in cm_o.values()), "B2: original tem desenho viavel"
    opt2, _ = opt_and_argmin(cost_map(merged2))
    assert opt2 == 0, f"B2: OPT fundido = {opt2}, esperado 0"
    print("[B2] contraexemplo de viabilidade (inviavel -> viavel): PASS")


# ---------------------------------------------------------------------------
# [C] subaditividade da fusao de produtos
# ---------------------------------------------------------------------------

def battery_C():
    n_inst = n_checks = n_strict = 0
    for seed in range(6200, 6230):
        s = seed
        inst = gen_instance(s, max_i=3, max_j=3, max_k=3, max_l=2, vmax=6)
        while inst["nL"] < 2:
            s += 977
            inst = gen_instance(s, max_i=3, max_j=3, max_k=3, max_l=2, vmax=6)
        # iguala colunas de custo dos produtos 0 e 1 (b,p,q independentes)
        for i in range(inst["nI"]):
            for j in range(inst["nJ"]):
                inst["c"][i][j][1] = inst["c"][i][j][0]
        for j in range(inst["nJ"]):
            for k in range(inst["nK"]):
                inst["d"][j][k][1] = inst["d"][j][k][0]
        merged = merge_products(inst, 0, 1)
        cmo, cmm = cost_map(inst), cost_map(merged)
        for key in cmo:
            if cmo[key] is not None:
                assert cmm[key] is not None, \
                    f"[C seed={seed}] viavel virou inviavel apos fusao"
                assert cmm[key] <= cmo[key], \
                    f"[C seed={seed}] fusao aumentou custo em {key}"
                if cmm[key] < cmo[key]:
                    n_strict += 1
                n_checks += 1
        n_inst += 1
    assert n_strict > 0, "[C] nenhum caso estrito observado (bateria fraca)"
    print(f"[C] subaditividade: {n_inst} instancias, {n_checks} desenhos "
          f"viaveis, {n_strict} estritamente menores: PASS")


# ---------------------------------------------------------------------------
# [D] produtos proporcionais
# ---------------------------------------------------------------------------

def battery_D():
    n_inst = n_checks = 0
    for seed in range(6300, 6330):
        s = seed
        base = gen_instance(s, max_i=3, max_j=3, max_k=3, max_l=1, vmax=6)
        lam = 1 + seed % 3
        nI, nJ, nK = base["nI"], base["nJ"], base["nK"]
        two = {"nI": nI, "nJ": nJ, "nK": nK, "nL": 2,
               "f": base["f"][:], "g": base["g"][:],
               "c": [[[base["c"][i][j][0]] * 2 for j in range(nJ)]
                     for i in range(nI)],
               "d": [[[base["d"][j][k][0]] * 2 for k in range(nK)]
                     for j in range(nJ)],
               "b": [[base["b"][i][0], lam * base["b"][i][0]]
                     for i in range(nI)],
               "p": [[base["p"][j][0], lam * base["p"][j][0]]
                     for j in range(nJ)],
               "q": [[base["q"][k][0], lam * base["q"][k][0]]
                     for k in range(nK)]}
        merged = merge_products(two, 0, 1)
        n_checks += check_design_equality(two, merged,
                                          f"D seed={seed} lam={lam}")
        n_inst += 1
    print(f"[D] produtos proporcionais: {n_inst} instancias "
          f"(lambda em 1..3), {n_checks} comparacoes por desenho: PASS")


# ---------------------------------------------------------------------------
# [E] capping de capacidades
# ---------------------------------------------------------------------------

def battery_E():
    n_inst = n_checks = n_capped = 0
    for seed in range(6100, 6130):
        inst = gen_instance(seed, max_i=3, max_j=3, max_k=3, max_l=2, vmax=9)
        capped = copy.deepcopy(inst)
        changed = False
        for l in range(inst["nL"]):
            D = sum(inst["q"][k][l] for k in range(inst["nK"]))
            for i in range(inst["nI"]):
                if capped["b"][i][l] > D:
                    capped["b"][i][l] = D
                    changed = True
            for j in range(inst["nJ"]):
                if capped["p"][j][l] > D:
                    capped["p"][j][l] = D
                    changed = True
        if changed:
            n_capped += 1
        n_checks += check_design_equality(inst, capped, f"E seed={seed}")
        n_inst += 1
    print(f"[E] capping b,p <= D_l: {n_inst} instancias "
          f"({n_capped} efetivamente alteradas), {n_checks} comparacoes: PASS")


if __name__ == "__main__":
    battery_A()
    battery_B()
    battery_C()
    battery_D()
    battery_E()
    print("verify_A6_aggregation.py: TODAS AS BATERIAS PASSARAM")
