"""
Verificacao do algoritmo XP do artigo (algoritmo XP em k como B&B com as
tres podas P1/P2/P3).

Implementacao de referencia (Python) do algoritmo do artigo sobre a arvore
binaria de decisoes abrir/fechar (fabricas primeiro, depois depositos),
na versao de OTIMIZACAO por cardinalidade: para cada k in {0..n} computa
OPT_k = min { custo(S) : S viavel, |S| <= k }. Isso subsume TODOS os
orcamentos B de uma vez (MP-TSCFLP(B,k) e SIM sse OPT_k <= B) — e a
varredura completa (todas as cardinalidades x todos os orcamentos).

Podas (exatamente as do artigo):
  P1 (cobertura, o lema de contagem de cobertura): no no (O,C), com
     r = k - |O| aberturas
     restantes, se max_l s_I(l;O,C) + max_l s_J(l;O,C) > r, nenhuma
     completacao e viavel dentro da cardinalidade -> descarta. s_side(l)
     e o numero minimo de instalacoes livres do lado, por capacidades
     ordenadas decrescentes (prefixo guloso), para fechar F1/F2.
  P2 (cota admissivel, via monotonicidade): LB(O,C) = custo fixo
     de O + v(abre O e todos os livres); descarta se v = +inf (nenhuma
     completacao viavel, por monotonicidade) ou LB >= incumbente.
  P3 (dominancia CNUF): na folha, se o fluxo otimo devolvido
     pelo oraculo deixa alguma instalacao aberta sem uso (throughput 0
     em todos os produtos), a folha e descartada sem atualizar o
     incumbente (a testemunha protegida — um otimo de cardinalidade
     minima — nunca e descartada).

Baterias:
  [A] 60 instancias aleatorias semeadas (|I|,|J| <= 5, |K| <= 4,
      |L| <= 2, valores <= 9);
  [B] 12 adversariais: 6 com custos fixos nulos (f=g=0; otimos com
      instalacoes de custo marginal zero — estresse maximo para P3) e
      6 com transporte nulo (c=d=0; a troca e 100%% custo fixo x
      cobertura — estresse para P1).

Para CADA instancia e CADA k in {0..n}:
  (1) forca bruta independente: OPT_k por enumeracao de TODOS os
      desenhos (oraculo MCMF do modulo comum, all_designs);
  (2) B&B com as tres podas == OPT_k  (as podas nunca descartam o otimo);
  (3) B&B sem podas (mesma arvore, so o corte estrutural |O| <= k)
      == OPT_k (sanidade da arvore);
  (4) contagem de nos com/sem podas (registro da reducao).

Custos de desenho memoizados por bitmask (a forca bruta e a fonte; o
B&B consome o mesmo oraculo, mas a IGUALDADE testada e contra o minimo
da enumeracao completa, que nao passa por nenhuma poda).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_mp_tscfl import (MinCostFlow, gen_instance,  # noqa: E402
                             demand_total)

INF = float("inf")
SEED0 = 9000


# ---------------------------------------------------------------------------
# Oraculo com extracao de uso (throughput por instalacao)
# ---------------------------------------------------------------------------

def eval_design(inst, y, z):
    """Roteia todos os produtos no desenho (y,z).

    Retorna (viavel, valor, usadaI, usadaJ):
      usadaI[i] = True sse a fabrica i transporta fluxo > 0 em ALGUM
      produto (arco S->F_i), no fluxo otimo devolvido; idem usadaJ[j]
      (arco Din_j->Dout_j).
    """
    nI, nJ, nK = inst["nI"], inst["nJ"], inst["nK"]
    usedI = [False] * nI
    usedJ = [False] * nJ
    total = 0
    for l in range(inst["nL"]):
        D = demand_total(inst, l)
        if D == 0:
            continue
        # rede identica a build_network do modulo comum, com handles de arco
        S = 0
        F = lambda i: 1 + i
        Din = lambda j: 1 + nI + j
        Dout = lambda j: 1 + nI + nJ + j
        C = lambda k: 1 + nI + 2 * nJ + k
        T = 1 + nI + 2 * nJ + nK
        BIG = 1 + sum(sum(r) for r in inst["b"]) \
                + sum(sum(r) for r in inst["p"]) \
                + sum(sum(r) for r in inst["q"])
        mc = MinCostFlow(T + 1)
        hI, hJ = [], []
        for i in range(nI):
            mc.add_edge(S, F(i), inst["b"][i][l] * y[i], 0)
            hI.append((mc.graph[S][-1], inst["b"][i][l] * y[i]))
        for i in range(nI):
            for j in range(nJ):
                mc.add_edge(F(i), Din(j), BIG, inst["c"][i][j][l])
        for j in range(nJ):
            mc.add_edge(Din(j), Dout(j), inst["p"][j][l] * z[j], 0)
            hJ.append((mc.graph[Din(j)][-1], inst["p"][j][l] * z[j]))
        for j in range(nJ):
            for k in range(nK):
                mc.add_edge(Dout(j), C(k), BIG, inst["d"][j][k][l])
        for k in range(nK):
            mc.add_edge(C(k), T, inst["q"][k][l], 0)
        sent, cost = mc.flow(S, T, D)
        if sent < D:
            return False, None, None, None
        total += cost
        for i in range(nI):
            if hI[i][1] - hI[i][0][1] > 0:      # cap original - residual
                usedI[i] = True
        for j in range(nJ):
            if hJ[j][1] - hJ[j][0][1] > 0:
                usedJ[j] = True
    return True, total, usedI, usedJ


class DesignCache:
    """Memoizacao por bitmask (yI | zJ << nI) de eval_design + custo total."""

    def __init__(self, inst):
        self.inst = inst
        self.nI, self.nJ = inst["nI"], inst["nJ"]
        self.memo = {}

    def get(self, ymask, zmask):
        key = (ymask, zmask)
        if key not in self.memo:
            y = [(ymask >> i) & 1 for i in range(self.nI)]
            z = [(zmask >> j) & 1 for j in range(self.nJ)]
            feas, rout, uI, uJ = eval_design(self.inst, y, z)
            if not feas:
                self.memo[key] = (False, None, 0, 0)
            else:
                fixed = sum(self.inst["f"][i] for i in range(self.nI)
                            if (ymask >> i) & 1)
                fixed += sum(self.inst["g"][j] for j in range(self.nJ)
                             if (zmask >> j) & 1)
                umI = sum(1 << i for i in range(self.nI) if uI[i])
                umJ = sum(1 << j for j in range(self.nJ) if uJ[j])
                self.memo[key] = (True, fixed + rout, umI, umJ)
        return self.memo[key]


# ---------------------------------------------------------------------------
# Forca bruta (fonte da verdade)
# ---------------------------------------------------------------------------

def brute_force(cache):
    """OPT_k para todo k, por enumeracao completa dos desenhos."""
    nI, nJ = cache.nI, cache.nJ
    n = nI + nJ
    opt = [INF] * (n + 1)
    for ymask in range(1 << nI):
        for zmask in range(1 << nJ):
            feas, cost, _, _ = cache.get(ymask, zmask)
            if not feas:
                continue
            card = bin(ymask).count("1") + bin(zmask).count("1")
            for k in range(card, n + 1):
                if cost < opt[k]:
                    opt[k] = cost
    return opt


# ---------------------------------------------------------------------------
# B&B do algoritmo do artigo
# ---------------------------------------------------------------------------

def covering_extra(caps_open, caps_free_sorted, D):
    """s = minimo de instalacoes livres (capacidades ja ordenadas desc)
    para caps_open + prefixo >= D; INF se impossivel (lema de contagem
    de cobertura)."""
    if caps_open >= D:
        return 0
    need = D - caps_open
    acc = 0
    for s, cap in enumerate(caps_free_sorted, start=1):
        acc += cap
        if acc >= need:
            return s
    return INF


def bnb(inst, cache, k, use_prunings):
    """Retorna (OPT_k, numero de nos visitados)."""
    nI, nJ, nL = inst["nI"], inst["nJ"], inst["nL"]
    n = nI + nJ
    best = [INF]
    nodes = [0]

    def rec(idx, ymask, zmask, nopen):
        nodes[0] += 1
        r = k - nopen
        if r < 0:
            return
        if idx == n:                                   # folha
            feas, cost, umI, umJ = cache.get(ymask, zmask)
            if not feas:
                return
            if use_prunings:
                # P3: instalacao aberta sem uso -> folha dominada
                if (ymask & ~umI) or (zmask & ~umJ):
                    return
            if cost < best[0]:
                best[0] = cost
            return
        if use_prunings:
            # decididos: 0..idx-1; livres: idx..n-1
            # P1 — poda de cobertura
            sI_max, sJ_max = 0, 0
            for l in range(nL):
                D = demand_total(inst, l)
                if D == 0:
                    continue
                capI = sum(inst["b"][i][l] for i in range(nI)
                           if (ymask >> i) & 1)
                freeI = sorted((inst["b"][i][l] for i in range(nI)
                                if i >= idx), reverse=True)
                sI = covering_extra(capI, freeI, D)
                capJ = sum(inst["p"][j][l] for j in range(nJ)
                           if (zmask >> j) & 1)
                freeJ = sorted((inst["p"][j][l] for j in range(nJ)
                                if nI + j >= idx), reverse=True)
                sJ = covering_extra(capJ, freeJ, D)
                sI_max = max(sI_max, sI)
                sJ_max = max(sJ_max, sJ)
            if sI_max + sJ_max > r:
                return
            # P2 — cota admissivel (abre O e todos os livres)
            upY = ymask | sum(1 << i for i in range(nI) if i >= idx)
            upZ = zmask | sum(1 << j for j in range(nJ) if nI + j >= idx)
            feas, _, _, _ = cache.get(upY, upZ)
            if not feas:
                return                                  # inviavel por monotonicidade
            # LB = custo fixo acumulado de O + v(tudo-livre-aberto)
            fixedO = sum(inst["f"][i] for i in range(nI) if (ymask >> i) & 1)
            fixedO += sum(inst["g"][j] for j in range(nJ) if (zmask >> j) & 1)
            up_cost = cache.get(upY, upZ)[1]            # fixo(up) + v(up)
            fixedUp = sum(inst["f"][i] for i in range(nI) if (upY >> i) & 1)
            fixedUp += sum(inst["g"][j] for j in range(nJ) if (upZ >> j) & 1)
            LB = fixedO + (up_cost - fixedUp)           # v(up) isolado
            if LB >= best[0]:
                return
        # ramifica: abre e fecha a instalacao idx
        if idx < nI:
            rec(idx + 1, ymask | (1 << idx), zmask, nopen + 1)
            rec(idx + 1, ymask, zmask, nopen)
        else:
            j = idx - nI
            rec(idx + 1, ymask, zmask | (1 << j), nopen + 1)
            rec(idx + 1, ymask, zmask, nopen)

    rec(0, 0, 0, 0)
    return best[0], nodes[0]


# ---------------------------------------------------------------------------
# Baterias
# ---------------------------------------------------------------------------

def make_instances():
    insts = []
    for t in range(60):                       # [A] aleatorias
        insts.append(("rand%02d" % t,
                      gen_instance(SEED0 + t, max_i=5, max_j=5,
                                   max_k=4, max_l=2, vmax=9)))
    for t in range(6):                        # [B] f = g = 0
        inst = gen_instance(SEED0 + 100 + t, max_i=5, max_j=5,
                            max_k=4, max_l=2, vmax=9)
        inst["f"] = [0] * inst["nI"]
        inst["g"] = [0] * inst["nJ"]
        insts.append(("zfix%02d" % t, inst))
    for t in range(6):                        # [B] c = d = 0
        inst = gen_instance(SEED0 + 200 + t, max_i=5, max_j=5,
                            max_k=4, max_l=2, vmax=9)
        inst["c"] = [[[0] * inst["nL"] for _ in range(inst["nJ"])]
                     for _ in range(inst["nI"])]
        inst["d"] = [[[0] * inst["nL"] for _ in range(inst["nK"])]
                     for _ in range(inst["nJ"])]
        insts.append(("ztrn%02d" % t, inst))
    return insts


def main():
    insts = make_instances()
    fails = 0
    checks = 0
    tot_nodes_plain = 0
    tot_nodes_pruned = 0

    for name, inst in insts:
        n = inst["nI"] + inst["nJ"]
        cache = DesignCache(inst)
        opt = brute_force(cache)
        for k in range(n + 1):
            v_pruned, nd_p = bnb(inst, cache, k, use_prunings=True)
            v_plain, nd_0 = bnb(inst, cache, k, use_prunings=False)
            tot_nodes_pruned += nd_p
            tot_nodes_plain += nd_0
            checks += 2
            if v_pruned != opt[k]:
                fails += 1
                print(f"FAIL {name} k={k}: B&B podado {v_pruned} "
                      f"!= brute {opt[k]}")
            if v_plain != opt[k]:
                fails += 1
                print(f"FAIL {name} k={k}: B&B sem podas {v_plain} "
                      f"!= brute {opt[k]}")

    red = 100.0 * (1.0 - tot_nodes_pruned / tot_nodes_plain)
    print(f"instancias: {len(insts)} (60 aleatorias + 6 f=g=0 + 6 c=d=0)")
    print(f"comparacoes B&B == forca bruta (todas as cardinalidades, "
          f"logo todos os orcamentos): {checks}; falhas: {fails}")
    print(f"nos visitados: sem podas {tot_nodes_plain}, "
          f"com podas P1+P2+P3 {tot_nodes_pruned} "
          f"(reducao {red:.1f}%)")
    print("RESULTADO GLOBAL:", "PASS" if fails == 0 else "FAIL")
    return fails


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
