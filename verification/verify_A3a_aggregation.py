#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_A3a_aggregation.py — Verificacao computacional do Lema de agregacao
da Observacao A3a.5: com |J| = 1, os clientes colapsam.

Lema (enunciado verificado): seja P uma instancia com |J| = 1 e
Delta := sum_{k,l} d_{1kl} * q_{kl}. Seja R a instancia com K' = {1},
q'_{1l} := D_l = sum_k q_{kl}, d'_{11l} := 0, todo o resto inalterado.
Entao P e viavel <=> R e viavel, e OPT(P) = OPT(R) + Delta.

Verificacao por forca bruta TOTAL dos dois lados: enumeracao de todos os
desenhos (y,z) (common_mp_tscfl.all_designs) com roteamento exato pelo MCMF
inteiro (common_mp_tscfl.routing_value, Prop. A1.1) — nenhuma forma fechada.

Baterias:
  (A) >= 40 instancias semeadas com |J| = |L| = 1, |K| <= 4, |I| <= 4,
      dados gerais (f, g, c, d, b, p, q aleatorios) — a celula da
      Observacao A3a.5; inclui casos inviaveis e com demanda nula.
  (B) >= 20 instancias com |J| = 1, |L| = 2 (o lema vale para |L| geral;
      a agregacao e por produto).
Saida: contagens (viaveis / inviaveis / D=0) e PASS/FAIL.
"""

import random
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_mp_tscfl import all_designs, routing_value


def brute_force_mp_opt(inst):
    """OPT por forca bruta (todos os desenhos, MCMF exato por produto)."""
    best = None
    for y, z in all_designs(inst["nI"], inst["nJ"]):
        total = sum(inst["f"][i] * y[i] for i in range(inst["nI"])) \
              + sum(inst["g"][j] * z[j] for j in range(inst["nJ"]))
        feas = True
        for l in range(inst["nL"]):
            ok, val = routing_value(inst, l, y, z)
            if not ok:
                feas = False
                break
            total += val
        if feas and (best is None or total < best):
            best = total
    return best


def gen_singleJ_instance(seed, n_l):
    """Instancia aleatoria com |J| = 1, |L| = n_l, |K| <= 4, |I| <= 4."""
    rng = random.Random(seed)
    nI = rng.randint(1, 4)
    nK = rng.randint(1, 4)
    inst = {
        "nI": nI, "nJ": 1, "nK": nK, "nL": n_l,
        "f": [rng.randint(0, 9) for _ in range(nI)],
        "g": [rng.randint(0, 9)],
        "c": [[[rng.randint(0, 9) for _ in range(n_l)]] for _ in range(nI)],
        "d": [[[rng.randint(0, 9) for _ in range(n_l)] for _ in range(nK)]],
        "b": [[rng.randint(0, 12) for _ in range(n_l)] for _ in range(nI)],
        "p": [[rng.randint(0, 20) for _ in range(n_l)]],
        "q": [[rng.randint(0, 3) for _ in range(n_l)] for _ in range(nK)],
    }
    for l in range(n_l):
        if rng.random() < 0.15:  # produto sem demanda (caso de borda)
            for k in range(nK):
                inst["q"][k][l] = 0
    return inst


def aggregate(inst):
    """Constroi (R, Delta) do lema."""
    nK, nL = inst["nK"], inst["nL"]
    delta = sum(inst["d"][0][k][l] * inst["q"][k][l]
                for k in range(nK) for l in range(nL))
    D = [sum(inst["q"][k][l] for k in range(nK)) for l in range(nL)]
    red = dict(inst)
    red["nK"] = 1
    red["q"] = [D]
    red["d"] = [[[0] * nL]]
    return red, delta


def run_battery(label, count, seed0, n_l, failures):
    stats = {"feas": 0, "infeas": 0, "D0": 0}
    for s in range(count):
        inst = gen_singleJ_instance(seed0 + s, n_l)
        red, delta = aggregate(inst)
        opt_p = brute_force_mp_opt(inst)
        opt_r = brute_force_mp_opt(red)
        if all(q == 0 for row in inst["q"] for q in row):
            stats["D0"] += 1
        elif opt_p is None:
            stats["infeas"] += 1
        else:
            stats["feas"] += 1
        if (opt_p is None) != (opt_r is None):
            failures.append((inst, "viabilidade divergente: P=%s R=%s"
                             % (opt_p, opt_r)))
        elif opt_p is not None and opt_p != opt_r + delta:
            failures.append((inst, "OPT(P)=%s != OPT(R)+Delta=%s+%s"
                             % (opt_p, opt_r, delta)))
    print("[%s] %d instancias (sementes %d..%d, |L|=%d): %d viaveis, "
          "%d inviaveis, %d com demanda nula"
          % (label, count, seed0, seed0 + count - 1, n_l,
             stats["feas"], stats["infeas"], stats["D0"]))


def main():
    failures = []
    run_battery("A", 40, 500, 1, failures)   # celula da Obs. A3a.5
    run_battery("B", 20, 700, 2, failures)   # lema em |L| geral
    if failures:
        print("\nFALHAS (%d):" % len(failures))
        for f in failures[:10]:
            print("  %s : %s" % (f[0], f[1]))
        sys.exit(1)
    print("\nTODOS OS TESTES PASSARAM.")


if __name__ == "__main__":
    main()
