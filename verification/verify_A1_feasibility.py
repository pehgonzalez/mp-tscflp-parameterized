"""
Verificacao computacional da caracterizacao de viabilidade do artigo.

Para >= 30 instancias aleatorias pequenas com semente fixa, percorre
EXAUSTIVAMENTE todos os desenhos (y,z) em {0,1}^{|I|} x {0,1}^{|J|}
(ate 256 por instancia) e checa a equivalencia

   [ para todo l: sum_i b_il y_i >= D_l  E  sum_j p_jl z_j >= D_l ]
                        <=>
   [ para todo l: fluxo maximo S-T em N_l(y,z) >= D_l ]

onde o fluxo maximo e calculado por caminhos aumentantes com
aritmetica inteira exata (implementacao propria, sem PL).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common_mp_tscfl import (gen_instance, aggregate_condition,
                             flow_feasible, all_designs)


def main():
    n_instances = 30
    checks = fails = 0
    n_true = n_false = 0

    for seed in range(101, 101 + n_instances):
        inst = gen_instance(seed)
        for (y, z) in all_designs(inst["nI"], inst["nJ"]):
            checks += 1
            cond = aggregate_condition(inst, y, z)
            real = flow_feasible(inst, y, z)
            if cond != real:
                fails += 1
                print(f"[FAIL] seed={seed} y={y} z={z}: "
                      f"condicao_agregada={cond} viabilidade_fluxo={real}")
            if real:
                n_true += 1
            else:
                n_false += 1

    print(f"\nverify_A1_feasibility: {n_instances} instancias, "
          f"{checks} desenhos (y,z) checados exaustivamente: "
          f"{checks - fails} PASS, {fails} FAIL "
          f"({n_true} viaveis, {n_false} inviaveis)")
    if fails:
        sys.exit(1)
    print("TODOS OS TESTES PASSARAM")


if __name__ == "__main__":
    main()
