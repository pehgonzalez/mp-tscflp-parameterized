"""
Verificacao computacional da Prop. A1.3 (monotonicidade).

Para >= 20 instancias aleatorias pequenas com semente fixa
(|I|,|J| <= 3 para manter a enumeracao de pares tratavel), calcula
v_l(y,z) para TODOS os desenhos (convencao: +infinito se inviavel) e
checa exaustivamente, para todo par comparavel (y,z) <= (y',z')
componente a componente e todo produto l:

    v_l(y', z') <= v_l(y, z)

o que inclui, em particular: (y,z) viavel => (y',z') viavel.
Valores por implementacao propria de MCMF com aritmetica inteira.

Revisao independente: pares reflexivos (y,z) = (y',z')
tornam a desigualdade tautologica; sao agora EXCLUIDOS da contagem de
checks (contados a parte, apenas para registro de cobertura).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common_mp_tscfl import gen_instance, routing_value, all_designs

INF = float("inf")


def main():
    n_instances = 20
    checks = fails = reflexive_skipped = 0

    for seed in range(201, 201 + n_instances):
        inst = gen_instance(seed, max_i=3, max_j=3)
        nI, nJ, nL = inst["nI"], inst["nJ"], inst["nL"]

        designs = list(all_designs(nI, nJ))
        values = {}
        for (y, z) in designs:
            key = (tuple(y), tuple(z))
            values[key] = []
            for l in range(nL):
                feas, v = routing_value(inst, l, y, z)
                values[key].append(v if feas else INF)

        for (y, z) in designs:
            for (y2, z2) in designs:
                if all(a <= b for a, b in zip(y, y2)) and \
                   all(a <= b for a, b in zip(z, z2)):
                    if (tuple(y), tuple(z)) == (tuple(y2), tuple(z2)):
                        # C1: par reflexivo -- desigualdade tautologica,
                        # nao conta como verificacao efetiva.
                        reflexive_skipped += nL
                        continue
                    va = values[(tuple(y), tuple(z))]
                    vb = values[(tuple(y2), tuple(z2))]
                    for l in range(nL):
                        checks += 1
                        if not (vb[l] <= va[l]):
                            fails += 1
                            print(f"[FAIL] seed={seed} l={l} "
                                  f"(y,z)={y},{z} v={va[l]} <= "
                                  f"(y',z')={y2},{z2} v'={vb[l]} VIOLADO")

    print(f"\nverify_A1_monotonicity: {n_instances} instancias, "
          f"{checks} desigualdades (par comparavel ESTRITO x produto): "
          f"{checks - fails} PASS, {fails} FAIL "
          f"({reflexive_skipped} checagens reflexivas tautologicas puladas)")
    if fails:
        sys.exit(1)
    print("TODOS OS TESTES PASSARAM")


if __name__ == "__main__":
    main()
