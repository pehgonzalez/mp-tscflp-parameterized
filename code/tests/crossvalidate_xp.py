#!/usr/bin/env python3
"""crossvalidate_xp.py — Phase B validation of the Xp solver (Algorithm A4.1).

Protocol (batteries mirroring verify_A4_xp_bb.py):
  1. compile xp with g++ -std=c++20 -O2;
  2. >= 150 seeded random small instances (|I|,|J| <= 5, |K| <= 4, |L| <= 2),
     sweep ALL cardinalities k in {0..n} plus plain mode (k = -1); compare
     obj/feasibility against Python brute force over all designs
     (common_mp_tscfl.py oracle);
  3. >= 30 adversarial instances: zero costs, tight capacities, infeasible
     (both sides), q == 0 — same sweep and comparison;
  4. determinism: run xp twice on a sample, outputs must be identical up to
     the time= field (wall-clock is the only nondeterministic datum);
  5. kstar consistency: xp's printed kstar must equal the Python-computed
     root covering bound (Lemma A4.1.1) on every run.

Exit code 0 iff every assertion passes.
"""

import os
import random
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
ROOT = os.path.dirname(CODE)
sys.path.insert(0, os.path.join(ROOT, "verification"))

from common_mp_tscfl import gen_instance, routing_value, all_designs, demand_total  # noqa: E402

BUILD = os.path.join(CODE, "build")
XP = os.path.join(BUILD, "xp")

N_RANDOM = 150
N_DETERMINISM = 12


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def compile_xp():
    os.makedirs(BUILD, exist_ok=True)
    cmd = ["g++", "-std=c++20", "-O2", "-Wall", "-Wextra",
           "-o", XP,
           os.path.join(CODE, "src", "solver_xp.cpp"),
           os.path.join(CODE, "src", "main_xp.cpp")]
    subprocess.run(cmd, check=True)
    print("[compile] ok:", " ".join(cmd))


def write_psc(inst, path):
    """PSC writer from the common_mp_tscfl dict layout (c[i][j][l], d[j][k][l])."""
    nI, nJ, nK, nL = inst["nI"], inst["nJ"], inst["nK"], inst["nL"]
    lines = [f"{nI} {nJ} {nK} {nL}"]
    for k in range(nK):
        lines.append(" ".join(str(inst["q"][k][l]) for l in range(nL)))
    for i in range(nI):
        lines.append(" ".join(str(inst["b"][i][l]) for l in range(nL)) + f" {inst['f'][i]}")
    for l in range(nL):
        for i in range(nI):
            lines.append(" ".join(str(inst["c"][i][j][l]) for j in range(nJ)))
    for j in range(nJ):
        lines.append(" ".join(str(inst["p"][j][l]) for l in range(nL)) + f" {inst['g'][j]}")
    for l in range(nL):
        for j in range(nJ):
            lines.append(" ".join(str(inst["d"][j][k][l]) for k in range(nK)))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def kstar_prefix(vals, D):
    if D <= 0:
        return 0
    acc = 0
    for s, v in enumerate(sorted(vals, reverse=True), start=1):
        acc += v
        if acc >= D:
            return s
    return float("inf")


def python_kstar(inst):
    """Root covering bound of Lemma A4.1.1 on the ORIGINAL instance
    (invariant under the A6 preprocessing)."""
    kI = kJ = 0
    for l in range(inst["nL"]):
        D = demand_total(inst, l)
        kI = max(kI, kstar_prefix([inst["b"][i][l] for i in range(inst["nI"])], D))
        kJ = max(kJ, kstar_prefix([inst["p"][j][l] for j in range(inst["nJ"])], D))
    return kI + kJ  # inf-propagating


def brute_force(inst):
    """cost table: list of (cardinality, cost) over feasible designs; and
    opt[k] = min cost over designs with cardinality <= k (inf if none)."""
    n = inst["nI"] + inst["nJ"]
    feas = []
    for y, z in all_designs(inst["nI"], inst["nJ"]):
        tot = sum(inst["f"][i] * y[i] for i in range(inst["nI"])) \
            + sum(inst["g"][j] * z[j] for j in range(inst["nJ"]))
        ok = True
        for l in range(inst["nL"]):
            fe, val = routing_value(inst, l, y, z)
            if not fe:
                ok = False
                break
            tot += val
        if ok:
            feas.append((sum(y) + sum(z), tot))
    opt = []
    for k in range(n + 1):
        cands = [c for card, c in feas if card <= k]
        opt.append(min(cands) if cands else float("inf"))
    return opt


def run_xp(path, k, timelimit=60):
    out = subprocess.run([XP, path, str(timelimit), str(k)],
                         capture_output=True, text=True, check=True).stdout.strip()
    fields = dict(tok.split("=", 1) for tok in out.split())
    return fields, out


def parse_obj(fields):
    return float("inf") if fields["obj"] == "inf" else int(fields["obj"])


def parse_kstar(fields):
    return float("inf") if fields["kstar"] == "inf" else int(fields["kstar"])


def check_instance(inst, tag, tmpdir, counters):
    path = os.path.join(tmpdir, f"{tag}.txt")
    write_psc(inst, path)
    n = inst["nI"] + inst["nJ"]
    opt = brute_force(inst)
    pk = python_kstar(inst)
    for k in list(range(n + 1)) + [-1]:
        fields, _ = run_xp(path, k)
        expected = opt[n] if k == -1 else opt[k]
        got = parse_obj(fields)
        assert fields["timeout"] == "0", f"{tag} k={k}: unexpected timeout"
        assert got == expected, \
            f"{tag} k={k}: xp obj={got} != brute force {expected}"
        st = fields["status"]
        assert (st == "INFEASIBLE") == (expected == float("inf")), \
            f"{tag} k={k}: status {st} vs expected obj {expected}"
        assert parse_kstar(fields) == pk, \
            f"{tag} k={k}: xp kstar={fields['kstar']} != python {pk}"
        ku = n if k == -1 else min(k, n)
        assert int(fields["k_used"]) == ku, f"{tag} k={k}: k_used mismatch"
        counters["comparisons"] += 1
    counters["instances"] += 1


# ---------------------------------------------------------------------------
# adversarial builders (common_mp_tscfl dict layout)
# ---------------------------------------------------------------------------

def adv_zero_costs(seed):
    inst = gen_instance(seed, max_i=4, max_j=4, max_k=4, max_l=2)
    inst["f"] = [0] * inst["nI"]
    inst["g"] = [0] * inst["nJ"]
    for i in range(inst["nI"]):
        for j in range(inst["nJ"]):
            for l in range(inst["nL"]):
                inst["c"][i][j][l] = 0
    for j in range(inst["nJ"]):
        for k in range(inst["nK"]):
            for l in range(inst["nL"]):
                inst["d"][j][k][l] = 0
    return inst


def _partition(rng, total, parts):
    """Deterministic split of `total` into `parts` non-negative integers."""
    cuts = sorted(rng.randint(0, total) for _ in range(parts - 1))
    vals = []
    prev = 0
    for c in cuts + [total]:
        vals.append(c - prev)
        prev = c
    return vals


def adv_tight(seed):
    """Capacities sum EXACTLY to D_l on both sides for every product."""
    rng = random.Random(f"tight|{seed}")
    inst = gen_instance(seed, max_i=3, max_j=3, max_k=3, max_l=2)
    for l in range(inst["nL"]):
        D = demand_total(inst, l)
        bi = _partition(rng, D, inst["nI"])
        pj = _partition(rng, D, inst["nJ"])
        for i in range(inst["nI"]):
            inst["b"][i][l] = bi[i]
        for j in range(inst["nJ"]):
            inst["p"][j][l] = pj[j]
    return inst


def adv_infeasible(seed, side):
    """Total capacity on one side strictly below D_l for some product."""
    inst = gen_instance(seed, max_i=3, max_j=3, max_k=3, max_l=2)
    # force positive demand on product 0
    if demand_total(inst, 0) == 0:
        inst["q"][0][0] = 5
    D = demand_total(inst, 0)
    if side == "I":
        short = _partition(random.Random(f"inf|{seed}"), max(D - 1, 0), inst["nI"])
        for i in range(inst["nI"]):
            inst["b"][i][0] = short[i]
    else:
        short = _partition(random.Random(f"inf|{seed}"), max(D - 1, 0), inst["nJ"])
        for j in range(inst["nJ"]):
            inst["p"][j][0] = short[j]
    return inst


def adv_zero_demand(seed):
    inst = gen_instance(seed, max_i=4, max_j=4, max_k=4, max_l=2)
    for k in range(inst["nK"]):
        for l in range(inst["nL"]):
            inst["q"][k][l] = 0
    return inst


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    compile_xp()
    counters = {"instances": 0, "comparisons": 0}
    with tempfile.TemporaryDirectory() as tmpdir:
        # battery 1: random small instances
        for seed in range(1, N_RANDOM + 1):
            inst = gen_instance(seed, max_i=5, max_j=5, max_k=4, max_l=2)
            check_instance(inst, f"rand{seed}", tmpdir, counters)
        n_rand = counters["instances"]
        print(f"[random] {n_rand} instances, {counters['comparisons']} comparisons ok")

        # battery 2: adversarial
        for s in range(1, 9):
            check_instance(adv_zero_costs(s), f"zerocost{s}", tmpdir, counters)
        for s in range(1, 9):
            check_instance(adv_tight(s), f"tight{s}", tmpdir, counters)
        for s in range(1, 5):
            check_instance(adv_infeasible(s, "I"), f"infeasI{s}", tmpdir, counters)
        for s in range(1, 5):
            check_instance(adv_infeasible(s, "J"), f"infeasJ{s}", tmpdir, counters)
        for s in range(1, 7):
            check_instance(adv_zero_demand(s), f"zeroq{s}", tmpdir, counters)
        n_adv = counters["instances"] - n_rand
        print(f"[adversarial] {n_adv} instances ok "
              f"(total comparisons {counters['comparisons']})")

        # battery 3: determinism (identical output modulo the time= field)
        for seed in range(1, N_DETERMINISM + 1):
            inst = gen_instance(1000 + seed, max_i=5, max_j=5, max_k=4, max_l=2)
            path = os.path.join(tmpdir, f"det{seed}.txt")
            write_psc(inst, path)
            _, out1 = run_xp(path, -1)
            _, out2 = run_xp(path, -1)
            strip = lambda s: " ".join(t for t in s.split() if not t.startswith("time="))
            assert strip(out1) == strip(out2), f"det{seed}: nondeterministic output"
        print(f"[determinism] {N_DETERMINISM} instances ok")

    print(f"ALL TESTS PASSED: {counters['instances']} instances, "
          f"{counters['comparisons']} obj/kstar comparisons, "
          f"{N_DETERMINISM} determinism checks")


if __name__ == "__main__":
    main()
