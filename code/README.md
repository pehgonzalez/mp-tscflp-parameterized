# MP-TSCFLP — Xp reference solver

Reference implementation of Algorithm 1 of the paper (Section 5.1): the
XP branch-and-bound over facility subsets for MP-TSCFLP(B,k), with the
prunings P1 (covering counts, the covering-count lemma), P2 (accrued
fixed costs + all-open routing bound) and P3 (unused-facility dominance
at leaves), plus the safe preprocessing of Section 5.4 (customer
aggregation by identical d-columns; capacity capping b,p <= D_l). No
external dependencies; no Gurobi needed.

## File map

```
code/
  src/instance.hpp        PSC parser (long long, integrality + non-negativity enforced)
  src/mcmf.hpp            exact integer min-cost flow (SSP + Johnson potentials)
  src/solver_xp.hpp/.cpp  Algorithm 1: preprocessing, P1/P2/P3, DFS B&B, deadline
  src/main_xp.cpp         CLI + one-line output contract
  tools/generate_kstar.py deterministic generator with the k* knob (+ MANIFEST.csv/MD5)
                          (--nK/--nL take comma lists; --nIJ takes a:b pairs for nI != nJ)
  tools/kstar_check.py    independent per-product covering bounds k*_I(l)/k*_J(l),
                          cross-checked vs the solver's kstar= field
  tools/assemble_q1.py    joins one-line records + MANIFEST into results/q1_xp.csv
  tests/crossvalidate_xp.py  validation battery vs Python brute force
  run_xp_batch.ps1        campaign (Windows): Xp 60 s on instances_kstar + the 100 benchmark instances
  run_mip_batch.ps1       campaign (Windows): Mip/Bd via gurobi_port mptscfl.exe
  run_boundary600_par.sh  campaign (Linux): budget-600 boundary family, resumable, 2 workers
  gurobi_port/            Gurobi-based baselines (compact MIP and Benders), own README
  data/kstar/             pilot family (36 files + MANIFEST.csv)
  data/instances_kstar/   Q1 main grid (400 files + MANIFEST.csv)
  data/kstar_boundary/    Q1 boundary refinement (24 files + MANIFEST.csv)
  data/boundary600/       budget-600 boundary family (MANIFEST.csv; instances regenerable)
  data/BENCHMARK_MANIFEST.md5  MD5 of the 100 third-party benchmark instances (not redistributed here)
```

## Build

```
mkdir -p build && g++ -std=c++20 -O2 -o build/xp src/solver_xp.cpp src/main_xp.cpp
```

(Windows/MinGW: same line with `build\xp.exe`.)

## CLI contract

```
xp <instance.psc> [time_limit_s=3600] [k|-1]
```

* `k >= 0`: mode (a) — optimize under cardinality `sum y + sum z <= k`
  (MP-TSCFLP(B,k); sweep k externally for the k-profile).
* `k = -1`: mode (b) — plain optimization (k = n = |I|+|J|).

Exactly ONE line on stdout:

```
instance= status= obj= k_used= kstar= nodes= time= agg= timeout= p1= p2i= p2b= p3= rneg= cap=
```

* `status` in {OPTIMAL, INFEASIBLE, TIMEOUT}; `obj=inf` when no feasible
  design of cardinality <= k is known; on TIMEOUT `obj` is the incumbent.
* `kstar` = root covering lower bound `max_l k*_I(l) + max_l k*_J(l)`
  (root case of the covering-count lemma; the Q3 predictor). Always computed and printed,
  including on TIMEOUT/INFEASIBLE; `inf` if no covering design exists.
  Invariant under the preprocessing of Section 5.4 (re-checked by the
  test harness against the ORIGINAL instance).
* `k_used` = k after normalization `k <- min(k, n)` (step 1).
* `nodes` = B&B nodes visited; `p1/p2i/p2b/p3/rneg` = prunings fired
  (covering / all-open infeasible / bound / unused-facility leaf / r<0);
  `agg` = customers merged by aggregation; `cap` = capacity entries capped.
* `time` = wall-clock seconds (the only nondeterministic field; everything
  else is deterministic — checked by the test suite).

Deadline is checked every 4096 node visits AND after every oracle call
(the oracle dominates node cost, so the deadline is respected tightly).

## PSC instance format

Whitespace-separated integers: header `nI nJ nK nL`; K rows x L demands
`q[k][l]`; I rows `b_i1..b_iL f_i`; L blocks of IxJ stage-1 costs
`c[l][i][j]`; J rows `p_j1..p_jL g_j`; L blocks of JxK stage-2 costs
`d[l][j][k]`. Identical to `gurobi_port/src/instance.hpp`.

## Implementation notes

Faithful to steps 1–9. Two documented, correctness-preserving choices:

1. **Branching order** (step 3 allows any fixed order, factories first):
   within each side, elements are ordered by decreasing total capacity
   ("most-constrained first"), so P1 resolves covering questions early.
   The open-child is explored before the close-child (finds feasible
   incumbents fast). Completeness is order-independent (the correctness proof tracks a
   distinguished optimum through the search whatever the order).
2. **Middle-arc capacities** in the oracle network are tightened to
   `min(b_il, p_jl)` and `min(p_jl, q_kl)` instead of BIG: every feasible
   flow already satisfies these bounds through the capacity arcs, so the
   optimum is unchanged (routing values cross-validated exactly against
   the BIG-cap Python oracle).

Feasibility inside the oracle uses the aggregate tests F1/F2 of the
paper (exact under complete stages); the min-cost flow then certifies
the value.

## Generator (k* knob)

```
python3 tools/generate_kstar.py            # default grid -> data/kstar/
```

Default grid: nI = nJ in {10,15,20}, nK = 30, nL = 3, per-side targets
tI = tJ = t in {2,4,6,8}, seeds {1,2,3} -> 36 instances, k* = 2t in
{4,8,12,16}. Mechanism: on the binding product, t-1 facilities share total
capacity D-1 (large but insufficient, forcing >= t), the rest get moderate
capacities in [1, base] so the t-th prefix reaches D — hence k*_side = t
exactly; other products get capacities in [ceil(D_l/t), D_l] so they never
dominate the max. The generator recomputes k* on the finished instance and
asserts it equals the target; byte-identical output for the same
(params, seed); MANIFEST.csv carries MD5 per file.

## Validation (all green, 2026-07-10, g++ 12 / Linux; re-run 2026-07-24, g++ 13.3)

`python3 tests/crossvalidate_xp.py` — exit 0 iff everything passes:

* compile with `g++ -std=c++20 -O2 -Wall -Wextra`;
* **150 random small instances** (|I|,|J| <= 5, |K| <= 4, |L| <= 2, seeded)
  swept over ALL k in {0..n} plus k = -1, objective and feasibility
  compared EXACTLY against Python brute force over all designs
  (`verification/common_mp_tscfl.py` oracle) — 1207 comparisons;
* **30 adversarial instances** (8 zero costs, 8 tight capacities with
  side sums exactly D_l, 8 infeasible — 4 per side, 6 with q == 0), same
  sweep — total 1381 obj comparisons, 0 divergences;
* **kstar consistency** on every one of the 1381 runs vs the
  Python-computed covering bound;
* **determinism**: 12 instances run twice, outputs identical up to `time=`.

## Campaign scripts

`run_xp_batch.ps1` / `run_mip_batch.ps1` are Windows PowerShell loops
writing one-line records to `results\`, with the paths to xp.exe, to the
gurobi_port binary and to the benchmark data directory set as variables
at the top of each script (relative placeholders, adjust to the local
layout). `run_boundary600_par.sh` is the resumable Linux runner of the
budget-600 boundary campaign.
