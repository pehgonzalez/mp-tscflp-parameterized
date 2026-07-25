# MP-TSCFLP — Gurobi-based baselines (Mip and Bd)

C++20/CMake codes for the exact baselines of the computational study,
built on Gurobi 13. The compact mixed-integer model (Mip, `lb_method 0`)
and the branch-and-Benders-cut code (Bd, `lb_method 1`, with a
Lagrangian-guided variant at `lb_method 2`) share the instance parser
and the exact flow oracle with the Xp solver one directory up.

## File map

```
gurobi_port/
  CMakeLists.txt          build (finds Gurobi via GUROBI_HOME)
  src/instance.hpp        PSC parser (identical contract to ../src/instance.hpp)
  src/mcmf.hpp            exact integer min-cost flow (SSP + Johnson potentials)
  src/flow_evaluator.hpp  routing oracle given (y,z), solver-independent
  src/exact_model.*       compact MIP with branching priorities, cutoff, MIP start
  src/benders_model.*     Benders master, disaggregated per-product optimality
                          cuts via lazy constraints, a-priori feasibility cuts,
                          root user cuts, optional Papadakos points
  src/lagrangian.*        Lagrangian bound, Polyak subgradient, primal heuristic
                          with exact routing
  src/single_stage_model.*  capacitated single-stage FLP submodel
  src/two_steps_solver.*  two-steps heuristic, merge, unused-facility cleanup,
                          warm start for the exact model
  src/main.cpp            CLI driver
  tests/test_core.cpp     solver-free tests: parser, MCMF, flow evaluator
  python/                 gurobipy mirrors and independent validators
  tools/extract_sp.py     rebuilds the single-product SP-TSCFLP instances
                          (Fernandes et al. 2014) from product 1
  data/bks_reference.csv  best-known values from the literature, source of
                          the certified-optimum comparisons
```

## Build and CLI

```bash
cmake -B build -DGUROBI_HOME=/opt/gurobi1300/linux64   # or the win64 layout
cmake --build build && ctest --test-dir build
build/mptscfl <instance.psc> <lb_method> <time_limit_s> [exact|two-steps] [seed] [threads]
```

Without a Gurobi installation only `test_core` is built, so the
solver-free core remains testable. The campaign drivers
`../run_mip_batch.ps1` and `../run_mip_mauri.ps1` consume the one-line
output contract, and the raw Gurobi logs of the published campaign are
in `../../results/logs_campanha/`.

## Validation record

- `test_core`: MCMF on a known network, evaluator on a hand-built
  instance, parse plus routing of a benchmark file. PASSED.
- `python/validate_bruteforce.py`: compact-MIP optimum equals exhaustive
  enumeration over all designs with an independent routing LP, seeded
  small instances. PASSED.
- C++/Gurobi cross-validation: the exact `FlowEvaluator` equals Gurobi's
  transportation LP value on every seed tried. PASSED.
- `python/validate_benders.py`: branch-and-Benders-cut equals the full
  MIP and the enumeration, with and without Papadakos, and the returned
  design re-evaluated by an independent LP. PASSED.
- `python/validate_lagrangian.py`: every Lagrangian bound at most the
  optimum, checked against the model value. PASSED.
