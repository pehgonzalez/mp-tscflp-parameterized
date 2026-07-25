# MP-TSCFLP — A Parameterized Complexity Analysis

Code, data, verification scripts and raw results accompanying the paper
on the parameterized complexity of the Multiproduct Two-Stage
Capacitated Facility Location Problem (MP-TSCFLP). Every number, table
and figure in the manuscript can be regenerated from this repository,
published at https://github.com/pehgonzalez/mp-tscflp-parameterized.

## Layout

    paper/          LaTeX sources of the manuscript
    code/           C++20 Xp solver (src/), instance generator (tools/),
                    cross-validation (tests/), Gurobi baselines (gurobi_port/),
                    campaign scripts, instance manifests (data/)
    verification/   one Python script per theoretical result, brute force
                    against each construction
    scripts/        regeneration of every published table and figure
    results/        raw campaign records (CSV) and the Gurobi logs of the
                    published campaign (logs_campanha/)

## Environment

The published campaigns ran with Gurobi 13.0.2 (deterministic settings
and seeds recorded in the paper and in the campaign scripts) on the
hardware documented in the manuscript. The verification suite and the
regeneration scripts need only Python 3 with matplotlib, plus g++ 12 or
later for the solver.

## Reproducing the results

1. Theoretical verification. Each script under `verification/` checks
   one construction of the paper by exhaustive comparison against brute
   force and prints its pass count, `python3 verification/verify_A1_oracle.py`
   and so on. All scripts exit 0. The A1 to A6 filename prefixes group
   the scripts by result family, structural facts, covering reductions,
   product and partition constructions, algorithms and bounds, numeric
   layer, and kernelization.
2. Solver cross-validation. `python3 code/tests/crossvalidate_xp.py`
   compiles the Xp solver, sweeps 180 seeded instances over every
   cardinality against a Python brute force (1,381 comparisons) and
   checks determinism.
3. Tables and figures. `python3 scripts/make_paper_tables.py` and
   `python3 scripts/make_paper_figures.py` regenerate every result table
   and figure of the paper from the raw CSVs in `results/`. The outputs
   are deterministic, so a diff against the committed versions is empty
   apart from PDF timestamps.
4. Statistical analysis. `python3 scripts/analyze_q3_correlation.py`
   recomputes the Q3 correlations with pinned seeds, and
   `python3 scripts/extract_rootgap.py` re-extracts the root-relaxation
   data of the mediation analysis from the raw logs in
   `results/logs_campanha/`.
5. Campaigns. `code/run_xp_batch.ps1` and `code/run_mip_batch.ps1`
   rerun the Windows campaigns (paths set as variables at the top), and
   `code/run_boundary600_par.sh` is the resumable Linux runner of the
   budget-600 boundary family. Instances of the generated families are
   reproduced byte-identically by `code/tools/generate_kstar.py` and
   checked against the MD5 manifests in `code/data/`.

## Data

The generated instance families are fully determined by the generator
and the MANIFEST files in `code/data/`. The 100 benchmark instances of
Mauri and coauthors are third-party data and are not redistributed
here, `code/data/BENCHMARK_MANIFEST.md5` carries their MD5 checksums so
a local copy can be verified before a rerun.

## Building the solver

    g++ -std=c++20 -O2 -o code/build/xp code/src/solver_xp.cpp code/src/main_xp.cpp

The Gurobi baselines build with CMake, see `code/gurobi_port/README.md`.

## Building the paper

    cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
