// CLI: mptscfl <instance> <lb_method> <time_limit> [mode] [seed] [threads]
//   lb_method: 0 = B&B, 1 = branch-and-Benders-cut, 2 = Lagrangian-guided Benders
//   mode:      "two-steps" (default) | "exact"
//   seed:      Gurobi Seed (default 0); threads: Gurobi Threads (default 0 = auto)
//
// Logging (audit-hardened schema, nothing lost if console truncates):
//   logs/<inst>_m<method>_<mode>_s<seed>_<timestamp>.gurobi.log — full solver log
//   logs/results.csv — one row per run; legacy-schema files are renamed aside.
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "benders_model.hpp"
#include "exact_model.hpp"
#include "flow_evaluator.hpp"
#include "instance.hpp"
#include "lagrangian.hpp"
#include "two_steps_solver.hpp"

using namespace mptscfl;

static const char* CSV_HEADER =
    "datetime,instance,mode,method,seed,threads,time_limit_s,status,obj,bound,gap,"
    "solver_time_s,total_wall_s,lag_lb,lag_ub,lag_time_s,benders_cuts,"
    "heuristic_cost,verified_cost,verified_ok,gurobi_version,gurobi_log";

static void usage(const char* prog) {
    std::cout << "Usage: " << prog
              << " <instance> <lb_method> <time_limit> [mode] [seed] [threads]\n"
              << "  lb_method: 0 = B&B, 1 = branch-and-Benders-cut,\n"
              << "             2 = Lagrangian-guided Benders (docs/LAGRANGIAN.md)\n"
              << "  mode: two-steps (default) | exact;  seed: default 0;  threads: 0=auto\n";
}

static std::string status_str(Status s) {
    switch (s) {
        case Status::OptimalFound: return "OPTIMAL";
        case Status::SolutionFound: return "FEASIBLE";
        case Status::Infeasible: return "INFEASIBLE";
        default: return "NOTFOUND";
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) { usage(argv[0]); return 1; }
    const std::string datafile = argv[1];
    const int method = std::atoi(argv[2]);
    const double time_limit = std::atof(argv[3]);
    const std::string mode = (argc > 4) ? argv[4] : "two-steps";
    const int seed = (argc > 5) ? std::atoi(argv[5]) : 0;
    const int threads = (argc > 6) ? std::atoi(argv[6]) : 0;
    const bool papadakos = (argc > 7) && std::atoi(argv[7]) != 0; // ablação
    // Qualquer modo iniciado por "exact" é exato (permite rótulos como "exact-pap"
    // que distinguem configs de ablação no CSV sem coluna nova).
    const bool exact_mode = mode.rfind("exact", 0) == 0;

    Instance inst;
    try {
        inst.load_file(datafile);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    std::cout << "Instancia carregada - " << datafile << " (" << inst.nfactories << "x"
              << inst.nwarehouses << "x" << inst.ncustomers << ", L=" << inst.ncommodities
              << ") seed=" << seed << " threads=" << threads << "\n";

    const std::string stem = std::filesystem::path(datafile).stem().string();
    std::time_t tt = std::time(nullptr);
    char ts[32];
    std::strftime(ts, sizeof(ts), "%Y%m%d-%H%M%S", std::localtime(&tt));
    std::filesystem::create_directories("logs");
    std::ostringstream basen;
    basen << "logs/" << stem << "_m" << method << "_" << mode << "_s" << seed << "_" << ts;
    const std::string gurobilog = basen.str() + ".gurobi.log";
    std::cout << "Gurobi log: " << gurobilog << "\n";

    std::ostringstream gv;
    gv << GRB_VERSION_MAJOR << "." << GRB_VERSION_MINOR << "." << GRB_VERSION_TECHNICAL;

    const auto wall0 = std::chrono::steady_clock::now();
    // Remaining TOTAL budget (fix for observed overrun: model-construction time — the
    // Lagrangian B model, the Benders subproblem LPs and the all-open theta bounds —
    // was not discounted from any phase; now every solve gets wall-clock remainder).
    auto remaining = [&](double floor_s = 1.0) {
        const double spent =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
        return std::max(floor_s, time_limit - spent);
    };
    try {
        ExactResult res;
        double heuristic_cost = -1.0;
        long long cuts = 0;
        double lag_lb = -1.0, lag_ub = -1.0, lag_time = -1.0;
        std::vector<int> heur_y, heur_z; // for CUTOFF promotion (audit #8)

        if (method == 2) {
            const double lag_budget = std::min(0.15 * time_limit, 600.0);
            LagrangianSolver lag(inst);
            LagrangianResult ld = lag.solve(/*iters=*/1000, lag_budget);
            lag_lb = ld.best_lb > -1e99 ? ld.best_lb : -1.0;
            lag_ub = ld.has_solution ? ld.best_ub : -1.0;
            lag_time = ld.runtime;
            BendersOptions bo;
            bo.threads = threads;
            bo.papadakos = papadakos;
            BendersModel benders(inst, bo);
            benders.set_log(gurobilog);
            benders.set_seed(seed);
            if (ld.best_lb > -1e99) benders.add_global_lower_bound(ld.best_lb);
            double cutoff = -1.0;
            if (ld.has_solution) {
                heuristic_cost = ld.best_ub;
                heur_y = ld.y;
                heur_z = ld.z;
                benders.set_start(ld.y, ld.z);
                cutoff = ld.best_ub + 0.499; // integral optimum: noise-robust (audit #4)
            }
            res = benders.run(remaining(), cutoff);
            cuts = benders.cuts_added();
        } else if (method == 1) {
            BendersOptions bo;
            bo.threads = threads;
            bo.papadakos = papadakos;
            BendersModel benders(inst, bo);
            benders.set_log(gurobilog);
            benders.set_seed(seed);
            if (!exact_mode) {
                TwoStepsSolver tss(inst);
                auto h = tss.solve(time_limit, /*method=*/0, /*run_exact=*/false);
                if (h.heuristic_feasible) {
                    heuristic_cost = h.heuristic_cost;
                    heur_y = h.y;
                    heur_z = h.z;
                    benders.set_start(h.y, h.z);
                    res = benders.run(remaining(), h.heuristic_cost + 0.499);
                } else {
                    res = benders.run(remaining());
                }
            } else {
                res = benders.run(remaining());
            }
            cuts = benders.cuts_added();
        } else if (exact_mode) {
            ExactModel exact(inst);
            exact.set_log(gurobilog);
            exact.set_seed(seed);
            res = exact.run(remaining(), -1.0, method, threads);
        } else {
            TwoStepsSolver tss(inst);
            auto h = tss.solve(time_limit, method, /*run_exact=*/true, gurobilog);
            heuristic_cost = h.heuristic_feasible ? h.heuristic_cost : -1.0;
            if (h.heuristic_feasible) { heur_y = h.y; heur_z = h.z; }
            res = h.exact;
        }

        // Audit #8: CUTOFF/timeout without incumbent but with a certified heuristic UB
        // and bound >= UB - 0.5 proves the heuristic solution optimal (integral value).
        if (res.status == Status::NotFound && heuristic_cost > 0 && !heur_y.empty() &&
            res.bound >= heuristic_cost - 0.5 && res.bound > 0) {
            res.status = Status::OptimalFound;
            res.obj = heuristic_cost;
            res.y = heur_y;
            res.z = heur_z;
            std::cout << "[main] heuristic UB certified optimal by dual bound "
                      << res.bound << "\n";
        } else if (res.status == Status::NotFound && heuristic_cost > 0 &&
                   !heur_y.empty()) {
            // Campaign-1 finding: with an active cutoff the master may end without
            // any incumbent even though the heuristic solution is the best known.
            // Report it as the incumbent instead of swallowing it as NOTFOUND.
            res.status = Status::SolutionFound;
            res.obj = heuristic_cost;
            res.y = heur_y;
            res.z = heur_z;
            if (res.bound > 0) res.gap = (res.obj - res.bound) / std::abs(res.obj);
            std::cout << "[main] no improving incumbent; reporting heuristic UB "
                      << heuristic_cost << " (bound " << res.bound << ")\n";
        }

        // Independent certification (solver-free re-routing of (y,z)).
        double verified = -1.0;
        bool verified_ok = false;
        if (!res.y.empty()) {
            FlowEvaluator ev(inst);
            FlowResult fr = ev.evaluate(res.y, res.z);
            if (fr.feasible) {
                verified = ev.solution_cost(res.y, res.z, fr);
                verified_ok = true;
            }
        }
        const double total_wall =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();

        std::cout << std::fixed << std::setprecision(2)
                  << "status=" << status_str(res.status) << " obj=" << res.obj
                  << " bound=" << res.bound << " gap=" << std::setprecision(6) << res.gap
                  << std::setprecision(2) << " solver_time=" << res.runtime
                  << " total_wall=" << total_wall << "\n";
        if (!res.y.empty())
            std::cout << "verified_cost="
                      << (verified_ok ? std::to_string(verified) : "INFEASIBLE (bug!)")
                      << "\n";

        // Summary row (audit #1 schema). Legacy-schema CSV is moved aside once.
        const std::string csvpath = "logs/results.csv";
        if (std::filesystem::exists(csvpath)) {
            std::ifstream in(csvpath);
            std::string first;
            std::getline(in, first);
            in.close();
            if (first != CSV_HEADER)
                std::filesystem::rename(csvpath, "logs/results_legacy.csv");
        }
        const bool fresh = !std::filesystem::exists(csvpath);
        std::ofstream csv(csvpath, std::ios::app);
        if (fresh) csv << CSV_HEADER << "\n";
        csv << ts << "," << stem << "," << mode << "," << method << "," << seed << ","
            << threads << "," << time_limit << "," << status_str(res.status) << ","
            << std::fixed << std::setprecision(2) << res.obj << "," << res.bound << ","
            << std::setprecision(8) << res.gap << std::setprecision(2) << ","
            << res.runtime << "," << total_wall << "," << lag_lb << "," << lag_ub << ","
            << lag_time << "," << cuts << "," << heuristic_cost << ","
            << (verified_ok ? verified : -1.0) << "," << (verified_ok ? 1 : 0) << ","
            << gv.str() << "," << gurobilog << "\n";
        std::cout << "Resumo acrescentado em " << csvpath << "\n";
    } catch (GRBException& e) {
        std::cerr << "Gurobi error " << e.getErrorCode() << ": " << e.getMessage() << "\n";
        return 2;
    }
    return 0;
}
