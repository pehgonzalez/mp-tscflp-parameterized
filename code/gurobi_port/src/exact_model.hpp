// Exact MIP for the MP-TSCFLP, model (1)-(10) of Mauri et al. (2021).
// Port of Old_Project/src/exato.* from CPLEX Concert to the Gurobi C++ API.
//
// CPLEX -> Gurobi mapping used here:
//   IloCplex::setPriorities            -> GRB_IntAttr_BranchPriority
//   IloCplex::CutUp                    -> GRB_DoubleParam_Cutoff
//   IloCplex::TiLim                    -> GRB_DoubleParam_TimeLimit
//   IloCplex::NodeFileInd = 3          -> GRB_DoubleParam_NodefileStart
//   refineConflict                     -> GRBModel::computeIIS
//   Benders::Strategy = BendersFull    -> NO Gurobi equivalent; the manual
//     branch-and-Benders-cut (lazy-constraint callback) is benders_model.*.
#ifndef MPTSCFL_EXACT_MODEL_HPP
#define MPTSCFL_EXACT_MODEL_HPP

#include <string>
#include <vector>

#include "gurobi_c++.h"
#include "instance.hpp"

namespace mptscfl {

struct ExactResult {
    Status status = Status::NotFound;
    double obj = 0.0;
    double bound = 0.0;
    double gap = 0.0;
    double runtime = 0.0;
    std::vector<int> y; // open factories
    std::vector<int> z; // open warehouses
};

class ExactModel {
public:
    explicit ExactModel(const Instance& inst);

    // MIP start (heuristic solution) for y/z; flows are completed by the solver.
    void set_start(const std::vector<int>& ybar, const std::vector<int>& zbar);

    // Full Gurobi log to file (survives console truncation).
    void set_log(const std::string& path);
    void set_seed(int seed);

    // method: 0 = plain branch-and-bound (Gurobi default cuts/heuristics on).
    //         1 = reserved for manual Benders (benders_model.*); currently falls back to 0
    //             with a warning, since Gurobi has no automatic Benders.
    ExactResult run(double time_limit, double cutoff = -1.0, int method = 0, int threads = 0);

    // Local branching around (ybar, zbar): adds the two Hamming-ball constraints of
    // Fischetti & Lodi (2003) exactly as in Exato::localbranching, solves, removes them.
    ExactResult local_branching(const std::vector<int>& ybar, const std::vector<int>& zbar,
                                int delta1, int delta2, double time_limit,
                                double cutoff = -1.0, int method = 0);

    GRBModel& model() { return model_; }

private:
    void build();

    const Instance& p_;
    GRBEnv env_;
    GRBModel model_;
    std::vector<GRBVar> y_, z_;
    std::vector<std::vector<std::vector<GRBVar>>> x_; // [i][j][l]
    std::vector<std::vector<std::vector<GRBVar>>> w_; // [j][k][l]
};

} // namespace mptscfl
#endif
