#include "exact_model.hpp"

#include <iostream>
#include <sstream>

namespace mptscfl {

ExactModel::ExactModel(const Instance& inst) : p_(inst), env_(true), model_((env_.start(), env_)) {
    build();
}

void ExactModel::build() {
    const int I = p_.nfactories, J = p_.nwarehouses, K = p_.ncustomers, L = p_.ncommodities;

    y_.reserve(I);
    for (int i = 0; i < I; ++i) {
        std::ostringstream n; n << "y[" << i << "]";
        y_.push_back(model_.addVar(0, 1, p_.fixedcost_factory[i], GRB_BINARY, n.str()));
        y_.back().set(GRB_IntAttr_BranchPriority, 10); // ordpri = 10 in Exato::run
    }
    z_.reserve(J);
    for (int j = 0; j < J; ++j) {
        std::ostringstream n; n << "z[" << j << "]";
        z_.push_back(model_.addVar(0, 1, p_.fixedcost_warehouse[j], GRB_BINARY, n.str()));
        z_.back().set(GRB_IntAttr_BranchPriority, 10);
    }

    x_.assign(I, std::vector<std::vector<GRBVar>>(J, std::vector<GRBVar>(L)));
    for (int i = 0; i < I; ++i)
        for (int j = 0; j < J; ++j)
            for (int l = 0; l < L; ++l) {
                std::ostringstream n; n << "x[" << i << "," << j << "," << l << "]";
                x_[i][j][l] = model_.addVar(0, GRB_INFINITY, p_.flowcost_fw[l][i][j],
                                            GRB_CONTINUOUS, n.str());
            }
    w_.assign(J, std::vector<std::vector<GRBVar>>(K, std::vector<GRBVar>(L)));
    for (int j = 0; j < J; ++j)
        for (int k = 0; k < K; ++k)
            for (int l = 0; l < L; ++l) {
                std::ostringstream n; n << "w[" << j << "," << k << "," << l << "]";
                w_[j][k][l] = model_.addVar(0, GRB_INFINITY, p_.flowcost_wc[l][j][k],
                                            GRB_CONTINUOUS, n.str());
            }
    model_.set(GRB_IntAttr_ModelSense, GRB_MINIMIZE);

    // (3) demand:  sum_j w_jkl >= q_kl
    for (int k = 0; k < K; ++k)
        for (int l = 0; l < L; ++l) {
            GRBLinExpr e;
            for (int j = 0; j < J; ++j) e += w_[j][k][l];
            std::ostringstream n; n << "ClientDemand[" << k << "," << l << "]";
            model_.addConstr(e >= p_.customer_demand[k][l], n.str());
        }
    // (4) flow conservation:  sum_i x_ijl >= sum_k w_jkl
    for (int j = 0; j < J; ++j)
        for (int l = 0; l < L; ++l) {
            GRBLinExpr in, out;
            for (int i = 0; i < I; ++i) in += x_[i][j][l];
            for (int k = 0; k < K; ++k) out += w_[j][k][l];
            std::ostringstream n; n << "FlowConservation[" << j << "," << l << "]";
            model_.addConstr(in >= out, n.str());
        }
    // (5) factory capacity:  sum_j x_ijl <= b_il * y_i
    for (int i = 0; i < I; ++i)
        for (int l = 0; l < L; ++l) {
            GRBLinExpr e;
            for (int j = 0; j < J; ++j) e += x_[i][j][l];
            std::ostringstream n; n << "FCapacity[" << i << "," << l << "]";
            model_.addConstr(e <= p_.factory_capacity[i][l] * y_[i], n.str());
        }
    // (6) warehouse capacity:  sum_k w_jkl <= p_jl * z_j
    for (int j = 0; j < J; ++j)
        for (int l = 0; l < L; ++l) {
            GRBLinExpr e;
            for (int k = 0; k < K; ++k) e += w_[j][k][l];
            std::ostringstream n; n << "WCapacity[" << j << "," << l << "]";
            model_.addConstr(e <= p_.warehouse_capacity[j][l] * z_[j], n.str());
        }
}

void ExactModel::set_log(const std::string& path) {
    model_.set(GRB_StringParam_LogFile, path);
}

void ExactModel::set_seed(int seed) {
    model_.set(GRB_IntParam_Seed, seed);
}

void ExactModel::set_start(const std::vector<int>& ybar, const std::vector<int>& zbar) {
    for (int i = 0; i < p_.nfactories; ++i) y_[i].set(GRB_DoubleAttr_Start, ybar[i]);
    for (int j = 0; j < p_.nwarehouses; ++j) z_[j].set(GRB_DoubleAttr_Start, zbar[j]);
}

ExactResult ExactModel::run(double time_limit, double cutoff, int method, int threads) {
    if (method == 1)
        std::cerr << "[ExactModel] Gurobi has no automatic Benders; running plain B&B. "
                     "Manual branch-and-Benders-cut is Phase 1 of the roadmap.\n";
    model_.set(GRB_DoubleParam_TimeLimit, time_limit);
    model_.set(GRB_DoubleParam_NodefileStart, 4.0); // ~ NodeFileInd
    // Proof mode: the optimal value is integral (per-product flow LPs are min-cost
    // flows with integral data; see BENDERS.md Prop. 5), so absolute gap < 1 certifies
    // exact optimality. Gurobi's default MIPGap=1e-4 would declare OPTIMAL early.
    model_.set(GRB_DoubleParam_MIPGap, 0.0);
    model_.set(GRB_DoubleParam_MIPGapAbs, 0.9999);
    // Audit #11: always reset (params persist across run() calls on this model).
    model_.set(GRB_DoubleParam_Cutoff, cutoff > 0 ? cutoff : GRB_INFINITY);
    if (threads > 0) model_.set(GRB_IntParam_Threads, threads);

    model_.optimize();

    ExactResult r;
    const int st = model_.get(GRB_IntAttr_Status);
    const int nsol = model_.get(GRB_IntAttr_SolCount);
    if (st == GRB_INFEASIBLE) {
        r.status = Status::Infeasible;
        model_.computeIIS(); // ~ chkConflicts
        model_.write("mptscfl_iis.ilp");
        return r;
    }
    if (nsol == 0) {
        r.status = Status::NotFound;
        try { r.bound = model_.get(GRB_DoubleAttr_ObjBound); } catch (GRBException&) {}
        return r;
    }
    r.status = (st == GRB_OPTIMAL) ? Status::OptimalFound : Status::SolutionFound;
    r.obj = model_.get(GRB_DoubleAttr_ObjVal);
    r.bound = model_.get(GRB_DoubleAttr_ObjBound);
    r.gap = model_.get(GRB_DoubleAttr_MIPGap);
    r.runtime = model_.get(GRB_DoubleAttr_Runtime);
    r.y.resize(p_.nfactories);
    r.z.resize(p_.nwarehouses);
    for (int i = 0; i < p_.nfactories; ++i) r.y[i] = y_[i].get(GRB_DoubleAttr_X) > 0.5;
    for (int j = 0; j < p_.nwarehouses; ++j) r.z[j] = z_[j].get(GRB_DoubleAttr_X) > 0.5;
    return r;
}

ExactResult ExactModel::local_branching(const std::vector<int>& ybar, const std::vector<int>& zbar,
                                        int delta1, int delta2, double time_limit,
                                        double cutoff, int method) {
    GRBLinExpr sy, sz;
    for (int i = 0; i < p_.nfactories; ++i) sy += ybar[i] ? (1 - y_[i]) : y_[i];
    for (int j = 0; j < p_.nwarehouses; ++j) sz += zbar[j] ? (1 - z_[j]) : z_[j];
    GRBConstr c1 = model_.addConstr(sy <= delta1, "LB1");
    GRBConstr c2 = model_.addConstr(sz <= delta2, "LB2");
    set_start(ybar, zbar);
    ExactResult r = run(time_limit, cutoff, method);
    model_.remove(c1);
    model_.remove(c2);
    model_.update();
    return r;
}

} // namespace mptscfl
