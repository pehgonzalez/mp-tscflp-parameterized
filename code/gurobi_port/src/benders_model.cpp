#include "benders_model.hpp"

#include <cmath>
#include <iostream>
#include <sstream>

namespace mptscfl {

// ---------------------------------------------------------------- subproblem
ProductSubproblem::ProductSubproblem(const Instance& inst, int l, GRBEnv& env)
    : p_(inst), l_(l), lp_(env) {
    const int I = p_.nfactories, J = p_.nwarehouses, K = p_.ncustomers;
    lp_.set(GRB_IntParam_OutputFlag, 0);

    std::vector<std::vector<GRBVar>> x(I, std::vector<GRBVar>(J));
    std::vector<std::vector<GRBVar>> w(J, std::vector<GRBVar>(K));
    for (int i = 0; i < I; ++i)
        for (int j = 0; j < J; ++j)
            x[i][j] = lp_.addVar(0, GRB_INFINITY, p_.flowcost_fw[l_][i][j], GRB_CONTINUOUS,
                                 "x");
    for (int j = 0; j < J; ++j)
        for (int k = 0; k < K; ++k)
            w[j][k] = lp_.addVar(0, GRB_INFINITY, p_.flowcost_wc[l_][j][k], GRB_CONTINUOUS,
                                 "w");
    lp_.set(GRB_IntAttr_ModelSense, GRB_MINIMIZE);

    dem_.reserve(K);
    for (int k = 0; k < K; ++k) {
        GRBLinExpr e;
        for (int j = 0; j < J; ++j) e += w[j][k];
        dem_.push_back(lp_.addConstr(e >= p_.customer_demand[k][l_], "dem"));
    }
    for (int j = 0; j < J; ++j) {
        GRBLinExpr in, out;
        for (int i = 0; i < I; ++i) in += x[i][j];
        for (int k = 0; k < K; ++k) out += w[j][k];
        lp_.addConstr(in - out >= 0.0, "cons");
    }
    fcap_.reserve(I);
    for (int i = 0; i < I; ++i) {
        GRBLinExpr e;
        for (int j = 0; j < J; ++j) e += x[i][j];
        fcap_.push_back(lp_.addConstr(e <= 0.0, "fcap")); // RHS set per call
    }
    wcap_.reserve(J);
    for (int j = 0; j < J; ++j) {
        GRBLinExpr e;
        for (int k = 0; k < K; ++k) e += w[j][k];
        wcap_.push_back(lp_.addConstr(e <= 0.0, "wcap"));
    }
}

BendersCut ProductSubproblem::solve(const std::vector<double>& yv,
                                    const std::vector<double>& zv) {
    const int I = p_.nfactories, J = p_.nwarehouses, K = p_.ncustomers;
    for (int i = 0; i < I; ++i)
        fcap_[i].set(GRB_DoubleAttr_RHS, p_.factory_capacity[i][l_] * yv[i]);
    for (int j = 0; j < J; ++j)
        wcap_[j].set(GRB_DoubleAttr_RHS, p_.warehouse_capacity[j][l_] * zv[j]);
    lp_.optimize();

    BendersCut cut;
    if (lp_.get(GRB_IntAttr_Status) != GRB_OPTIMAL) return cut; // infeasible safeguard
    cut.feasible = true;
    cut.value = lp_.get(GRB_DoubleAttr_ObjVal);
    for (int k = 0; k < K; ++k)
        cut.constant += p_.customer_demand[k][l_] * dem_[k].get(GRB_DoubleAttr_Pi);
    cut.fac_coef.resize(I);
    for (int i = 0; i < I; ++i)
        cut.fac_coef[i] = p_.factory_capacity[i][l_] * fcap_[i].get(GRB_DoubleAttr_Pi);
    cut.ware_coef.resize(J);
    for (int j = 0; j < J; ++j)
        cut.ware_coef[j] = p_.warehouse_capacity[j][l_] * wcap_[j].get(GRB_DoubleAttr_Pi);
    return cut;
}

// ----------------------------------------------------------------- callback
class BendersCallback : public GRBCallback {
public:
    explicit BendersCallback(BendersModel& o) : o_(o) {}

protected:
    void callback() override {
        // A swallowed exception would silently ACCEPT the incumbent (no lazy
        // cut added). Any failure must abort the solve and poison the result.
        try {
            if (where == GRB_CB_MIPSOL) {
                separate(/*integer=*/true);
            } else if (where == GRB_CB_MIPNODE &&
                       getIntInfo(GRB_CB_MIPNODE_STATUS) == GRB_OPTIMAL) {
                inject_pending(); // theta-repair (any node)
                if (o_.opt_.root_cuts && getDoubleInfo(GRB_CB_MIPNODE_NODCNT) < 0.5)
                    separate(/*integer=*/false);
            }
        } catch (GRBException& e) {
            std::cerr << "[Benders callback] FATAL: " << e.getMessage() << "\n";
            o_.callback_failed_ = true;
            abort();
        } catch (...) {
            std::cerr << "[Benders callback] FATAL: unknown exception\n";
            o_.callback_failed_ = true;
            abort();
        }
    }

private:
    // Sparsified cut with VALIDITY-PRESERVING compensation:
    // coefficients are <= 0, so moving a dropped term into the constant (its value
    // at y=1) only weakens the cut: coef*y >= coef for y in [0,1].
    GRBLinExpr cut_expr(const BendersCut& c) {
        double cst = c.constant;
        GRBLinExpr e;
        for (int i = 0; i < (int)c.fac_coef.size(); ++i) {
            if (std::abs(c.fac_coef[i]) > o_.opt_.eps) e += c.fac_coef[i] * o_.y_[i];
            else cst += c.fac_coef[i];
        }
        for (int j = 0; j < (int)c.ware_coef.size(); ++j) {
            if (std::abs(c.ware_coef[j]) > o_.opt_.eps) e += c.ware_coef[j] * o_.z_[j];
            else cst += c.ware_coef[j];
        }
        return e + cst;
    }

    // theta-repair injection: post the stored solution with theta_l = v_l exact.
    void inject_pending() {
        if (!o_.has_pending_) return;
        const int I = o_.p_.nfactories, J = o_.p_.nwarehouses, L = o_.p_.ncommodities;
        for (int i = 0; i < I; ++i) setSolution(o_.y_[i], o_.pend_y_[i]);
        for (int j = 0; j < J; ++j) setSolution(o_.z_[j], o_.pend_z_[j]);
        for (int l = 0; l < L; ++l) setSolution(o_.theta_[l], o_.pend_theta_[l]);
        useSolution();
        o_.has_pending_ = false;
    }

    void separate(bool integer) {
        const int I = o_.p_.nfactories, J = o_.p_.nwarehouses, L = o_.p_.ncommodities;
        std::vector<double> yv(I), zv(J), tv(L);
        for (int i = 0; i < I; ++i)
            yv[i] = integer ? std::round(getSolution(o_.y_[i])) : getNodeRel(o_.y_[i]);
        for (int j = 0; j < J; ++j)
            zv[j] = integer ? std::round(getSolution(o_.z_[j])) : getNodeRel(o_.z_[j]);
        for (int l = 0; l < L; ++l)
            tv[l] = integer ? getSolution(o_.theta_[l]) : getNodeRel(o_.theta_[l]);

        std::vector<double> vl(L, -1.0); // exact subproblem values (for theta-repair)
        bool cut_added = false;
        for (int l = 0; l < L; ++l) {
            BendersCut c = o_.sp_[l]->solve(yv, zv);
            if (!c.feasible) { // impossible under (F1),(F2); combinatorial safeguard
                if (integer) {
                    GRBLinExpr e;
                    for (int i = 0; i < I; ++i) if (yv[i] < 0.5) e += o_.y_[i];
                    for (int j = 0; j < J; ++j) if (zv[j] < 0.5) e += o_.z_[j];
                    addLazy(e >= 1.0);
                    ++o_.ncuts_;
                    cut_added = true; // never schedule theta-repair here
                }
                continue;
            }
            vl[l] = c.value;
            // Relative tolerance alone allows theta to sit up to
            // eps*|v_l| BELOW v_l; summed over products this can exceed the absolute
            // gap-1 certificate. Cap the tolerance so L*tol < 0.5.
            const double tol =
                std::min(o_.opt_.eps * std::max(1.0, std::abs(c.value)), 0.4 / L);
            if (tv[l] < c.value - tol) {
                cut_added = true;
                if (integer) addLazy(o_.theta_[l] >= cut_expr(c));
                else addCut(o_.theta_[l] >= cut_expr(c));
                ++o_.ncuts_;
                if (integer && o_.opt_.papadakos) {
                    BendersCut pc = o_.sp_[l]->solve(o_.core_y_, o_.core_z_);
                    if (pc.feasible) {
                        addLazy(o_.theta_[l] >= cut_expr(pc));
                        ++o_.ncuts_;
                    }
                }
            }
        }
        // Papadakos core point is updated ONCE per round, not per product.
        if (integer && o_.opt_.papadakos && cut_added) {
            for (int i = 0; i < I; ++i) o_.core_y_[i] = 0.5 * (o_.core_y_[i] + yv[i]);
            for (int j = 0; j < J; ++j) o_.core_z_[j] = 0.5 * (o_.core_z_[j] + zv[j]);
        }
        // theta-repair: incumbent will be accepted (no cut) but some theta_l
        // overestimates v_l -> schedule re-injection with exact values.
        if (integer && !cut_added) {
            double slack = 0.0;
            for (int l = 0; l < L; ++l)
                if (vl[l] >= 0.0) slack += std::max(0.0, tv[l] - vl[l]);
            if (slack > o_.opt_.eps) {
                o_.pend_y_.assign(yv.begin(), yv.end());
                o_.pend_z_.assign(zv.begin(), zv.end());
                o_.pend_theta_.assign(vl.begin(), vl.end());
                o_.has_pending_ = true;
            }
        }
    }

    BendersModel& o_;
};

// -------------------------------------------------------------------- master
BendersModel::BendersModel(const Instance& inst, BendersOptions opt)
    : p_(inst), opt_(opt), env_(true), master_((env_.start(), env_)) {
    const int I = p_.nfactories, J = p_.nwarehouses, L = p_.ncommodities;

    sp_.reserve(L);
    for (int l = 0; l < L; ++l) sp_.push_back(std::make_unique<ProductSubproblem>(p_, l, env_));

    for (int i = 0; i < I; ++i) {
        std::ostringstream n; n << "y[" << i << "]";
        y_.push_back(master_.addVar(0, 1, p_.fixedcost_factory[i], GRB_BINARY, n.str()));
        y_.back().set(GRB_IntAttr_BranchPriority, 10);
    }
    for (int j = 0; j < J; ++j) {
        std::ostringstream n; n << "z[" << j << "]";
        z_.push_back(master_.addVar(0, 1, p_.fixedcost_warehouse[j], GRB_BINARY, n.str()));
        z_.back().set(GRB_IntAttr_BranchPriority, 10);
    }
    for (int l = 0; l < L; ++l) {
        std::ostringstream n; n << "theta[" << l << "]";
        theta_.push_back(master_.addVar(0, GRB_INFINITY, 1.0, GRB_CONTINUOUS, n.str()));
    }
    master_.set(GRB_IntAttr_ModelSense, GRB_MINIMIZE);

    // (F1_l),(F2_l): a-priori feasibility (Prop. 1).
    for (int l = 0; l < L; ++l) {
        const double D = p_.total_demand(l);
        GRBLinExpr ef, ew;
        for (int i = 0; i < I; ++i) ef += p_.factory_capacity[i][l] * y_[i];
        for (int j = 0; j < J; ++j) ew += p_.warehouse_capacity[j][l] * z_[j];
        std::ostringstream nf, nw;
        nf << "F1[" << l << "]"; nw << "F2[" << l << "]";
        master_.addConstr(ef >= D, nf.str());
        master_.addConstr(ew >= D, nw.str());
    }
    // theta_l >= v_l(all-open) (Prop. 4).
    std::vector<double> ones_y(I, 1.0), ones_z(J, 1.0);
    for (int l = 0; l < L; ++l) {
        BendersCut c = sp_[l]->solve(ones_y, ones_z);
        if (c.feasible) theta_[l].set(GRB_DoubleAttr_LB, c.value);
    }

    core_y_.assign(I, 1.0);
    core_z_.assign(J, 1.0);
    master_.set(GRB_IntParam_LazyConstraints, 1);
    if (opt_.root_cuts) master_.set(GRB_IntParam_PreCrush, 1);
    if (opt_.threads > 0) master_.set(GRB_IntParam_Threads, opt_.threads);
}

void BendersModel::set_log(const std::string& path) {
    master_.set(GRB_StringParam_LogFile, path);
}

void BendersModel::set_seed(int seed) {
    master_.set(GRB_IntParam_Seed, seed);
    // subproblem LPs are deterministic re-solves; master seed governs the search
}

void BendersModel::add_global_lower_bound(double v_ld) {
    GRBLinExpr e;
    for (int i = 0; i < p_.nfactories; ++i) e += p_.fixedcost_factory[i] * y_[i];
    for (int j = 0; j < p_.nwarehouses; ++j) e += p_.fixedcost_warehouse[j] * z_[j];
    for (int l = 0; l < p_.ncommodities; ++l) e += theta_[l];
    master_.addConstr(e >= v_ld, "LD_bound");
}

void BendersModel::set_start(const std::vector<int>& ybar, const std::vector<int>& zbar) {
    for (int i = 0; i < p_.nfactories; ++i) y_[i].set(GRB_DoubleAttr_Start, ybar[i]);
    for (int j = 0; j < p_.nwarehouses; ++j) z_[j].set(GRB_DoubleAttr_Start, zbar[j]);
}

ExactResult BendersModel::run(double time_limit, double cutoff) {
    master_.set(GRB_DoubleParam_TimeLimit, time_limit);
    // Proof mode: integral optimum. In the BENDERS master obj = fixed + sum
    // theta, and acceptance allows theta_l >= v_l - tol_l with sum_l tol_l <= 0.4
    // (capped per-product tolerance above). MIPGapAbs = 0.5 then certifies:
    // bound > obj - 0.5 >= true - 0.9,
    // and integrality closes the argument (see the paper's Benders section).
    master_.set(GRB_DoubleParam_MIPGap, 0.0);
    master_.set(GRB_DoubleParam_MIPGapAbs, 0.5);
    master_.set(GRB_DoubleParam_Cutoff, cutoff > 0 ? cutoff : GRB_INFINITY);

    BendersCallback cb(*this);
    master_.setCallback(&cb);
    master_.optimize();

    ExactResult r;
    if (callback_failed_) { // never trust a run whose separation failed
        std::cerr << "[Benders] result POISONED by callback failure; discarding.\n";
        r.status = Status::NotFound;
        return r;
    }
    const int st = master_.get(GRB_IntAttr_Status);
    if (st == GRB_INFEASIBLE) { r.status = Status::Infeasible; return r; }
    if (master_.get(GRB_IntAttr_SolCount) == 0) {
        r.status = Status::NotFound;
        // On CUTOFF (or timeout) without incumbent the dual bound is
        // still valid and lets the caller certify a heuristic UB.
        try { r.bound = master_.get(GRB_DoubleAttr_ObjBound); } catch (GRBException&) {}
        return r;
    }
    r.status = (st == GRB_OPTIMAL) ? Status::OptimalFound : Status::SolutionFound;
    r.obj = master_.get(GRB_DoubleAttr_ObjVal);
    r.bound = master_.get(GRB_DoubleAttr_ObjBound);
    r.gap = master_.get(GRB_DoubleAttr_MIPGap);
    r.runtime = master_.get(GRB_DoubleAttr_Runtime);
    r.y.resize(p_.nfactories);
    r.z.resize(p_.nwarehouses);
    for (int i = 0; i < p_.nfactories; ++i) r.y[i] = y_[i].get(GRB_DoubleAttr_X) > 0.5;
    for (int j = 0; j < p_.nwarehouses; ++j) r.z[j] = z_[j].get(GRB_DoubleAttr_X) > 0.5;
    std::cout << "[Benders] cuts added: " << ncuts_ << "\n";
    return r;
}

} // namespace mptscfl
