// Branch-and-Benders-cut for the MP-TSCFLP (math and proofs: see the paper's
// Benders section; reference implementation validated by brute force:
// python/benders_gurobipy.py).
//
// Master in (y, z, theta_l), one transportation-LP subproblem per product,
// disaggregated optimality cuts as lazy constraints, a-priori aggregate capacity
// inequalities (no feasibility cuts needed, Prop. 1), theta_l >= v_l(all-open)
// initial bounds (Prop. 4), optional Papadakos core-point cuts and root user cuts.
#ifndef MPTSCFL_BENDERS_MODEL_HPP
#define MPTSCFL_BENDERS_MODEL_HPP

#include <memory>
#include <vector>

#include "gurobi_c++.h"
#include "exact_model.hpp" // ExactResult, Status
#include "instance.hpp"

namespace mptscfl {

struct BendersCut {
    double value = 0.0;             // v_l(y,z)
    double constant = 0.0;          // sum_k q_kl * alpha_k
    std::vector<double> fac_coef;   // b_il * Pi_fcap_i   (<= 0)
    std::vector<double> ware_coef;  // p_jl * Pi_wcap_j   (<= 0)
    bool feasible = false;
};

// Transportation LP of one product; capacities enter as mutable RHS, so the same
// model is re-solved (dual simplex warm start) at every separation call.
class ProductSubproblem {
public:
    ProductSubproblem(const Instance& inst, int l, GRBEnv& env);
    BendersCut solve(const std::vector<double>& yv, const std::vector<double>& zv);

private:
    const Instance& p_;
    int l_;
    GRBModel lp_;
    std::vector<GRBConstr> dem_, fcap_, wcap_;
};

struct BendersOptions {
    bool papadakos = false;  // extra core-point cuts (validity unconditional, sec. 5)
    bool root_cuts = true;   // user cuts at the root LP
    int threads = 0;
    double eps = 1e-6;       // relative cut-violation tolerance
};

class BendersModel {
public:
    BendersModel(const Instance& inst, BendersOptions opt = {});

    void set_start(const std::vector<int>& ybar, const std::vector<int>& zbar);
    void set_log(const std::string& path);
    void set_seed(int seed);

    // Valid by weak duality (Theorem L1): fixed costs + sum_l theta_l >= v_LD.
    void add_global_lower_bound(double v_ld);

    ExactResult run(double time_limit, double cutoff = -1.0);

    long long cuts_added() const { return ncuts_; }

private:
    friend class BendersCallback;

    const Instance& p_;
    BendersOptions opt_;
    GRBEnv env_;
    GRBModel master_;
    std::vector<GRBVar> y_, z_, theta_;
    std::vector<std::unique_ptr<ProductSubproblem>> sp_;
    std::vector<double> core_y_, core_z_; // Papadakos core point
    long long ncuts_ = 0;

    // theta-repair: incumbent accepted with theta_l > v_l gets re-injected at the
    // next MIPNODE with theta_l := v_l exactly (reported obj = true cost).
    bool has_pending_ = false;
    std::vector<double> pend_y_, pend_z_, pend_theta_;

    // set by the callback on any exception: poisons the result
    bool callback_failed_ = false;
};

} // namespace mptscfl
#endif
