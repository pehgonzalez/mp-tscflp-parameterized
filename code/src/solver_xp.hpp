// solver_xp.hpp — reference implementation of the branch-and-bound of the
// paper: depth-first branch and bound over facility subsets with prunings
// P1 (the covering-count lemma), P2 (accrued fixed costs + all-open
// routing bound), P3 (CNUF dominance at leaves, protected-witness
// correctness; see the correctness proposition of the paper).
//
// Preprocessing (safe under (B,k)): customer aggregation by identical
// d-columns and capacity capping b,p <= D_l. Both preserve feasibility,
// cost of every design, and the root covering bound k*.

#pragma once

#include <limits>
#include <string>

#include "instance.hpp"

namespace mptscfl {

struct XpStats {
    long long nodes = 0;       // B&B nodes visited (incl. r<0 discards)
    long long p1 = 0;          // nodes discarded by P1 (covering)
    long long p2_infeas = 0;   // nodes discarded by P2 (all-open infeasible)
    long long p2_bound = 0;    // nodes discarded by P2 (LB >= min(best, B+1))
    long long p3 = 0;          // leaves discarded by P3 (unused facility)
    long long rneg = 0;        // nodes discarded by step 4 (r < 0)
    int agg_merged = 0;        // customers removed by aggregation
    int capped = 0;            // capacity entries reduced by capping
};

struct XpResult {
    // status: "OPTIMAL" (search completed, finite optimum),
    //         "INFEASIBLE" (search completed, no feasible design of card <= k),
    //         "TIMEOUT" (deadline hit; obj = best incumbent, possibly inf).
    std::string status;
    bool has_obj = false;      // false <=> obj = +inf
    long long obj = 0;         // OPT_k when has_obj
    int k_used = 0;            // k after normalization k <- min(k, n)
    bool kstar_finite = false; // false <=> no covering design exists at all
    long long kstar = 0;       // root covering bound (the covering-count lemma, root case)
    bool timed_out = false;
    double seconds = 0.0;
    XpStats stats;
};

// k_request < 0 selects plain optimization (k = n). time_limit_s <= 0 means
// no deadline. B is the decision budget; LLONG_MAX gives pure optimization
// (the CLI always uses that).
XpResult solve_xp(const Instance& original, int k_request, double time_limit_s,
                  long long B = std::numeric_limits<long long>::max());

}  // namespace mptscfl
