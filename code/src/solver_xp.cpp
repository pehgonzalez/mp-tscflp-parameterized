// solver_xp.cpp — the branch-and-bound of the paper, literal steps 1-9, plus
// the paper's safe preprocessing. See solver_xp.hpp for the contract.

#include "solver_xp.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <map>
#include <stdexcept>

#include "mcmf.hpp"

namespace mptscfl {

namespace {

constexpr long long LLINF = std::numeric_limits<long long>::max();

// ---------------------------------------------------------------------------
// Preprocessing — exact, safe under (B,k)
// ---------------------------------------------------------------------------

// Customer aggregation: merge customers with identical d-columns (d[l][j][k] equal for
// all l, j), summing demands. Preserves feasibility and cost of every design.
Instance aggregate_customers(const Instance& X, int& merged) {
    std::map<std::vector<long long>, int> seen;  // d-column -> new index
    std::vector<int> map_k(X.nK, -1);
    std::vector<int> reps;
    for (int k = 0; k < X.nK; ++k) {
        std::vector<long long> col;
        col.reserve(static_cast<size_t>(X.nL) * X.nJ);
        for (int l = 0; l < X.nL; ++l)
            for (int j = 0; j < X.nJ; ++j) col.push_back(X.d[l][j][k]);
        auto it = seen.find(col);
        if (it == seen.end()) {
            seen.emplace(std::move(col), static_cast<int>(reps.size()));
            map_k[k] = static_cast<int>(reps.size());
            reps.push_back(k);
        } else {
            map_k[k] = it->second;
        }
    }
    merged = X.nK - static_cast<int>(reps.size());
    if (merged == 0) return X;
    Instance Y = X;
    Y.nK = static_cast<int>(reps.size());
    Y.q.assign(Y.nK, std::vector<long long>(X.nL, 0));
    for (int k = 0; k < X.nK; ++k)
        for (int l = 0; l < X.nL; ++l) Y.q[map_k[k]][l] += X.q[k][l];
    Y.d.assign(X.nL, std::vector<std::vector<long long>>(X.nJ, std::vector<long long>(Y.nK)));
    for (int l = 0; l < X.nL; ++l)
        for (int j = 0; j < X.nJ; ++j)
            for (int r = 0; r < Y.nK; ++r) Y.d[l][j][r] = X.d[l][j][reps[r]];
    return Y;
}

// Capacity capping: cap b_il, p_jl at D_l. Preserves feasibility and cost of every
// design; also preserves k* (a capped entry >= D_l already gives prefix >= D_l
// at s = 1, and entries < D_l are untouched).
void cap_capacities(Instance& X, int& capped) {
    capped = 0;
    for (int l = 0; l < X.nL; ++l) {
        const long long D = X.demand_total(l);
        for (int i = 0; i < X.nI; ++i)
            if (X.b[i][l] > D) { X.b[i][l] = D; ++capped; }
        for (int j = 0; j < X.nJ; ++j)
            if (X.p[j][l] > D) { X.p[j][l] = D; ++capped; }
    }
}

// ---------------------------------------------------------------------------
// The covering-count lemma of the paper — minimum covering count via
// sorted-capacity prefix scan
// ---------------------------------------------------------------------------

// k*(a; D): smallest s with the sum of the s largest entries of a >= D;
// LLINF if even the full sum falls short; 0 if D <= 0.
long long kstar_prefix(std::vector<long long>& a, long long D) {
    if (D <= 0) return 0;
    std::sort(a.begin(), a.end(), std::greater<long long>());
    long long acc = 0;
    for (size_t s = 0; s < a.size(); ++s) {
        acc += a[s];
        if (acc >= D) return static_cast<long long>(s) + 1;
    }
    return LLINF;
}

// ---------------------------------------------------------------------------
// Routing oracle: |L| exact min-cost flows on N_l(y,z)
// ---------------------------------------------------------------------------

struct OracleOut {
    bool feasible = false;
    long long value = 0;
    std::vector<long long> fac_use;  // total flow leaving factory i, all l
    std::vector<long long> dep_use;  // total flow through depot j, all l
};

// yopen/zopen: which facilities are open. Under complete stages, feasibility
// is exactly F1/F2 (the feasibility proposition of the paper); middle arcs may be capped at
// min(b_il, p_jl) (resp. min(p_jl, q_kl)) without loss, since every feasible
// flow already respects those bounds through the capacity arcs.
OracleOut routing_oracle(const Instance& X, const std::vector<char>& yopen,
                         const std::vector<char>& zopen) {
    OracleOut out;
    out.fac_use.assign(X.nI, 0);
    out.dep_use.assign(X.nJ, 0);
    const int S = 0;
    const int T = 1 + X.nI + 2 * X.nJ + X.nK;
    for (int l = 0; l < X.nL; ++l) {
        const long long D = X.demand_total(l);
        if (D == 0) continue;
        long long capI = 0, capJ = 0;
        for (int i = 0; i < X.nI; ++i)
            if (yopen[i]) capI += X.b[i][l];
        for (int j = 0; j < X.nJ; ++j)
            if (zopen[j]) capJ += X.p[j][l];
        if (capI < D || capJ < D) return out;  // F1/F2 fail: infeasible
        MCMF net(T + 1);
        std::vector<MCMF::Handle> hb(X.nI), hp(X.nJ);
        auto Fi = [&](int i) { return 1 + i; };
        auto Din = [&](int j) { return 1 + X.nI + j; };
        auto Dout = [&](int j) { return 1 + X.nI + X.nJ + j; };
        auto Ck = [&](int k) { return 1 + X.nI + 2 * X.nJ + k; };
        for (int i = 0; i < X.nI; ++i)
            if (yopen[i] && X.b[i][l] > 0) hb[i] = net.add_edge(S, Fi(i), X.b[i][l], 0);
        for (int j = 0; j < X.nJ; ++j)
            if (zopen[j] && X.p[j][l] > 0) hp[j] = net.add_edge(Din(j), Dout(j), X.p[j][l], 0);
        for (int i = 0; i < X.nI; ++i) {
            if (!yopen[i] || X.b[i][l] == 0) continue;
            for (int j = 0; j < X.nJ; ++j) {
                if (!zopen[j] || X.p[j][l] == 0) continue;
                net.add_edge(Fi(i), Din(j), std::min(X.b[i][l], X.p[j][l]), X.c[l][i][j]);
            }
        }
        for (int j = 0; j < X.nJ; ++j) {
            if (!zopen[j] || X.p[j][l] == 0) continue;
            for (int k = 0; k < X.nK; ++k) {
                if (X.q[k][l] == 0) continue;
                net.add_edge(Dout(j), Ck(k), std::min(X.p[j][l], X.q[k][l]), X.d[l][j][k]);
            }
        }
        for (int k = 0; k < X.nK; ++k)
            if (X.q[k][l] > 0) net.add_edge(Ck(k), T, X.q[k][l], 0);
        auto [sent, cost] = net.min_cost_flow(S, T, D);
        if (sent < D) return out;  // defensive; cannot happen when F1/F2 hold
        out.value += cost;
        for (int i = 0; i < X.nI; ++i) out.fac_use[i] += net.flow_on(hb[i]);
        for (int j = 0; j < X.nJ; ++j) out.dep_use[j] += net.flow_on(hp[j]);
    }
    out.feasible = true;
    return out;
}

// ---------------------------------------------------------------------------
// The B&B (steps 3-9 of the paper's algorithm)
// ---------------------------------------------------------------------------

struct Search {
    const Instance& X;
    int k = 0;
    long long B = LLINF;
    std::vector<long long> D;              // D_l
    std::vector<std::pair<int, int>> order;  // (side 0=I/1=J, index); factories first
    // Node state (element e = order[depth]): decided prefix < depth.
    std::vector<char> ydec, zdec;          // decided?
    std::vector<char> yopen_dec, zopen_dec;  // decided-open?
    std::vector<long long> openCapI, openCapJ;  // per l: capacity opened in O
    int cardO = 0;
    long long fixedO = 0;
    long long best = LLINF;
    XpStats st;
    std::chrono::steady_clock::time_point deadline;
    bool has_deadline = false;
    bool timed_out = false;

    explicit Search(const Instance& x) : X(x) {
        D.resize(X.nL);
        for (int l = 0; l < X.nL; ++l) D[l] = X.demand_total(l);
        ydec.assign(X.nI, 0);
        zdec.assign(X.nJ, 0);
        yopen_dec.assign(X.nI, 0);
        zopen_dec.assign(X.nJ, 0);
        openCapI.assign(X.nL, 0);
        openCapJ.assign(X.nL, 0);
        // Branching order: factories first (spec step 3); within each side,
        // most-constrained-first = decreasing total (capped) capacity, so the
        // covering prunings P1 become informative as early as possible.
        std::vector<int> is(X.nI), js(X.nJ);
        std::vector<long long> wi(X.nI, 0), wj(X.nJ, 0);
        for (int i = 0; i < X.nI; ++i) { is[i] = i; for (int l = 0; l < X.nL; ++l) wi[i] += X.b[i][l]; }
        for (int j = 0; j < X.nJ; ++j) { js[j] = j; for (int l = 0; l < X.nL; ++l) wj[j] += X.p[j][l]; }
        std::stable_sort(is.begin(), is.end(), [&](int a, int b) { return wi[a] > wi[b]; });
        std::stable_sort(js.begin(), js.end(), [&](int a, int b) { return wj[a] > wj[b]; });
        for (int i : is) order.emplace_back(0, i);
        for (int j : js) order.emplace_back(1, j);
    }

    bool deadline_hit() {
        if (!has_deadline || timed_out) return timed_out;
        if (std::chrono::steady_clock::now() >= deadline) timed_out = true;
        return timed_out;
    }

    // Ablation switch. With MPTSCFL_NO_P1=1 in the environment the covering
    // pruning is disabled while every other component of the search is left
    // untouched, which isolates the contribution of rule P1. The default is
    // the full algorithm, and the flag is read once per process.
    static bool p1_disabled() {
        static const bool off = [] {
            const char* e = std::getenv("MPTSCFL_NO_P1");
            return e != nullptr && e[0] == '1';
        }();
        return off;
    }

    // Node P1 quantities (node form of the covering-count lemma): s_I, s_J
    // via prefix counts over the
    // capacities of FREE (undecided) facilities against residual demand.
    bool p1_prune(int r) {
        if (p1_disabled()) return false;
        long long maxSI = 0, maxSJ = 0;
        std::vector<long long> freeCap;
        for (int l = 0; l < X.nL; ++l) {
            freeCap.clear();
            for (int i = 0; i < X.nI; ++i)
                if (!ydec[i]) freeCap.push_back(X.b[i][l]);
            long long s = kstar_prefix(freeCap, D[l] - openCapI[l]);
            if (s == LLINF) return true;
            maxSI = std::max(maxSI, s);
        }
        for (int l = 0; l < X.nL; ++l) {
            freeCap.clear();
            for (int j = 0; j < X.nJ; ++j)
                if (!zdec[j]) freeCap.push_back(X.p[j][l]);
            long long s = kstar_prefix(freeCap, D[l] - openCapJ[l]);
            if (s == LLINF) return true;
            maxSJ = std::max(maxSJ, s);
        }
        return maxSI + maxSJ > r;
    }

    void dfs(int depth) {
        if (timed_out) return;
        ++st.nodes;
        if ((st.nodes & 4095) == 0 && deadline_hit()) {
            --st.nodes;  // the aborted node is classified by no counter
            return;
        }
        const int r = k - cardO;
        if (r < 0) { ++st.rneg; return; }  // step 4
        if (p1_prune(r)) { ++st.p1; return; }  // step 5 [P1]
        // step 6 [P2]: all-open completion (y^, z^): open = decided-open or free
        std::vector<char> yup(X.nI), zup(X.nJ);
        for (int i = 0; i < X.nI; ++i) yup[i] = ydec[i] ? yopen_dec[i] : 1;
        for (int j = 0; j < X.nJ; ++j) zup[j] = zdec[j] ? zopen_dec[j] : 1;
        OracleOut orc = routing_oracle(X, yup, zup);
        if (deadline_hit()) {
            --st.nodes;  // aborted after the oracle, classified by no counter
            return;
        }
        if (!orc.feasible) { ++st.p2_infeas; return; }
        const long long LB = fixedO + orc.value;
        const long long cut = (B >= LLINF - 1) ? best : std::min(best, B + 1);
        if (LB >= cut) { ++st.p2_bound; return; }
        if (depth == static_cast<int>(order.size())) {
            // step 7: leaf, S = O, LB = custo(S) exactly. [P3]
            for (int i = 0; i < X.nI; ++i)
                if (yopen_dec[i] && orc.fac_use[i] == 0) { ++st.p3; return; }
            for (int j = 0; j < X.nJ; ++j)
                if (zopen_dec[j] && orc.dep_use[j] == 0) { ++st.p3; return; }
            best = std::min(best, LB);
            return;
        }
        // step 8: branch on the next element in the fixed order
        auto [side, e] = order[depth];
        // child OPEN first (reaches feasible incumbents quickly)
        if (side == 0) {
            ydec[e] = 1; yopen_dec[e] = 1; ++cardO; fixedO += X.f[e];
            for (int l = 0; l < X.nL; ++l) openCapI[l] += X.b[e][l];
            dfs(depth + 1);
            for (int l = 0; l < X.nL; ++l) openCapI[l] -= X.b[e][l];
            fixedO -= X.f[e]; --cardO; yopen_dec[e] = 0;
            if (!timed_out) dfs(depth + 1);  // child CLOSED (ydec stays 1, open flag 0)
            ydec[e] = 0;
        } else {
            zdec[e] = 1; zopen_dec[e] = 1; ++cardO; fixedO += X.g[e];
            for (int l = 0; l < X.nL; ++l) openCapJ[l] += X.p[e][l];
            dfs(depth + 1);
            for (int l = 0; l < X.nL; ++l) openCapJ[l] -= X.p[e][l];
            fixedO -= X.g[e]; --cardO; zopen_dec[e] = 0;
            if (!timed_out) dfs(depth + 1);
            zdec[e] = 0;
        }
    }
};

}  // namespace

XpResult solve_xp(const Instance& original, int k_request, double time_limit_s, long long B) {
    const auto t0 = std::chrono::steady_clock::now();
    XpResult R;

    // Preprocessing (A6 §1): aggregation then capping. Both preserve the
    // optimum, the feasible designs, and the root covering bound k*.
    Instance X = aggregate_customers(original, R.stats.agg_merged);
    cap_capacities(X, R.stats.capped);

    const int n = X.nI + X.nJ;
    // Step 1: normalization k <- min(k, n); k < 0 encodes plain optimization.
    R.k_used = (k_request < 0) ? n : std::min(k_request, n);

    // Step 2: a priori P1 and the experimental knob k* (root covering bound).
    long long maxKI = 0, maxKJ = 0;
    for (int l = 0; l < X.nL; ++l) {
        std::vector<long long> bs(X.nI), ps(X.nJ);
        for (int i = 0; i < X.nI; ++i) bs[i] = X.b[i][l];
        for (int j = 0; j < X.nJ; ++j) ps[j] = X.p[j][l];
        const long long D = X.demand_total(l);
        maxKI = std::max(maxKI, kstar_prefix(bs, D));
        maxKJ = std::max(maxKJ, kstar_prefix(ps, D));
    }
    R.kstar_finite = (maxKI != LLINF && maxKJ != LLINF);
    R.kstar = R.kstar_finite ? maxKI + maxKJ : 0;
    // The root covering test is part of rule P1, so the ablation switch
    // disables it too and the search then proves any infeasibility through
    // the oracle-based prunes, keeping the knockout complete under bounded k.
    if (!Search::p1_disabled() && (!R.kstar_finite || R.kstar > R.k_used)) {
        R.status = "INFEASIBLE";
        R.has_obj = false;
        R.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return R;
    }

    // Steps 3-8: DFS branch and bound.
    Search S(X);
    S.k = R.k_used;
    S.B = B;
    if (time_limit_s > 0) {
        S.has_deadline = true;
        S.deadline = t0 + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                              std::chrono::duration<double>(time_limit_s));
    }
    S.dfs(0);

    // Step 9.
    R.stats.nodes = S.st.nodes;
    R.stats.p1 = S.st.p1;
    R.stats.p2_infeas = S.st.p2_infeas;
    R.stats.p2_bound = S.st.p2_bound;
    R.stats.p3 = S.st.p3;
    R.stats.rneg = S.st.rneg;
    R.timed_out = S.timed_out;
    R.has_obj = (S.best < LLINF);
    R.obj = R.has_obj ? S.best : 0;
    R.status = S.timed_out ? "TIMEOUT" : (R.has_obj ? "OPTIMAL" : "INFEASIBLE");
    R.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return R;
}

}  // namespace mptscfl
