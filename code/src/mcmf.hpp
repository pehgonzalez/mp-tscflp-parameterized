// mcmf.hpp — exact integer min-cost flow: successive shortest paths (SSP)
// with Johnson potentials (Dijkstra on reduced costs). All arithmetic in
// long long; requires all original arc costs >= 0 (true for the layered
// network N_l(y,z) of Prop. A1.1: c, d >= 0, capacity arcs cost 0).
//
// Correctness of keeping potentials of unreachable nodes unchanged: in SSP
// a node unreachable in the residual graph can never become reachable
// later (residual arcs only change along augmenting paths, whose endpoints
// are reachable), so its potential is never consulted again.

#pragma once

#include <cstdint>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

namespace mptscfl {

class MCMF {
public:
    static constexpr long long INF = std::numeric_limits<long long>::max();

    struct Edge {
        int to;
        long long cap;   // residual capacity
        long long cost;
        int rev;         // index of reverse edge in graph_[to]
    };

    struct Handle { int u = -1, idx = -1; long long orig_cap = 0; };

    explicit MCMF(int n) : n_(n), graph_(n) {}

    Handle add_edge(int u, int v, long long cap, long long cost) {
        graph_[u].push_back({v, cap, cost, static_cast<int>(graph_[v].size())});
        graph_[v].push_back({u, 0, -cost, static_cast<int>(graph_[u].size()) - 1});
        return {u, static_cast<int>(graph_[u].size()) - 1, cap};
    }

    long long flow_on(const Handle& h) const {
        if (h.u < 0) return 0;
        return h.orig_cap - graph_[h.u][h.idx].cap;
    }

    // Send up to `target` units s->t at minimum cost.
    // Returns {sent, cost}; sent < target means max-flow < target.
    std::pair<long long, long long> min_cost_flow(int s, int t, long long target) {
        std::vector<long long> pot(n_, 0), dist(n_);
        std::vector<int> pv(n_, -1), pe(n_, -1);
        long long sent = 0, total_cost = 0;
        using QN = std::pair<long long, int>;
        while (sent < target) {
            std::fill(dist.begin(), dist.end(), INF);
            dist[s] = 0;
            std::priority_queue<QN, std::vector<QN>, std::greater<QN>> pq;
            pq.push({0, s});
            while (!pq.empty()) {
                auto [du, u] = pq.top();
                pq.pop();
                if (du != dist[u]) continue;
                for (int idx = 0; idx < static_cast<int>(graph_[u].size()); ++idx) {
                    const Edge& e = graph_[u][idx];
                    if (e.cap <= 0) continue;
                    long long nd = du + e.cost + pot[u] - pot[e.to];
                    if (nd < dist[e.to]) {
                        dist[e.to] = nd;
                        pv[e.to] = u;
                        pe[e.to] = idx;
                        pq.push({nd, e.to});
                    }
                }
            }
            if (dist[t] == INF) break;  // no augmenting path: max-flow reached
            for (int v = 0; v < n_; ++v)
                if (dist[v] < INF) pot[v] += dist[v];
            long long push = target - sent;
            for (int v = t; v != s; v = pv[v])
                push = std::min(push, graph_[pv[v]][pe[v]].cap);
            for (int v = t; v != s; v = pv[v]) {
                Edge& e = graph_[pv[v]][pe[v]];
                e.cap -= push;
                graph_[v][e.rev].cap += push;
            }
            sent += push;
            total_cost += push * pot[t];  // pot[s] == 0 throughout
        }
        return {sent, total_cost};
    }

private:
    int n_;
    std::vector<std::vector<Edge>> graph_;
};

}  // namespace mptscfl
