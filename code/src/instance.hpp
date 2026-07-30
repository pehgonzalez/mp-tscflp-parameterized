// instance.hpp — PSC-format parser for the MP-TSCFLP.
//
// PSC layout (whitespace-separated text, all integers; mirrors
// gurobi_port/src/instance.hpp of the exact-methods companion):
//   1. header: nI nJ nK nL
//   2. K rows x L values: q[k][l]        (customer demands)
//   3. I rows: b_i1 .. b_iL f_i          (factory capacities + fixed cost)
//   4. L blocks of I x J: c[l][i][j]     (stage-1 flow costs)
//   5. J rows: p_j1 .. p_jL g_j          (warehouse capacities + fixed cost)
//   6. L blocks of J x K: d[l][j][k]     (stage-2 flow costs)
//
// Every token is parsed strictly as a decimal integer (a token such as
// "7.5", "1e3" or "abc" aborts, wherever it appears), non-negativity is
// enforced, and each datum must not exceed MAX_DATUM = 10^9, a documented
// bound that keeps every downstream sum and cost product within long long
// with ample headroom. Trailing content of any kind aborts.

#pragma once

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mptscfl {

struct Instance {
    int nI = 0, nJ = 0, nK = 0, nL = 0;
    std::vector<long long> f;                            // [i]
    std::vector<long long> g;                            // [j]
    std::vector<std::vector<long long>> b;               // [i][l]
    std::vector<std::vector<long long>> p;               // [j][l]
    std::vector<std::vector<long long>> q;               // [k][l]
    std::vector<std::vector<std::vector<long long>>> c;  // [l][i][j]
    std::vector<std::vector<std::vector<long long>>> d;  // [l][j][k]

    long long demand_total(int l) const {
        long long D = 0;
        for (int k = 0; k < nK; ++k) D += q[k][l];
        return D;
    }
};

namespace detail {
constexpr long long MAX_DATUM = 1000000000LL;  // 10^9, documented input bound
inline long long read_int(std::istream& in, const char* what) {
    std::string tok;
    if (!(in >> tok))
        throw std::runtime_error(std::string("PSC parse error: expected integer for ") + what +
                                 " (missing datum; proof-mode assumptions violated)");
    errno = 0;
    char* end = nullptr;
    const long long v = std::strtoll(tok.c_str(), &end, 10);
    if (errno == ERANGE || end == tok.c_str() || *end != '\0')
        throw std::runtime_error(std::string("PSC parse error: non-integral token '") + tok +
                                 "' for " + what);
    if (v < 0)
        throw std::runtime_error(std::string("PSC parse error: negative datum for ") + what +
                                 " (the paper assumes non-negative integer data)");
    if (v > MAX_DATUM)
        throw std::runtime_error(std::string("PSC parse error: datum for ") + what +
                                 " exceeds the documented bound of 10^9");
    return v;
}
}  // namespace detail

inline Instance read_psc(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open instance file: " + path);
    Instance X;
    using detail::read_int;
    X.nI = static_cast<int>(read_int(in, "nI"));
    X.nJ = static_cast<int>(read_int(in, "nJ"));
    X.nK = static_cast<int>(read_int(in, "nK"));
    X.nL = static_cast<int>(read_int(in, "nL"));
    if (X.nI <= 0 || X.nJ <= 0 || X.nK <= 0 || X.nL <= 0)
        throw std::runtime_error("PSC parse error: non-positive dimension in header");

    X.q.assign(X.nK, std::vector<long long>(X.nL));
    for (int k = 0; k < X.nK; ++k)
        for (int l = 0; l < X.nL; ++l) X.q[k][l] = read_int(in, "q[k][l]");

    X.b.assign(X.nI, std::vector<long long>(X.nL));
    X.f.assign(X.nI, 0);
    for (int i = 0; i < X.nI; ++i) {
        for (int l = 0; l < X.nL; ++l) X.b[i][l] = read_int(in, "b[i][l]");
        X.f[i] = read_int(in, "f[i]");
    }

    X.c.assign(X.nL, std::vector<std::vector<long long>>(X.nI, std::vector<long long>(X.nJ)));
    for (int l = 0; l < X.nL; ++l)
        for (int i = 0; i < X.nI; ++i)
            for (int j = 0; j < X.nJ; ++j) X.c[l][i][j] = read_int(in, "c[l][i][j]");

    X.p.assign(X.nJ, std::vector<long long>(X.nL));
    X.g.assign(X.nJ, 0);
    for (int j = 0; j < X.nJ; ++j) {
        for (int l = 0; l < X.nL; ++l) X.p[j][l] = read_int(in, "p[j][l]");
        X.g[j] = read_int(in, "g[j]");
    }

    X.d.assign(X.nL, std::vector<std::vector<long long>>(X.nJ, std::vector<long long>(X.nK)));
    for (int l = 0; l < X.nL; ++l)
        for (int j = 0; j < X.nJ; ++j)
            for (int k = 0; k < X.nK; ++k) X.d[l][j][k] = read_int(in, "d[l][j][k]");

    std::string trailing;
    if (in >> trailing)
        throw std::runtime_error("PSC parse error: trailing data after expected end of instance");
    return X;
}

}  // namespace mptscfl
