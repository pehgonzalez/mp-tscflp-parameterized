// main_xp.cpp — CLI for the Xp solver (the branch-and-bound of the paper).
//
// Usage: xp <instance.psc> [time_limit_s] [k|-1]
//   time_limit_s : wall-clock deadline in seconds (default 3600; <=0 = none)
//   k            : cardinality bound (mode a); -1 = plain optimize, k = n
//                  (mode b). Default -1.
//
// Output contract (ONE line on stdout):
//   instance= status= obj= k_used= kstar= nodes= time= agg= timeout=
// followed by extra diagnostic fields (p1= p2i= p2b= p3= rneg= cap=).
// kstar is the root covering lower bound of the covering-count lemma (the Q3 predictor);
// it is always computed and printed, including on TIMEOUT and INFEASIBLE
// ("inf" when no covering design exists at all). obj= is "inf" when no
// feasible design of cardinality <= k is known.

#include <cstdio>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <limits>
#include <string>

#include "instance.hpp"
#include "solver_xp.hpp"

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::fprintf(stderr, "usage: %s <instance.psc> [time_limit_s] [k|-1]\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    auto parse_num = [](const char* a, bool as_int) -> double {
        errno = 0;
        char* end = nullptr;
        const double v = as_int ? static_cast<double>(std::strtol(a, &end, 10))
                                : std::strtod(a, &end);
        if (errno == ERANGE || end == a || *end != '\0') {
            std::fprintf(stderr, "bad numeric argument '%s'\n", a);
            std::exit(2);
        }
        return v;
    };
    // Range guards. The time limit is capped so the duration conversion in
    // the solver stays representable, and k must fit in int before the cast.
    const double kMaxTimeLimitS = 1e7;  // ~115 days, far above any protocol
    const double tl = (argc >= 3) ? parse_num(argv[2], false) : 3600.0;
    if (!std::isfinite(tl) || tl > kMaxTimeLimitS) {
        std::fprintf(stderr, "bad numeric argument '%s'\n", argv[2]);
        return 2;
    }
    const double k_raw = (argc >= 4) ? parse_num(argv[3], true) : -1.0;
    if (k_raw < static_cast<double>(std::numeric_limits<int>::min()) ||
        k_raw > static_cast<double>(std::numeric_limits<int>::max())) {
        std::fprintf(stderr, "bad numeric argument '%s'\n", argv[3]);
        return 2;
    }
    const int k = static_cast<int>(k_raw);

    std::string name = path;
    if (auto pos = name.find_last_of("/\\"); pos != std::string::npos) name = name.substr(pos + 1);

    mptscfl::Instance X;
    try {
        X = mptscfl::read_psc(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }

    mptscfl::XpResult R = mptscfl::solve_xp(X, k, tl);

    char objbuf[32], ksbuf[32];
    if (R.has_obj) std::snprintf(objbuf, sizeof objbuf, "%lld", R.obj);
    else std::snprintf(objbuf, sizeof objbuf, "inf");
    if (R.kstar_finite) std::snprintf(ksbuf, sizeof ksbuf, "%lld", R.kstar);
    else std::snprintf(ksbuf, sizeof ksbuf, "inf");

    std::printf(
        "instance=%s status=%s obj=%s k_used=%d kstar=%s nodes=%lld time=%.3f agg=%d timeout=%d "
        "p1=%lld p2i=%lld p2b=%lld p3=%lld rneg=%lld cap=%d\n",
        name.c_str(), R.status.c_str(), objbuf, R.k_used, ksbuf, R.stats.nodes, R.seconds,
        R.stats.agg_merged, R.timed_out ? 1 : 0, R.stats.p1, R.stats.p2_infeas, R.stats.p2_bound,
        R.stats.p3, R.stats.rneg, R.stats.capped);
    return 0;
}
