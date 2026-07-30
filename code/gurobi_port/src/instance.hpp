// MP-TSCFLP instance loader (PSC format, Mauri et al. 2021 / Fernandes et al. 2014).
// Faithful port of the legacy instance.{h,cpp} to modern C++20 (no raw memory).
// Every token is parsed strictly as a non-negative decimal integer bounded by
// 10^9 (the archive's documented input bound, same contract as ../../src),
// then stored as double, exact well below 2^53. Trailing content aborts.
#ifndef MPTSCFL_INSTANCE_HPP
#define MPTSCFL_INSTANCE_HPP

#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mptscfl {

enum class Status { OptimalFound, SolutionFound, Infeasible, NotFound };

class Instance {
public:
    int nfactories = 0;   // |I|
    int nwarehouses = 0;  // |J|
    int ncustomers = 0;   // |K|
    int ncommodities = 0; // |L|

    // Indexing follows the original code.
    std::vector<std::vector<double>> customer_demand;    // [k][l]  q_kl
    std::vector<std::vector<double>> factory_capacity;   // [i][l]  b_il
    std::vector<std::vector<double>> warehouse_capacity; // [j][l]  p_jl
    std::vector<double> fixedcost_factory;               // [i]     f_i
    std::vector<double> fixedcost_warehouse;             // [j]     g_j
    std::vector<std::vector<std::vector<double>>> flowcost_fw; // [l][i][j] c_ijl
    std::vector<std::vector<std::vector<double>>> flowcost_wc; // [l][j][k] d_jkl

    // Transposed helpers, as in the original code.
    std::vector<std::vector<double>> t_customer_demand;    // [l][k]
    std::vector<std::vector<double>> t_factory_capacity;   // [l][i]
    std::vector<std::vector<double>> t_warehouse_capacity; // [l][j]

    double TotalCapFac = 0.0;
    double TotalCapWare = 0.0;
    std::string file_name;

    static double read_datum(std::ifstream& f, const std::string& fname) {
        std::string tok;
        if (!(f >> tok))
            throw std::runtime_error("Truncated instance file: " + fname);
        errno = 0;
        char* end = nullptr;
        const long long v = std::strtoll(tok.c_str(), &end, 10);
        if (errno == ERANGE || end == tok.c_str() || *end != '\0' || v < 0 ||
            v > 1000000000LL)
            throw std::runtime_error("Bad datum '" + tok + "' in " + fname +
                                     ": expected a non-negative integer at most 10^9");
        return static_cast<double>(v);
    }

    void load_file(const std::string& fname) {
        file_name = fname;
        std::ifstream f(fname);
        if (!f) throw std::runtime_error("Instance file not found: " + fname);

        nfactories = static_cast<int>(read_datum(f, fname));
        nwarehouses = static_cast<int>(read_datum(f, fname));
        ncustomers = static_cast<int>(read_datum(f, fname));
        ncommodities = static_cast<int>(read_datum(f, fname));
        if (nfactories <= 0 || nwarehouses <= 0 || ncustomers <= 0 || ncommodities <= 0)
            throw std::runtime_error("Bad header in " + fname);

        customer_demand.assign(ncustomers, std::vector<double>(ncommodities));
        for (int k = 0; k < ncustomers; ++k)
            for (int l = 0; l < ncommodities; ++l) customer_demand[k][l] = read_datum(f, fname);

        fixedcost_factory.assign(nfactories, 0.0);
        factory_capacity.assign(nfactories, std::vector<double>(ncommodities));
        for (int i = 0; i < nfactories; ++i) {
            for (int l = 0; l < ncommodities; ++l) {
                factory_capacity[i][l] = read_datum(f, fname);
                TotalCapFac += factory_capacity[i][l];
            }
            fixedcost_factory[i] = read_datum(f, fname);
        }

        flowcost_fw.assign(ncommodities,
            std::vector<std::vector<double>>(nfactories, std::vector<double>(nwarehouses)));
        for (int l = 0; l < ncommodities; ++l)
            for (int i = 0; i < nfactories; ++i)
                for (int j = 0; j < nwarehouses; ++j) flowcost_fw[l][i][j] = read_datum(f, fname);

        fixedcost_warehouse.assign(nwarehouses, 0.0);
        warehouse_capacity.assign(nwarehouses, std::vector<double>(ncommodities));
        for (int j = 0; j < nwarehouses; ++j) {
            for (int l = 0; l < ncommodities; ++l) {
                warehouse_capacity[j][l] = read_datum(f, fname);
                TotalCapWare += warehouse_capacity[j][l];
            }
            fixedcost_warehouse[j] = read_datum(f, fname);
        }

        flowcost_wc.assign(ncommodities,
            std::vector<std::vector<double>>(nwarehouses, std::vector<double>(ncustomers)));
        for (int l = 0; l < ncommodities; ++l)
            for (int j = 0; j < nwarehouses; ++j)
                for (int k = 0; k < ncustomers; ++k) flowcost_wc[l][j][k] = read_datum(f, fname);

        {
            std::string trailing;
            if (f >> trailing)
                throw std::runtime_error("Trailing data after expected end of instance in " + fname);
        }

        // The whole certification chain (FlowEvaluator, integral proof mode,
        // MIPGapAbs<1) assumes integral data. Fail loudly if violated.
        auto chk = [&](double v) {
            if (v != static_cast<double>(static_cast<long long>(v)))
                throw std::runtime_error("Non-integral datum in " + fname +
                                         ": proof mode assumptions violated");
        };
        for (const auto& r : customer_demand) for (double v : r) chk(v);
        for (const auto& r : factory_capacity) for (double v : r) chk(v);
        for (const auto& r : warehouse_capacity) for (double v : r) chk(v);
        for (double v : fixedcost_factory) chk(v);
        for (double v : fixedcost_warehouse) chk(v);
        for (const auto& m : flowcost_fw) for (const auto& r : m) for (double v : r) chk(v);
        for (const auto& m : flowcost_wc) for (const auto& r : m) for (double v : r) chk(v);

        t_customer_demand.assign(ncommodities, std::vector<double>(ncustomers));
        t_factory_capacity.assign(ncommodities, std::vector<double>(nfactories));
        t_warehouse_capacity.assign(ncommodities, std::vector<double>(nwarehouses));
        for (int l = 0; l < ncommodities; ++l) {
            for (int k = 0; k < ncustomers; ++k) t_customer_demand[l][k] = customer_demand[k][l];
            for (int i = 0; i < nfactories; ++i) t_factory_capacity[l][i] = factory_capacity[i][l];
            for (int j = 0; j < nwarehouses; ++j) t_warehouse_capacity[l][j] = warehouse_capacity[j][l];
        }
    }

    double total_demand(int l) const {
        double s = 0.0;
        for (int k = 0; k < ncustomers; ++k) s += customer_demand[k][l];
        return s;
    }
};

} // namespace mptscfl
#endif
