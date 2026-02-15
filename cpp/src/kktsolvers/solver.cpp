#include "clarabel/kktsolvers/solver.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>

namespace clarabel {
namespace kktsolvers {

namespace {
std::vector<double> gaussian_elimination(std::vector<double> A, std::size_t n, std::vector<double> b) {
    for (std::size_t k = 0; k < n; ++k) {
        std::size_t pivot = k;
        double max_val = std::abs(A[k * n + k]);
        for (std::size_t i = k + 1; i < n; ++i) {
            double val = std::abs(A[i * n + k]);
            if (val > max_val) {
                max_val = val;
                pivot = i;
            }
        }
        if (pivot != k) {
            for (std::size_t j = 0; j < n; ++j) std::swap(A[k * n + j], A[pivot * n + j]);
            std::swap(b[k], b[pivot]);
        }

        double diag = A[k * n + k];
        if (std::abs(diag) < 1e-12) continue;
        for (std::size_t j = k; j < n; ++j) A[k * n + j] /= diag;
        b[k] /= diag;
        for (std::size_t i = k + 1; i < n; ++i) {
            double factor = A[i * n + k];
            if (std::abs(factor) < 1e-14) continue;
            for (std::size_t j = k; j < n; ++j) {
                A[i * n + j] -= factor * A[k * n + j];
            }
            b[i] -= factor * b[k];
        }
    }
    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
        for (int j = i + 1; j < static_cast<int>(n); ++j) {
            b[i] -= A[i * n + j] * b[j];
        }
    }
    return b;
}
}

SolveResult solve(const clarabel::kktsystem::KKTSystem& system, const std::vector<double>& rhs, const clarabel::core::Settings&) {
    const std::size_t n = system.K.cols;
    assert(rhs.size() == n);
    std::vector<double> row_major(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            row_major[i * n + j] = system.K(i, j);
        }
    }
    auto solution = gaussian_elimination(row_major, n, rhs);

    // split into primal/dual pieces assuming ordering [x; y]
    SolveResult res;
    std::size_t primal_dim = n / 2 + n % 2; // approximate split when m != n
    res.delta_x.assign(solution.begin(), solution.begin() + primal_dim);
    res.delta_y.assign(solution.begin() + primal_dim, solution.end());
    return res;
}

} // namespace kktsolvers
} // namespace clarabel

