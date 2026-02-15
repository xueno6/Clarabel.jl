#include "clarabel/kktsystem/kkt.hpp"

#include <cassert>

namespace clarabel {
namespace kktsystem {

KKTSystem assemble(const clarabel::core::ProblemData& data) {
    // Build dense KKT matrix [P A^T; A 0]
    const std::size_t n = data.P.cols;
    const std::size_t m = data.A.rows;
    KKTSystem system{clarabel::core::DenseMatrix(n + m, n + m, std::vector<double>((n + m) * (n + m), 0.0)), n, m};

    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            system.K(i, j) = data.P(i, j);
        }
    }

    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t idx = data.A.row_ptr[i]; idx < data.A.row_ptr[i + 1]; ++idx) {
            std::size_t j = data.A.col_ind[idx];
            system.K(n + i, j) = data.A.values[idx];
            system.K(j, n + i) = data.A.values[idx];
        }
    }
    return system;
}

} // namespace kktsystem
} // namespace clarabel
