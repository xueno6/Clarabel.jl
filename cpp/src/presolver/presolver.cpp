#include "clarabel/presolver/presolver.hpp"

#include <algorithm>
#include <cmath>

namespace clarabel {
namespace presolver {

clarabel::core::ProblemData equilibrate(const clarabel::core::ProblemData& data) {
    auto result = data;
    // Simple row scaling for A to unit row norms
    for (std::size_t i = 0; i < result.A.rows; ++i) {
        double norm = 0.0;
        for (std::size_t idx = result.A.row_ptr[i]; idx < result.A.row_ptr[i + 1]; ++idx) {
            norm += std::abs(result.A.values[idx]);
        }
        if (norm > 0) {
            double scale = 1.0 / norm;
            for (std::size_t idx = result.A.row_ptr[i]; idx < result.A.row_ptr[i + 1]; ++idx) {
                result.A.values[idx] *= scale;
            }
            result.b[i] *= scale;
        }
    }
    return result;
}

} // namespace presolver
} // namespace clarabel

