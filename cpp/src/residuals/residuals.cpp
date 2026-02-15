#include "clarabel/residuals/residuals.hpp"

#include <cmath>
#include <limits>

namespace clarabel {
namespace residuals {

Residuals compute(const clarabel::core::ProblemData& data, const clarabel::core::Variables& vars) {
    Residuals res;

    if (vars.x.size() != data.q.size() || vars.y.size() != data.b.size() || vars.s.size() < data.q.size()) {
        const double inf = std::numeric_limits<double>::infinity();
        res.primal = inf;
        res.dual = inf;
        return res;
    }

    std::vector<double> Ax(data.A.rows, 0.0);
    for (std::size_t i = 0; i < data.A.rows; ++i) {
        for (std::size_t idx = data.A.row_ptr[i]; idx < data.A.row_ptr[i + 1]; ++idx) {
            std::size_t j = data.A.col_ind[idx];
            Ax[i] += data.A.values[idx] * vars.x[j];
        }
        double diff = Ax[i] - data.b[i];
        res.primal += diff * diff;
    }
    res.primal = std::sqrt(res.primal);

    res.dual = 0.0;
    std::vector<double> Px(data.P.rows, 0.0);
    for (std::size_t i = 0; i < data.P.rows; ++i) {
        for (std::size_t j = 0; j < data.P.cols; ++j) {
            Px[i] += data.P(i, j) * vars.x[j];
        }
    }
    for (std::size_t j = 0; j < data.P.cols; ++j) {
        double Aty = 0.0;
        for (std::size_t i = 0; i < data.A.rows; ++i) {
            for (std::size_t idx = data.A.row_ptr[i]; idx < data.A.row_ptr[i + 1]; ++idx) {
                if (data.A.col_ind[idx] == j) {
                    Aty += data.A.values[idx] * vars.y[i];
                }
            }
        }
        double diff = Px[j] + data.q[j] + Aty - vars.s[j];
        res.dual += diff * diff;
    }
    res.dual = std::sqrt(res.dual);
    return res;
}

} // namespace residuals
} // namespace clarabel
