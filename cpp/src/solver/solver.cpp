#include "clarabel/solver/solver.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "clarabel/kktsolvers/solver.hpp"
#include "clarabel/kktsystem/kkt.hpp"
#include "clarabel/residuals/residuals.hpp"

namespace clarabel {
namespace solver {

namespace {
std::vector<double> matvec(const clarabel::core::DenseMatrix& M, const std::vector<double>& x) {
    std::vector<double> y(M.rows, 0.0);
    for (std::size_t i = 0; i < M.rows; ++i) {
        for (std::size_t j = 0; j < M.cols; ++j) {
            y[i] += M(i, j) * x[j];
        }
    }
    return y;
}

std::vector<double> At_times(const clarabel::core::SparseMatrix& A, const std::vector<double>& y) {
    std::vector<double> out(A.cols, 0.0);
    for (std::size_t i = 0; i < A.rows; ++i) {
        for (std::size_t idx = A.row_ptr[i]; idx < A.row_ptr[i + 1]; ++idx) {
            std::size_t j = A.col_ind[idx];
            out[j] += A.values[idx] * y[i];
        }
    }
    return out;
}

std::vector<double> A_times(const clarabel::core::SparseMatrix& A, const std::vector<double>& x) {
    std::vector<double> out(A.rows, 0.0);
    for (std::size_t i = 0; i < A.rows; ++i) {
        for (std::size_t idx = A.row_ptr[i]; idx < A.row_ptr[i + 1]; ++idx) {
            std::size_t j = A.col_ind[idx];
            out[i] += A.values[idx] * x[j];
        }
    }
    return out;
}
} // namespace

SolveSummary solve(const clarabel::core::ProblemData& data, const clarabel::cones::Cone& cone, const clarabel::core::Settings& settings) {
    SolveSummary summary;

    const auto data_validation = clarabel::core::validate_problem_data(data);
    if (!data_validation.ok()) {
        summary.status = clarabel::core::SolveStatus::InvalidProblemData;
        summary.status_detail = "Problem data failed validation.";
        summary.validation_issues = data_validation.issues;
        return summary;
    }

    const std::size_t n = data.q.size();
    const std::size_t m = data.b.size();

    summary.vars.x.assign(n, 0.0);
    summary.vars.s.assign(cone.dimension(), 0.0);
    summary.vars.y.assign(m, 0.0);

    const auto var_validation = clarabel::core::validate_variables(data, summary.vars, cone.dimension());
    if (!var_validation.ok()) {
        summary.status = clarabel::core::SolveStatus::InvalidProblemData;
        summary.status_detail = "Initial variables failed validation.";
        summary.validation_issues = var_validation.issues;
        return summary;
    }

    const auto backend = clarabel::cones::choose_backend(settings, n + m);
    (void)backend; // placeholder for future CPU/GPU dispatch

    const double tau = 0.5;   // primal step
    const double sigma = 0.5; // dual step

    auto kkt = clarabel::kktsystem::assemble(data);

    for (std::size_t iter = 0; iter < settings.max_iter; ++iter) {
        auto res = clarabel::residuals::compute(data, summary.vars);
        summary.stats.primal_residuals.push_back(res.primal);
        summary.stats.dual_residuals.push_back(res.dual);
        summary.stats.iterations = iter + 1;

        if (res.primal <= settings.atol && res.dual <= settings.atol) {
            summary.converged = true;
            summary.status = clarabel::core::SolveStatus::Solved;
            summary.status_detail = "Converged to configured tolerances.";
            return summary;
        }

        auto Ax = A_times(data.A, summary.vars.x);
        for (std::size_t i = 0; i < m; ++i) {
            const double rp = Ax[i] - data.b[i];
            summary.vars.y[i] += sigma * rp;
        }

        auto Px = matvec(data.P, summary.vars.x);
        auto Aty = At_times(data.A, summary.vars.y);
        std::vector<double> grad(n, 0.0);
        for (std::size_t j = 0; j < n; ++j) {
            grad[j] = Px[j] + data.q[j] + Aty[j] - summary.vars.s[j];
        }

        std::vector<double> rhs;
        rhs.reserve(n + m);
        for (std::size_t j = 0; j < n; ++j) rhs.push_back(-grad[j]);
        for (std::size_t i = 0; i < m; ++i) rhs.push_back(data.b[i] - Ax[i]);

        auto delta = clarabel::kktsolvers::solve(kkt, rhs, settings);
        for (std::size_t j = 0; j < n && j < delta.delta_x.size(); ++j) {
            summary.vars.x[j] += tau * delta.delta_x[j];
        }
        for (std::size_t i = 0; i < m && i < delta.delta_y.size(); ++i) {
            summary.vars.y[i] += tau * delta.delta_y[i];
        }

        std::vector<double> dual_candidate(n, 0.0);
        for (std::size_t j = 0; j < n; ++j) dual_candidate[j] = Px[j] + data.q[j] + Aty[j];
        auto proj = cone.project(dual_candidate);
        summary.vars.s = proj.projected;
    }

    summary.status = clarabel::core::SolveStatus::MaxIterations;
    summary.status_detail = "Reached max_iter before meeting tolerances.";
    return summary;
}

} // namespace solver
} // namespace clarabel
