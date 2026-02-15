#include <cassert>
#include <cmath>
#include <iostream>

#include "clarabel/cones/cones.hpp"
#include "clarabel/core/types.hpp"
#include "clarabel/kktsolvers/solver.hpp"
#include "clarabel/kktsystem/kkt.hpp"
#include "clarabel/presolver/presolver.hpp"
#include "clarabel/residuals/residuals.hpp"
#include "clarabel/solver/solver.hpp"

using clarabel::cones::NonnegativeCone;
using clarabel::cones::PSDCone;
using clarabel::cones::SecondOrderCone;

bool approx_equal(double a, double b, double tol = 1e-6) { return std::abs(a - b) <= tol; }

int main() {
    {
        NonnegativeCone cone(2);
        auto proj = cone.project({-1.0, 2.0});
        assert(approx_equal(proj.projected[0], 0.0));
        assert(approx_equal(proj.projected[1], 2.0));
    }
    {
        SecondOrderCone soc(3);
        auto proj = soc.project({-1.0, 2.0, 2.0});
        assert(proj.projected[0] >= 0.0);
        double norm = std::sqrt(proj.projected[1] * proj.projected[1] + proj.projected[2] * proj.projected[2]);
        assert(proj.projected[0] >= norm - 1e-9);
    }
    {
        PSDCone psd(2);
        auto proj = psd.project({-1.0, 0.0, 0.0, 2.0});
        assert(proj.projected[0] >= 0.0);
        assert(proj.projected[3] >= 0.0);
    }

    clarabel::core::ProblemData data;
    data.P = clarabel::core::DenseMatrix(2, 2, {1.0, 0.0, 0.0, 1.0});
    data.A = clarabel::core::SparseMatrix(1, 2, {0, 2}, {0, 1}, {1.0, 1.0});
    data.q = {1.0, 1.0};
    data.b = {1.0};

    auto validation = clarabel::core::validate_problem_data(data);
    assert(validation.ok());

    auto var_validation_default = clarabel::core::validate_variables(data, {std::vector<double>{0.0, 0.0}, std::vector<double>{0.0}, std::vector<double>{0.0, 0.0}});
    assert(var_validation_default.ok());
    auto var_validation_custom = clarabel::core::validate_variables(data, {std::vector<double>{0.0, 0.0}, std::vector<double>{0.0}, std::vector<double>{0.0, 0.0, 0.0}}, 3);
    assert(var_validation_custom.ok());
    auto scaled = clarabel::presolver::equilibrate(data);
    (void)scaled;

    auto kkt = clarabel::kktsystem::assemble(data);
    std::vector<double> rhs = {-data.q[0], -data.q[1], data.b[0]};
    clarabel::core::Settings settings;
    settings.use_gpu = true;
    auto decision = clarabel::cones::choose_backend(settings, 3);
    assert(decision.backend == clarabel::core::Backend::CPU);

    auto solve_result = clarabel::kktsolvers::solve(kkt, rhs, settings);
    assert(solve_result.delta_x.size() == 2);
    assert(solve_result.delta_y.size() == 1);
    assert(approx_equal(solve_result.delta_x[0], 0.5, 1e-4));
    assert(approx_equal(solve_result.delta_x[1], 0.5, 1e-4));

    clarabel::core::Variables vars{solve_result.delta_x, {-1.5}, {0.0, 0.0}};
    auto res = clarabel::residuals::compute(data, vars);
    assert(res.primal < 1e-8);
    assert(res.dual < 1e-4);

    NonnegativeCone cone(data.q.size());
    settings.max_iter = 50;
    auto summary = clarabel::solver::solve(data, cone, settings);
    assert(summary.converged);
    assert(summary.status == clarabel::core::SolveStatus::Solved);
    assert(summary.validation_issues.empty());
    assert(summary.vars.x.size() == 2);
    assert(approx_equal(summary.vars.x[0], 0.5, 1e-2));
    assert(approx_equal(summary.vars.x[1], 0.5, 1e-2));
    assert(!summary.stats.primal_residuals.empty());

    clarabel::core::ProblemData invalid = data;
    invalid.A.row_ptr = {0};
    auto bad_summary = clarabel::solver::solve(invalid, cone, settings);
    assert(!bad_summary.converged);
    assert(bad_summary.status == clarabel::core::SolveStatus::InvalidProblemData);
    assert(!bad_summary.validation_issues.empty());

    std::cout << "All clarabel_cpp tests passed" << std::endl;
    return 0;
}
