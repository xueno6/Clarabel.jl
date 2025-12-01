#pragma once

#include <memory>

#include "clarabel/cones/cones.hpp"
#include "clarabel/core/types.hpp"

namespace clarabel {
namespace solver {

struct SolveSummary {
    clarabel::core::Variables vars;
    clarabel::core::SolverStats stats;
    bool converged{false};
};

// A lightweight primal-dual iteration that mirrors the Clarabel loop layout.
// It alternates KKT solves with cone projections to drive primal/dual residuals
// toward zero. This is **not** a full reproduction of the Julia solver but
// captures the predictor-corrector structure with a CPU/GPU backend hook.
SolveSummary solve(const clarabel::core::ProblemData& data,
                   const clarabel::cones::Cone& cone,
                   const clarabel::core::Settings& settings);

} // namespace solver
} // namespace clarabel

