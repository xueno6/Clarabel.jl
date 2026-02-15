#pragma once

#include <memory>
#include <string>
#include <vector>

#include "clarabel/cones/cones.hpp"
#include "clarabel/core/types.hpp"

namespace clarabel {
namespace solver {

struct SolveSummary {
    clarabel::core::Variables vars;
    clarabel::core::SolverStats stats;
    clarabel::core::SolveStatus status{clarabel::core::SolveStatus::NotImplemented};
    std::string status_detail{"not started"};
    std::vector<clarabel::core::ValidationIssue> validation_issues;
    bool converged{false};
};

// Framework-level primal-dual driver scaffold.
// The control flow and interfaces are intentionally complete (validation,
// status reporting, backend selection, KKT solve/projection hooks), while
// numerical methods can be upgraded incrementally.
SolveSummary solve(const clarabel::core::ProblemData& data,
                   const clarabel::cones::Cone& cone,
                   const clarabel::core::Settings& settings);

} // namespace solver
} // namespace clarabel
