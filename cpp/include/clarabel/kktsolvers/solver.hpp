#pragma once

#include <vector>

#include "clarabel/core/types.hpp"
#include "clarabel/kktsystem/kkt.hpp"

namespace clarabel {
namespace kktsolvers {

struct SolveResult {
    std::vector<double> delta_x;
    std::vector<double> delta_y;
};

SolveResult solve(const clarabel::kktsystem::KKTSystem& system, const std::vector<double>& rhs, const clarabel::core::Settings& settings);

} // namespace kktsolvers
} // namespace clarabel

