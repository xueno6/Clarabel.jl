#pragma once

#include <vector>

#include "clarabel/core/types.hpp"

namespace clarabel {
namespace residuals {

struct Residuals {
    double primal{0.0};
    double dual{0.0};
};

Residuals compute(const clarabel::core::ProblemData& data, const clarabel::core::Variables& vars);

} // namespace residuals
} // namespace clarabel

