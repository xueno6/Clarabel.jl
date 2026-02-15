#pragma once

#include <vector>

#include "clarabel/core/types.hpp"

namespace clarabel {
namespace kktsystem {

struct KKTSystem {
    clarabel::core::DenseMatrix K;
    std::size_t n_primal{0};
    std::size_t n_dual{0};
};

KKTSystem assemble(const clarabel::core::ProblemData& data);

} // namespace kktsystem
} // namespace clarabel
