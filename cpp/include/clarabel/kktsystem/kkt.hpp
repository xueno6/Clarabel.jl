#pragma once

#include <vector>

#include "clarabel/core/types.hpp"

namespace clarabel {
namespace kktsystem {

struct KKTSystem {
    clarabel::core::DenseMatrix K;
};

KKTSystem assemble(const clarabel::core::ProblemData& data);

} // namespace kktsystem
} // namespace clarabel

