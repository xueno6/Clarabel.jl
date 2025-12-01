#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace clarabel {
namespace core {

struct SparseMatrix {
    std::size_t rows{0};
    std::size_t cols{0};
    std::vector<std::size_t> row_ptr; // size rows + 1
    std::vector<std::size_t> col_ind; // size nnz
    std::vector<double> values;       // size nnz

    SparseMatrix() = default;
    SparseMatrix(std::size_t r, std::size_t c, std::vector<std::size_t> rp,
                 std::vector<std::size_t> ci, std::vector<double> vals)
        : rows(r), cols(c), row_ptr(std::move(rp)), col_ind(std::move(ci)), values(std::move(vals)) {}
};

struct DenseMatrix {
    std::size_t rows{0};
    std::size_t cols{0};
    std::vector<double> values; // column-major

    DenseMatrix() = default;
    DenseMatrix(std::size_t r, std::size_t c, std::vector<double> vals)
        : rows(r), cols(c), values(std::move(vals)) {}

    double& operator()(std::size_t r, std::size_t c) {
        return values[c * rows + r];
    }
    double operator()(std::size_t r, std::size_t c) const {
        return values[c * rows + r];
    }
};

struct ProblemData {
    DenseMatrix P; // quadratic term
    SparseMatrix A; // constraint matrix
    std::vector<double> q; // linear term
    std::vector<double> b; // rhs
};

struct Variables {
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> s;
};

struct Settings {
    double atol{1e-8};
    double rtol{1e-6};
    std::size_t max_iter{100};
    bool use_gpu{false};
    std::size_t gpu_threshold{5000};
};

enum class Backend { CPU, GPU };

struct BackendDecision {
    Backend backend{Backend::CPU};
    std::string reason{"Default to CPU"};
};

struct SolverStats {
    std::size_t iterations{0};
    std::vector<double> primal_residuals;
    std::vector<double> dual_residuals;
};

struct HostVector {
    std::vector<double> data;
    explicit HostVector(std::size_t n = 0, double value = 0.0) : data(n, value) {}
};

struct DeviceVector {
    std::vector<double> data; // CPU fallback; CUDA implementation lives in cones_cuda.cu when enabled
    explicit DeviceVector(std::size_t n = 0, double value = 0.0) : data(n, value) {}
};

} // namespace core
} // namespace clarabel

