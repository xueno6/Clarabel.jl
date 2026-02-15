#include "clarabel/cones/cones.hpp"

#include <cassert>
#include <numeric>

namespace clarabel {
namespace cones {

ConeProjectionResult NonnegativeCone::project(const std::vector<double>& x) const {
    ConeProjectionResult res;
    res.projected.resize(n_);
    for (std::size_t i = 0; i < n_; ++i) {
        res.projected[i] = std::max(0.0, x[i]);
    }
    return res;
}

ConeProjectionResult SecondOrderCone::project(const std::vector<double>& x) const {
    ConeProjectionResult res;
    res.projected.resize(n_);
    assert(x.size() == n_);
    const double t = x[0];
    double norm_v = 0.0;
    for (std::size_t i = 1; i < n_; ++i) {
        norm_v += x[i] * x[i];
    }
    norm_v = std::sqrt(norm_v);
    if (norm_v <= -t) {
        std::fill(res.projected.begin(), res.projected.end(), 0.0);
    } else if (norm_v <= t) {
        res.projected = x;
    } else {
        const double scale = 0.5 * (1.0 + t / norm_v);
        res.projected[0] = norm_v * scale;
        for (std::size_t i = 1; i < n_; ++i) {
            res.projected[i] = scale * x[i];
        }
    }
    return res;
}

namespace {
// Jacobi eigenvalue algorithm for symmetric matrices stored column-major
std::vector<double> jacobi_eigenvalues(const std::vector<double>& A, std::size_t n, std::size_t max_sweeps = 10) {
    std::vector<double> mat = A;
    std::vector<double> eigenvalues(n, 0.0);
    for (std::size_t sweep = 0; sweep < max_sweeps; ++sweep) {
        bool converged = true;
        for (std::size_t p = 0; p < n; ++p) {
            for (std::size_t q = p + 1; q < n; ++q) {
                double apq = mat[q * n + p];
                if (std::abs(apq) < 1e-10) continue;
                converged = false;
                double app = mat[p * n + p];
                double aqq = mat[q * n + q];
                double tau = (aqq - app) / (2.0 * apq);
                double t = ((tau >= 0) ? 1.0 : -1.0) / (std::abs(tau) + std::sqrt(1 + tau * tau));
                double c = 1.0 / std::sqrt(1 + t * t);
                double s = t * c;
                for (std::size_t k = 0; k < n; ++k) {
                    double aik = mat[k * n + p];
                    double akq = mat[k * n + q];
                    mat[k * n + p] = c * aik - s * akq;
                    mat[k * n + q] = s * aik + c * akq;
                }
                for (std::size_t k = 0; k < n; ++k) {
                    double aip = mat[p * n + k];
                    double aiq = mat[q * n + k];
                    mat[p * n + k] = c * aip - s * aiq;
                    mat[q * n + k] = s * aip + c * aiq;
                }
                mat[p * n + p] = c * c * app - 2 * s * c * apq + s * s * aqq;
                mat[q * n + q] = s * s * app + 2 * s * c * apq + c * c * aqq;
                mat[q * n + p] = mat[p * n + q] = 0.0;
            }
        }
        if (converged) break;
    }
    for (std::size_t i = 0; i < n; ++i) {
        eigenvalues[i] = mat[i * n + i];
    }
    return eigenvalues;
}
}

ConeProjectionResult PSDCone::project(const std::vector<double>& x) const {
    assert(x.size() == n_ * n_);
    ConeProjectionResult res;
    res.projected = x;
    auto eigenvalues = jacobi_eigenvalues(x, n_);
    // shift negative eigenvalues to zero by scaling diagonal
    for (std::size_t i = 0; i < n_; ++i) {
        if (eigenvalues[i] < 0) {
            double delta = -eigenvalues[i];
            res.projected[i * n_ + i] += delta;
        }
    }
    return res;
}

BackendDecision choose_backend(const clarabel::core::Settings& settings, std::size_t problem_size) {
    BackendDecision decision;
    if (settings.use_gpu && problem_size >= settings.gpu_threshold) {
        decision.backend = clarabel::core::Backend::GPU;
        decision.reason = "Problem large enough for GPU backend";
    } else if (settings.use_gpu) {
        decision.reason = "GPU requested but problem below threshold";
    } else {
        decision.reason = "GPU disabled in settings";
    }
    return decision;
}

DeviceVector to_device(const HostVector& host, BackendDecision decision) {
    DeviceVector device(host.data.size());
    device.data = host.data; // CPU fallback copy
#ifdef CLARABEL_USE_CUDA
    // CUDA copy implemented in cones_cuda.cu
#endif
    return device;
}

HostVector to_host(const DeviceVector& device) {
    HostVector host(device.data.size());
    host.data = device.data;
    return host;
}

} // namespace cones
} // namespace clarabel

