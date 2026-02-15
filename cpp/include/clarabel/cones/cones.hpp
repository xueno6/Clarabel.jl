#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "clarabel/core/types.hpp"

namespace clarabel {
namespace cones {

using clarabel::core::BackendDecision;
using clarabel::core::DeviceVector;
using clarabel::core::HostVector;

struct ConeProjectionResult {
    std::vector<double> projected;
    double dual_scale{1.0};
};

class Cone {
public:
    virtual ~Cone() = default;
    virtual ConeProjectionResult project(const std::vector<double>& x) const = 0;
    virtual std::size_t dimension() const = 0;
};

class NonnegativeCone : public Cone {
public:
    explicit NonnegativeCone(std::size_t n) : n_(n) {}
    ConeProjectionResult project(const std::vector<double>& x) const override;
    std::size_t dimension() const override { return n_; }
private:
    std::size_t n_;
};

class SecondOrderCone : public Cone {
public:
    explicit SecondOrderCone(std::size_t n) : n_(n) {}
    ConeProjectionResult project(const std::vector<double>& x) const override;
    std::size_t dimension() const override { return n_; }
private:
    std::size_t n_;
};

class PSDCone : public Cone {
public:
    explicit PSDCone(std::size_t n) : n_(n) {}
    ConeProjectionResult project(const std::vector<double>& x) const override;
    std::size_t dimension() const override { return n_ * n_; }
private:
    std::size_t n_;
};

BackendDecision choose_backend(const clarabel::core::Settings& settings, std::size_t problem_size);
DeviceVector to_device(const HostVector& host, BackendDecision decision);
HostVector to_host(const DeviceVector& device);

} // namespace cones
} // namespace clarabel

