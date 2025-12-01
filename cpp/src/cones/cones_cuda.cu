#include "clarabel/cones/cones.hpp"

#ifdef CLARABEL_USE_CUDA
#include <cuda_runtime.h>

namespace clarabel {
namespace cones {

DeviceVector to_device(const HostVector& host, BackendDecision) {
    DeviceVector device(host.data.size());
    cudaMemcpy(device.data.data(), host.data.data(), host.data.size() * sizeof(double), cudaMemcpyHostToDevice);
    return device;
}

HostVector to_host(const DeviceVector& device) {
    HostVector host(device.data.size());
    cudaMemcpy(host.data.data(), device.data.data(), device.data.size() * sizeof(double), cudaMemcpyDeviceToHost);
    return host;
}

} // namespace cones
} // namespace clarabel

#endif
