// !!! This is a file automatically generated by hipify!!!
#include <torch/torch.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>

#include "bev_sum_pool_hip.h"
#include "bev_max_pool_hip.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bev_sum_pool_forward", &bev_sum_pool_forward,
        "bev_sum_pool_forward");
  m.def("bev_sum_pool_backward", &bev_sum_pool_backward,
        "bev_sum_pool_backward");
  m.def("bev_max_pool_forward", &bev_max_pool_forward,
        "bev_max_pool_forward");
  m.def("bev_max_pool_backward", &bev_max_pool_backward,
        "bev_max_pool_backward");
}