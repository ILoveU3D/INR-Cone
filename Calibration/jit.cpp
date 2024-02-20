#include <torch/extension.h>
#include "include/differentiableFanFlatGradient.cuh"
#include "include/fanFlatTransform.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("differentiableFanFlatGradient", &differentiableFanFlatGradient, "Get Error of projection matrix in fanbeam");
  m.def("backward", &backward, "backward projection");
  m.def("forward", &forward, "forward projection");
}