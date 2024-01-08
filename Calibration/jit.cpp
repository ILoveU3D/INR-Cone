#include <torch/extension.h>
#include "include/differentiableFanFlatGradient.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("differentiableFanFlatGradient", &differentiableFanFlatGradient, "Get Error of projection matrix in fanbeam");
}