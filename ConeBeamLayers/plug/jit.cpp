#include <torch/extension.h>
#include "include/forward.h"
#include "include/backward.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Standard forward (CUDA)");
  m.def("backward", &backward, "Standard backward (CUDA)");
}