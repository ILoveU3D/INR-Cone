#include <torch/extension.h>

torch::Tensor differentiableFanFlatGradient(torch::Tensor sino, torch::Tensor volume, torch::Tensor projectVector);