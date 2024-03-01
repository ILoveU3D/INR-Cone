#include <torch/extension.h>

//beta
torch::Tensor differentiableConeGradient(torch::Tensor sino_x, torch::Tensor sino_y, torch::Tensor volume, torch::Tensor projectVector);