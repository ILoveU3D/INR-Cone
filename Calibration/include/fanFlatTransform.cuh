#include <torch/extension.h>

torch::Tensor backward(torch::Tensor sino, torch::Tensor projectVector);
torch::Tensor forward(torch::Tensor volume, torch::Tensor projectVector);