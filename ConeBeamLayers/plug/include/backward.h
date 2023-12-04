#include <torch/extension.h>

torch::Tensor backward(torch::Tensor sino, torch::Tensor _volumeSize, torch::Tensor _detectorSize, torch::Tensor projectVector, const int gap, const long device);