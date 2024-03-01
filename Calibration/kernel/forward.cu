#include <cmath>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../include/helper_math.h"

#define BLOCK_X 16
#define BLOCK_Y 16
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define DEBUG

__global__ void forwardKernel(float* sino, float* volume, float* projectVector, const uint angleNum, const uint2 volumeSize, const float2 volumeCenter, const uint detectorSize, const float detectorCenter){
    uint2 volumeIdx = make_uint2(blockIdx.x*blockDim.x+threadIdx.x, blockIdx.y*blockDim.y+threadIdx.y);
    if (volumeIdx.x >= volumeSize.x || volumeIdx.y >= volumeSize.y) return;

    float value = volume[volumeIdx.y*volumeSize.x+volumeIdx.x];
    float2 volumeCoordination = make_float2(volumeIdx) + (float2)volumeCenter;
    float3 point = make_float3(volumeCoordination, 1);
    float3 projectVectorSp1 = make_float3(projectVector[0], projectVector[1], projectVector[2]);
    float3 projectVectorSp2 = make_float3(projectVector[3], projectVector[4], projectVector[5]);
    float u = dot(point, projectVectorSp1);
    float v = dot(point, projectVectorSp2);
    uint sinoIdx = angleNum*detectorSize + round(u/v - detectorCenter);
    atomicAdd(sino+sinoIdx, value);
}

torch::Tensor forward(torch::Tensor volume, torch::Tensor projectVector){
    CHECK_INPUT(volume);
    CHECK_INPUT(projectVector);
    AT_ASSERTM(projectVector.size(1) == 2, "project vector's shape 1 must be 2");
    AT_ASSERTM(projectVector.size(2) == 3, "project vector's shape 2 must be 3");

    const uint2 volumeSize = make_uint2(512, 512);
    const float2 volumeCenter = make_float2(volumeSize) / -2.0;
    const uint detectorSize = 900;
    const float detectorCenter = detectorSize / -2.0;

    int angleNum = projectVector.size(0);
    auto out = torch::zeros({angleNum, detectorSize}).to(volume.device());
    auto outPtr = out.data<float>();
    auto volumePtr = volume.data<float>();

    for(int idx=0;idx<angleNum;idx++){
      auto projectVectorPtr = projectVector.index({idx,"..."}).data<float>();
      const dim3 blockSize = dim3(BLOCK_X, BLOCK_Y);
      const dim3 gridSize = dim3(volumeSize.x/BLOCK_X, volumeSize.y/BLOCK_Y);
      forwardKernel<<<gridSize, blockSize>>>(outPtr, volumePtr, projectVectorPtr, idx, volumeSize, volumeCenter, detectorSize, detectorCenter);
    }

    return out;
}