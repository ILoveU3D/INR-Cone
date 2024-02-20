#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../include/helper_math.h"
#include "../include/helper.h"

#define BLOCK_X 16
#define BLOCK_Y 16
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
// #define DEBUG

__global__ void differentiableKernel(float* error, float* sino, float* volume, float* projectVector, 
const uint angleNum, const uint2 volumeSize, const float2 volumeCenter, const uint detectorSize, const float detectorCenter){

    uint2 volumeIdx = make_uint2(blockIdx.x*blockDim.x+threadIdx.x, blockIdx.y*blockDim.y+threadIdx.y);
    if (volumeIdx.x >= volumeSize.x || volumeIdx.y >= volumeSize.y || volumeIdx.x <= 0 || volumeIdx.y <= 0) return;

    float2 volumeCoordination = make_float2(volumeIdx) + (float2)volumeCenter;
    float3 point = make_float3(volumeCoordination, 1);
    float volumeError = volume[volumeIdx.y * volumeSize.x + volumeIdx.x];
    float3 projectVectorSp1 = make_float3(projectVector[angleNum*6+0], projectVector[angleNum*6+1], projectVector[angleNum*6+2]);
    float3 projectVectorSp2 = make_float3(projectVector[angleNum*6+3], projectVector[angleNum*6+4], projectVector[angleNum*6+5]);
    float u = dot(point, projectVectorSp1);
    float v = dot(point, projectVectorSp2);
    float gsino = interpolate_1d(sino, angleNum*detectorSize + u/v - detectorCenter, angleNum*detectorSize, angleNum*detectorSize + detectorSize);
    float3 errorSp1 = point * gsino * volumeError;
    float3 errorSp2 = point * -u * gsino * volumeError;
    if(fabs(v)>1e-5){
      errorSp1 /= v;
      errorSp2 /= (v*v);
    }

    #if defined DEBUG
    if(volumeIdx.x == 256 && volumeIdx.y == 256 && angleNum == 0){
      printf("point(%f,%f,%f)\n", point.x, point.y, point.z);
      printf("u=%f v=%f\n",u,v);
      printf("w=%f\n", __fdividef(u, v) - detectorCenter);
      printf("grad(D)=%f\n", gsino);
      printf("errorSp1(%f,%f,%f)\n", errorSp1.x, errorSp1.y, errorSp1.z);
      printf("errorSp2(%f,%f,%f)\n", errorSp2.x, errorSp2.y, errorSp2.z);
      printf("idx=%d\n", volumeIdx.y * volumeSize.x + volumeIdx.x);
      printf("volume error=%f\n", volumeError);
    }
    #endif

    atomicAdd(error+angleNum*6, errorSp1.x);
    atomicAdd(error+angleNum*6+1, errorSp1.y);
    atomicAdd(error+angleNum*6+2, errorSp1.z);
    atomicAdd(error+angleNum*6+3, errorSp2.x);
    atomicAdd(error+angleNum*6+4, errorSp2.y);
    atomicAdd(error+angleNum*6+5, errorSp2.z);
}

torch::Tensor differentiableFanFlatGradient(torch::Tensor sino, torch::Tensor volume, torch::Tensor projectVector){
    CHECK_INPUT(sino);
    CHECK_INPUT(volume);
    CHECK_INPUT(projectVector);
    AT_ASSERTM(projectVector.size(1) == 2, "project vector's shape 1 must be 2");
    AT_ASSERTM(projectVector.size(2) == 3, "project vector's shape 2 must be 3");

    const uint2 volumeSize = make_uint2(512, 512);
    const float2 volumeCenter = make_float2(volumeSize) / -2.0;
    const uint detectorSize = 900;
    const float detectorCenter = detectorSize / -2.0;

    int angleNum = projectVector.size(0);
    auto out = torch::zeros({angleNum,2,3}).to(volume.device());
    auto sinoPtr = sino.data<float>();
    auto volumePtr = volume.data<float>();

    for(int idx=0;idx<angleNum;idx++){
      auto outPtr = out.data<float>();
      auto projectVectorPtr = projectVector.data<float>();
      const dim3 blockSize = dim3(BLOCK_X, BLOCK_Y);
      const dim3 gridSize = dim3(volumeSize.x/BLOCK_X, volumeSize.y/BLOCK_Y);
      differentiableKernel<<<gridSize, blockSize>>>(outPtr, sinoPtr, volumePtr, projectVectorPtr, idx, volumeSize, volumeCenter, detectorSize, detectorCenter);
    }

    return out;
}