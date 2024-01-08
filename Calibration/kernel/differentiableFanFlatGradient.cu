#include <iostream>
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
texture<float, cudaTextureType1D, cudaReadModeElementType> sinoTexture;

__device__ inline void _atomicAdd(float *address, float val){
      int i_val = __float_as_int(val);
      int tmp0 = 0, tmp1;
      while((tmp1 = atomicCAS((int *)address, tmp0, i_val)) != tmp0){
        tmp0 = tmp1;
        i_val = __float_as_int(val + __int_as_float(tmp1));
      }
}

__global__ void differentiableKernel(float* error, float* volume, float* projectVector, const uint2 volumeSize, const float2 volumeCenter, const float detectorCenter){
    uint2 volumeIdx = make_uint2(blockIdx.x*blockDim.x+threadIdx.x, blockIdx.y*blockDim.y+threadIdx.y);
    if (volumeIdx.x >= volumeSize.x || volumeIdx.y >= volumeSize.y) return;

    float2 volumeCoordination = make_float2(volumeIdx) + (float2)volumeCenter;
    float3 point = make_float3(volumeCoordination, 1);
    float volumeError = volume[volumeIdx.y * volumeSize.x + volumeIdx.x];
    float3 projectVectorSp1 = make_float3(projectVector[0], projectVector[1], projectVector[2]);
    float3 projectVectorSp2 = make_float3(projectVector[3], projectVector[4], projectVector[5]);
    float u = dot(point, projectVectorSp1);
    float v = dot(point, projectVectorSp2);
    float gsino = tex1Dfetch(sinoTexture, __fdividef(u, v) - detectorCenter);
    float3 errorSp1 = point * gsino * volumeError;
    float3 errorSp2 = point * -u * gsino * volumeError;
    if(fabs(v)>1e-5){
      errorSp1 /= v;
      errorSp2 /= (v*v);
    }

    #if defined DEBUG
    if(volumeIdx.x == 256 && volumeIdx.y == 256){
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

    _atomicAdd(error, errorSp1.x);
    _atomicAdd(error+1, errorSp1.y);
    _atomicAdd(error+2, errorSp1.z);
    _atomicAdd(error+3, errorSp2.x);
    _atomicAdd(error+4, errorSp2.y);
    _atomicAdd(error+5, errorSp2.z);
}

torch::Tensor differentiableFanFlatGradient(torch::Tensor sino, torch::Tensor volume, torch::Tensor projectVector){
    CHECK_INPUT(sino);
    CHECK_INPUT(volume);
    CHECK_INPUT(projectVector);
    AT_ASSERTM(projectVector.size(0) == 2, "project vector's shape 0 must be 2");
    AT_ASSERTM(projectVector.size(1) == 3, "project vector's shape 1 must be 3");

    auto out = torch::zeros({2,3}).to(volume.device());
    float* outPtr = out.data<float>();
    float* sinoPtr = sino.data<float>();
    float* volumePtr = volume.data<float>();
    float* projectVectorPtr = projectVector.data<float>();

    const uint2 volumeSize = make_uint2(512, 512);
    const float2 volumeCenter = make_float2(volumeSize) / -2.0;
    const uint detectorSize = 900;
    const float detectorCenter = detectorSize / -2.0;

    sinoTexture.filterMode = cudaFilterModeLinear;
    sinoTexture.normalized = false;
    cudaBindTexture(NULL, sinoTexture, sinoPtr, detectorSize*sizeof(float));
    const dim3 blockSize = dim3(BLOCK_X, BLOCK_Y);
    const dim3 gridSize = dim3(volumeSize.x/BLOCK_X, volumeSize.y/BLOCK_Y);
    differentiableKernel<<<gridSize, blockSize>>>(outPtr, volumePtr, projectVectorPtr, volumeSize, volumeCenter, detectorCenter);
    cudaUnbindTexture(sinoTexture);

    return out;
}