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

__global__ void differentiableConeKernel(float* error, float* sinoX, float* sinoY, float* volume, float* projectVector, 
const uint angleNum, const uint3 volumeSize, const float3 volumeCenter, const uint2 detectorSize, const float2 detectorCenter){

    uint2 volumeIdx = make_uint2(blockIdx.x*blockDim.x+threadIdx.x, blockIdx.y*blockDim.y+threadIdx.y);
    if (volumeIdx.x >= volumeSize.x || volumeIdx.y >= volumeSize.y || volumeIdx.x <= 0 || volumeIdx.y <= 0) return;

    for(uint z=1;z<volumeSize.z;z++){
      float3 volumeCoordination = make_float3(make_uint3(volumeIdx, z)) + (float3)volumeCenter;
      float4 point = make_float4(volumeCoordination, 1);
      uint volumePathIdx = z * volumeSize.y * volumeSize.x + volumeIdx.y * volumeSize.x + volumeIdx.x;
      float volumeError = volume[volumePathIdx];
      float4 projectVectorSp1 = make_float4(projectVector[angleNum*6+0], projectVector[angleNum*6+1], projectVector[angleNum*6+2], projectVector[angleNum*6+3]);
      float4 projectVectorSp2 = make_float4(projectVector[angleNum*6+4], projectVector[angleNum*6+5], projectVector[angleNum*6+6], projectVector[angleNum*6+7]);
      float4 projectVectorSp3 = make_float4(projectVector[angleNum*6+8], projectVector[angleNum*6+9], projectVector[angleNum*6+10], projectVector[angleNum*6+11]);
      float u = dot(point, projectVectorSp1);
      float v = dot(point, projectVectorSp2);
      float w = dot(point, projectVectorSp3);
      //interpolate 2d
      int pos_x = u/w - detectorCenter.x;
      int pos_y = v/w - detectorCenter.y;
      int pos_x_floor = floor(pos_x);
      int pos_y_floor = floor(pos_y);
      int pos_x_ceil = pos_x_floor + 1;
      int pos_y_ceil = pos_y_floor + 1;
      float delta_x = pos_x - pos_x_floor;
      float delta_y = pos_y - pos_x_floor;
      float gsinoX, gsinoY = 0;
      if(!(pos_x_floor<=0||pos_x_ceil>=detectorSize.x||pos_y_floor<=0||pos_y_ceil>=detectorSize.y)){
        uint base = angleNum*detectorSize.x*detectorSize.y;
        float a = sinoX[base + pos_x_floor*detectorSize.x+pos_y_floor];
        float b = sinoX[base + pos_x_ceil*detectorSize.x+pos_y_floor];
        float c = sinoX[base + pos_x_floor*detectorSize.x+pos_y_ceil];
        float d = sinoX[base + pos_x_ceil*detectorSize.x+pos_y_ceil];
        gsinoX = delta_y * (delta_x * d + (1-delta_x) * c) + (1-delta_y) * (delta_x * b + (1-delta_x) * a);
        a = sinoY[base + pos_x_floor*detectorSize.x+pos_y_floor];
        b = sinoY[base + pos_x_ceil*detectorSize.x+pos_y_floor];
        c = sinoY[base + pos_x_floor*detectorSize.x+pos_y_ceil];
        d = sinoY[base + pos_x_ceil*detectorSize.x+pos_y_ceil];
        gsinoY = delta_y * (delta_x * d + (1-delta_x) * c) + (1-delta_y) * (delta_x * b + (1-delta_x) * a);
      }
      float4 errorSp1 = point * gsinoX * volumeError;
      float4 errorSp2 = point * gsinoY * volumeError;
      float4 errorSp3 = (point * gsinoX * u + point * gsinoY * v)*-volumeError;
      if(fabs(w)>1e-5){
        errorSp1 /= w;
        errorSp2 /= w;
        errorSp3 /= (w*w);
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

      error[volumePathIdx] = errorSp1.x;
      error[volumePathIdx+1] = errorSp1.y;
      error[volumePathIdx+2] = errorSp1.z;
      error[volumePathIdx+3] = errorSp1.w;
      error[volumePathIdx+4] = errorSp2.x;
      error[volumePathIdx+5] = errorSp2.y;
      error[volumePathIdx+6] = errorSp2.z;
      error[volumePathIdx+7] = errorSp2.w;
      error[volumePathIdx+8] = errorSp3.x;
      error[volumePathIdx+9] = errorSp3.y;
      error[volumePathIdx+10] = errorSp3.z;
      error[volumePathIdx+11] = errorSp3.w;
    }
}

// beta
torch::Tensor differentiableConeGradient(torch::Tensor sino_x, torch::Tensor sino_y, torch::Tensor volume, torch::Tensor projectVector){
    CHECK_INPUT(sino_x);
    CHECK_INPUT(sino_y);
    CHECK_INPUT(volume);
    CHECK_INPUT(projectVector);
    AT_ASSERTM(projectVector.size(1) == 3, "project vector's shape 1 must be 2");
    AT_ASSERTM(projectVector.size(2) == 4, "project vector's shape 2 must be 3");

    const uint3 volumeSize = make_uint3(512, 512, 16);
    const float3 volumeCenter = make_float3(volumeSize) / -2.0;
    const uint2 detectorSize = make_uint2(78, 160);
    const float2 detectorCenter = make_float2(detectorSize) / -2.0;

    int angleNum = projectVector.size(0);
    auto out = torch::zeros({angleNum,3,4}).to(volume.device());
    auto temp = torch::zeros({volumeSize.x*volumeSize.y*volumeSize.z, 3, 4}).to(volume.device());
    auto sinoPtrX = sino_x.data<float>();
    auto sinoPtrY = sino_y.data<float>();
    auto volumePtr = volume.data<float>();

    for(int idx=0;idx<angleNum;idx++){
      auto outPtr = temp.data<float>();
      auto projectVectorPtr = projectVector.data<float>();
      const dim3 blockSize = dim3(BLOCK_X, BLOCK_Y);
      const dim3 gridSize = dim3(volumeSize.x/BLOCK_X, volumeSize.y/BLOCK_Y);
      differentiableConeKernel<<<gridSize, blockSize>>>(outPtr, sinoPtrX, sinoPtrY, volumePtr, projectVectorPtr, idx, volumeSize, volumeCenter, detectorSize, detectorCenter);
      out += torch::sum(temp, 0);
      temp.zero_();
    }

    return out;
}