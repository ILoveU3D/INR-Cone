#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../include/helper_math.h"

#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_A 180
#define TEXA (21*1080)
#define PI 3.14159265359
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// 弦图的纹理内存
texture<float, cudaTextureType3D, cudaReadModeElementType> sinoTexture;

__global__ void backwardKernel(float* volume, const uint3 volumeSize, const uint2 detectorSize, const float* projectVector, const uint index, const int anglesNum, const float bias, const float3 volumeCenter, const float2 detectorCenter){
    // 体素驱动，代表一个体素点
    uint3 volumeIdx = make_uint3(blockIdx.x*blockDim.x+threadIdx.x, blockIdx.y*blockDim.y+threadIdx.y, blockIdx.z*blockDim.z+threadIdx.z);
    if (volumeIdx.x >= volumeSize.x || volumeIdx.y >= volumeSize.y){
        return;
    }

    for(int k=0;k<volumeSize.z;k++){
        float value = 0.0f;
        bool found = false;
        for(int angleIdx = index;angleIdx<index+BLOCK_A;angleIdx++){
            float3 sourcePosition = make_float3(projectVector[angleIdx*12], projectVector[angleIdx*12+1], -projectVector[angleIdx*12+2]);
            float3 detectorPosition = make_float3(projectVector[angleIdx*12+3], projectVector[angleIdx*12+4], -projectVector[angleIdx*12+5]);
            float3 v = make_float3(projectVector[angleIdx*12+6], projectVector[angleIdx*12+7], projectVector[angleIdx*12+8]);
            float3 u = make_float3(projectVector[angleIdx*12+9], projectVector[angleIdx*12+10], projectVector[angleIdx*12+11]);
            float3 coordinates = make_float3(volumeCenter.x + volumeIdx.x, volumeCenter.y + volumeIdx.y,volumeCenter.z+k);
            float fScale = __fdividef(1.0f, det3(u, v, sourcePosition-coordinates));
            float detectorX = fScale * det3(coordinates-sourcePosition,v,sourcePosition-detectorPosition) - detectorCenter.x + bias;
            float detectorY = fScale * det3(u, coordinates-sourcePosition,sourcePosition-detectorPosition) - detectorCenter.y;
            float fr = fScale * det3(u, v, sourcePosition-detectorPosition);
            // if(detectorY >= 0 && detectorY <= detectorSize.y){
            //     if(detectorX >= 0 && detectorX <= detectorSize.x)
            //         found = true;
            //     else if(detectorX < 0 && detectorX >= -2){
            //         found = true;
            //         detectorX = 0;
            //     }else if(detectorX > detectorSize.x && detectorX <= detectorSize.x+2){
            //         found = true;
            //         detectorX = detectorSize.x;
            //     }else continue;
            // }
            if(detectorX < -1 || detectorX > detectorSize.x+1 || detectorY < -1 || detectorY > detectorSize.y+1) continue;
            else found = true;
            value += fr * tex3D(sinoTexture, detectorX, detectorY, angleIdx%TEXA+0.5f);
        }
        int idx = k * volumeSize.x * volumeSize.y + volumeIdx.y * volumeSize.x + volumeIdx.x;
        if(found) volume[idx] += value * 2 * PI / anglesNum;
    }
}

torch::Tensor backward(torch::Tensor sino, torch::Tensor _volumeSize, torch::Tensor _detectorSize, torch::Tensor projectVector, const int gap, const long device){
    CHECK_INPUT(sino);
    CHECK_INPUT(_volumeSize);
    AT_ASSERTM(_volumeSize.size(0) == 3, "volume size's length must be 3");
    CHECK_INPUT(_detectorSize);
    AT_ASSERTM(_detectorSize.size(0) == 2, "detector size's length must be 2");
    CHECK_INPUT(projectVector);
    AT_ASSERTM(projectVector.size(1) == 12, "project vector's shape must be [angle's number, 12]");

    int angles = projectVector.size(0);
    auto out = torch::zeros({sino.size(0), 1, _volumeSize[2].item<int>(), _volumeSize[1].item<int>(), _volumeSize[0].item<int>()}).to(sino.device());
    float* outPtr = out.data<float>();
    float* sinoPtr = sino.data<float>();

    // 初始化纹理
    cudaSetDevice(device);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    sinoTexture.addressMode[0] = cudaAddressModeBorder;
    sinoTexture.addressMode[1] = cudaAddressModeBorder;
    sinoTexture.addressMode[2] = cudaAddressModeBorder;
    sinoTexture.filterMode = cudaFilterModeLinear;
    sinoTexture.normalized = false;

    // 体块和探测器的大小位置向量化
    uint3 volumeSize = make_uint3(_volumeSize[0].item<int>(), _volumeSize[1].item<int>(), _volumeSize[2].item<int>());
    uint2 detectorSize = make_uint2(_detectorSize[0].item<int>(), _detectorSize[1].item<int>());
    uint2 sinoTextureSize = make_uint2(detectorSize.x + gap, detectorSize.y);
    float3 volumeCenter = make_float3(volumeSize) / -2.0;
    float2 detectorCenter = make_float2(detectorSize) / -2.0;

    for(int batch = 0;batch < sino.size(0); batch++){
        for (int subAngle = 0;subAngle<angles;subAngle+=TEXA){
            float* sinoPtrPitch = sinoPtr + sinoTextureSize.x * sinoTextureSize.y * subAngle + sinoTextureSize.x * sinoTextureSize.y * angles * batch;
            float* outPtrPitch = outPtr + volumeSize.x * volumeSize.y * volumeSize.z * batch;

            // 绑定纹理
            cudaExtent m_extent = make_cudaExtent(sinoTextureSize.x, sinoTextureSize.y, TEXA);
            cudaArray *sinoArray;
            cudaMalloc3DArray(&sinoArray, &channelDesc, m_extent);
            cudaMemcpy3DParms copyParams = {0};
            copyParams.srcPtr = make_cudaPitchedPtr((void*)sinoPtrPitch, sinoTextureSize.x*sizeof(float), sinoTextureSize.x, sinoTextureSize.y);
            copyParams.dstArray = sinoArray;
            copyParams.kind = cudaMemcpyDeviceToDevice;
            copyParams.extent = m_extent;
            cudaMemcpy3D(&copyParams);
            cudaBindTextureToArray(sinoTexture, sinoArray, channelDesc);

            // 以角度为单位做体素驱动的反投影
            const dim3 blockSize = dim3(BLOCK_X, BLOCK_Y, 1);
            const dim3 gridSize = dim3(volumeSize.x / BLOCK_X + 1, volumeSize.y / BLOCK_Y + 1, 1);
            for (int angle = subAngle; angle < subAngle+TEXA; angle+=BLOCK_A){
               backwardKernel<<<gridSize, blockSize>>>(outPtrPitch, volumeSize, detectorSize, (float*)projectVector.data<float>(), angle, angles/21, gap/2.0, volumeCenter, detectorCenter);
            }
            // 解绑纹理
            cudaUnbindTexture(sinoTexture);
            cudaFreeArray(sinoArray);
        }
    }
    return out;
}
