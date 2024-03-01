#include "cuda_runtime.h"

inline __device__ float interpolate_1d(float* array, float pos, uint down, uint up){
    int left = floor(pos);
    int right = left + 1;
    float delta = pos - left;
    if(left<down||right>=up)
      return 0;
    return array[left] * delta + (1-delta) * array[right];
}