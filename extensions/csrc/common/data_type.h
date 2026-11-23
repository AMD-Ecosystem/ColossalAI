#pragma once

#if defined(COLOSSAL_WITH_CUDA) || defined(COLOSSAL_WITH_HIP)
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#if defined(USE_ROCM)
// CUDA bf16 conversions that HIP's bf16 header does not provide.
__host__ __device__ inline __nv_bfloat16 __float2bfloat16_rn(float val) {
  return __float2bfloat16(val);
}

__host__ __device__ inline __nv_bfloat162 __float2bfloat162_rn(float val) {
  return __float22bfloat162_rn(make_float2(val, val));
}

__host__ __device__ inline __nv_bfloat162 __floats2bfloat162_rn(float lo,
                                                                float hi) {
  return __float22bfloat162_rn(make_float2(lo, hi));
}
#endif
#endif

namespace colossalAI {
namespace dtype {

struct bfloat164 {
#if defined(COLOSSAL_WITH_CUDA) || defined(COLOSSAL_WITH_HIP)
  __nv_bfloat162 x;
  __nv_bfloat162 y;
#endif
};

struct bfloat168 {
#if defined(COLOSSAL_WITH_CUDA) || defined(COLOSSAL_WITH_HIP)
  __nv_bfloat162 x;
  __nv_bfloat162 y;
  __nv_bfloat162 z;
  __nv_bfloat162 w;
#endif
};

struct half4 {
#if defined(COLOSSAL_WITH_CUDA) || defined(COLOSSAL_WITH_HIP)
  half2 x;
  half2 y;
#endif
};

struct half8 {
#if defined(COLOSSAL_WITH_CUDA) || defined(COLOSSAL_WITH_HIP)
  half2 x;
  half2 y;
  half2 z;
  half2 w;
#endif
};

struct float8 {
#if defined(COLOSSAL_WITH_CUDA) || defined(COLOSSAL_WITH_HIP)
  float2 x;
  float2 y;
  float2 z;
  float2 w;
#endif
};

}  // namespace dtype
}  // namespace colossalAI
