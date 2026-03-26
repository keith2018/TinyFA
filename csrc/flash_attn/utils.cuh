/*
 * TinyFA
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <type_traits>

namespace tfa {

#define TFA_CUDA_CHECK(call)                                                                  \
  do {                                                                                        \
    cudaError_t err = call;                                                                   \
    if (err != cudaSuccess) {                                                                 \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      abort();                                                                                \
    }                                                                                         \
  } while (0)

template <typename T>
__host__ __device__ __forceinline__ constexpr T ceilDiv(T m, T n) {
  return (m + n - 1) / n;
}

template <typename T>
__host__ __device__ __forceinline__ float toFloat(T val) {
  if constexpr (std::is_same_v<T, __half>) {
    return __half2float(val);
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return __bfloat162float(val);
  } else {
    return static_cast<float>(val);
  }
}

template <typename T>
__host__ __device__ __forceinline__ T fromFloat(float val) {
  if constexpr (std::is_same_v<T, __half>) {
    return __float2half(val);
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return __float2bfloat16(val);
  } else {
    return static_cast<T>(val);
  }
}

namespace detail {

template <int N>
struct InvSqrt;

// clang-format off
template <> struct InvSqrt<32>  { static constexpr float value = 0.17677669529663689f; };   // 1/sqrt(32)
template <> struct InvSqrt<64>  { static constexpr float value = 0.125f;               };   // 1/sqrt(64)
template <> struct InvSqrt<96>  { static constexpr float value = 0.10206207261596576f; };   // 1/sqrt(96)
template <> struct InvSqrt<128> { static constexpr float value = 0.08838834764831845f; };   // 1/sqrt(128)
template <> struct InvSqrt<192> { static constexpr float value = 0.07216878364870323f; };   // 1/sqrt(192)
template <> struct InvSqrt<256> { static constexpr float value = 0.0625f;              };   // 1/sqrt(256)
// clang-format on

}  // namespace detail

// scale = 1/sqrt(d) * log2(e), expf(x / sqrt(d)) -> exp2f(x * scale)
template <int kHeadDim>
struct AttentionScale {
  static constexpr float kLog2e = 1.4426950408889634f;
  static constexpr float value = detail::InvSqrt<kHeadDim>::value * kLog2e;
};

template <int kReduceWidth = 32, typename T>
__device__ __forceinline__ T warpReduceMax(T val) {
#pragma unroll
  for (int delta = kReduceWidth / 2; delta > 0; delta >>= 1) {
    val = max(val, __shfl_xor_sync(0xffffffff, val, delta));
  }
  return val;
}

template <int kReduceWidth = 32, typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
#pragma unroll
  for (int delta = kReduceWidth / 2; delta > 0; delta >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, delta);
  }
  return val;
}

#define TFA_DEFINE_KERNEL_WRAPPER(DEVICE_FN)                  \
  template <typename Config, bool kIsCausal, typename Params> \
  __global__ void flashAttentionKernel(Params params) {       \
    extern __shared__ char smemBuf[];                         \
    DEVICE_FN<Config, kIsCausal>(params, smemBuf);            \
  }

inline int getRuntimeArch() {
  static int cachedArch = [] {
    int device;
    TFA_CUDA_CHECK(cudaGetDevice(&device));
    int major, minor;
    TFA_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    TFA_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    return major * 10 + minor;
  }();
  return cachedArch;
}

inline int getMaxSmemPerBlock() {
  static int cached = [] {
    int device = 0;
    TFA_CUDA_CHECK(cudaGetDevice(&device));
    int maxSmem = 0;
    TFA_CUDA_CHECK(cudaDeviceGetAttribute(&maxSmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    return maxSmem;
  }();
  return cached;
}

template <typename KernelFunc>
bool configureSmem(KernelFunc kernel, size_t smemSize) {
  if (smemSize < 48 * 1024) {
    return true;
  }

  int maxSmem = getMaxSmemPerBlock();
  if (maxSmem <= 0 || static_cast<size_t>(maxSmem) < smemSize) {
    fprintf(stderr,
            "[TinyFA] ERROR: kernel requires %zu bytes (%.1f KB) shared memory, "
            "but device only supports %d bytes (%.1f KB). \n",
            smemSize, static_cast<float>(smemSize) / 1024.0f, maxSmem, static_cast<float>(maxSmem) / 1024.0f);
    return false;
  }

  TFA_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smemSize));
  return true;
}

}  // namespace tfa
