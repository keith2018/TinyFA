/*
 * TinyFA
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <cstdint>

namespace tfa {

__device__ __forceinline__ void cpAsync128bZfill(void* smemPtr, const void* gmemPtr, bool pred) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  uint32_t smemAddr = static_cast<uint32_t>(__cvta_generic_to_shared(smemPtr));
  int srcSize = pred ? 16 : 0;
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::"r"(smemAddr), "l"(gmemPtr), "r"(srcSize));
#endif
}

__device__ __forceinline__ void cpAsyncCommit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ __forceinline__ void cpAsyncWaitAll() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_all;\n" ::);
#endif
}

template <int N>
__device__ __forceinline__ void cpAsyncWaitGroup() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

}  // namespace tfa
