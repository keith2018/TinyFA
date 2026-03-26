/*
 * TinyFA
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cassert>
#include <cstdio>

namespace tfa {

template <typename DType_, int HeadDim_, int Br_, int Bc_, int NumWarps_, bool UseTensorCore_, bool UseCpAsync_ = false>
struct KernelConfig {
  using DType = DType_;

  static constexpr int kHeadDim = HeadDim_;
  static constexpr int kBr = Br_;
  static constexpr int kBc = Bc_;
  static constexpr int kNumWarps = NumWarps_;
  static constexpr bool kUseTensorCore = UseTensorCore_;
  static constexpr bool kUseCpAsync = UseCpAsync_;  // SM80+ cp.async

  static constexpr int kWarpSize = 32;
  static constexpr int kNumThreads = kNumWarps * kWarpSize;

  // swizzle requires power-of-two HeadDim
  static constexpr bool kUseSwizzle = (kHeadDim & (kHeadDim - 1)) == 0;
  static constexpr int kBytesPerVecLoad = 16;  // uint4
  static constexpr int kElemsPerVecLoad = kBytesPerVecLoad / sizeof(DType);

  static_assert(kBr % kNumWarps == 0, "kBr must be divisible by kNumWarps");
  static_assert(kUseTensorCore || kBc % kWarpSize == 0, "kBc must be divisible by kWarpSize (FMA)");
  static_assert(kUseTensorCore || kHeadDim % kWarpSize == 0, "kHeadDim must be divisible by kWarpSize (FMA)");

  static constexpr int kRowsPerWarp = kBr / kNumWarps;
};

#define TFA_HEAD_DIM_CASE(N, HEAD_DIM_VAR, ...) \
  case N: {                                     \
    constexpr int HEAD_DIM_VAR = N;             \
    __VA_ARGS__                                 \
    break;                                      \
  }

#ifdef TFA_TARGET_HEADDIM
#define TFA_DISPATCH_HEAD_DIM(headDim, HEAD_DIM_VAR, ...)                                                   \
  [&] {                                                                                                     \
    switch (headDim) {                                                                                      \
      TFA_HEAD_DIM_CASE(TFA_TARGET_HEADDIM, HEAD_DIM_VAR, __VA_ARGS__)                                      \
      default:                                                                                              \
        fprintf(stderr, "TinyFA: unsupported headDim: %d (compiled with TFA_TARGET_HEADDIM=%d)\n", headDim, \
                TFA_TARGET_HEADDIM);                                                                        \
        abort();                                                                                            \
        break;                                                                                              \
    }                                                                                                       \
  }()
#else
#define TFA_DISPATCH_HEAD_DIM(headDim, HEAD_DIM_VAR, ...)              \
  [&] {                                                                \
    switch (headDim) {                                                 \
      TFA_HEAD_DIM_CASE(32, HEAD_DIM_VAR, __VA_ARGS__)                 \
      TFA_HEAD_DIM_CASE(64, HEAD_DIM_VAR, __VA_ARGS__)                 \
      TFA_HEAD_DIM_CASE(96, HEAD_DIM_VAR, __VA_ARGS__)                 \
      TFA_HEAD_DIM_CASE(128, HEAD_DIM_VAR, __VA_ARGS__)                \
      TFA_HEAD_DIM_CASE(192, HEAD_DIM_VAR, __VA_ARGS__)                \
      TFA_HEAD_DIM_CASE(256, HEAD_DIM_VAR, __VA_ARGS__)                \
      default:                                                         \
        fprintf(stderr, "TinyFA: unsupported headDim: %d\n", headDim); \
        abort();                                                       \
        break;                                                         \
    }                                                                  \
  }()
#endif

template <typename Arch, typename DType, int HeadDim, bool IsCausal = false>
struct ConfigForArch {
  using Config = KernelConfig<DType, HeadDim, 32, 32, 4, false>;
};

#define TFA_DEFINE_ARCH_CONFIG(ARCH, DTYPE, HEADDIM, ISCAUSAL, BR, BC, WARPS, ...) \
  template <>                                                                      \
  struct ConfigForArch<ARCH, DTYPE, HEADDIM, ISCAUSAL> {                           \
    using Config = KernelConfig<DTYPE, HEADDIM, BR, BC, WARPS, ##__VA_ARGS__>;     \
  }

using FP32 = float;
using FP16 = __half;
using BF16 = __nv_bfloat16;

struct SM75 {};
struct SM80 {};
struct SM8x {};

// clang-format off

// sm75
TFA_DEFINE_ARCH_CONFIG(SM75, FP32, 32,  false, 128, 64, 4, false);
TFA_DEFINE_ARCH_CONFIG(SM75, FP32, 32,  true,  128, 64, 4, false);
TFA_DEFINE_ARCH_CONFIG(SM75, FP32, 64,  false, 64,  64, 4, false);
TFA_DEFINE_ARCH_CONFIG(SM75, FP32, 64,  true,  64,  64, 4, false);
TFA_DEFINE_ARCH_CONFIG(SM75, FP32, 96,  false, 64,  32, 4, false);
TFA_DEFINE_ARCH_CONFIG(SM75, FP32, 96,  true,  64,  32, 4, false);
TFA_DEFINE_ARCH_CONFIG(SM75, FP32, 128, false, 32,  32, 4, false);
TFA_DEFINE_ARCH_CONFIG(SM75, FP32, 128, true,  32,  32, 4, false);
TFA_DEFINE_ARCH_CONFIG(SM75, FP32, 192, false, 32,  32, 4, false);
TFA_DEFINE_ARCH_CONFIG(SM75, FP32, 192, true,  32,  32, 4, false);
TFA_DEFINE_ARCH_CONFIG(SM75, FP32, 256, false, 32,  32, 4, false);
TFA_DEFINE_ARCH_CONFIG(SM75, FP32, 256, true,  32,  32, 4, false);

TFA_DEFINE_ARCH_CONFIG(SM75, FP16, 32,  false, 128, 128, 4, false);
TFA_DEFINE_ARCH_CONFIG(SM75, FP16, 32,  true,  128, 128, 4, false);
TFA_DEFINE_ARCH_CONFIG(SM75, FP16, 64,  false, 128, 128, 4, false);
TFA_DEFINE_ARCH_CONFIG(SM75, FP16, 64,  true,  128, 128, 4, false);
TFA_DEFINE_ARCH_CONFIG(SM75, FP16, 96,  false, 128, 64,  4, false);
TFA_DEFINE_ARCH_CONFIG(SM75, FP16, 96,  true,  64,  64,  4, false);
TFA_DEFINE_ARCH_CONFIG(SM75, FP16, 128, false, 128, 32,  4, false);
TFA_DEFINE_ARCH_CONFIG(SM75, FP16, 128, true,  64,  64,  4, false);
TFA_DEFINE_ARCH_CONFIG(SM75, FP16, 192, false, 64,  32,  4, false);
TFA_DEFINE_ARCH_CONFIG(SM75, FP16, 192, true,  64,  32,  4, false);
TFA_DEFINE_ARCH_CONFIG(SM75, FP16, 256, false, 64,  32,  4, false);
TFA_DEFINE_ARCH_CONFIG(SM75, FP16, 256, true,  64,  32,  4, false);

TFA_DEFINE_ARCH_CONFIG(SM75, BF16, 32,  false, 128, 128, 4, false);
TFA_DEFINE_ARCH_CONFIG(SM75, BF16, 32,  true,  128, 128, 4, false);
TFA_DEFINE_ARCH_CONFIG(SM75, BF16, 64,  false, 128, 128, 4, false);
TFA_DEFINE_ARCH_CONFIG(SM75, BF16, 64,  true,  128, 128, 4, false);
TFA_DEFINE_ARCH_CONFIG(SM75, BF16, 96,  false, 128, 64,  4, false);
TFA_DEFINE_ARCH_CONFIG(SM75, BF16, 96,  true,  64,  64,  4, false);
TFA_DEFINE_ARCH_CONFIG(SM75, BF16, 128, false, 128, 32,  4, false);
TFA_DEFINE_ARCH_CONFIG(SM75, BF16, 128, true,  64,  64,  4, false);
TFA_DEFINE_ARCH_CONFIG(SM75, BF16, 192, false, 64,  32,  4, false);
TFA_DEFINE_ARCH_CONFIG(SM75, BF16, 192, true,  64,  32,  4, false);
TFA_DEFINE_ARCH_CONFIG(SM75, BF16, 256, false, 64,  32,  4, false);
TFA_DEFINE_ARCH_CONFIG(SM75, BF16, 256, true,  64,  32,  4, false);

// sm80
TFA_DEFINE_ARCH_CONFIG(SM80, FP32, 32,  false, 128, 64, 4, false, true);
TFA_DEFINE_ARCH_CONFIG(SM80, FP32, 32,  true,  128, 64, 4, false, true);
TFA_DEFINE_ARCH_CONFIG(SM80, FP32, 64,  false, 64,  64, 4, false, true);
TFA_DEFINE_ARCH_CONFIG(SM80, FP32, 64,  true,  64,  64, 4, false, true);
TFA_DEFINE_ARCH_CONFIG(SM80, FP32, 96,  false, 64,  32, 4, false, true);
TFA_DEFINE_ARCH_CONFIG(SM80, FP32, 96,  true,  64,  32, 4, false, true);
TFA_DEFINE_ARCH_CONFIG(SM80, FP32, 128, false, 32,  32, 4, false, true);
TFA_DEFINE_ARCH_CONFIG(SM80, FP32, 128, true,  32,  32, 4, false, true);
TFA_DEFINE_ARCH_CONFIG(SM80, FP32, 192, false, 32,  32, 4, false, true);
TFA_DEFINE_ARCH_CONFIG(SM80, FP32, 192, true,  32,  32, 4, false, true);
TFA_DEFINE_ARCH_CONFIG(SM80, FP32, 256, false, 32,  32, 4, false, true);
TFA_DEFINE_ARCH_CONFIG(SM80, FP32, 256, true,  32,  32, 4, false, true);

TFA_DEFINE_ARCH_CONFIG(SM80, FP16, 32,  false, 128, 128, 4, true, true);
TFA_DEFINE_ARCH_CONFIG(SM80, FP16, 32,  true,  128, 128, 4, true, true);
TFA_DEFINE_ARCH_CONFIG(SM80, FP16, 64,  false, 128, 128, 4, true, true);
TFA_DEFINE_ARCH_CONFIG(SM80, FP16, 64,  true,  128, 128, 4, true, true);
TFA_DEFINE_ARCH_CONFIG(SM80, FP16, 96,  false, 128, 64,  4, true, true);
TFA_DEFINE_ARCH_CONFIG(SM80, FP16, 96,  true,  128, 64,  4, true, true);
TFA_DEFINE_ARCH_CONFIG(SM80, FP16, 128, false, 128, 64,  4, true, true);
TFA_DEFINE_ARCH_CONFIG(SM80, FP16, 128, true,  128, 64,  4, true, true);
TFA_DEFINE_ARCH_CONFIG(SM80, FP16, 192, false, 128, 64,  8, true, true);
TFA_DEFINE_ARCH_CONFIG(SM80, FP16, 192, true,  128, 64,  8, true, true);
TFA_DEFINE_ARCH_CONFIG(SM80, FP16, 256, false, 128, 64,  8, true, true);
TFA_DEFINE_ARCH_CONFIG(SM80, FP16, 256, true,  128, 64,  8, true, true);

TFA_DEFINE_ARCH_CONFIG(SM80, BF16, 32,  false, 128, 128, 4, true, true);
TFA_DEFINE_ARCH_CONFIG(SM80, BF16, 32,  true,  128, 128, 4, true, true);
TFA_DEFINE_ARCH_CONFIG(SM80, BF16, 64,  false, 128, 128, 4, true, true);
TFA_DEFINE_ARCH_CONFIG(SM80, BF16, 64,  true,  128, 128, 4, true, true);
TFA_DEFINE_ARCH_CONFIG(SM80, BF16, 96,  false, 128, 64,  4, true, true);
TFA_DEFINE_ARCH_CONFIG(SM80, BF16, 96,  true,  128, 64,  4, true, true);
TFA_DEFINE_ARCH_CONFIG(SM80, BF16, 128, false, 128, 64,  4, true, true);
TFA_DEFINE_ARCH_CONFIG(SM80, BF16, 128, true,  128, 64,  4, true, true);
TFA_DEFINE_ARCH_CONFIG(SM80, BF16, 192, false, 128, 64,  8, true, true);
TFA_DEFINE_ARCH_CONFIG(SM80, BF16, 192, true,  128, 64,  8, true, true);
TFA_DEFINE_ARCH_CONFIG(SM80, BF16, 256, false, 128, 64,  8, true, true);
TFA_DEFINE_ARCH_CONFIG(SM80, BF16, 256, true,  128, 64,  8, true, true);

// sm8x
TFA_DEFINE_ARCH_CONFIG(SM8x, FP32, 32,  false, 128, 64,  4, false, true);
TFA_DEFINE_ARCH_CONFIG(SM8x, FP32, 32,  true,  128, 64,  4, false, true);
TFA_DEFINE_ARCH_CONFIG(SM8x, FP32, 64,  false, 128, 64,  4, false, true);
TFA_DEFINE_ARCH_CONFIG(SM8x, FP32, 64,  true,  128, 64,  4, false, true);
TFA_DEFINE_ARCH_CONFIG(SM8x, FP32, 96,  false, 64,  64,  4, false, true);
TFA_DEFINE_ARCH_CONFIG(SM8x, FP32, 96,  true,  64,  64,  4, false, true);
TFA_DEFINE_ARCH_CONFIG(SM8x, FP32, 128, false, 64,  64,  4, false, true);
TFA_DEFINE_ARCH_CONFIG(SM8x, FP32, 128, true,  64,  64,  4, false, true);
TFA_DEFINE_ARCH_CONFIG(SM8x, FP32, 192, false, 64,  32,  4, false, true);
TFA_DEFINE_ARCH_CONFIG(SM8x, FP32, 192, true,  64,  32,  4, false, true);
TFA_DEFINE_ARCH_CONFIG(SM8x, FP32, 256, false, 32,  32,  4, false, true);
TFA_DEFINE_ARCH_CONFIG(SM8x, FP32, 256, true,  32,  32,  4, false, true);

TFA_DEFINE_ARCH_CONFIG(SM8x, FP16, 32,  false, 128, 128, 4, true, true);
TFA_DEFINE_ARCH_CONFIG(SM8x, FP16, 32,  true,  128, 128, 4, true, true);
TFA_DEFINE_ARCH_CONFIG(SM8x, FP16, 64,  false, 128, 128, 4, true, true);
TFA_DEFINE_ARCH_CONFIG(SM8x, FP16, 64,  true,  128, 128, 4, true, true);
TFA_DEFINE_ARCH_CONFIG(SM8x, FP16, 96,  false, 128, 64,  4, true, true);
TFA_DEFINE_ARCH_CONFIG(SM8x, FP16, 96,  true,  128, 64,  4, true, true);
TFA_DEFINE_ARCH_CONFIG(SM8x, FP16, 128, false, 128, 64,  4, true, true);
TFA_DEFINE_ARCH_CONFIG(SM8x, FP16, 128, true,  128, 64,  4, true, true);
TFA_DEFINE_ARCH_CONFIG(SM8x, FP16, 192, false, 128, 64,  8, true, true);
TFA_DEFINE_ARCH_CONFIG(SM8x, FP16, 192, true,  128, 64,  8, true, true);
TFA_DEFINE_ARCH_CONFIG(SM8x, FP16, 256, false, 64,  64,  4, true, true);
TFA_DEFINE_ARCH_CONFIG(SM8x, FP16, 256, true,  64,  64,  4, true, true);

TFA_DEFINE_ARCH_CONFIG(SM8x, BF16, 32,  false, 128, 128, 4, true, true);
TFA_DEFINE_ARCH_CONFIG(SM8x, BF16, 32,  true,  128, 128, 4, true, true);
TFA_DEFINE_ARCH_CONFIG(SM8x, BF16, 64,  false, 128, 128, 4, true, true);
TFA_DEFINE_ARCH_CONFIG(SM8x, BF16, 64,  true,  128, 128, 4, true, true);
TFA_DEFINE_ARCH_CONFIG(SM8x, BF16, 96,  false, 128, 64,  4, true, true);
TFA_DEFINE_ARCH_CONFIG(SM8x, BF16, 96,  true,  128, 64,  4, true, true);
TFA_DEFINE_ARCH_CONFIG(SM8x, BF16, 128, false, 128, 64,  4, true, true);
TFA_DEFINE_ARCH_CONFIG(SM8x, BF16, 128, true,  128, 64,  4, true, true);
TFA_DEFINE_ARCH_CONFIG(SM8x, BF16, 192, false, 128, 64,  8, true, true);
TFA_DEFINE_ARCH_CONFIG(SM8x, BF16, 192, true,  128, 64,  8, true, true);
TFA_DEFINE_ARCH_CONFIG(SM8x, BF16, 256, false, 64,  64,  4, true, true);
TFA_DEFINE_ARCH_CONFIG(SM8x, BF16, 256, true,  64,  64,  4, true, true);

// clang-format on

#ifdef TFA_TARGET_SM

#if TFA_TARGET_SM == 75
#define TFA_DISPATCH_ARCH(archInt, ...)                        \
  [&] {                                                        \
    assert((archInt) >= 75 && "unsupported GPU architecture"); \
    {                                                          \
      using ArchTag = SM75;                                    \
      __VA_ARGS__                                              \
    }                                                          \
  }()
#elif TFA_TARGET_SM == 80
#define TFA_DISPATCH_ARCH(archInt, ...)                        \
  [&] {                                                        \
    assert((archInt) >= 80 && "unsupported GPU architecture"); \
    {                                                          \
      using ArchTag = SM80;                                    \
      __VA_ARGS__                                              \
    }                                                          \
  }()
#elif TFA_TARGET_SM == 89
#define TFA_DISPATCH_ARCH(archInt, ...)                       \
  [&] {                                                       \
    assert((archInt) > 80 && "unsupported GPU architecture"); \
    {                                                         \
      using ArchTag = SM8x;                                   \
      __VA_ARGS__                                             \
    }                                                         \
  }()
#else
#error "TFA_TARGET_SM has unsupported value. Valid: 75, 80, 89 (sm8x)"
#endif

#else

#define TFA_DISPATCH_ARCH(archInt, ...)                \
  [&] {                                                \
    if ((archInt) > 80) {                              \
      {                                                \
        using ArchTag = SM8x;                          \
        __VA_ARGS__                                    \
      }                                                \
    } else if ((archInt) >= 80) {                      \
      {                                                \
        using ArchTag = SM80;                          \
        __VA_ARGS__                                    \
      }                                                \
    } else if ((archInt) >= 75) {                      \
      {                                                \
        using ArchTag = SM75;                          \
        __VA_ARGS__                                    \
      }                                                \
    } else {                                           \
      assert(false && "unsupported GPU architecture"); \
    }                                                  \
  }()

#endif  // TFA_TARGET_SM

}  // namespace tfa
