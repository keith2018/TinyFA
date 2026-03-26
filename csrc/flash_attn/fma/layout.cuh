/*
 * TinyFA
 * @author 	: keith@robot9.me
 *
 */

#pragma once

namespace tfa::fma {

// row-major
struct LayoutIdentity {
  __device__ __forceinline__ static int map(int row, int col, int stride) { return row * stride + col; }
};

// Ref https://leimao.github.io/blog/CuTe-Swizzle/.
template <int BBits = 3, int MBase = 0, int SShift = 3>
struct CuteSwizzle {
  static constexpr int kMBase = MBase;
  static constexpr int kMaskBits = BBits;
  static constexpr int kMaskShift = SShift;

  static constexpr int kBitMask = (1 << kMaskBits) - 1;
  static constexpr int kRowMask = kBitMask << (kMBase + kMaskShift);

  __device__ __forceinline__ constexpr static int apply(int offset) {
    const int rowShifted = (offset & kRowMask) >> kMaskShift;
    return offset ^ rowShifted;
  }
};

template <typename DType, int HeadDim>
struct LayoutSwizzle {
  static constexpr int kVecBytes = 16;
  static constexpr int kDTypeBytes = sizeof(DType);
  static constexpr int kVecElem = kVecBytes / kDTypeBytes;

  static constexpr int kMBase = (kVecElem == 8) ? 3 : (kVecElem == 4) ? 2 : 0;

  static constexpr int kHeadDimBits = (HeadDim == 256)   ? 8
                                      : (HeadDim == 128) ? 7
                                      : (HeadDim == 64)  ? 6
                                      : (HeadDim == 32)  ? 5
                                                         : 0;

  static_assert(kHeadDimBits > 0, "Unsupported HeadDim for swizzle layout");
  static constexpr int kSShift = kHeadDimBits - kMBase;

  using Swizzle = CuteSwizzle<3, kMBase, kSShift>;

  __device__ __forceinline__ static int map(int row, int col, int stride) {
    int offset = row * stride + col;
    return Swizzle::apply(offset);
  }
};

template <typename Config>
using TileLayout =
    typename std::conditional<Config::kUseSwizzle, LayoutSwizzle<typename Config::DType, Config::kHeadDim>,
                              LayoutIdentity>::type;

}  // namespace tfa::fma
