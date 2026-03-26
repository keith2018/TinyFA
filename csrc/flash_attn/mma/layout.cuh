/*
 * TinyFA
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cutlass/numeric_conversion.h>

#include <cute/tensor.hpp>

namespace tfa::mma {

template <typename DType>
struct CutlassElemType;

template <>
struct CutlassElemType<__half> {
  using type = cutlass::half_t;
};

template <>
struct CutlassElemType<__nv_bfloat16> {
  using type = cutlass::bfloat16_t;
};

template <typename Config>
struct SmemLayout {
  using DType = typename Config::DType;
  using Element = typename CutlassElemType<DType>::type;

  static constexpr int kHeadDim = Config::kHeadDim;
  static constexpr int kBr = Config::kBr;
  static constexpr int kBc = Config::kBc;
  static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;  // prefer 64 if HeadDim is aligned

  using SmemLayoutAtomNoSwizzle =
      cute::Layout<cute::Shape<cute::_8, cute::Int<kBlockKSmem>>, cute::Stride<cute::Int<kBlockKSmem>, cute::_1>>;

  static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;
  using SmemLayoutSwizzle = cute::Swizzle<kSwizzle, 3, 3>;
  using SmemLayoutAtom = decltype(cute::composition(SmemLayoutSwizzle{}, SmemLayoutAtomNoSwizzle{}));

  using SmemLayoutQ =
      decltype(cute::tile_to_shape(SmemLayoutAtom{}, cute::Shape<cute::Int<kBr>, cute::Int<kHeadDim>>{}));
  using SmemLayoutKV =
      decltype(cute::tile_to_shape(SmemLayoutAtom{}, cute::Shape<cute::Int<kBc>, cute::Int<kHeadDim>>{}));
  using SmemLayoutO =
      decltype(cute::tile_to_shape(SmemLayoutAtomNoSwizzle{}, cute::Shape<cute::Int<kBr>, cute::Int<kHeadDim>>{}));

  // V transposed layout (P @ V)
  using SmemLayoutVTransposed = decltype(cute::composition(
      SmemLayoutKV{}, cute::make_layout(cute::Shape<cute::Int<kHeadDim>, cute::Int<kBc>>{}, cute::GenRowMajor{})));
  using SmemLayoutVTransposedNoSwizzle = decltype(cute::get_nonswizzle_portion(SmemLayoutVTransposed{}));

  static constexpr int kSmemSize = (cute::size(SmemLayoutQ{}) + cute::size(SmemLayoutKV{}) * 2) * sizeof(Element);
};

// MMA accumulator layout (MMA=4, MMA_M, MMA_N) -> (row, col)
template <typename Layout_>
__forceinline__ __device__ auto convertAccRowCol(Layout_ accLayout) {
  using namespace cute;

  static_assert(decltype(size<0>(accLayout))::value == 4);
  static_assert(decltype(rank(accLayout))::value == 3);
  auto divided = logical_divide(accLayout, Shape<_2>{});
  return make_layout(make_layout(get<0, 1>(divided), get<1>(divided)),
                     make_layout(get<0, 0>(divided), get<2>(divided)));
}

// MMA C-fragment layout -> A-register layout
template <int kAtomKDivN, typename Layout_>
__forceinline__ __device__ auto convertARegsLayout(Layout_ accLayout) {
  using namespace cute;

  static_assert(decltype(size<0>(accLayout))::value == 4);
  static_assert(decltype(rank(accLayout))::value == 3);
  static_assert(kAtomKDivN == 2, "SM80+ MMA atom requires kAtomKDivN=2");
  auto divided = logical_divide(accLayout, Shape<_1, _1, _2>{});
  return make_layout(make_layout(get<0>(divided), get<2, 0>(divided)), get<1>(divided), get<2, 1>(divided));
}

// P(accS): float -> fp16/bf16 -> A-register layout
template <typename Config, typename AccS>
__forceinline__ __device__ auto convertAccToP(AccS& accS) {
  using namespace cute;

  using ElemTypeG = typename CutlassElemType<typename Config::DType>::type;
  constexpr int numel = decltype(size(accS))::value;
  cutlass::NumericArrayConverter<ElemTypeG, float, numel> converter;
  auto frag = converter(*reinterpret_cast<const cutlass::Array<float, numel>*>(accS.data()));
  auto rP = make_tensor(make_rmem_ptr<ElemTypeG>(&frag), accS.layout());
  return make_tensor(rP.data(), convertARegsLayout<2>(accS.layout()));
}

}  // namespace tfa::mma
