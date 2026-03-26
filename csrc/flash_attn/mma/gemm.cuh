/*
 * TinyFA
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "layout.cuh"

namespace tfa::mma {

template <typename Config>
struct GemmConfig {
  using DType = typename Config::DType;
  using Element = typename CutlassElemType<DType>::type;

  static constexpr int kNumWarps = Config::kNumWarps;

  using ElemType = Element;
  using MmaAtomArch =
      std::conditional_t<std::is_same_v<Element, cutlass::half_t>, cute::MMA_Atom<cute::SM80_16x8x16_F32F16F16F32_TN>,
                         cute::MMA_Atom<cute::SM80_16x8x16_F32BF16BF16F32_TN>>;
  static constexpr int kMmaAtomKDivN = 2;

  // LDSM (smem -> reg)
  using SmemCopyAtom = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, ElemType>;
  using SmemCopyAtomTransposed = cute::Copy_Atom<cute::SM75_U16x8_LDSM_T, ElemType>;

  using TiledMma = cute::TiledMMA<MmaAtomArch, cute::Layout<cute::Shape<cute::Int<kNumWarps>, cute::_1, cute::_1>>,
                                  cute::Tile<cute::Int<16 * kNumWarps>, cute::_16, cute::_16>>;
};

struct GemmOp {
  // S = Q @ K^T
  template <typename TiledMma, typename SmemCopyQ, typename SmemCopyK, typename TsQ, typename TsK, typename TrQCopy,
            typename TrKCopy, typename TrQ, typename TrK, typename AccS>
  __device__ __forceinline__ static void computeScore(AccS& accS, const TiledMma& tiledMma,
                                                      const SmemCopyQ& smemTiledCopyQ, const SmemCopyK& smemTiledCopyK,
                                                      const TsQ& tSsQ, const TsK& tSsK, TrQCopy& tSrQCopy,
                                                      TrKCopy& tSrKCopy, const TrQ& tSrQ, const TrK& tSrK) {
    using namespace cute;

    copy(smemTiledCopyQ, tSsQ(_, _, _0{}), tSrQCopy(_, _, _0{}));
    copy(smemTiledCopyK, tSsK(_, _, _0{}), tSrKCopy(_, _, _0{}));

#pragma unroll
    for (int ki = 0; ki < size<2>(tSrQ); ki++) {
      if (ki < size<2>(tSrQ) - 1) {
        copy(smemTiledCopyQ, tSsQ(_, _, ki + 1), tSrQCopy(_, _, ki + 1));
        copy(smemTiledCopyK, tSsK(_, _, ki + 1), tSrKCopy(_, _, ki + 1));
      }
      gemm(tiledMma, tSrQ(_, _, ki), tSrK(_, _, ki), accS);
    }
  }

  // O += P @ V
  template <typename Config, typename TiledMma, typename SmemCopyV, typename TsVt, typename TrVtCopy, typename TrVt,
            typename AccS, typename AccO>
  __device__ __forceinline__ static void computeOutput(AccO& accO, AccS& accS, const TiledMma& tiledMma,
                                                       const SmemCopyV& smemTiledCopyV, const TsVt& tOsVt,
                                                       TrVtCopy& tOrVtCopy, const TrVt& tOrVt) {
    using namespace cute;

    // convert P: float -> fp16/bf16 -> A-register layout
    auto tOrP = convertAccToP<Config>(accS);

    copy(smemTiledCopyV, tOsVt(_, _, _0{}), tOrVtCopy(_, _, _0{}));
#pragma unroll
    for (int ki = 0; ki < size<2>(tOrP); ki++) {
      if (ki < size<2>(tOrP) - 1) {
        copy(smemTiledCopyV, tOsVt(_, _, ki + 1), tOrVtCopy(_, _, ki + 1));
      }
      gemm(tiledMma, tOrP(_, _, ki), tOrVt(_, _, ki), accO);
    }
  }
};

}  // namespace tfa::mma
