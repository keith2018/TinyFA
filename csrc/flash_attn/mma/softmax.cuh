/*
 * TinyFA
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "../utils.cuh"
#include "layout.cuh"

namespace tfa::mma {

template <typename Config, int kRowsPerWarp>
struct Softmax {
  float rowMax[kRowsPerWarp];
  float rowSum[kRowsPerWarp];

  __device__ void init() {
#pragma unroll
    for (int i = 0; i < kRowsPerWarp; i++) {
      rowMax[i] = -INFINITY;
      rowSum[i] = 0.0f;
    }
  }

  template <bool kIsCausal, typename BlockInfo_>
  __device__ __forceinline__ static int numMaskingSteps(const BlockInfo_& blockInfo) {
    static constexpr int kBr = Config::kBr;
    static constexpr int kBc = Config::kBc;
    if constexpr (!kIsCausal) {
      return (blockInfo.seqLenKV % kBc == 0) ? 0 : 1;
    } else {
      int numTilesKV = blockInfo.template numTilesKV<kIsCausal>();
      bool isAlignedKV = (blockInfo.seqLenKV % kBc == 0);
      int numMasking = ceilDiv(kBr, kBc) + (isAlignedKV ? 0 : 1);
      return min(numMasking, numTilesKV);
    }
  }

  template <bool kIsCausal, typename AccS, typename TiledMma, typename BlockInfo_>
  __device__ __forceinline__ static void applyMask(AccS& accS, const TiledMma& tiledMma, int tidx,
                                                   const BlockInfo_& blockInfo, int tileIdx) {
    using namespace cute;
    static constexpr int kBr = Config::kBr;
    static constexpr int kBc = Config::kBc;

    auto thrMma = tiledMma.get_thread_slice(tidx);
    auto scores = make_tensor(accS.data(), convertAccRowCol(accS.layout()));
    auto cS = make_identity_tensor(Shape<Int<kBr>, Int<kBc>>{});
    auto tScS = thrMma.partition_C(cS);
    auto tScSRowCol = make_tensor(tScS.data(), convertAccRowCol(tScS.layout()));

#pragma unroll
    for (int mi = 0; mi < size<0>(scores); mi++) {
#pragma unroll
      for (int ni = 0; ni < size<1>(scores); ni++) {
        const int row = blockInfo.tileQ + get<0>(tScSRowCol(mi, ni));
        const int col = tileIdx * kBc + get<1>(tScSRowCol(mi, ni));
        if (isMasked<kIsCausal>(row, col, blockInfo.seqLenKV)) {
          scores(mi, ni) = -INFINITY;
        }
      }
    }
  }

  template <typename AccS, typename AccO>
  __device__ __forceinline__ void update(AccS& accS, AccO& accO, float attnScale) {
    using namespace cute;

    auto scores = make_tensor(accS.data(), convertAccRowCol(accS.layout()));
    auto accORowCol = make_tensor(accO.data(), convertAccRowCol(accO.layout()));

    // row max
    float scoresMaxCur[kRowsPerWarp];
#pragma unroll
    for (int mi = 0; mi < size<0>(scores); mi++) {
      scoresMaxCur[mi] = scores(mi, 0);
#pragma unroll
      for (int ni = 1; ni < size<1>(scores); ni++) {
        scoresMaxCur[mi] = max(scoresMaxCur[mi], scores(mi, ni));
      }
      scoresMaxCur[mi] = warpAllReduceMax(scoresMaxCur[mi]);
    }

    // rescale
#pragma unroll
    for (int mi = 0; mi < kRowsPerWarp; mi++) {
      float oldMax = rowMax[mi];
      float newMax = max(oldMax, scoresMaxCur[mi]);
      float scale = (oldMax == -INFINITY) ? 0.0f : exp2f((oldMax - newMax) * attnScale);

      rowSum[mi] *= scale;
#pragma unroll
      for (int ni = 0; ni < size<1>(accORowCol); ni++) {
        accORowCol(mi, ni) *= scale;
      }
      rowMax[mi] = newMax;
    }

    // exp and row sum
#pragma unroll
    for (int mi = 0; mi < size<0>(scores); mi++) {
      float maxScaled = (rowMax[mi] == -INFINITY) ? 0.0f : rowMax[mi] * attnScale;
#pragma unroll
      for (int ni = 0; ni < size<1>(scores); ni++) {
        scores(mi, ni) = exp2f(scores(mi, ni) * attnScale - maxScaled);
        rowSum[mi] += scores(mi, ni);
      }
    }
  }

  template <typename AccO>
  __device__ __forceinline__ void finalize(AccO& accO) {
    using namespace cute;

#pragma unroll
    for (int mi = 0; mi < kRowsPerWarp; mi++) {
      rowSum[mi] = warpAllReduceSum(rowSum[mi]);
    }

    auto accORowCol = make_tensor(accO.data(), convertAccRowCol(accO.layout()));

#pragma unroll
    for (int mi = 0; mi < size<0>(accORowCol); mi++) {
      float invSum = (rowSum[mi] > 0.f) ? (1.f / rowSum[mi]) : 0.f;
#pragma unroll
      for (int ni = 0; ni < size<1>(accORowCol); ni++) {
        accORowCol(mi, ni) *= invSum;
      }
    }
  }

 private:
  template <bool kIsCausal>
  __device__ __forceinline__ static bool isMasked(int row, int col, int seqLenKV) {
    return col >= seqLenKV || (kIsCausal && col > row);
  }

  template <typename T>
  __device__ __forceinline__ static T warpAllReduceMax(T val) {
    return warpReduceMax<4>(val);
  }

  template <typename T>
  __device__ __forceinline__ static T warpAllReduceSum(T val) {
    return warpReduceSum<4>(val);
  }
};

}  // namespace tfa::mma
