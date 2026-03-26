/*
 * TinyFA
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "../utils.cuh"

namespace tfa::fma {

template <typename Config>
struct Softmax {
  static constexpr int kWarpSize = Config::kWarpSize;
  static constexpr int kRowsPerWarp = Config::kRowsPerWarp;
  static constexpr int kColsPerLane = Config::kBc / kWarpSize;
  static constexpr int kDimsPerLane = Config::kHeadDim / kWarpSize;

  float rowMax[kRowsPerWarp];
  float rowSum[kRowsPerWarp];

  __device__ void init() {
#pragma unroll
    for (int i = 0; i < kRowsPerWarp; i++) {
      rowMax[i] = -INFINITY;
      rowSum[i] = 0.f;
    }
  }

  template <bool kIsCausal, typename Context>
  __device__ __forceinline__ static int numMaskingSteps(const Context& ctx) {
    if constexpr (!kIsCausal) {
      static constexpr int kBc = Context::Block::kTileKV;
      return (ctx.seqLenKV % kBc == 0) ? 0 : 1;
    } else {
      static constexpr int kBr = Context::Block::kTileQ;
      static constexpr int kBc = Context::Block::kTileKV;
      int numTilesKV = ctx.template numTilesKV<kIsCausal>();
      bool isAlignedKV = (ctx.seqLenKV % kBc == 0);
      int numMasking = ceilDiv(kBr, kBc) + (isAlignedKV ? 0 : 1);
      return min(numMasking, numTilesKV);
    }
  }

  template <bool kIsCausal, typename S, typename Context>
  __device__ static __forceinline__ void applyMask(S& s, const Context& ctx, int tileKV, int tileSizeKV) {
    bool needsCausal = kIsCausal && (tileKV + Context::Block::kTileKV > ctx.tileQ);
    bool isPartial = tileSizeKV < Context::Block::kTileKV;

    if (needsCausal) {
      isPartial ? applyMaskImpl<true, true>(s, ctx, tileKV, tileSizeKV)
                : applyMaskImpl<true, false>(s, ctx, tileKV, tileSizeKV);
    } else {
      isPartial ? applyMaskImpl<false, true>(s, ctx, tileKV, tileSizeKV)
                : applyMaskImpl<false, false>(s, ctx, tileKV, tileSizeKV);
    }
  }

  template <typename AccS, typename AccO>
  __device__ __forceinline__ void update(AccS& accS, AccO& accO, float attnScale) {
#pragma unroll
    for (int m = 0; m < kRowsPerWarp; m++) {
      float newMax = computeRowMax(accS[m]);
      float correction = rescaleState(m, newMax, attnScale);
      float localSum = computeExpAndSum(accS[m], rowMax[m], attnScale);
      updateRowSum(m, correction, localSum);
      rescaleOutput(accO[m], correction);
    }
  }

  template <typename AccO>
  __device__ __forceinline__ void finalize(AccO& accO) const {
#pragma unroll
    for (int m = 0; m < kRowsPerWarp; m++) {
      float norm = (rowSum[m] > 0.f) ? (1.f / rowSum[m]) : 0.f;
#pragma unroll
      for (int k = 0; k < kDimsPerLane; k++) {
        accO[m][k] *= norm;
      }
    }
  }

 private:
  template <bool kCausalMask, bool kBoundaryMask, typename S, typename Context>
  __device__ static __forceinline__ void applyMaskImpl(S& s, const Context& ctx, int tileKV, int tileSizeKV) {
#pragma unroll
    for (int m = 0; m < kRowsPerWarp; m++) {
      int globalQ = ctx.globalRowQ(m);
      int globalKV = tileKV + ctx.laneId;
#pragma unroll
      for (int n = 0; n < kColsPerLane; n++) {
        bool masked = false;
        if constexpr (kBoundaryMask) {
          masked |= (ctx.laneId + n * kWarpSize >= tileSizeKV);
        }
        if constexpr (kCausalMask) {
          masked |= (globalKV > globalQ);
          globalKV += kWarpSize;
        }
        if (masked) {
          s[m][n] = -INFINITY;
        }
      }
    }
  }

  template <typename Row>
  __device__ __forceinline__ static float computeRowMax(const Row& row) {
    float localMax = row[0];
#pragma unroll
    for (int n = 1; n < kColsPerLane; n++) {
      localMax = fmaxf(localMax, row[n]);
    }
    return warpAllReduceMax(localMax);
  }

  __device__ __forceinline__ float rescaleState(int m, float newMax, float attnScale) {
    float prevMax = rowMax[m];
    rowMax[m] = fmaxf(prevMax, newMax);
    return (prevMax == -INFINITY) ? 0.f : exp2f((prevMax - rowMax[m]) * attnScale);
  }

  template <typename SRow>
  __device__ __forceinline__ static float computeExpAndSum(SRow& accSRow, float maxVal, float attnScale) {
    float maxScaled = (maxVal == -INFINITY) ? 0.f : maxVal * attnScale;
    float localSum = 0.f;
#pragma unroll
    for (int n = 0; n < kColsPerLane; n++) {
      float p = (maxVal == -INFINITY) ? 0.f : exp2f(accSRow[n] * attnScale - maxScaled);
      accSRow[n] = p;
      localSum += p;
    }
    return localSum;
  }

  __device__ __forceinline__ void updateRowSum(int m, float correction, float localSum) {
    float warpSum = warpAllReduceSum(localSum);
    rowSum[m] = rowSum[m] * correction + warpSum;
  }

  template <typename AccRow>
  __device__ __forceinline__ static void rescaleOutput(AccRow& accRow, float correction) {
#pragma unroll
    for (int k = 0; k < kDimsPerLane; k++) {
      accRow[k] *= correction;
    }
  }

  __device__ __forceinline__ static float warpAllReduceMax(float val) { return warpReduceMax<kWarpSize>(val); }

  __device__ __forceinline__ static float warpAllReduceSum(float val) { return warpReduceSum<kWarpSize>(val); }
};

}  // namespace tfa::fma
