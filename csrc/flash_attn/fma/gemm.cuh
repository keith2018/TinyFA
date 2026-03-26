/*
 * TinyFA
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "../utils.cuh"
#include "tile.cuh"

namespace tfa::fma {

template <typename Config>
struct GemmOp {
  using DType = typename Config::DType;

  static constexpr int kDimsPerLane = Config::kHeadDim / Config::kWarpSize;
  static constexpr int kRowsPerWarp = Config::kRowsPerWarp;
  static constexpr int kColsPerLane = Config::kBc / Config::kWarpSize;
  static constexpr int kElemsPerVec = Config::kElemsPerVecLoad;
  static constexpr int kWarpSize = Config::kWarpSize;

  static_assert(Config::kHeadDim % kElemsPerVec == 0, "kHeadDim must be divisible by kElemsPerVec");
  static constexpr int kNumVecs = Config::kHeadDim / kElemsPerVec;

  // S = Q @ K^T
  template <typename AccS, typename Context>
  __device__ static void computeScore(AccS& accS, const QTile<Config>& qTile, const KTile<Config>& kTile,
                                      Context& ctx) {
    bool isPartial = ctx.isPartialTile();
    isPartial ? computeScoreImpl<true>(accS, qTile, kTile, ctx) : computeScoreImpl<false>(accS, qTile, kTile, ctx);
  }

 private:
  template <bool kBoundaryMask, typename AccS, typename Context>
  __device__ static void computeScoreImpl(AccS& accS, const QTile<Config>& qTile, const KTile<Config>& kTile,
                                          Context& ctx) {
    DType qReg[kRowsPerWarp][kElemsPerVec];
    DType kReg[kColsPerLane][kElemsPerVec];

    accumDotProducts<kBoundaryMask>(accS, qReg, kReg, qTile, kTile, ctx);
  }

  template <bool kBoundaryMask, typename AccS, typename Context>
  __device__ static __forceinline__ void accumDotProducts(AccS& accS, DType qReg[][kElemsPerVec],
                                                          DType kReg[][kElemsPerVec], const QTile<Config>& qTile,
                                                          const KTile<Config>& kTile, Context& ctx) {
#pragma unroll
    for (int v = 0; v < kNumVecs; v++) {
      loadQ(qReg, qTile, ctx, v);
      loadK<kBoundaryMask>(kReg, kTile, ctx, v);
      accumulate(accS, qReg, kReg);
    }
  }

  template <typename Context>
  __device__ static __forceinline__ void loadQ(DType qReg[][kElemsPerVec], const QTile<Config>& qTile,
                                               const Context& ctx, int v) {
#pragma unroll
    for (int m = 0; m < kRowsPerWarp; m++) {
      qTile.sm2reg(&qReg[m][0], ctx.warpRowOffset + m, v);
    }
  }

  template <bool kBoundaryMask, typename Context>
  __device__ static __forceinline__ void loadK(DType kReg[][kElemsPerVec], const KTile<Config>& kTile,
                                               const Context& ctx, int v) {
#pragma unroll
    for (int n = 0; n < kColsPerLane; n++) {
      int colN = ctx.laneId + n * kWarpSize;
      if constexpr (kBoundaryMask) {
        if (colN < ctx.tileSizeKV) {
          kTile.sm2reg(&kReg[n][0], colN, v);
        } else {
#pragma unroll
          for (int e = 0; e < kElemsPerVec; e++) {
            kReg[n][e] = fromFloat<DType>(0.f);
          }
        }
      } else {
        kTile.sm2reg(&kReg[n][0], colN, v);
      }
    }
  }

  template <typename AccS>
  __device__ static __forceinline__ void accumulate(AccS& accS, const DType qReg[][kElemsPerVec],
                                                    const DType kReg[][kElemsPerVec]) {
#pragma unroll
    for (int e = 0; e < kElemsPerVec; e++) {
#pragma unroll
      for (int m = 0; m < kRowsPerWarp; m++) {
        float q = toFloat(qReg[m][e]);
#pragma unroll
        for (int n = 0; n < kColsPerLane; n++) {
          accS[m][n] += q * toFloat(kReg[n][e]);
        }
      }
    }
  }

 public:
  // O += P @ V
  template <typename AccO, typename AccS, typename Context>
  __device__ static void computeOutput(AccO& accO, const AccS& accS, const VTile<Config>& vTile, const Context& ctx) {
    static constexpr int kUnrollV = (kDimsPerLane <= 4) ? 4 : 2;
    static_assert(Config::kBc % kUnrollV == 0, "kBc must be divisible by kUnrollV");

#pragma unroll
    for (int n = 0; n < Config::kBc; n += kUnrollV) {
      DType vRegs[kUnrollV][kDimsPerLane];
      float pRegs[kUnrollV][kRowsPerWarp];

      loadPV(vRegs, pRegs, accS, vTile, ctx, n, kUnrollV);
      accumOutput(accO, vRegs, pRegs, kUnrollV);
    }
  }

 private:
  template <typename AccS, typename Context>
  __device__ static __forceinline__ void loadPV(DType vRegs[][kDimsPerLane], float pRegs[][kRowsPerWarp],
                                                const AccS& accS, const VTile<Config>& vTile, const Context& ctx,
                                                int nBase, int kUnrollV) {
    static constexpr int kVecsPerThread = (kDimsPerLane % kElemsPerVec == 0) ? (kDimsPerLane / kElemsPerVec) : 0;
    static constexpr int kScalarElems = kDimsPerLane - kVecsPerThread * kElemsPerVec;

    const int vectorBase = ctx.laneId * kVecsPerThread;
    const int scalarBase = ctx.laneId * kDimsPerLane + kVecsPerThread * kElemsPerVec;

#pragma unroll
    for (int u = 0; u < kUnrollV; u++) {
      int rowIdx = nBase + u;

      // vector load
#pragma unroll
      for (int i = 0; i < kVecsPerThread; i++) {
        vTile.sm2reg(&vRegs[u][i * kElemsPerVec], rowIdx, vectorBase + i);
      }
      // scalar load
#pragma unroll
      for (int i = 0; i < kScalarElems; i++) {
        vRegs[u][kVecsPerThread * kElemsPerVec + i] = vTile.at(rowIdx, scalarBase + i);
      }

      int srcLane = rowIdx % kWarpSize;
      int colIdx = rowIdx / kWarpSize;
#pragma unroll
      for (int m = 0; m < kRowsPerWarp; m++) {
        pRegs[u][m] = __shfl_sync(0xffffffff, accS[m][colIdx], srcLane);
      }
    }
  }

  template <typename AccO>
  __device__ static __forceinline__ void accumOutput(AccO& accO, DType vRegs[][kDimsPerLane],
                                                     float pRegs[][kRowsPerWarp], int kUnrollV) {
#pragma unroll
    for (int m = 0; m < kRowsPerWarp; m++) {
#pragma unroll
      for (int u = 0; u < kUnrollV; u++) {
        float p = pRegs[u][m];
#pragma unroll
        for (int k = 0; k < kDimsPerLane; k++) {
          accO[m][k] += p * toFloat(vRegs[u][k]);
        }
      }
    }
  }
};

}  // namespace tfa::fma
