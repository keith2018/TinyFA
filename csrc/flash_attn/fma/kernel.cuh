/*
 * TinyFA
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "../params.cuh"
#include "../utils.cuh"
#include "gemm.cuh"
#include "softmax.cuh"
#include "tile.cuh"

namespace tfa::fma {

template <typename Config, typename Params>
struct KernelContext : ThreadInfo<Config>, BlockInfo<Config, Params> {
  using DType = typename Config::DType;
  using Thread = ThreadInfo<Config>;
  using Block = BlockInfo<Config, Params>;

  static constexpr float kAttnScale = AttentionScale<Config::kHeadDim>::value;

  using Block::batchIdx;
  using Block::headIdx;
  using Block::headIdxKV;
  using Block::isValidTile;
  using Block::kTileKV;
  using Block::numTilesKV;
  using Block::seqLenKV;
  using Block::seqLenQ;
  using Block::tileQ;
  using Block::tileSizeQ;

  using Thread::blockSize;
  using Thread::kRowsPerWarp;
  using Thread::laneId;
  using Thread::threadId;
  using Thread::warpRowOffset;

  __device__ explicit KernelContext(const Params& params) : Thread(), Block(params), params(params) {}

  int tileKV = 0;
  int tileSizeKV = 0;

  template <bool kIsCausal>
  __device__ __forceinline__ void setTileKV(int tileIdx) {
    tileKV = tileIdx * kTileKV;
    tileSizeKV = min(kTileKV, seqLenKV - tileKV);
    if constexpr (kIsCausal) {
      tileSizeKV = min(tileSizeKV, tileQ + tileSizeQ - tileKV);
    }
  }

  template <bool kIsCausal>
  __device__ __forceinline__ bool needsCausalMask() const {
    return kIsCausal && (tileKV + kTileKV > tileQ);
  }

  __device__ __forceinline__ bool isPartialTile() const { return tileSizeKV < kTileKV; }

  __device__ __forceinline__ int validWarpRows() const {
    int warpStartQ = tileQ + warpRowOffset;
    return (warpStartQ < seqLenQ) ? min(kRowsPerWarp, seqLenQ - warpStartQ) : 0;
  }

  __device__ __forceinline__ int globalRowQ(int localRow) const { return tileQ + warpRowOffset + localRow; }

  __device__ __forceinline__ int globalKV(int n) const { return tileKV + laneId + n * Config::kWarpSize; }

  __device__ __forceinline__ const DType* qPtr() const { return params.qPtr(*this); }
  __device__ __forceinline__ const DType* kPtr() const { return params.kPtr(*this, tileKV); }
  __device__ __forceinline__ const DType* vPtr() const { return params.vPtr(*this, tileKV); }
  __device__ __forceinline__ DType* oPtr() const { return params.oPtr(*this); }
  __device__ __forceinline__ int seqDimQ() const { return params.seqDimQ; }
  __device__ __forceinline__ int seqDimKV() const { return params.seqDimKV; }

 private:
  const Params& params;
};

template <typename Config, bool kIsCausal, typename Params>
__device__ void flashAttnFma(const Params& params, char* smemBuf) {
  using DType = typename Config::DType;
  using Context = KernelContext<Config, Params>;

  static constexpr bool kUseCpAsync = Config::kUseCpAsync;
  static constexpr int kRowsPerWarp = Config::kRowsPerWarp;
  static constexpr int kColsPerLane = Config::kBc / Config::kWarpSize;
  static constexpr int kDimsPerLane = Config::kHeadDim / Config::kWarpSize;

  Context ctx(params);
  if (!ctx.isValidTile()) {
    return;
  }

  // smem:
  //   async: Q | K | V
  //   sync:  Q | KV
  auto* smem = reinterpret_cast<DType*>(smemBuf);
  QTile<Config> qTile(smem);
  DType* kSmemBase = smem + qTile.numElems();
  KTile<Config> kTile(kSmemBase);
  VTile<Config> vTile(kUseCpAsync ? (kSmemBase + KTile<Config>::numElems()) : kSmemBase);

  float accO[kRowsPerWarp][kDimsPerLane]{};

  Softmax<Config> softmax;
  softmax.init();

  // load Q tile
  qTile.gm2sm(ctx.qPtr(), ctx.seqDimQ(), ctx.tileSizeQ, ctx);
  __syncthreads();

  const int numTilesKV = ctx.template numTilesKV<kIsCausal>();
  if (numTilesKV <= 0) {
    return;
  }
  int kvTileIdx = numTilesKV - 1;

  const int nMaskingSteps = Softmax<Config>::template numMaskingSteps<kIsCausal>(ctx);
  const int nNoMaskEnd = numTilesKV - nMaskingSteps;

  // load first K tile [tileIdx]
  ctx.template setTileKV<kIsCausal>(kvTileIdx);
  if constexpr (kUseCpAsync) {
    kTile.gm2sm_async(ctx.kPtr(), ctx.seqDimKV(), ctx.tileSizeKV, ctx);
    cpAsyncCommit();
    cpAsyncWaitAll();
  } else {
    kTile.gm2sm(ctx.kPtr(), ctx.seqDimKV(), ctx.tileSizeKV, ctx);
  }
  __syncthreads();

  // kv tile iteration (async)
  auto tileIterationAsync = [&](int tileIdx, auto kNeedsMaskTag) {
    constexpr bool kNeedsMask = kNeedsMaskTag.value;

    // async load V[tileIdx]
    vTile.gm2sm_async(ctx.vPtr(), ctx.seqDimKV(), ctx.tileSizeKV, ctx);
    cpAsyncCommit();

    // S = Q @ K^T
    float accS[kRowsPerWarp][kColsPerLane]{};
    GemmOp<Config>::computeScore(accS, qTile, kTile, ctx);

    // wait for V load
    cpAsyncWaitAll();
    __syncthreads();

    int curTileKV = ctx.tileKV;
    int tileSizeKV = ctx.tileSizeKV;

    // async load next K [tileIdx - 1]
    bool hasNext = tileIdx > 0;
    if (hasNext) {
      ctx.template setTileKV<kIsCausal>(tileIdx - 1);
      kTile.gm2sm_async(ctx.kPtr(), ctx.seqDimKV(), ctx.tileSizeKV, ctx);
      cpAsyncCommit();
    }

    // masking
    if constexpr (kNeedsMask) {
      Softmax<Config>::template applyMask<kIsCausal>(accS, ctx, curTileKV, tileSizeKV);
    }

    // softmax
    softmax.update(accS, accO, ctx.kAttnScale);

    // O += P @ V
    GemmOp<Config>::computeOutput(accO, accS, vTile, ctx);

    // wait for next K load
    if (hasNext) {
      cpAsyncWaitAll();
      __syncthreads();
    }
  };

  // kv tile iteration (sync)
  auto tileIterationSync = [&](int tileIdx, auto kNeedsMaskTag) {
    constexpr bool kNeedsMask = kNeedsMaskTag.value;

    // S = Q @ K^T
    float accS[kRowsPerWarp][kColsPerLane]{};
    GemmOp<Config>::computeScore(accS, qTile, kTile, ctx);

    // load V[tileIdx]
    __syncthreads();
    vTile.gm2sm(ctx.vPtr(), ctx.seqDimKV(), ctx.tileSizeKV, ctx);
    __syncthreads();

    // masking
    if constexpr (kNeedsMask) {
      Softmax<Config>::template applyMask<kIsCausal>(accS, ctx, ctx.tileKV, ctx.tileSizeKV);
    }

    // softmax
    softmax.update(accS, accO, ctx.kAttnScale);

    // O += P @ V
    GemmOp<Config>::computeOutput(accO, accS, vTile, ctx);

    // load next K [tileIdx - 1]
    bool hasNext = tileIdx > 0;
    __syncthreads();
    if (hasNext) {
      ctx.template setTileKV<kIsCausal>(tileIdx - 1);
      kTile.gm2sm(ctx.kPtr(), ctx.seqDimKV(), ctx.tileSizeKV, ctx);
      __syncthreads();
    }
  };

  if constexpr (kUseCpAsync) {
    // masking iterations
    for (; kvTileIdx >= nNoMaskEnd; --kvTileIdx) {
      tileIterationAsync(kvTileIdx, std::integral_constant<bool, true>{});
    }
    // no masking iterations
    for (; kvTileIdx >= 0; --kvTileIdx) {
      tileIterationAsync(kvTileIdx, std::integral_constant<bool, false>{});
    }
  } else {
    // masking iterations
    for (; kvTileIdx >= nNoMaskEnd; --kvTileIdx) {
      tileIterationSync(kvTileIdx, std::integral_constant<bool, true>{});
    }
    // no masking iterations
    for (; kvTileIdx >= 0; --kvTileIdx) {
      tileIterationSync(kvTileIdx, std::integral_constant<bool, false>{});
    }
  }

  // finalize softmax
  softmax.finalize(accO);

  // store O
  MemStore<Config>::storeO(ctx.oPtr(), smem, ctx.seqDimQ(), accO, ctx);
}

TFA_DEFINE_KERNEL_WRAPPER(flashAttnFma)

}  // namespace tfa::fma
