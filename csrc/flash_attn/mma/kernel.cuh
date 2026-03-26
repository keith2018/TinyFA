/*
 * TinyFA
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "../params.cuh"
#include "../utils.cuh"
#include "gemm.cuh"
#include "layout.cuh"
#include "memory.cuh"
#include "softmax.cuh"

namespace tfa::mma {

template <typename Config, bool kIsCausal, typename Params>
__device__ void flashAttnMma(const Params& params, char* smemBuf) {
  using namespace cute;

  using Block = BlockInfo<Config, Params>;
  using Smem = SmemLayout<Config>;
  using Gemm = GemmConfig<Config>;
  using Mem = MemConfig<Config>;
  using ElemType = typename Smem::Element;

  static constexpr int kBr = Config::kBr;
  static constexpr int kBc = Config::kBc;
  static constexpr int kHeadDim = Config::kHeadDim;

  const unsigned int tidx = threadIdx.x;

  Block blockInfo(params);
  if (!blockInfo.isValidTile()) {
    return;
  }

  const int numTilesKV = blockInfo.template numTilesKV<kIsCausal>();
  if (numTilesKV <= 0) {
    return;
  }

  const auto* qBase = reinterpret_cast<const ElemType*>(params.qPtr(blockInfo));
  const auto* kBase = reinterpret_cast<const ElemType*>(params.kPtr(blockInfo, 0));
  const auto* vBase = reinterpret_cast<const ElemType*>(params.vPtr(blockInfo, 0));
  auto* oBase = reinterpret_cast<ElemType*>(params.oPtr(blockInfo));
  const int rowStrideQ = params.seqDimQ;
  const int rowStrideKV = params.seqDimKV;

  // smem tensors
  auto* smemQ = reinterpret_cast<ElemType*>(smemBuf);
  auto* smemK = smemQ + size(typename Smem::SmemLayoutQ{});
  auto* smemV = smemK + size(typename Smem::SmemLayoutKV{});

  auto sQ = make_tensor(make_smem_ptr(smemQ), typename Smem::SmemLayoutQ{});
  auto sK = make_tensor(make_smem_ptr(smemK), typename Smem::SmemLayoutKV{});
  auto sV = make_tensor(make_smem_ptr(smemV), typename Smem::SmemLayoutKV{});
  auto sVt = make_tensor(make_smem_ptr(smemV), typename Smem::SmemLayoutVTransposed{});
  auto sVtNoSwizzle = make_tensor(smemV, typename Smem::SmemLayoutVTransposedNoSwizzle{});

  // gmem tensors
  auto mQ = make_tensor(make_gmem_ptr(qBase), make_shape(blockInfo.seqLenQ - blockInfo.tileQ, Int<kHeadDim>{}),
                        make_stride(rowStrideQ, _1{}));
  auto gQ = local_tile(mQ, Shape<Int<kBr>, Int<kHeadDim>>{}, make_coord(0, 0));

  auto mK = make_tensor(make_gmem_ptr(kBase), make_shape(blockInfo.seqLenKV, Int<kHeadDim>{}),
                        make_stride(rowStrideKV, _1{}));
  auto gK = local_tile(mK, Shape<Int<kBc>, Int<kHeadDim>>{}, make_coord(_, 0));

  auto mV = make_tensor(make_gmem_ptr(vBase), make_shape(blockInfo.seqLenKV, Int<kHeadDim>{}),
                        make_stride(rowStrideKV, _1{}));
  auto gV = local_tile(mV, Shape<Int<kBc>, Int<kHeadDim>>{}, make_coord(_, 0));

  auto mO = make_tensor(make_gmem_ptr(oBase), make_shape(blockInfo.seqLenQ - blockInfo.tileQ, Int<kHeadDim>{}),
                        make_stride(rowStrideQ, _1{}));
  auto gO = local_tile(mO, Shape<Int<kBr>, Int<kHeadDim>>{}, make_coord(0, 0));

  // gmem copy partitions
  typename Mem::GmemTiledCopyQKV gmemTiledCopyQKV;
  auto gmemThrCopyQKV = gmemTiledCopyQKV.get_thread_slice(tidx);

  auto tQgQ = gmemThrCopyQKV.partition_S(gQ);
  auto tQsQ = gmemThrCopyQKV.partition_D(sQ);
  auto tKgK = gmemThrCopyQKV.partition_S(gK);
  auto tKsK = gmemThrCopyQKV.partition_D(sK);
  auto tVgV = gmemThrCopyQKV.partition_S(gV);
  auto tVsV = gmemThrCopyQKV.partition_D(sV);

  // coordinate tensors for boundary checks
  auto cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
  auto cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));
  auto tQcQ = gmemThrCopyQKV.partition_S(cQ);
  auto tKVcKV = gmemThrCopyQKV.partition_S(cKV);

  // MMA partitions
  typename Gemm::TiledMma tiledMma;
  auto thrMma = tiledMma.get_thread_slice(tidx);

  auto accO = partition_fragment_C(tiledMma, Shape<Int<kBr>, Int<kHeadDim>>{});
  clear(accO);

  auto tSrQ = thrMma.partition_fragment_A(sQ);
  auto tSrK = thrMma.partition_fragment_B(sK);

  auto smemTiledCopyQ = make_tiled_copy_A(typename Gemm::SmemCopyAtom{}, tiledMma);
  auto smemThrCopyQ = smemTiledCopyQ.get_thread_slice(tidx);
  auto tSsQ = smemThrCopyQ.partition_S(sQ);
  auto tSrQCopy = smemThrCopyQ.retile_D(tSrQ);

  auto smemTiledCopyK = make_tiled_copy_B(typename Gemm::SmemCopyAtom{}, tiledMma);
  auto smemThrCopyK = smemTiledCopyK.get_thread_slice(tidx);
  auto tSsK = smemThrCopyK.partition_S(sK);
  auto tSrKCopy = smemThrCopyK.retile_D(tSrK);

  auto tOrVt = thrMma.partition_fragment_B(sVtNoSwizzle);

  auto smemTiledCopyV = make_tiled_copy_B(typename Gemm::SmemCopyAtomTransposed{}, tiledMma);
  auto smemThrCopyV = smemTiledCopyV.get_thread_slice(tidx);
  auto tOsVt = smemThrCopyV.partition_S(sVt);
  auto tOrVtCopy = smemThrCopyV.retile_D(tOrVt);

  constexpr int kNRows = 2 * size<1>(accO);
  Softmax<Config, kNRows> softmax;
  softmax.init();

  constexpr float kAttnScale = AttentionScale<kHeadDim>::value;

  // load Q tile
  MemLoader::gm2sm(gmemTiledCopyQKV, tQsQ, tQgQ, tQcQ, blockInfo.tileSizeQ);
  cp_async_fence();

  int kvTileIdx = numTilesKV - 1;

  // load first K tile [tileIdx]
  MemLoader::gm2sm(gmemTiledCopyQKV, tKsK, tKgK, tKVcKV, blockInfo.seqLenKV - kvTileIdx * kBc, kvTileIdx);
  cp_async_fence();
  cpAsyncWaitGroup<0>();
  __syncthreads();

  const int nMaskingSteps = Softmax<Config, kNRows>::template numMaskingSteps<kIsCausal>(blockInfo);
  const int nNoMaskEnd = numTilesKV - nMaskingSteps;  // tiles [nNoMaskEnd, numTilesKV-1] need masking

  // kv tile iteration
  auto tileIteration = [&](int tileIdx, auto kNeedsMaskTag) {
    constexpr bool kNeedsMask = kNeedsMaskTag.value;

    // async load V[tileIdx]
    MemLoader::gm2sm(gmemTiledCopyQKV, tVsV, tVgV, tKVcKV, blockInfo.seqLenKV - tileIdx * kBc, tileIdx);
    cp_async_fence();

    // S = Q @ K^T
    auto accS = partition_fragment_C(tiledMma, Shape<Int<kBr>, Int<kBc>>{});
    clear(accS);
    GemmOp::computeScore(accS, tiledMma, smemTiledCopyQ, smemTiledCopyK, tSsQ, tSsK, tSrQCopy, tSrKCopy, tSrQ, tSrK);

    // wait for V load
    cpAsyncWaitGroup<0>();
    __syncthreads();

    // async load next K [tileIdx - 1]
    bool hasNext = tileIdx > 0;
    if (hasNext) {
      MemLoader::gm2sm(gmemTiledCopyQKV, tKsK, tKgK, tKVcKV, blockInfo.seqLenKV - (tileIdx - 1) * kBc, tileIdx - 1);
      cp_async_fence();
    }

    // masking
    if constexpr (kNeedsMask) {
      Softmax<Config, kNRows>::template applyMask<kIsCausal>(accS, tiledMma, tidx, blockInfo, tileIdx);
    }

    // softmax
    softmax.update(accS, accO, kAttnScale);

    // O += P @ V
    GemmOp::computeOutput<Config>(accO, accS, tiledMma, smemTiledCopyV, tOsVt, tOrVtCopy, tOrVt);

    // wait for next K load
    if (hasNext) {
      cpAsyncWaitGroup<0>();
      __syncthreads();
    }
  };

  // masking iterations
  for (; kvTileIdx >= nNoMaskEnd; --kvTileIdx) {
    tileIteration(kvTileIdx, std::integral_constant<bool, true>{});
  }

  // no masking iterations
  for (; kvTileIdx >= 0; --kvTileIdx) {
    tileIteration(kvTileIdx, std::integral_constant<bool, false>{});
  }

  // finalize softmax
  softmax.finalize(accO);

  // store O
  MemStore::storeO<Config>(accO, smemQ, tiledMma, tidx, gO, blockInfo);
}

TFA_DEFINE_KERNEL_WRAPPER(flashAttnMma)

}  // namespace tfa::mma
