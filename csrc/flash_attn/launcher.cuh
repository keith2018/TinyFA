/*
 * TinyFA
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "config.cuh"
#include "fma/kernel.cuh"
#include "mma/kernel.cuh"
#include "params.cuh"
#include "utils.cuh"

namespace tfa {

namespace detail {

template <typename Config>
size_t getSmemSize() {
  if constexpr (Config::kUseTensorCore) {
    return mma::SmemLayout<Config>::kSmemSize;
  } else {
    return fma::SmemSize<Config>::kSmemSize;
  }
}

template <typename Config, typename Params>
void launchKernel(const Params& params, int gridX, int numHeads, int batchSize, bool isCausal, cudaStream_t stream) {
  size_t smemSize = getSmemSize<Config>();

  dim3 grid(gridX, batchSize, numHeads);
  dim3 block(Config::kNumThreads);

  auto launch = [&](auto kernel) {
    if (!configureSmem(kernel, smemSize)) {
      return;
    }
    kernel<<<grid, block, smemSize, stream>>>(params);
  };

  if constexpr (Config::kUseTensorCore) {
    launch(isCausal ? mma::flashAttentionKernel<Config, true, Params>
                    : mma::flashAttentionKernel<Config, false, Params>);
  } else {
    launch(isCausal ? fma::flashAttentionKernel<Config, true, Params>
                    : fma::flashAttentionKernel<Config, false, Params>);
  }
}

}  // namespace detail

namespace impl {

template <typename Params, typename DType>
void initParams(Params& params, const DType* Q, const DType* K, const DType* V, DType* O, int numHeadsQ,
                int numHeadsKV) {
  static constexpr int kHeadDim = Params::kHeadDim;
  params.Q = Q;
  params.K = K;
  params.V = V;
  params.O = O;
  params.seqDimQ = numHeadsQ * kHeadDim;
  params.seqDimKV = numHeadsKV * kHeadDim;
  params.groupSize = numHeadsQ / numHeadsKV;
}

template <typename ArchTag, typename DType, int kHeadDim, bool IsCausal>
void runFwd(const DType* Q, const DType* K, const DType* V, DType* O, int batchSize, int seqLenQ, int seqLenKV,
            int numHeadsQ, int numHeadsKV, cudaStream_t stream) {
  using Config = typename ConfigForArch<ArchTag, DType, kHeadDim, IsCausal>::Config;
  using Params = FixLenParams<DType, kHeadDim>;

  Params params;
  initParams(params, Q, K, V, O, numHeadsQ, numHeadsKV);
  params.seqLenQ = seqLenQ;
  params.seqLenKV = seqLenKV;
  params.numKVTiles = ceilDiv(seqLenKV, Config::kBc);

  detail::launchKernel<Config>(params, ceilDiv(seqLenQ, Config::kBr), numHeadsQ, batchSize, IsCausal, stream);
}

template <typename ArchTag, typename DType, int kHeadDim, bool IsCausal>
void runFwdVarlen(const DType* Q, const DType* K, const DType* V, DType* O, const int* cuSeqLensQ,
                  const int* cuSeqLensKV, int batchSize, int maxSeqLenQ, int maxSeqLenKV, int numHeadsQ, int numHeadsKV,
                  cudaStream_t stream) {
  using Config = typename ConfigForArch<ArchTag, DType, kHeadDim, IsCausal>::Config;
  using Params = VarLenParams<DType, kHeadDim>;

  Params params;
  initParams(params, Q, K, V, O, numHeadsQ, numHeadsKV);
  params.cuSeqLensQ = cuSeqLensQ;
  params.cuSeqLensKV = cuSeqLensKV;
  params.maxSeqLenQ = maxSeqLenQ;
  params.maxSeqLenKV = maxSeqLenKV;
  params.maxKVTiles = ceilDiv(maxSeqLenKV, Config::kBc);

  detail::launchKernel<Config>(params, ceilDiv(maxSeqLenQ, Config::kBr), numHeadsQ, batchSize, IsCausal, stream);
}

}  // namespace impl

}  // namespace tfa
