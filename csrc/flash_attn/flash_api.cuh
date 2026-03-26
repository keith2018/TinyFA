/*
 * TinyFA
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "config.cuh"
#include "params.cuh"
#include "utils.cuh"

namespace tfa {

namespace impl {

template <typename ArchTag, typename DType, int kHeadDim, bool IsCausal>
void runFwd(const DType* Q, const DType* K, const DType* V, DType* O, int batchSize, int seqLenQ, int seqLenKV,
            int numHeadsQ, int numHeadsKV, cudaStream_t stream);

template <typename ArchTag, typename DType, int kHeadDim, bool IsCausal>
void runFwdVarlen(const DType* Q, const DType* K, const DType* V, DType* O, const int* cuSeqLensQ,
                  const int* cuSeqLensKV, int batchSize, int maxSeqLenQ, int maxSeqLenKV, int numHeadsQ, int numHeadsKV,
                  cudaStream_t stream);
}  // namespace impl

#define TFA_DISPATCH_KERNEL(headDim, isCausal, ...) \
  do {                                              \
    const int arch_ = getRuntimeArch();             \
    TFA_DISPATCH_ARCH(arch_, {                      \
      TFA_DISPATCH_HEAD_DIM(headDim, kHeadDim, {    \
        if (isCausal) {                             \
          constexpr bool IsCausal = true;           \
          __VA_ARGS__                               \
        } else {                                    \
          constexpr bool IsCausal = false;          \
          __VA_ARGS__                               \
        }                                           \
      });                                           \
    });                                             \
  } while (0)

template <typename DType>
void flashAttn(const DType* Q, const DType* K, const DType* V, DType* O, int batchSize, int seqLenQ, int seqLenKV,
               int numHeadsQ, int numHeadsKV, int headDim, bool isCausal = false, cudaStream_t stream = nullptr) {
  TFA_DISPATCH_KERNEL(headDim, isCausal, {
    impl::runFwd<ArchTag, DType, kHeadDim, IsCausal>(Q, K, V, O, batchSize, seqLenQ, seqLenKV, numHeadsQ, numHeadsKV,
                                                     stream);
  });
}

template <typename DType>
void flashAttnVarLen(const DType* Q, const DType* K, const DType* V, DType* O, const int* cuSeqLensQ,
                     const int* cuSeqLensKV, int batchSize, int maxSeqLenQ, int maxSeqLenKV, int numHeadsQ,
                     int numHeadsKV, int headDim, bool isCausal = false, cudaStream_t stream = nullptr) {
  TFA_DISPATCH_KERNEL(headDim, isCausal, {
    impl::runFwdVarlen<ArchTag, DType, kHeadDim, IsCausal>(Q, K, V, O, cuSeqLensQ, cuSeqLensKV, batchSize, maxSeqLenQ,
                                                           maxSeqLenKV, numHeadsQ, numHeadsKV, stream);
  });
}

}  // namespace tfa
