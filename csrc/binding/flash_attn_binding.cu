/*
 * TinyFA
 * @author 	: keith@robot9.me
 *
 */

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "flash_attn/flash_api.cuh"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)   \
  do {                   \
    CHECK_CUDA(x);       \
    CHECK_CONTIGUOUS(x); \
  } while (0)
#define CHECK_SHAPE(x, ...) \
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

// TFA_TARGET_DTYPE: 1=FP16, 2=BF16, 3=FP32
#ifdef TFA_TARGET_DTYPE

#if TFA_TARGET_DTYPE == 1
#define DISPATCH_DTYPE(TENSOR, NAME, ...)                                                      \
  [&] {                                                                                        \
    if ((TENSOR).dtype() == torch::kFloat16) {                                                 \
      using DType = __half;                                                                    \
      __VA_ARGS__                                                                              \
    } else {                                                                                   \
      TORCH_CHECK(false, NAME " compiled with TFA_TARGET_DTYPE=fp16, got ", (TENSOR).dtype()); \
    }                                                                                          \
  }()
#elif TFA_TARGET_DTYPE == 2
#define DISPATCH_DTYPE(TENSOR, NAME, ...)                                                      \
  [&] {                                                                                        \
    if ((TENSOR).dtype() == torch::kBFloat16) {                                                \
      using DType = __nv_bfloat16;                                                             \
      __VA_ARGS__                                                                              \
    } else {                                                                                   \
      TORCH_CHECK(false, NAME " compiled with TFA_TARGET_DTYPE=bf16, got ", (TENSOR).dtype()); \
    }                                                                                          \
  }()
#elif TFA_TARGET_DTYPE == 3
#define DISPATCH_DTYPE(TENSOR, NAME, ...)                                                      \
  [&] {                                                                                        \
    if ((TENSOR).dtype() == torch::kFloat32) {                                                 \
      using DType = float;                                                                     \
      __VA_ARGS__                                                                              \
    } else {                                                                                   \
      TORCH_CHECK(false, NAME " compiled with TFA_TARGET_DTYPE=fp32, got ", (TENSOR).dtype()); \
    }                                                                                          \
  }()
#else
#error "TFA_TARGET_DTYPE has unsupported value. Valid: 1 (fp16), 2 (bf16), 3 (fp32)"
#endif

#else

#define DISPATCH_DTYPE(TENSOR, NAME, ...)                                                    \
  [&] {                                                                                      \
    if ((TENSOR).dtype() == torch::kFloat16) {                                               \
      using DType = __half;                                                                  \
      __VA_ARGS__                                                                            \
    } else if ((TENSOR).dtype() == torch::kBFloat16) {                                       \
      using DType = __nv_bfloat16;                                                           \
      __VA_ARGS__                                                                            \
    } else if ((TENSOR).dtype() == torch::kFloat32) {                                        \
      using DType = float;                                                                   \
      __VA_ARGS__                                                                            \
    } else {                                                                                 \
      TORCH_CHECK(false, NAME " only supports fp16, bf16 and fp32, got ", (TENSOR).dtype()); \
    }                                                                                        \
  }()

#endif  // TFA_TARGET_DTYPE

// Q/K/V: [batch, seq, heads, dim]
torch::Tensor flash_attn_forward(torch::Tensor Q,  // [batch, seqQ,  numHeadsQ,  headDim]
                                 torch::Tensor K,  // [batch, seqKV, numHeadsKV, headDim]
                                 torch::Tensor V,  // [batch, seqKV, numHeadsKV, headDim]
                                 bool is_causal) {
  CHECK_INPUT(Q);
  CHECK_INPUT(K);
  CHECK_INPUT(V);

  TORCH_CHECK(Q.dim() == 4, "Q must be 4D [batch, seqQ, numHeadsQ, headDim]");
  TORCH_CHECK(K.dim() == 4, "K must be 4D [batch, seqKV, numHeadsKV, headDim]");
  TORCH_CHECK(V.dim() == 4, "V must be 4D [batch, seqKV, numHeadsKV, headDim]");
  TORCH_CHECK(Q.dtype() == K.dtype() && Q.dtype() == V.dtype(), "Q, K, V must have the same dtype");

  const int batch = Q.size(0);
  const int seqLenQ = Q.size(1);
  const int numHeadsQ = Q.size(2);
  const int headDim = Q.size(3);
  const int seqLenKV = K.size(1);
  const int numHeadsKV = K.size(2);

  TORCH_CHECK(K.size(0) == batch && V.size(0) == batch, "Batch sizes must match");
  TORCH_CHECK(V.size(1) == seqLenKV, "K and V must have the same seqlen");
  TORCH_CHECK(K.size(3) == headDim && V.size(3) == headDim, "Head dims must match");
  TORCH_CHECK(V.size(2) == numHeadsKV, "K and V must have the same number of heads");
  TORCH_CHECK(numHeadsQ % numHeadsKV == 0, "numHeadsQ must be divisible by numHeadsKV (GQA)");

  auto O = torch::zeros_like(Q);

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

  DISPATCH_DTYPE(Q, "flash_attn_forward", {
    tfa::flashAttn<DType>(reinterpret_cast<const DType*>(Q.data_ptr()), reinterpret_cast<const DType*>(K.data_ptr()),
                          reinterpret_cast<const DType*>(V.data_ptr()), reinterpret_cast<DType*>(O.data_ptr()), batch,
                          seqLenQ, seqLenKV, numHeadsQ, numHeadsKV, headDim, is_causal, stream);
  });

  return O;
}

torch::Tensor flash_attn_varlen_forward(torch::Tensor Q,              // [totalQ,  numHeadsQ,  headDim]
                                        torch::Tensor K,              // [totalKV, numHeadsKV, headDim]
                                        torch::Tensor V,              // [totalKV, numHeadsKV, headDim]
                                        torch::Tensor cu_seqlens_q,   // [batch + 1], cumulative offsets
                                        torch::Tensor cu_seqlens_kv,  // [batch + 1], cumulative offsets
                                        int max_seqlen_q, int max_seqlen_kv, bool is_causal) {
  CHECK_INPUT(Q);
  CHECK_INPUT(K);
  CHECK_INPUT(V);
  CHECK_INPUT(cu_seqlens_q);
  CHECK_INPUT(cu_seqlens_kv);

  TORCH_CHECK(Q.dim() == 3, "Q must be 3D [totalQ, numHeadsQ, headDim]");
  TORCH_CHECK(K.dim() == 3, "K must be 3D [totalKV, numHeadsKV, headDim]");
  TORCH_CHECK(V.dim() == 3, "V must be 3D [totalKV, numHeadsKV, headDim]");
  TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_kv.dim() == 1, "cu_seqlens must be 1D");
  TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32 && cu_seqlens_kv.dtype() == torch::kInt32,
              "cu_seqlens must be int32");
  TORCH_CHECK(Q.dtype() == K.dtype() && Q.dtype() == V.dtype(), "Q, K, V must have the same dtype");

  const int batchSize = cu_seqlens_q.size(0) - 1;
  const int numHeadsQ = Q.size(1);
  const int headDim = Q.size(2);
  const int numHeadsKV = K.size(1);

  TORCH_CHECK(K.size(2) == headDim && V.size(2) == headDim, "Head dims must match");
  TORCH_CHECK(V.size(1) == numHeadsKV, "K and V must have the same number of heads");
  TORCH_CHECK(numHeadsQ % numHeadsKV == 0, "numHeadsQ must be divisible by numHeadsKV (GQA)");

  auto O = torch::zeros_like(Q);

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

  DISPATCH_DTYPE(Q, "flash_attn_varlen_forward", {
    tfa::flashAttnVarLen<DType>(reinterpret_cast<const DType*>(Q.data_ptr()),
                                reinterpret_cast<const DType*>(K.data_ptr()),
                                reinterpret_cast<const DType*>(V.data_ptr()), reinterpret_cast<DType*>(O.data_ptr()),
                                cu_seqlens_q.data_ptr<int>(), cu_seqlens_kv.data_ptr<int>(), batchSize, max_seqlen_q,
                                max_seqlen_kv, numHeadsQ, numHeadsKV, headDim, is_causal, stream);
  });

  return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "TinyFA: Flash Attention CUDA implementation";

  m.def("flash_attn_forward", &flash_attn_forward, "Flash Attention forward (fixed length)", py::arg("Q"), py::arg("K"),
        py::arg("V"), py::arg("is_causal") = false);

  m.def("flash_attn_varlen_forward", &flash_attn_varlen_forward, "Flash Attention forward (variable length)",
        py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("cu_seqlens_q"), py::arg("cu_seqlens_kv"),
        py::arg("max_seqlen_q"), py::arg("max_seqlen_kv"), py::arg("is_causal") = false);
}
