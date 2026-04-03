/*
 * TinyFA
 * @author 	: keith@robot9.me
 *
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <random>
#include <vector>

#include "cpu_reference.h"
#include "flash_attn/flash_api.cuh"
#include "gtest/gtest.h"

using namespace testing;

namespace tfa::test {

#define CUDA_CHECK_TEST(call)                               \
  do {                                                      \
    cudaError_t err = call;                                 \
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err); \
  } while (0)

void initRandomData(std::vector<float>& data, float minVal = -1.0f, float maxVal = 1.0f, unsigned seed = 42) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(minVal, maxVal);
  for (auto& v : data) {
    v = dist(gen);
  }
}

template <typename DType>
DType floatToType(float val);

template <>
float floatToType<float>(float val) {
  return val;
}

template <>
__half floatToType<__half>(float val) {
  return __float2half(val);
}

template <>
__nv_bfloat16 floatToType<__nv_bfloat16>(float val) {
  return __float2bfloat16(val);
}

template <typename DType>
float typeToFloat(DType val);

template <>
float typeToFloat<float>(float val) {
  return val;
}

template <>
float typeToFloat<__half>(__half val) {
  return __half2float(val);
}

template <>
float typeToFloat<__nv_bfloat16>(__nv_bfloat16 val) {
  return __bfloat162float(val);
}

template <typename DType>
const char* getTypeName();

template <>
const char* getTypeName<float>() {
  return "FP32";
}

template <>
const char* getTypeName<__half>() {
  return "FP16";
}

template <>
const char* getTypeName<__nv_bfloat16>() {
  return "BF16";
}

template <typename DType>
void getTolerance(float& relTol, float& absTol);

template <>
void getTolerance<float>(float& relTol, float& absTol) {
  relTol = 1e-2f;
  absTol = 1e-3f;
}

template <>
void getTolerance<__half>(float& relTol, float& absTol) {
  relTol = 5e-2f;
  absTol = 5e-3f;
}

template <>
void getTolerance<__nv_bfloat16>(float& relTol, float& absTol) {
  relTol = 1e-1f;
  absTol = 1e-2f;
}

AssertionResult compareResults(const std::vector<float>& ref, const std::vector<float>& test, float relTol = 1e-2f,
                               float absTol = 1e-3f) {
  if (ref.size() != test.size()) {
    return AssertionFailure() << "Size mismatch: " << ref.size() << " vs " << test.size();
  }

  int mismatchCount = 0;
  float maxDiff = 0.0f;
  int maxDiffIdx = 0;

  for (size_t i = 0; i < ref.size(); i++) {
    float diff = std::abs(ref[i] - test[i]);
    float tol = absTol + relTol * std::abs(ref[i]);
    if (diff > tol) {
      mismatchCount++;
      if (diff > maxDiff) {
        maxDiff = diff;
        maxDiffIdx = i;
      }
    }
  }

  if (mismatchCount > 0) {
    return AssertionFailure() << "Mismatch: " << mismatchCount << "/" << ref.size() << ", max diff at [" << maxDiffIdx
                              << "]: " << ref[maxDiffIdx] << " vs " << test[maxDiffIdx] << " (diff=" << maxDiff << ")";
  }
  return AssertionSuccess();
}

// TFA_TARGET_DTYPE values: 1=FP16, 2=BF16, 3=FP32
#ifdef TFA_TARGET_DTYPE
#if TFA_TARGET_DTYPE == 1
#define TFA_HAS_FP16 1
#elif TFA_TARGET_DTYPE == 2
#define TFA_HAS_BF16 1
#elif TFA_TARGET_DTYPE == 3
#define TFA_HAS_FP32 1
#endif
#else
// All dtypes available
#define TFA_HAS_FP16 1
#define TFA_HAS_BF16 1
#define TFA_HAS_FP32 1
#endif

template <typename DType>
void callFlashAttn(DType* dQ, DType* dK, DType* dV, DType* dO, int batchSize, int seqLenQ, int seqLenKV, int numHeadsQ,
                   int numHeadsKV, int headDim, bool isCausal) {
  tfa::flashAttn(dQ, dK, dV, dO, batchSize, seqLenQ, seqLenKV, numHeadsQ, numHeadsKV, headDim, isCausal);
}

template <typename DType>
void callFlashAttnVarLen(DType* dQ, DType* dK, DType* dV, DType* dO, int* dCuSeqLensQ, int* dCuSeqLensKV, int batchSize,
                         int maxSeqLenQ, int maxSeqLenKV, int numHeadsQ, int numHeadsKV, int headDim, bool isCausal) {
  tfa::flashAttnVarLen(dQ, dK, dV, dO, dCuSeqLensQ, dCuSeqLensKV, batchSize, maxSeqLenQ, maxSeqLenKV, numHeadsQ,
                       numHeadsKV, headDim, isCausal);
}

class FlashAttnTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
      GTEST_SKIP() << "No CUDA device available";
    }
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
      GTEST_SKIP() << "Skipping due to sticky CUDA error from a prior test: " << cudaGetErrorString(err);
    }
  }

  void TearDown() override { cudaDeviceSynchronize(); }

  void runFlashAttnTest(int batchSize, int seqLenQ, int seqLenKV, int numHeadsQ, int numHeadsKV, int headDim,
                        bool isCausal = false) {
#if defined(TFA_TARGET_DTYPE) && TFA_TARGET_DTYPE == 1
    runFlashAttnTestTyped<__half>(batchSize, seqLenQ, seqLenKV, numHeadsQ, numHeadsKV, headDim, isCausal);
#elif defined(TFA_TARGET_DTYPE) && TFA_TARGET_DTYPE == 2
    runFlashAttnTestTyped<__nv_bfloat16>(batchSize, seqLenQ, seqLenKV, numHeadsQ, numHeadsKV, headDim, isCausal);
#elif defined(TFA_TARGET_DTYPE) && TFA_TARGET_DTYPE == 3
    runFlashAttnTestTyped<float>(batchSize, seqLenQ, seqLenKV, numHeadsQ, numHeadsKV, headDim, isCausal);
#elif defined(USE_CUTE_FLASH)
    runFlashAttnTestTyped<__half>(batchSize, seqLenQ, seqLenKV, numHeadsQ, numHeadsKV, headDim, isCausal);
#else
    runFlashAttnTestTyped<float>(batchSize, seqLenQ, seqLenKV, numHeadsQ, numHeadsKV, headDim, isCausal);
#endif
  }

  template <typename DType>
  void runFlashAttnTestTyped(int batchSize, int seqLenQ, int seqLenKV, int numHeadsQ, int numHeadsKV, int headDim,
                             bool isCausal = false) {
    if (!tfa::isHeadDimCompiled(headDim)) {
      GTEST_SKIP() << "headDim=" << headDim << " not compiled, skipping";
    }
    const int qSize = batchSize * seqLenQ * numHeadsQ * headDim;
    const int kvSize = batchSize * seqLenKV * numHeadsKV * headDim;

    std::vector<float> hQFloat(qSize), hKFloat(kvSize), hVFloat(kvSize);
    std::vector<float> hORef(qSize), hOTest(qSize);

    initRandomData(hQFloat);
    initRandomData(hKFloat);
    initRandomData(hVFloat);

    cpuAttentionFixedLen(hQFloat.data(), hKFloat.data(), hVFloat.data(), hORef.data(), batchSize, seqLenQ, seqLenKV,
                         numHeadsQ, numHeadsKV, headDim, isCausal);

    std::vector<DType> hQ(qSize), hK(kvSize), hV(kvSize), hO(qSize);
    for (int i = 0; i < qSize; i++) hQ[i] = floatToType<DType>(hQFloat[i]);
    for (int i = 0; i < kvSize; i++) {
      hK[i] = floatToType<DType>(hKFloat[i]);
      hV[i] = floatToType<DType>(hVFloat[i]);
    }

    DType *dQ, *dK, *dV, *dO;
    CUDA_CHECK_TEST(cudaMalloc(&dQ, qSize * sizeof(DType)));
    CUDA_CHECK_TEST(cudaMalloc(&dK, kvSize * sizeof(DType)));
    CUDA_CHECK_TEST(cudaMalloc(&dV, kvSize * sizeof(DType)));
    CUDA_CHECK_TEST(cudaMalloc(&dO, qSize * sizeof(DType)));

    CUDA_CHECK_TEST(cudaMemcpy(dQ, hQ.data(), qSize * sizeof(DType), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemcpy(dK, hK.data(), kvSize * sizeof(DType), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemcpy(dV, hV.data(), kvSize * sizeof(DType), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemset(dO, 0, qSize * sizeof(DType)));

    callFlashAttn(dQ, dK, dV, dO, batchSize, seqLenQ, seqLenKV, numHeadsQ, numHeadsKV, headDim, isCausal);
    CUDA_CHECK_TEST(cudaDeviceSynchronize());

    CUDA_CHECK_TEST(cudaMemcpy(hO.data(), dO, qSize * sizeof(DType), cudaMemcpyDeviceToHost));

    for (int i = 0; i < qSize; i++) hOTest[i] = typeToFloat(hO[i]);

    float relTol, absTol;
    getTolerance<DType>(relTol, absTol);
    EXPECT_TRUE(compareResults(hORef, hOTest, relTol, absTol)) << "Failed for type: " << getTypeName<DType>();

    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dO);
  }

  void runFlashAttnVarLenTest(const std::vector<int>& seqLensQ, const std::vector<int>& seqLensKV, int numHeadsQ,
                              int numHeadsKV, int headDim, bool isCausal = false) {
#if defined(TFA_TARGET_DTYPE) && TFA_TARGET_DTYPE == 1
    runFlashAttnVarLenTestTyped<__half>(seqLensQ, seqLensKV, numHeadsQ, numHeadsKV, headDim, isCausal);
#elif defined(TFA_TARGET_DTYPE) && TFA_TARGET_DTYPE == 2
    runFlashAttnVarLenTestTyped<__nv_bfloat16>(seqLensQ, seqLensKV, numHeadsQ, numHeadsKV, headDim, isCausal);
#elif defined(TFA_TARGET_DTYPE) && TFA_TARGET_DTYPE == 3
    runFlashAttnVarLenTestTyped<float>(seqLensQ, seqLensKV, numHeadsQ, numHeadsKV, headDim, isCausal);
#else
    runFlashAttnVarLenTestTyped<float>(seqLensQ, seqLensKV, numHeadsQ, numHeadsKV, headDim, isCausal);
#endif
  }

  template <typename DType>
  void runFlashAttnVarLenTestTyped(const std::vector<int>& seqLensQ, const std::vector<int>& seqLensKV, int numHeadsQ,
                                   int numHeadsKV, int headDim, bool isCausal = false) {
    if (!tfa::isHeadDimCompiled(headDim)) {
      GTEST_SKIP() << "headDim=" << headDim << " not compiled, skipping";
    }
    int batchSize = seqLensQ.size();

    std::vector<int> cuSeqLensQ(batchSize + 1), cuSeqLensKV(batchSize + 1);
    cuSeqLensQ[0] = 0;
    cuSeqLensKV[0] = 0;
    int totalQ = 0, totalKV = 0;
    int maxSeqLenQ = 0, maxSeqLenKV = 0;

    for (int i = 0; i < batchSize; i++) {
      totalQ += seqLensQ[i];
      totalKV += seqLensKV[i];
      cuSeqLensQ[i + 1] = totalQ;
      cuSeqLensKV[i + 1] = totalKV;
      maxSeqLenQ = std::max(maxSeqLenQ, seqLensQ[i]);
      maxSeqLenKV = std::max(maxSeqLenKV, seqLensKV[i]);
    }

    const int qSize = totalQ * numHeadsQ * headDim;
    const int kvSize = totalKV * numHeadsKV * headDim;

    std::vector<float> hQFloat(qSize), hKFloat(kvSize), hVFloat(kvSize);
    std::vector<float> hORef(qSize), hOTest(qSize);

    initRandomData(hQFloat);
    initRandomData(hKFloat);
    initRandomData(hVFloat);

    cpuAttentionVarLen(hQFloat.data(), hKFloat.data(), hVFloat.data(), hORef.data(), cuSeqLensQ.data(),
                       cuSeqLensKV.data(), batchSize, numHeadsQ, numHeadsKV, headDim, isCausal);

    std::vector<DType> hQ(qSize), hK(kvSize), hV(kvSize), hO(qSize);
    for (int i = 0; i < qSize; i++) hQ[i] = floatToType<DType>(hQFloat[i]);
    for (int i = 0; i < kvSize; i++) {
      hK[i] = floatToType<DType>(hKFloat[i]);
      hV[i] = floatToType<DType>(hVFloat[i]);
    }

    DType *dQ, *dK, *dV, *dO;
    int *dCuSeqLensQ, *dCuSeqLensKV;

    CUDA_CHECK_TEST(cudaMalloc(&dQ, qSize * sizeof(DType)));
    CUDA_CHECK_TEST(cudaMalloc(&dK, kvSize * sizeof(DType)));
    CUDA_CHECK_TEST(cudaMalloc(&dV, kvSize * sizeof(DType)));
    CUDA_CHECK_TEST(cudaMalloc(&dO, qSize * sizeof(DType)));
    CUDA_CHECK_TEST(cudaMalloc(&dCuSeqLensQ, (batchSize + 1) * sizeof(int)));
    CUDA_CHECK_TEST(cudaMalloc(&dCuSeqLensKV, (batchSize + 1) * sizeof(int)));

    CUDA_CHECK_TEST(cudaMemcpy(dQ, hQ.data(), qSize * sizeof(DType), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemcpy(dK, hK.data(), kvSize * sizeof(DType), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemcpy(dV, hV.data(), kvSize * sizeof(DType), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemset(dO, 0, qSize * sizeof(DType)));
    CUDA_CHECK_TEST(cudaMemcpy(dCuSeqLensQ, cuSeqLensQ.data(), (batchSize + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(
        cudaMemcpy(dCuSeqLensKV, cuSeqLensKV.data(), (batchSize + 1) * sizeof(int), cudaMemcpyHostToDevice));

    callFlashAttnVarLen(dQ, dK, dV, dO, dCuSeqLensQ, dCuSeqLensKV, batchSize, maxSeqLenQ, maxSeqLenKV, numHeadsQ,
                        numHeadsKV, headDim, isCausal);
    CUDA_CHECK_TEST(cudaDeviceSynchronize());

    CUDA_CHECK_TEST(cudaMemcpy(hO.data(), dO, qSize * sizeof(DType), cudaMemcpyDeviceToHost));

    for (int i = 0; i < qSize; i++) hOTest[i] = typeToFloat(hO[i]);

    float relTol, absTol;
    getTolerance<DType>(relTol, absTol);
    EXPECT_TRUE(compareResults(hORef, hOTest, relTol, absTol)) << "Failed for type: " << getTypeName<DType>();

    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dO);
    cudaFree(dCuSeqLensQ);
    cudaFree(dCuSeqLensKV);
  }
};

TEST_F(FlashAttnTest, Basic_SmallSequence) { runFlashAttnTest(1, 16, 16, 1, 1, 32); }
TEST_F(FlashAttnTest, Basic_MediumSequence) { runFlashAttnTest(2, 64, 64, 4, 4, 64); }
TEST_F(FlashAttnTest, Basic_LargeSequence) { runFlashAttnTest(2, 256, 256, 8, 8, 128); }

// HeadDim
TEST_F(FlashAttnTest, HeadDim32) { runFlashAttnTest(2, 64, 64, 4, 4, 32); }
TEST_F(FlashAttnTest, HeadDim96) { runFlashAttnTest(2, 64, 64, 4, 4, 96); }
TEST_F(FlashAttnTest, HeadDim128) { runFlashAttnTest(2, 64, 64, 4, 4, 128); }
TEST_F(FlashAttnTest, HeadDim192) { runFlashAttnTest(2, 64, 64, 4, 4, 192); }
TEST_F(FlashAttnTest, HeadDim256) { runFlashAttnTest(2, 64, 64, 4, 4, 256); }

TEST_F(FlashAttnTest, HeadDim32_Causal) { runFlashAttnTest(2, 64, 64, 4, 4, 32, true); }
TEST_F(FlashAttnTest, HeadDim96_Causal) { runFlashAttnTest(2, 64, 64, 4, 4, 96, true); }
TEST_F(FlashAttnTest, HeadDim128_Causal) { runFlashAttnTest(2, 64, 64, 4, 4, 128, true); }
TEST_F(FlashAttnTest, HeadDim192_Causal) { runFlashAttnTest(2, 64, 64, 4, 4, 192, true); }
TEST_F(FlashAttnTest, HeadDim256_Causal) { runFlashAttnTest(2, 64, 64, 4, 4, 256, true); }

TEST_F(FlashAttnTest, HeadDim32_GQA) { runFlashAttnTest(2, 64, 64, 8, 2, 32); }

#define VARLEN_HEADDIM_TEST(name, hQ, hKV, hd, ...)              \
  TEST_F(FlashAttnTest, name) {                                  \
    std::vector<int> sq = {32, 48, 64}, skv = {32, 48, 64};      \
    runFlashAttnVarLenTest(sq, skv, hQ, hKV, hd, ##__VA_ARGS__); \
  }

VARLEN_HEADDIM_TEST(HeadDim32_VarLen, 4, 4, 32)
VARLEN_HEADDIM_TEST(HeadDim64_VarLen, 4, 4, 64)
VARLEN_HEADDIM_TEST(HeadDim96_VarLen, 4, 4, 96)
VARLEN_HEADDIM_TEST(HeadDim128_VarLen, 4, 4, 128)
VARLEN_HEADDIM_TEST(HeadDim192_VarLen, 4, 4, 192)
VARLEN_HEADDIM_TEST(HeadDim256_VarLen, 4, 4, 256)

VARLEN_HEADDIM_TEST(HeadDim32_VarLen_Causal, 4, 4, 32, true)
VARLEN_HEADDIM_TEST(HeadDim64_VarLen_Causal, 4, 4, 64, true)

#undef VARLEN_HEADDIM_TEST

// seqLen
TEST_F(FlashAttnTest, SeqLen_VeryShort) { runFlashAttnTest(2, 1, 16, 4, 4, 64); }
TEST_F(FlashAttnTest, SeqLen_Unequal_QShorter) { runFlashAttnTest(2, 32, 128, 4, 4, 64); }
TEST_F(FlashAttnTest, SeqLen_Unequal_QLonger) { runFlashAttnTest(2, 128, 32, 4, 4, 64); }
TEST_F(FlashAttnTest, SeqLen_NotMultipleOfTile) { runFlashAttnTest(2, 67, 83, 4, 4, 64); }
TEST_F(FlashAttnTest, SeqLen_PrimeNumbers) { runFlashAttnTest(3, 97, 101, 4, 4, 64); }
TEST_F(FlashAttnTest, SeqLen_Long) { runFlashAttnTest(1, 512, 512, 8, 8, 64); }

// batch
TEST_F(FlashAttnTest, Batch_Single) { runFlashAttnTest(1, 64, 64, 4, 4, 64); }
TEST_F(FlashAttnTest, Batch_Multiple) { runFlashAttnTest(8, 64, 64, 4, 4, 64); }
TEST_F(FlashAttnTest, Batch_Large) { runFlashAttnTest(16, 32, 32, 4, 4, 64); }
TEST_F(FlashAttnTest, SingleHead) { runFlashAttnTest(2, 64, 64, 1, 1, 64); }
TEST_F(FlashAttnTest, ManyHeads) { runFlashAttnTest(2, 64, 64, 16, 16, 64); }

// GQA
TEST_F(FlashAttnTest, GQA_2Groups) { runFlashAttnTest(2, 64, 64, 8, 4, 64); }
TEST_F(FlashAttnTest, GQA_4Groups) { runFlashAttnTest(2, 64, 64, 8, 2, 64); }
TEST_F(FlashAttnTest, GQA_8Groups) { runFlashAttnTest(2, 64, 64, 8, 1, 64); }
TEST_F(FlashAttnTest, GQA_WithLargerHeadDim) { runFlashAttnTest(2, 64, 64, 16, 4, 128); }
TEST_F(FlashAttnTest, GQA_MQA) { runFlashAttnTest(2, 64, 64, 32, 1, 64); }
TEST_F(FlashAttnTest, GQA_MQA_Causal) { runFlashAttnTest(2, 64, 64, 32, 1, 64, true); }
TEST_F(FlashAttnTest, GQA_MQA_HeadDim128) { runFlashAttnTest(2, 64, 64, 16, 1, 128); }

// causal
TEST_F(FlashAttnTest, Causal_Basic) { runFlashAttnTest(2, 64, 64, 4, 4, 64, true); }
TEST_F(FlashAttnTest, Causal_LongSequence) { runFlashAttnTest(2, 256, 256, 8, 8, 64, true); }
TEST_F(FlashAttnTest, Causal_WithGQA) { runFlashAttnTest(2, 64, 64, 8, 2, 64, true); }
TEST_F(FlashAttnTest, Causal_NotAligned) { runFlashAttnTest(2, 67, 67, 4, 4, 64, true); }
TEST_F(FlashAttnTest, Causal_SingleToken) { runFlashAttnTest(2, 1, 64, 4, 4, 64, true); }
TEST_F(FlashAttnTest, Causal_UnequalQKV) { runFlashAttnTest(2, 32, 128, 4, 4, 64, true); }
TEST_F(FlashAttnTest, Causal_QLongerThanKV) { runFlashAttnTest(2, 128, 32, 4, 4, 64, true); }
TEST_F(FlashAttnTest, Causal_SingleKVToken) { runFlashAttnTest(2, 64, 1, 4, 4, 64, true); }
TEST_F(FlashAttnTest, Causal_QLongerNotAligned) { runFlashAttnTest(2, 97, 53, 4, 4, 64, true); }

// varLen
TEST_F(FlashAttnTest, VarLen_Basic) {
  std::vector<int> sq = {32, 64, 48}, skv = {32, 64, 48};
  runFlashAttnVarLenTest(sq, skv, 4, 4, 64);
}

TEST_F(FlashAttnTest, VarLen_DifferentLengths) {
  std::vector<int> sq = {16, 64, 32, 128}, skv = {16, 64, 32, 128};
  runFlashAttnVarLenTest(sq, skv, 4, 4, 64);
}

TEST_F(FlashAttnTest, VarLen_UnequalQKV) {
  std::vector<int> sq = {32, 64, 16}, skv = {64, 128, 32};
  runFlashAttnVarLenTest(sq, skv, 4, 4, 64);
}

TEST_F(FlashAttnTest, VarLen_WithGQA) {
  std::vector<int> sq = {32, 48, 64}, skv = {32, 48, 64};
  runFlashAttnVarLenTest(sq, skv, 8, 2, 64);
}

TEST_F(FlashAttnTest, VarLen_Causal) {
  std::vector<int> sq = {32, 64, 48}, skv = {32, 64, 48};
  runFlashAttnVarLenTest(sq, skv, 4, 4, 64, true);
}

TEST_F(FlashAttnTest, VarLen_Causal_NotAligned) {
  std::vector<int> sq = {17, 53, 89}, skv = {17, 53, 89};
  runFlashAttnVarLenTest(sq, skv, 4, 4, 64, true);
}

TEST_F(FlashAttnTest, VarLen_NotAligned) {
  std::vector<int> sq = {17, 53, 89}, skv = {23, 67, 101};
  runFlashAttnVarLenTest(sq, skv, 4, 4, 64);
}

TEST_F(FlashAttnTest, VarLen_SingleBatch) {
  std::vector<int> sq = {128}, skv = {256};
  runFlashAttnVarLenTest(sq, skv, 4, 4, 64);
}

TEST_F(FlashAttnTest, VarLen_ManyBatches) {
  std::vector<int> sq = {16, 32, 24, 48, 64, 40, 56, 72};
  std::vector<int> skv = {16, 32, 24, 48, 64, 40, 56, 72};
  runFlashAttnVarLenTest(sq, skv, 4, 4, 64);
}

TEST_F(FlashAttnTest, VarLen_Causal_UnequalQKV) {
  std::vector<int> sq = {16, 32, 48}, skv = {32, 64, 96};
  runFlashAttnVarLenTest(sq, skv, 4, 4, 64, true);
}

TEST_F(FlashAttnTest, VarLen_Causal_UnequalQKV_GQA) {
  std::vector<int> sq = {16, 48, 24}, skv = {64, 96, 128};
  runFlashAttnVarLenTest(sq, skv, 8, 2, 64, true);
}

TEST_F(FlashAttnTest, VarLen_Causal_UnequalQKV_NotAligned) {
  std::vector<int> sq = {13, 37, 53}, skv = {29, 71, 97};
  runFlashAttnVarLenTest(sq, skv, 4, 4, 64, true);
}

TEST_F(FlashAttnTest, VarLen_SingleToken) {
  std::vector<int> sq = {1, 64, 1, 32}, skv = {1, 64, 1, 32};
  runFlashAttnVarLenTest(sq, skv, 4, 4, 64);
}

TEST_F(FlashAttnTest, VarLen_ExtremeDisparity) {
  std::vector<int> sq = {1, 512, 2, 256}, skv = {1, 512, 2, 256};
  runFlashAttnVarLenTest(sq, skv, 4, 4, 64);
}

TEST_F(FlashAttnTest, VarLen_LargeBatch) {
  std::vector<int> sq = {8, 64, 16, 96, 24, 48, 32, 128, 12, 80, 20, 56, 36, 72, 44, 100};
  std::vector<int> skv = {8, 64, 16, 96, 24, 48, 32, 128, 12, 80, 20, 56, 36, 72, 44, 100};
  runFlashAttnVarLenTest(sq, skv, 4, 4, 64);
}

TEST_F(FlashAttnTest, VarLen_Causal_QLongerThanKV) {
  std::vector<int> sq = {64, 128, 96}, skv = {32, 64, 48};
  runFlashAttnVarLenTest(sq, skv, 4, 4, 64, true);
}

TEST_F(FlashAttnTest, VarLen_MQA) {
  std::vector<int> sq = {32, 64, 48}, skv = {32, 64, 48};
  runFlashAttnVarLenTest(sq, skv, 16, 1, 64);
}

TEST_F(FlashAttnTest, VarLen_MQA_Causal) {
  std::vector<int> sq = {32, 48, 64}, skv = {32, 48, 64};
  runFlashAttnVarLenTest(sq, skv, 16, 1, 64, true);
}

// others
TEST_F(FlashAttnTest, Edge_MinimalDimensions) { runFlashAttnTest(1, 1, 1, 1, 1, 32); }
TEST_F(FlashAttnTest, Edge_SingleQueryToken) { runFlashAttnTest(4, 1, 256, 8, 8, 64); }
TEST_F(FlashAttnTest, Edge_SingleKVToken) { runFlashAttnTest(4, 64, 1, 8, 8, 64); }
TEST_F(FlashAttnTest, Edge_BothSingleToken) { runFlashAttnTest(4, 1, 1, 8, 8, 64); }
TEST_F(FlashAttnTest, Edge_SingleQueryToken_Causal) { runFlashAttnTest(4, 1, 256, 8, 8, 64, true); }
TEST_F(FlashAttnTest, Edge_BothSingleToken_Causal) { runFlashAttnTest(4, 1, 1, 8, 8, 64, true); }

TEST_F(FlashAttnTest, Stress_LargeBatchAndSeq) { runFlashAttnTest(4, 512, 512, 8, 8, 128); }
TEST_F(FlashAttnTest, Stress_ManyHeadsSmallSeq) { runFlashAttnTest(8, 32, 32, 32, 32, 64); }
TEST_F(FlashAttnTest, Stress_HeadDim256_LongSeq) { runFlashAttnTest(2, 256, 256, 4, 4, 256); }
TEST_F(FlashAttnTest, Stress_HeadDim256_Causal_GQA) { runFlashAttnTest(2, 128, 128, 8, 2, 256, true); }
TEST_F(FlashAttnTest, Stress_HeadDim192_LongSeq) { runFlashAttnTest(2, 256, 256, 4, 4, 192); }

TEST_F(FlashAttnTest, Stress_VarLen_MixedSizes) {
  std::vector<int> sq = {8, 256, 32, 128, 16, 512, 64, 1};
  std::vector<int> skv = {8, 256, 32, 128, 16, 512, 64, 1};
  runFlashAttnVarLenTest(sq, skv, 8, 4, 64);
}

TEST_F(FlashAttnTest, Stress_VarLen_LargeHeadDim256) {
  std::vector<int> sq = {32, 64, 48, 96}, skv = {32, 64, 48, 96};
  runFlashAttnVarLenTest(sq, skv, 4, 4, 256);
}

TEST_F(FlashAttnTest, Combined_GQA_Causal_LargeHeadDim) { runFlashAttnTest(2, 128, 128, 16, 4, 128, true); }

TEST_F(FlashAttnTest, Combined_VarLen_GQA_Causal) {
  std::vector<int> sq = {32, 64, 48, 96}, skv = {32, 64, 48, 96};
  runFlashAttnVarLenTest(sq, skv, 8, 2, 64, true);
}

TEST_F(FlashAttnTest, Combined_AllFeatures) {
  std::vector<int> sq = {33, 67, 51}, skv = {45, 89, 73};
  runFlashAttnVarLenTest(sq, skv, 8, 2, 128, true);
}

TEST_F(FlashAttnTest, Combined_VarLen_GQA_Causal_HeadDim192) {
  std::vector<int> sq = {33, 67, 51}, skv = {45, 89, 73};
  runFlashAttnVarLenTest(sq, skv, 8, 2, 192, true);
}

TEST_F(FlashAttnTest, Combined_VarLen_GQA_Causal_HeadDim256) {
  std::vector<int> sq = {33, 67, 51}, skv = {45, 89, 73};
  runFlashAttnVarLenTest(sq, skv, 8, 2, 256, true);
}

TEST_F(FlashAttnTest, Combined_MQA_Causal_VarLen) {
  std::vector<int> sq = {16, 48, 24}, skv = {64, 96, 128};
  runFlashAttnVarLenTest(sq, skv, 32, 1, 128, true);
}

TEST_F(FlashAttnTest, Combined_VarLen_GQA_Causal_LargeBatch) {
  std::vector<int> sq = {17, 53, 31, 67, 43, 89, 23, 71, 37, 61, 19, 47};
  std::vector<int> skv = {17, 53, 31, 67, 43, 89, 23, 71, 37, 61, 19, 47};
  runFlashAttnVarLenTest(sq, skv, 8, 2, 64, true);
}

#define DTYPE_TEST_SUITE(PREFIX, TYPE)                                                                              \
                                                                                                                    \
  TEST_F(FlashAttnTest, PREFIX##_Basic) { runFlashAttnTestTyped<TYPE>(2, 64, 64, 4, 4, 64); }                       \
  TEST_F(FlashAttnTest, PREFIX##_HeadDim32) { runFlashAttnTestTyped<TYPE>(2, 64, 64, 4, 4, 32); }                   \
  TEST_F(FlashAttnTest, PREFIX##_HeadDim96) { runFlashAttnTestTyped<TYPE>(2, 64, 64, 4, 4, 96); }                   \
  TEST_F(FlashAttnTest, PREFIX##_HeadDim128) { runFlashAttnTestTyped<TYPE>(2, 64, 64, 4, 4, 128); }                 \
  TEST_F(FlashAttnTest, PREFIX##_HeadDim192) { runFlashAttnTestTyped<TYPE>(2, 64, 64, 4, 4, 192); }                 \
  TEST_F(FlashAttnTest, PREFIX##_HeadDim256) { runFlashAttnTestTyped<TYPE>(2, 64, 64, 4, 4, 256); }                 \
                                                                                                                    \
  TEST_F(FlashAttnTest, PREFIX##_Causal) { runFlashAttnTestTyped<TYPE>(2, 64, 64, 4, 4, 64, true); }                \
  TEST_F(FlashAttnTest, PREFIX##_Causal_HeadDim32) { runFlashAttnTestTyped<TYPE>(2, 64, 64, 4, 4, 32, true); }      \
  TEST_F(FlashAttnTest, PREFIX##_Causal_HeadDim96) { runFlashAttnTestTyped<TYPE>(2, 64, 64, 4, 4, 96, true); }      \
  TEST_F(FlashAttnTest, PREFIX##_Causal_HeadDim128) { runFlashAttnTestTyped<TYPE>(2, 64, 64, 4, 4, 128, true); }    \
  TEST_F(FlashAttnTest, PREFIX##_Causal_HeadDim192) { runFlashAttnTestTyped<TYPE>(2, 64, 64, 4, 4, 192, true); }    \
  TEST_F(FlashAttnTest, PREFIX##_Causal_HeadDim256) { runFlashAttnTestTyped<TYPE>(2, 64, 64, 4, 4, 256, true); }    \
                                                                                                                    \
  TEST_F(FlashAttnTest, PREFIX##_GQA) { runFlashAttnTestTyped<TYPE>(2, 64, 64, 8, 2, 64); }                         \
  TEST_F(FlashAttnTest, PREFIX##_GQA_HeadDim32) { runFlashAttnTestTyped<TYPE>(2, 64, 64, 8, 2, 32); }               \
  TEST_F(FlashAttnTest, PREFIX##_GQA_HeadDim128) { runFlashAttnTestTyped<TYPE>(2, 64, 64, 16, 4, 128); }            \
  TEST_F(FlashAttnTest, PREFIX##_GQA_MQA) { runFlashAttnTestTyped<TYPE>(2, 64, 64, 32, 1, 64); }                    \
                                                                                                                    \
  TEST_F(FlashAttnTest, PREFIX##_LongSequence) { runFlashAttnTestTyped<TYPE>(2, 256, 256, 8, 8, 64); }              \
  TEST_F(FlashAttnTest, PREFIX##_NotAligned) { runFlashAttnTestTyped<TYPE>(2, 67, 83, 4, 4, 64); }                  \
  TEST_F(FlashAttnTest, PREFIX##_Causal_QLongerThanKV) { runFlashAttnTestTyped<TYPE>(2, 128, 32, 4, 4, 64, true); } \
                                                                                                                    \
  TEST_F(FlashAttnTest, PREFIX##_VarLen) {                                                                          \
    std::vector<int> sq = {32, 64, 48}, skv = {32, 64, 48};                                                         \
    runFlashAttnVarLenTestTyped<TYPE>(sq, skv, 4, 4, 64);                                                           \
  }                                                                                                                 \
  TEST_F(FlashAttnTest, PREFIX##_VarLen_HeadDim32) {                                                                \
    std::vector<int> sq = {32, 48, 64}, skv = {32, 48, 64};                                                         \
    runFlashAttnVarLenTestTyped<TYPE>(sq, skv, 4, 4, 32);                                                           \
  }                                                                                                                 \
  TEST_F(FlashAttnTest, PREFIX##_VarLen_HeadDim96) {                                                                \
    std::vector<int> sq = {32, 48, 64}, skv = {32, 48, 64};                                                         \
    runFlashAttnVarLenTestTyped<TYPE>(sq, skv, 4, 4, 96);                                                           \
  }                                                                                                                 \
  TEST_F(FlashAttnTest, PREFIX##_VarLen_HeadDim128) {                                                               \
    std::vector<int> sq = {32, 48, 64}, skv = {32, 48, 64};                                                         \
    runFlashAttnVarLenTestTyped<TYPE>(sq, skv, 4, 4, 128);                                                          \
  }                                                                                                                 \
  TEST_F(FlashAttnTest, PREFIX##_VarLen_HeadDim192) {                                                               \
    std::vector<int> sq = {32, 48, 64}, skv = {32, 48, 64};                                                         \
    runFlashAttnVarLenTestTyped<TYPE>(sq, skv, 4, 4, 192);                                                          \
  }                                                                                                                 \
  TEST_F(FlashAttnTest, PREFIX##_VarLen_HeadDim256) {                                                               \
    std::vector<int> sq = {32, 48, 64}, skv = {32, 48, 64};                                                         \
    runFlashAttnVarLenTestTyped<TYPE>(sq, skv, 4, 4, 256);                                                          \
  }                                                                                                                 \
                                                                                                                    \
  TEST_F(FlashAttnTest, PREFIX##_VarLen_UnequalQKV) {                                                               \
    std::vector<int> sq = {32, 64, 16}, skv = {64, 128, 32};                                                        \
    runFlashAttnVarLenTestTyped<TYPE>(sq, skv, 4, 4, 64);                                                           \
  }                                                                                                                 \
  TEST_F(FlashAttnTest, PREFIX##_VarLen_Causal_GQA) {                                                               \
    std::vector<int> sq = {32, 64, 48}, skv = {32, 64, 48};                                                         \
    runFlashAttnVarLenTestTyped<TYPE>(sq, skv, 8, 2, 64, true);                                                     \
  }                                                                                                                 \
  TEST_F(FlashAttnTest, PREFIX##_VarLen_SingleToken) {                                                              \
    std::vector<int> sq = {1, 64, 1, 32}, skv = {1, 64, 1, 32};                                                     \
    runFlashAttnVarLenTestTyped<TYPE>(sq, skv, 4, 4, 64);                                                           \
  }                                                                                                                 \
  TEST_F(FlashAttnTest, PREFIX##_VarLen_Causal_UnequalQKV) {                                                        \
    std::vector<int> sq = {16, 32, 48}, skv = {32, 64, 96};                                                         \
    runFlashAttnVarLenTestTyped<TYPE>(sq, skv, 4, 4, 64, true);                                                     \
  }                                                                                                                 \
  TEST_F(FlashAttnTest, PREFIX##_VarLen_NotAligned) {                                                               \
    std::vector<int> sq = {17, 53, 89}, skv = {23, 67, 101};                                                        \
    runFlashAttnVarLenTestTyped<TYPE>(sq, skv, 4, 4, 64);                                                           \
  }                                                                                                                 \
  TEST_F(FlashAttnTest, PREFIX##_Combined) { runFlashAttnTestTyped<TYPE>(2, 128, 128, 16, 4, 128, true); }

#if TFA_HAS_FP16
DTYPE_TEST_SUITE(FP16, __half)
#endif

#if TFA_HAS_BF16
DTYPE_TEST_SUITE(BF16, __nv_bfloat16)
#endif

#if TFA_HAS_FP32
DTYPE_TEST_SUITE(FP32, float)
#endif

#undef DTYPE_TEST_SUITE

}  // namespace tfa::test
