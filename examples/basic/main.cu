/*
 * TinyFA
 * @author 	: keith@robot9.me
 *
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>

#include "flash_attn/flash_api.cuh"

int main() {
  constexpr int B = 1, S = 128, H = 8, D = 64;
  const size_t numel = B * S * H * D;

  half *dQ, *dK, *dV, *dO;
  cudaMalloc(&dQ, numel * sizeof(half));
  cudaMalloc(&dK, numel * sizeof(half));
  cudaMalloc(&dV, numel * sizeof(half));
  cudaMalloc(&dO, numel * sizeof(half));

  cudaMemset(dQ, 0, numel * sizeof(half));
  cudaMemset(dK, 0, numel * sizeof(half));
  cudaMemset(dV, 0, numel * sizeof(half));

  tfa::flashAttn<half>(dQ, dK, dV, dO,
                       /*batchSize=*/B,
                       /*seqLenQ=*/S,
                       /*seqLenKV=*/S,
                       /*numHeadsQ=*/H,
                       /*numHeadsKV=*/H,
                       /*headDim=*/D,
                       /*isCausal=*/false);

  cudaDeviceSynchronize();
  printf("TinyFA flash attention completed successfully.\n");

  cudaFree(dQ);
  cudaFree(dK);
  cudaFree(dV);
  cudaFree(dO);
  return 0;
}
