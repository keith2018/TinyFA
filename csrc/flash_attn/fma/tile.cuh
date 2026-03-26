/*
 * TinyFA
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "layout.cuh"
#include "memory.cuh"

namespace tfa::fma {

template <typename Config, int NumRows>
struct Tile {
  using DType = typename Config::DType;
  using Layout = TileLayout<Config>;
  static constexpr int kHeadDim = Config::kHeadDim;
  static constexpr int kElemsPerVec = Config::kElemsPerVecLoad;

  DType* smem;

  __device__ explicit Tile(DType* smemBase) : smem(smemBase) {}

  static constexpr size_t numElems() { return NumRows * kHeadDim; }
  static constexpr size_t smemSize() { return numElems() * sizeof(DType); }

  template <typename Context>
  __device__ __forceinline__ void gm2sm(const DType* __restrict__ globalPtr, int stride, int validRows,
                                        const Context& ctx) {
    MemLoader<Config>::template gm2sm<NumRows>(smem, globalPtr, stride, validRows, ctx);
  }

  template <typename Context>
  __device__ __forceinline__ void gm2sm_async(const DType* __restrict__ globalPtr, int stride, int validRows,
                                              const Context& ctx) {
    MemLoader<Config>::template gm2sm_async<NumRows>(smem, globalPtr, stride, validRows, ctx);
  }

  __device__ __forceinline__ DType at(int row, int col) const { return smem[Layout::map(row, col, kHeadDim)]; }

  __device__ __forceinline__ void sm2reg(DType* __restrict__ regPtr, int row, int vecIdx) const {
    int smemIdx = Layout::map(row, vecIdx * kElemsPerVec, kHeadDim);
    MemLoader<Config>::sm2reg(regPtr, &smem[smemIdx]);
  }
};

template <typename Config>
using QTile = Tile<Config, Config::kBr>;

template <typename Config>
using KTile = Tile<Config, Config::kBc>;

template <typename Config>
using VTile = Tile<Config, Config::kBc>;

template <typename Config>
struct SmemSize {
  static constexpr size_t kSmemSize =
      Config::kUseCpAsync ? (QTile<Config>::smemSize() + KTile<Config>::smemSize() + VTile<Config>::smemSize())
                          : (QTile<Config>::smemSize() + KTile<Config>::smemSize());
};

}  // namespace tfa::fma
