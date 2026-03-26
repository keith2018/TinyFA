# TinyFA

A lightweight, from-scratch Flash Attention CUDA implementation (forward pass only).

## Features

- Data Types: FP16, BF16, FP32
- Head Dimensions: 32, 64, 96, 128, 192, 256
- Causal Attention
- Grouped-Query Attention (`numHeadsQ % numHeadsKV == 0`)
- Fixed Length / Variable Length (`cu_seqlens` packed format)

## Installation

### Requirements

- NVIDIA GPU (Turing or newer, compute capability >= 7.5)
- CUDA Toolkit >= 11.4
- PyTorch (with CUDA support)
- Python >= 3.8
- C++17 compatible compiler

### Install from source

```bash
git clone --recursive https://github.com/keith2018/TinyFA.git
cd TinyFA/python
pip install --no-build-isolation .
```

### Faster compilation

By default, **all HeadDim variants** (32, 64, 96, 128, 192, 256) are compiled. Use environment variables to speed up compilation by targeting specific configurations:

```bash
# Target a specific head dimension
TFA_TARGET_HEADDIM=128 pip install .

# Target a specific GPU architecture
TFA_TARGET_SM=sm80 pip install .

# Target a specific data type
TFA_TARGET_DTYPE=fp16 pip install .

# Combine for fastest compilation
TFA_TARGET_SM=sm80 TFA_TARGET_DTYPE=fp16 TFA_TARGET_HEADDIM=128 pip install --no-build-isolation .
```

## Benchmarks

Speedup vs [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention): 94-96%

- Device: NVIDIA A100-SXM4-40GB
- Configuration: batch=2, heads=128, SeqLen=4096.

| Dtype | Causal | TinyFA (ms) | TinyFA (TFLOPS) | flash_attn (ms) | flash_attn (TFLOPS) | Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **fp16** | False | 2.832 | 194.09 | 2.654 | 207.14 | 0.94x |
| **fp16** | True | 1.636 | 168.01 | 1.557 | 176.54 | 0.95x |
| **bf16** | False | 2.743 | 200.40 | 2.623 | 209.58 | 0.96x |
| **bf16** | True | 1.624 | 169.28 | 1.537 | 178.82 | 0.95x |

### Run benchmarks

```bash
cd benchmarks

# Default: fp16, head_dim=128, SeqLen=[512, 1024, 2048, 4096]
python benchmark.py --dtype fp16 --head-dim 128

# With causal mask
python benchmark.py --dtype fp16 --head-dim 128 --causal

# Sweep all combinations (fp16/bf16 x causal x head_dim)
python benchmark.py --sweep
```

## Usage

### Fixed-length Attention

```python
import torch
from tiny_flash_attn import flash_attn_forward

# Q: [batch, seqQ,  numHeadsQ,  headDim]
# K: [batch, seqKV, numHeadsKV, headDim]
# V: [batch, seqKV, numHeadsKV, headDim]
Q = torch.randn(2, 1024, 32, 128, dtype=torch.float16, device="cuda")
K = torch.randn(2, 1024, 32, 128, dtype=torch.float16, device="cuda")
V = torch.randn(2, 1024, 32, 128, dtype=torch.float16, device="cuda")

O = flash_attn_forward(Q, K, V, is_causal=False)
# O: [batch, seqQ, numHeadsQ, headDim]
```

### Variable-length Attention

```python
from tiny_flash_attn import flash_attn_varlen_forward

# Q: [totalQ,  numHeadsQ,  headDim]  (packed sequences)
# K: [totalKV, numHeadsKV, headDim]
# V: [totalKV, numHeadsKV, headDim]
# cu_seqlens_q:  [batch + 1], int32, cumulative sequence lengths
# cu_seqlens_kv: [batch + 1], int32

O = flash_attn_varlen_forward(
    Q, K, V,
    cu_seqlens_q, cu_seqlens_kv,
    max_seqlen_q, max_seqlen_kv,
    is_causal=False
)
```

### GQA (Grouped-Query Attention)

```python
# GQA: numHeadsQ must be divisible by numHeadsKV
Q = torch.randn(2, 1024, 32, 128, dtype=torch.float16, device="cuda")  # 32 query heads
K = torch.randn(2, 1024,  8, 128, dtype=torch.float16, device="cuda")  #  8 KV heads (4 groups)
V = torch.randn(2, 1024,  8, 128, dtype=torch.float16, device="cuda")

O = flash_attn_forward(Q, K, V, is_causal=True)
```

### C++ API

```cpp
#include "flash_attn/flash_api.cuh"

// Fixed length
tfa::flashAttn<__half>(Q, K, V, O,
    batch, seqLenQ, seqLenKV,
    numHeadsQ, numHeadsKV, headDim,
    isCausal, stream);

// Variable length
tfa::flashAttnVarLen<__half>(Q, K, V, O,
    cu_seqlens_q, cu_seqlens_kv,
    batchSize, maxSeqLenQ, maxSeqLenKV,
    numHeadsQ, numHeadsKV, headDim,
    isCausal, stream);
```

## Building & Running Tests

```bash
git submodule update --init --recursive

mkdir build && cd build
cmake .. -DTFA_BUILD_TESTS=ON
make -j$(nproc)

# Run tests
ctest --test-dir . --output-on-failure
```

## Dependencies

- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) — CuTe sublibrary for tensor core abstractions

## Limitations

- Forward pass only
- No dropout

## License

This code is licensed under the MIT License (see [LICENSE](LICENSE)).
