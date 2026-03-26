#!/usr/bin/env python3
"""
Benchmark: TinyFA vs Official Flash Attention / PyTorch SDPA

Usage:
    python benchmarks/benchmark.py
    python benchmarks/benchmark.py --causal
    python benchmarks/benchmark.py --head-dim 64 --dtype bf16
    python benchmarks/benchmark.py --sweep  # all dtype x causal x head-dim
"""

import itertools
import torch
import torch.nn.functional as F

# disable pytorch TF32
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# config
BATCH = 2
SEQLENS = [512, 1024, 2048, 4096, 8192]
NHEADS = 32
HEADDIM = 128
DTYPE = torch.float16
CAUSAL = False
WARMUP = 20
REPEATS = 40


def flops_fwd(batch, seqlen, nheads, headdim, causal=False):
    f = 4 * batch * nheads * seqlen * seqlen * headdim
    return f // 2 if causal else f


def benchmark_fn(fn, warmup=WARMUP, repeats=REPEATS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def run_tinyfa(Q, K, V, causal):
    import tiny_flash_attn
    return tiny_flash_attn.flash_attn_forward(Q, K, V, is_causal=causal)


def run_flash_attn_official(Q, K, V, causal):
    from flash_attn import flash_attn_func
    return flash_attn_func(Q, K, V, causal=causal)


def run_torch_sdpa(Q, K, V, causal):
    # SDPA: [batch, heads, seq, dim]
    q, k, v = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal).transpose(1, 2)


def check_correctness(Q, K, V, causal, tinyfa_fn, ref_name, ref_fn):
    torch.manual_seed(42)
    ref = ref_fn(Q, K, V, causal).float()
    out = tinyfa_fn(Q, K, V, causal).float()

    diff = (out - ref).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    cos_sim = F.cosine_similarity(
        out.reshape(-1).unsqueeze(0), ref.reshape(-1).unsqueeze(0)
    ).item()

    status = "PASS" if max_err < 0.05 else "WARN"
    print(f"  [{status}] tinyfa vs {ref_name}: "
          f"max_err={max_err:.6f}, mean_err={mean_err:.6f}, cos_sim={cos_sim:.6f}")


def run_benchmark(seqlens, nheads, headdim, dtype, causal, baseline_name, baseline_fn):
    dtype_str = {torch.float16: "fp16", torch.bfloat16: "bf16", torch.float32: "fp32"}[dtype]
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"

    print(f"\n{'=' * 100}")
    print(f"  GPU: {gpu}")
    print(f"  Config: batch={BATCH}, heads={nheads}, dim={headdim}, "
          f"dtype={dtype_str}, causal={causal}")
    print(f"  Baseline: {baseline_name}")
    print(f"{'=' * 100}")

    header = (f"{'SeqLen':>8} | {'TinyFA (ms)':>12} | {'TinyFA (TFLOPS)':>16} "
              f"| {baseline_name + ' (ms)':>16} | {baseline_name + ' (TFLOPS)':>16} | {'Speedup':>8}")
    print(header)
    print("-" * len(header))

    results = []
    for seqlen in seqlens:
        Q = torch.randn(BATCH, seqlen, nheads, headdim, device="cuda", dtype=dtype)
        K = torch.randn(BATCH, seqlen, nheads, headdim, device="cuda", dtype=dtype)
        V = torch.randn(BATCH, seqlen, nheads, headdim, device="cuda", dtype=dtype)

        total_flops = flops_fwd(BATCH, seqlen, nheads, headdim, causal)

        # TinyFA
        tflops_tiny = None
        try:
            ms_tiny = benchmark_fn(lambda: run_tinyfa(Q, K, V, causal))
            tflops_tiny = total_flops / ms_tiny / 1e9
            tiny_str = f"{ms_tiny:>12.3f} | {tflops_tiny:>16.2f}"
        except Exception as e:
            ms_tiny = None
            tiny_str = f"{'ERROR':>12} | {'N/A':>16}"
            if seqlen == seqlens[0]:
                print(f"  [!] TinyFA failed: {e}")

        # Baseline
        tflops_base = None
        try:
            ms_base = benchmark_fn(lambda: baseline_fn(Q, K, V, causal))
            tflops_base = total_flops / ms_base / 1e9
            base_str = f"{ms_base:>16.3f} | {tflops_base:>16.2f}"
        except Exception as e:
            ms_base = None
            base_str = f"{'ERROR':>16} | {'N/A':>16}"
            if seqlen == seqlens[0]:
                print(f"  [!] {baseline_name} failed: {e}")

        # Speedup
        if ms_tiny and ms_base:
            speedup = ms_base / ms_tiny
            sp_str = f"{speedup:>7.2f}x"
        else:
            speedup = None
            sp_str = f"{'N/A':>8}"

        print(f"{seqlen:>8} | {tiny_str} | {base_str} | {sp_str}")

        results.append({
            "seqlen": seqlen,
            "ms_tiny": ms_tiny,
            "ms_base": ms_base,
            "tflops_tiny": tflops_tiny,
            "tflops_base": tflops_base,
            "speedup": speedup,
        })

        del Q, K, V
        torch.cuda.empty_cache()

    print()
    return results


def pick_baseline(dtype, has_flash_attn):
    if dtype != torch.float32 and has_flash_attn:
        return "flash_attn", run_flash_attn_official
    return "torch_sdpa", run_torch_sdpa


def run_single(dtype, headdim, causal, has_flash_attn):
    baseline_name, baseline_fn = pick_baseline(dtype, has_flash_attn)
    print(f"[Baseline] {baseline_name}")

    check_seqlen = min(SEQLENS[0], 512)
    Q = torch.randn(BATCH, check_seqlen, NHEADS, headdim, device="cuda", dtype=dtype)
    K = torch.randn(BATCH, check_seqlen, NHEADS, headdim, device="cuda", dtype=dtype)
    V = torch.randn(BATCH, check_seqlen, NHEADS, headdim, device="cuda", dtype=dtype)
    print(f"\nCorrectness (seq={check_seqlen}, dim={headdim}):")
    check_correctness(Q, K, V, causal, run_tinyfa, baseline_name, baseline_fn)
    del Q, K, V

    run_benchmark(SEQLENS, NHEADS, headdim, dtype, causal, baseline_name, baseline_fn)


# sweep

ALL_DTYPES = [("fp16", torch.float16), ("bf16", torch.bfloat16)]
ALL_CAUSALS = [False, True]
ALL_HEADDIMS = [64, 128, 256]


def run_sweep(has_flash_attn):
    combos = list(itertools.product(ALL_DTYPES, ALL_CAUSALS, ALL_HEADDIMS))
    total = len(combos)

    print(f"\n{'#' * 100}")
    print(f"  SWEEP: {total} configurations  "
          f"(dtypes={[d[0] for d in ALL_DTYPES]}, causal={ALL_CAUSALS}, head_dims={ALL_HEADDIMS})")
    print(f"{'#' * 100}")

    all_sweep_results = []

    for idx, ((dtype_str, dtype), causal, headdim) in enumerate(combos, 1):
        print(f"\n>>> [{idx}/{total}] dtype={dtype_str}, causal={causal}, head_dim={headdim}")

        baseline_name, baseline_fn = pick_baseline(dtype, has_flash_attn)
        results = run_benchmark(SEQLENS, NHEADS, headdim, dtype, causal, baseline_name, baseline_fn)
        all_sweep_results.append({
            "dtype": dtype_str,
            "causal": causal,
            "headdim": headdim,
            "baseline": baseline_name,
            "results": results,
        })

    _print_sweep_summary(all_sweep_results)


def _print_sweep_summary(all_sweep_results):
    print(f"\n{'#' * 120}")
    print(f"{'SWEEP SUMMARY':^120}")
    print(f"{'#' * 120}")

    header = (f"{'Dtype':>6} | {'Causal':>6} | {'HeadDim':>7} | {'Baseline':>12} "
              f"| {'Avg TinyFA (TFLOPS)':>20} | {'Avg Baseline (TFLOPS)':>22} "
              f"| {'Avg Speedup':>12} | {'Min Speedup':>12} | {'Max Speedup':>12}")
    print(header)
    print("-" * len(header))

    for entry in all_sweep_results:
        dtype_str = entry["dtype"]
        causal = entry["causal"]
        headdim = entry["headdim"]
        baseline = entry["baseline"]
        results = entry["results"]

        tflops_tiny_list = [r["tflops_tiny"] for r in results if r["tflops_tiny"] is not None]
        tflops_base_list = [r["tflops_base"] for r in results if r["tflops_base"] is not None]
        speedup_list = [r["speedup"] for r in results if r["speedup"] is not None]

        avg_tflops_tiny = sum(tflops_tiny_list) / len(tflops_tiny_list) if tflops_tiny_list else None
        avg_tflops_base = sum(tflops_base_list) / len(tflops_base_list) if tflops_base_list else None
        avg_speedup = sum(speedup_list) / len(speedup_list) if speedup_list else None
        min_speedup = min(speedup_list) if speedup_list else None
        max_speedup = max(speedup_list) if speedup_list else None

        tiny_str = f"{avg_tflops_tiny:>20.2f}" if avg_tflops_tiny is not None else f"{'N/A':>20}"
        base_str = f"{avg_tflops_base:>22.2f}" if avg_tflops_base is not None else f"{'N/A':>22}"
        avg_sp_str = f"{avg_speedup:>11.2f}x" if avg_speedup is not None else f"{'N/A':>12}"
        min_sp_str = f"{min_speedup:>11.2f}x" if min_speedup is not None else f"{'N/A':>12}"
        max_sp_str = f"{max_speedup:>11.2f}x" if max_speedup is not None else f"{'N/A':>12}"

        print(f"{dtype_str:>6} | {str(causal):>6} | {headdim:>7} | {baseline:>12} "
              f"| {tiny_str} | {base_str} "
              f"| {avg_sp_str} | {min_sp_str} | {max_sp_str}")

    print("-" * len(header))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark TinyFA")
    parser.add_argument("--head-dim", type=int, default=HEADDIM)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--sweep", action="store_true",
                        help="Run all dtype x causal x head-dim combos")
    args = parser.parse_args()

    # check TinyFA
    try:
        import tiny_flash_attn
        print("[OK] TinyFA loaded")
    except ImportError:
        print("[FAIL] TinyFA not installed. Run: cd python && pip install -e .")
        return

    # check official flash-attn
    has_flash_attn = False
    try:
        from flash_attn import flash_attn_func
        has_flash_attn = True
        print("[OK] Official flash-attn available")
    except ImportError:
        print("[INFO] Official flash-attn not found, will use PyTorch SDPA as baseline")

    if args.sweep:
        run_sweep(has_flash_attn)
    else:
        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        run_single(dtype_map[args.dtype], args.head_dim, args.causal, has_flash_attn)


if __name__ == "__main__":
    main()
