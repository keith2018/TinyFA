import os
import subprocess
import torch
from setuptools import setup

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
except ImportError:
    raise ImportError(
        "PyTorch is required to build this package. "
        "Please install it first:\n"
        "  conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia\n"
        "  # or: pip install torch"
    )


def get_cuda_arch_flags():
    """Get CUDA architecture flags. Uses TORCH_CUDA_ARCH_LIST env var if set,
    otherwise auto-detects from the current GPU."""
    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
    if arch_list is not None:
        return []  # torch will handle it via env var

    # Auto-detect GPU architecture
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        caps = set()
        for line in result.stdout.strip().split("\n"):
            cap = line.strip()
            if cap:
                caps.add(cap)
        if caps:
            arch_str = ";".join(sorted(caps))
            os.environ["TORCH_CUDA_ARCH_LIST"] = arch_str
    except Exception:
        pass  # Let torch auto-detect

    return []


get_cuda_arch_flags()

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(ROOT_DIR)
CSRC_DIR = os.path.join(PROJECT_DIR, "csrc")
KERNEL_DIR = os.path.join(CSRC_DIR, "flash_attn", "kernels")

# Collect kernel instantiation sources
# Optional: set TFA_TARGET_SM env var to only compile kernels for a specific arch.
# e.g. TFA_TARGET_SM=sm80 pip install -e .
_target_sm = os.environ.get("TFA_TARGET_SM", "").strip().lower()

# Auto-detect TFA_TARGET_SM from native GPU if not explicitly set
if not _target_sm:
    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            if major == 7 and minor == 5:
                _target_sm = "sm75"
            elif major == 8 and minor == 0:
                _target_sm = "sm80"
            elif major > 8 or (major == 8 and minor > 0):
                _target_sm = "sm8x"
            if _target_sm:
                print(f"[TinyFA] Auto-detected GPU compute capability {major}.{minor} -> TFA_TARGET_SM={_target_sm}")
            else:
                print(
                    f"[TinyFA] GPU compute capability {major}.{minor} does not map to a known SM target. Compiling all kernels.")
    except Exception as e:
        print(f"[TinyFA] Could not auto-detect GPU architecture: {e}. Compiling all kernels.")

# Optional: set TFA_TARGET_DTYPE env var to only compile kernels for a specific dtype.
# e.g. TFA_TARGET_DTYPE=fp16 pip install -e .
_target_dtype = os.environ.get("TFA_TARGET_DTYPE", "").strip().lower()

kernel_sources = sorted([
    os.path.join(KERNEL_DIR, f)
    for f in os.listdir(KERNEL_DIR)
    if f.endswith(".cu") and (not _target_sm or f"_{_target_sm}.cu" in f)
])
if _target_sm and not kernel_sources:
    raise ValueError(
        f"TFA_TARGET_SM='{_target_sm}' matched no kernel files. "
        f"Valid values: sm75, sm80, sm8x"
    )
if _target_sm:
    print(f"[TinyFA] Filtering kernels for {_target_sm}: {len(kernel_sources)} files")
if _target_dtype:
    kernel_sources = [f for f in kernel_sources if f"_{_target_dtype}_" in os.path.basename(f)]
    if not kernel_sources:
        raise ValueError(
            f"TFA_TARGET_DTYPE='{_target_dtype}' matched no kernel files. "
            f"Valid values: fp16, bf16, fp32"
        )
    print(f"[TinyFA] Filtering kernels for dtype={_target_dtype}: {len(kernel_sources)} files")

# TFA_TARGET_HEADDIM_<N>: each defaults to ON, set env var to 0/OFF/FALSE to disable.
# e.g. TFA_TARGET_HEADDIM_32=0 TFA_TARGET_HEADDIM_96=0 pip install -e .
_all_headdims = [32, 64, 96, 128, 192, 256]
_enabled_headdims = []
for _hd in _all_headdims:
    _env_val = os.environ.get(f"TFA_TARGET_HEADDIM_{_hd}", "1").strip()
    if _env_val.upper() not in ("0", "OFF", "FALSE", "NO"):
        _enabled_headdims.append(_hd)

if not _enabled_headdims:
    raise ValueError(
        "All HeadDims are disabled. At least one TFA_TARGET_HEADDIM_<N> must be enabled."
    )

print("=" * 40)
print(f"[TinyFA] TFA_TARGET_SM      = '{_target_sm}'")
print(f"[TinyFA] TFA_TARGET_DTYPE   = '{_target_dtype}'")
print(f"[TinyFA] HeadDims enabled   = {_enabled_headdims}")
print("=" * 40)

_headdim_nvcc_flags = [f"-DTFA_TARGET_HEADDIM_{hd}=1" for hd in _enabled_headdims]

# TFA_DISPATCH_DTYPE
_dtype_nvcc_flags = []
if _target_dtype:
    _dtype_name_to_num = {"fp16": "1", "bf16": "2", "fp32": "3"}
    _dtype_num = _dtype_name_to_num.get(_target_dtype)
    if _dtype_num is None:
        raise ValueError(
            f"TFA_TARGET_DTYPE='{_target_dtype}' cannot be mapped. "
            f"Valid values: fp16, bf16, fp32"
        )
    _dtype_nvcc_flags = [f"-DTFA_TARGET_DTYPE={_dtype_num}"]

# TFA_TARGET_SM
_sm_nvcc_flags = []
if _target_sm:
    _sm_name_to_num = {"sm75": "75", "sm80": "80", "sm8x": "89"}
    _sm_num = _sm_name_to_num.get(_target_sm)
    if _sm_num is None:
        raise ValueError(
            f"TFA_TARGET_SM='{_target_sm}' cannot be mapped. "
            f"Valid values: sm75, sm80, sm8x"
        )
    _sm_nvcc_flags = [f"-DTFA_TARGET_SM={_sm_num}"]

_common_nvcc_flags = [
                         "-O3",
                         "-std=c++17",
                         "-use_fast_math",
                         "--expt-relaxed-constexpr",
                         "--threads", "4",
                     ] + _headdim_nvcc_flags + _dtype_nvcc_flags + _sm_nvcc_flags

_common_include_dirs = [
    CSRC_DIR,
    os.path.join(CSRC_DIR, "flash_attn"),
    os.path.join(PROJECT_DIR, "third-party", "cutlass", "include"),
]

_common_link_args = [
    f"-Wl,-rpath,{os.path.dirname(torch.__file__)}/lib"
]

ext_modules = []
packages = ["tiny_flash_attn"]
package_dir = {"tiny_flash_attn": "tiny_flash_attn"}

ext_modules.append(
    CUDAExtension(
        name="tiny_flash_attn._C",
        sources=[
                    os.path.relpath(os.path.join(CSRC_DIR, "binding", "flash_attn_binding.cu"), ROOT_DIR),
                ] + [os.path.relpath(f, ROOT_DIR) for f in kernel_sources],
        include_dirs=_common_include_dirs,
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": _common_nvcc_flags,
        },
        extra_link_args=_common_link_args,
    ),
)

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    packages=packages,
    package_dir=package_dir,
)
