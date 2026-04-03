cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(TinyFA VERSION 1.0.0 LANGUAGES CXX CUDA)

option(TFA_BUILD_TESTS "Build tests" ON)
option(TFA_BUILD_EXAMPLES "Build examples" OFF)

set(TFA_CUDA_ARCHITECTURES "native" CACHE STRING
        "CUDA architectures to compile for (e.g. '80;89' or 'native')")

set(TFA_TARGET_SM "" CACHE STRING
        "Only compile kernels for this SM arch (e.g. sm80). Empty = auto-detect from GPU. Valid: sm75, sm80, sm8x")

# Auto-detect TFA_TARGET_SM from native GPU if not explicitly set
if (NOT TFA_TARGET_SM)
    # nvidia-smi
    execute_process(
            COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE _gpu_compute_cap
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE _nvsmi_result
    )
    if (NOT _nvsmi_result EQUAL 0)
        set(_gpu_compute_cap "")
    endif ()

    # compile and run a tiny CUDA program
    if (NOT _gpu_compute_cap)
        set(_detect_src "${CMAKE_BINARY_DIR}/_detect_cuda_arch.cu")
        file(WRITE "${_detect_src}"
                "#include <stdio.h>\n#include <cuda_runtime.h>\nint main(){cudaDeviceProp p;if(cudaGetDeviceProperties(&p,0)==cudaSuccess)printf(\"%d.%d\",p.major,p.minor);return 0;}\n")
        execute_process(
                COMMAND "${CMAKE_CUDA_COMPILER}" -o "${CMAKE_BINARY_DIR}/_detect_cuda_arch" "${_detect_src}" -lcudart
                RESULT_VARIABLE _compile_result
                ERROR_QUIET
        )
        if (_compile_result EQUAL 0)
            execute_process(
                    COMMAND "${CMAKE_BINARY_DIR}/_detect_cuda_arch"
                    OUTPUT_VARIABLE _gpu_compute_cap
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET
                    RESULT_VARIABLE _run_result
            )
            if (NOT _run_result EQUAL 0)
                set(_gpu_compute_cap "")
            endif ()
        endif ()
    endif ()

    if (_gpu_compute_cap)
        # first GPU
        string(REGEX MATCH "([0-9]+)\\.([0-9]+)" _cap_match "${_gpu_compute_cap}")
        set(_cap_major "${CMAKE_MATCH_1}")
        set(_cap_minor "${CMAKE_MATCH_2}")
        if (_cap_major STREQUAL "7" AND _cap_minor STREQUAL "5")
            set(TFA_TARGET_SM "sm75")
        elseif (_cap_major STREQUAL "8" AND _cap_minor STREQUAL "0")
            set(TFA_TARGET_SM "sm80")
        elseif (_cap_major GREATER_EQUAL 8)
            set(TFA_TARGET_SM "sm8x")
        endif ()
        if (TFA_TARGET_SM)
            message(STATUS "[TinyFA] Auto-detected GPU compute capability ${_cap_major}.${_cap_minor} -> TFA_TARGET_SM=${TFA_TARGET_SM}")
        else ()
            message(WARNING "[TinyFA] GPU compute capability ${_cap_major}.${_cap_minor} does not map to a known SM target (sm75, sm80, sm8x). Compiling all kernels.")
        endif ()
    else ()
        message(STATUS "[TinyFA] Could not auto-detect GPU architecture. Compiling all kernels.")
    endif ()
endif ()
set(TFA_TARGET_DTYPE "" CACHE STRING
        "Only compile kernels for this dtype (e.g. fp16). Empty = all. Valid: fp16, bf16, fp32")
option(TFA_TARGET_HEADDIM_32  "Compile HeadDim=32 kernels"  ON)
option(TFA_TARGET_HEADDIM_64  "Compile HeadDim=64 kernels"  ON)
option(TFA_TARGET_HEADDIM_96  "Compile HeadDim=96 kernels"  ON)
option(TFA_TARGET_HEADDIM_128 "Compile HeadDim=128 kernels" ON)
option(TFA_TARGET_HEADDIM_192 "Compile HeadDim=192 kernels" ON)
option(TFA_TARGET_HEADDIM_256 "Compile HeadDim=256 kernels" ON)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)

set(TFA_CSRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/csrc)
set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third-party)

include(${TFA_CSRC_DIR}/flash_attn/kernels/kernel_sources.cmake)

message(STATUS "========================================")
message(STATUS "[TinyFA] TFA_TARGET_SM      = '${TFA_TARGET_SM}'")
message(STATUS "[TinyFA] TFA_TARGET_DTYPE   = '${TFA_TARGET_DTYPE}'")
set(_enabled_headdims "")
foreach(_hd 32 64 96 128 192 256)
    if (TFA_TARGET_HEADDIM_${_hd})
        list(APPEND _enabled_headdims ${_hd})
    endif ()
endforeach()
message(STATUS "[TinyFA] HeadDims enabled   = '${_enabled_headdims}'")
if (NOT _enabled_headdims)
    message(FATAL_ERROR "All HeadDims are disabled. At least one TFA_TARGET_HEADDIM_<N> must be ON.")
endif ()
message(STATUS "========================================")

# TFA_TARGET_SM
if (TFA_TARGET_SM)
    string(TOLOWER "${TFA_TARGET_SM}" _tfa_sm_lower)
    set(_filtered "")
    foreach (_src ${TFA_KERNEL_SOURCES})
        if (_src MATCHES "_${_tfa_sm_lower}\\.cu$")
            list(APPEND _filtered ${_src})
        endif ()
    endforeach ()
    if (NOT _filtered)
        message(FATAL_ERROR "TFA_TARGET_SM='${TFA_TARGET_SM}' matched no kernel files. "
                "Valid: sm75, sm80, sm8x")
    endif ()
    message(STATUS "[TinyFA] Filtering kernels for SM=${TFA_TARGET_SM}")
    set(TFA_KERNEL_SOURCES ${_filtered})
endif ()

# TFA_TARGET_DTYPE
if (TFA_TARGET_DTYPE)
    string(TOLOWER "${TFA_TARGET_DTYPE}" _tfa_dtype_lower)
    set(_filtered "")
    foreach (_src ${TFA_KERNEL_SOURCES})
        if (_src MATCHES "_${_tfa_dtype_lower}_")
            list(APPEND _filtered ${_src})
        endif ()
    endforeach ()
    if (NOT _filtered)
        message(FATAL_ERROR "TFA_TARGET_DTYPE='${TFA_TARGET_DTYPE}' matched no kernel files. "
                "Valid: fp16, bf16, fp32")
    endif ()
    message(STATUS "[TinyFA] Filtering kernels for DTYPE=${TFA_TARGET_DTYPE}")
    set(TFA_KERNEL_SOURCES ${_filtered})
endif ()

message(STATUS "[TinyFA] Compiling ${CMAKE_CURRENT_LIST_DIR}: ${TFA_KERNEL_SOURCES}")


add_library(tinyfa STATIC ${TFA_KERNEL_SOURCES})
add_library(TinyFA::tinyfa ALIAS tinyfa)

target_compile_options(tinyfa PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 -use_fast_math --expt-relaxed-constexpr>
)

if (TFA_CUDA_ARCHITECTURES STREQUAL "native")
    set_property(TARGET tinyfa PROPERTY CUDA_ARCHITECTURES native)
else ()
    set_property(TARGET tinyfa PROPERTY CUDA_ARCHITECTURES ${TFA_CUDA_ARCHITECTURES})
endif ()

# TFA_TARGET_HEADDIM_<N>
foreach(_hd 32 64 96 128 192 256)
    if (TFA_TARGET_HEADDIM_${_hd})
        target_compile_definitions(tinyfa PUBLIC TFA_TARGET_HEADDIM_${_hd}=1)
    endif ()
endforeach()

if (TFA_TARGET_DTYPE)
    string(TOLOWER "${TFA_TARGET_DTYPE}" _tfa_dtype_lower2)
    if (_tfa_dtype_lower2 STREQUAL "fp16")
        set(_tfa_dtype_num 1)
    elseif (_tfa_dtype_lower2 STREQUAL "bf16")
        set(_tfa_dtype_num 2)
    elseif (_tfa_dtype_lower2 STREQUAL "fp32")
        set(_tfa_dtype_num 3)
    endif ()
    target_compile_definitions(tinyfa PUBLIC TFA_TARGET_DTYPE=${_tfa_dtype_num})
endif ()

if (TFA_TARGET_SM)
    string(TOLOWER "${TFA_TARGET_SM}" _tfa_sm_lower2)
    if (_tfa_sm_lower2 STREQUAL "sm75")
        set(_tfa_sm_num 75)
    elseif (_tfa_sm_lower2 STREQUAL "sm80")
        set(_tfa_sm_num 80)
    elseif (_tfa_sm_lower2 STREQUAL "sm8x")
        set(_tfa_sm_num 89)
    endif ()
    target_compile_definitions(tinyfa PUBLIC TFA_TARGET_SM=${_tfa_sm_num})
endif ()

target_include_directories(tinyfa
        PUBLIC
        ${TFA_CSRC_DIR}
        PRIVATE
        ${THIRD_PARTY_DIR}/cutlass/include
)

target_link_libraries(tinyfa PUBLIC CUDA::cudart)


if (TFA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/cpp)
endif ()

if (TFA_BUILD_EXAMPLES)
    add_subdirectory(examples/basic)
endif ()
