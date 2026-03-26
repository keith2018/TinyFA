cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(TinyFA_example LANGUAGES CXX CUDA)

if (NOT TARGET TinyFA::tinyfa)
    find_package(TinyFA REQUIRED)
endif ()

add_executable(tfa_example main.cu)
set_property(TARGET tfa_example PROPERTY CUDA_ARCHITECTURES native)
target_link_libraries(tfa_example PRIVATE TinyFA::tinyfa)
