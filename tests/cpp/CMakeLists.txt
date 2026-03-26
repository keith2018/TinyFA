set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../third-party)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory(${THIRD_PARTY_DIR}/googletest googletest)

file(GLOB test_cpp_src "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB test_cu_src "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

add_executable(TinyFA_test ${test_cpp_src} ${test_cu_src})

set_property(TARGET TinyFA_test PROPERTY CUDA_ARCHITECTURES native)

target_include_directories(TinyFA_test PRIVATE
        ${THIRD_PARTY_DIR}/googletest/googletest/include
        ${THIRD_PARTY_DIR}/googletest/googlemock/include
)

target_link_libraries(TinyFA_test
        PRIVATE
        TinyFA::tinyfa
        gtest
        gmock
        gtest_main
)

if (MSVC)
    set_source_files_properties(${test_cpp_src} PROPERTIES COMPILE_FLAGS "/utf-8")
endif ()

include(GoogleTest)
gtest_discover_tests(TinyFA_test WORKING_DIRECTORY $<TARGET_FILE_DIR:TinyFA_test>)
