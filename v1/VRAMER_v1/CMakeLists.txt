cmake_minimum_required(VERSION 3.24)
project(VRAMER VERSION 1.0.0 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Compile for the GPU in the build host by default. Override with e.g.
#   -DCMAKE_CUDA_ARCHITECTURES=86   (Ampere A2)
#   -DCMAKE_CUDA_ARCHITECTURES=120  (Blackwell RTX 50-series, CUDA >= 12.8)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

option(VRAMER_BUILD_TESTS "Build VRAMER tests" ON)
option(VRAMER_BUILD_SHARED "Also build libvramer.so (for ctypes/FFI bindings)" ON)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

add_library(vramer STATIC
  src/memory_source.cpp
  src/cuda_pager.cu
  src/vramer_c.cpp
)

target_include_directories(vramer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(vramer PUBLIC CUDA::cudart Threads::Threads)
target_compile_options(vramer PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>
)

# Shared library for ctypes / Rust FFI: static .a cannot be dlopen'd. Same
# sources, exports the C ABI (and C++) symbols. Produces libvramer.so.
if(VRAMER_BUILD_SHARED)
  add_library(vramer_shared SHARED
    src/memory_source.cpp
    src/cuda_pager.cu
    src/vramer_c.cpp
  )
  target_include_directories(vramer_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(vramer_shared PUBLIC CUDA::cudart Threads::Threads)
  set_target_properties(vramer_shared PROPERTIES OUTPUT_NAME vramer)
endif()

add_executable(vramer_cli src/main.cu)
set_target_properties(vramer_cli PROPERTIES OUTPUT_NAME vramer)
target_link_libraries(vramer_cli PRIVATE vramer CUDA::cudart)

if(VRAMER_BUILD_TESTS)
  enable_testing()
  add_executable(test_memory_source tests/test_memory_source.cpp)
  target_link_libraries(test_memory_source PRIVATE vramer)
  add_test(NAME memory_source COMMAND test_memory_source)

  add_executable(test_pager tests/test_pager.cu)
  target_link_libraries(test_pager PRIVATE vramer CUDA::cudart)
  add_test(NAME pager COMMAND test_pager)

  add_executable(test_writeback tests/test_writeback.cu)
  target_link_libraries(test_writeback PRIVATE vramer CUDA::cudart)
  add_test(NAME writeback COMMAND test_writeback)

  add_executable(test_pager_concurrency tests/test_pager_concurrency.cu)
  target_link_libraries(test_pager_concurrency PRIVATE vramer CUDA::cudart Threads::Threads)
  add_test(NAME pager_concurrency COMMAND test_pager_concurrency)

  add_executable(test_c_api tests/test_c_api.c)
  target_link_libraries(test_c_api PRIVATE vramer CUDA::cudart)
  set_target_properties(test_c_api PROPERTIES LINKER_LANGUAGE CXX)
  add_test(NAME c_api COMMAND test_c_api)
endif()
