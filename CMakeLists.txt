cmake_minimum_required(VERSION 3.18)
project(convolution2d LANGUAGES CXX) # Replace 'my_project' with the name of your project

option(USE_CUDA "Enable CUDA support" OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)
if(COMPILER_SUPPORTS_AVX512F)
    set(SIMD_FLAGS "-mavx512f")
    message(STATUS "Using AVX-512 instructions")
    add_definitions(-DUSE_AVX512)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SIMD_FLAGS} -Wall -Wextra")
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Ofast -O3 -std=c++20 -march=native -fpic -ftree-vectorize")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60")

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  message(STATUS "Linking OpenMP")
  link_libraries(OpenMP::OpenMP_CXX)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/nanobind)

set(CPP_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cpp/conv2d.cpp)

if(USE_CUDA)
  find_package(CUDA)
  include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
  )
  list(APPEND SOURCES cpp/conv2d_cudnn.cpp)
  add_definitions(-DUSE_CUDA)
  set(CUDA_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cpp/conv2d_cudnn.cpp)
  link_libraries(${CUDA_LIBRARIES})
endif()

nanobind_add_module(conv2d ${CPP_SRC_FILES} ${CUDA_SRC_FILES})

