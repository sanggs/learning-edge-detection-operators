cmake_minimum_required(VERSION 3.18)
project(convolution2d) # Replace 'my_project' with the name of your project

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)
find_package(OpenMP)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/nanobind)

nanobind_add_module(conv2d ${CMAKE_CURRENT_SOURCE_DIR}/cpp/conv2d_vanilla.cpp)

if(OpenMP_CXX_FOUND)
  message(STATUS "Linking OpenMP")
  link_libraries(OpenMP::OpenMP_CXX)
endif()

