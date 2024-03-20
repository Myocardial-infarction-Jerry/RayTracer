cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cuda_project LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

include_directories(include)

file(GLOB SOURCES "src/*.cu")

add_executable(RayTracer ${SOURCES})

set_property(TARGET RayTracer PROPERTY CUDA_SEPARABLE_COMPILATION ON)