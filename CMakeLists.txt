# 设置CMake最低版本要求
cmake_minimum_required(VERSION 3.18)

# 设置项目名称
project(RayTracer)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)

# 启用CUDA语言
# enable_language(CUDA)

# 如果你的CUDA版本不是默认的版本，你可以通过以下方式设置
# set(CMAKE_CUDA_COMPILER /usr/local/cuda-8.6/bin/nvcc)

# 设置CUDA标准
# set(CMAKE_CUDA_STANDARD 14)
# set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

# 添加子目录
add_subdirectory(src)