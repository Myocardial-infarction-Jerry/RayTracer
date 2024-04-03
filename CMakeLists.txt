# 设置CMake最低版本
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# 设置项目名称和版本
project(RayTracer LANGUAGES CXX CUDA)

# 设置CUDA的版本
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

# 设置CUDA编译器为nvcc
set(CMAKE_CUDA_COMPILER nvcc)

# 添加OpenCV库
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 设置可执行文件的输出路径
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(include)

# 将src目录下的每个.cu文件添加到项目中
file(GLOB CUDA_FILES CONFIGURE_DEPENDS "src/*.cu")
foreach(file ${CUDA_FILES})
    get_filename_component(file_name ${file} NAME_WE)
    add_executable(${file_name} ${file})
    set_target_properties(${file_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    # 链接OpenCV库到每个目标
    target_link_libraries(${file_name} ${OpenCV_LIBS})
endforeach()