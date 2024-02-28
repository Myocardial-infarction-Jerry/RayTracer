# 创建一个变量，包含所有的源文件
file(GLOB SOURCE_FILES *.cpp *.cu)

# 添加一个可执行文件
add_executable(RayTracer ${SOURCE_FILES})

# 设置目标属性以使用CUDA
set_property(TARGET RayTracer PROPERTY CUDA_ARCHITECTURES 86)

# 设置可执行文件的输出路径
set_target_properties(RayTracer PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})