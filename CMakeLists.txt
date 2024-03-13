cmake_minimum_required(VERSION 3.10)
project(RayTracer)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

include_directories(include ${OpenCV_INCLUDE_DIRS})

add_library(vec3 src/vec3.cpp)
add_library(render src/render.cpp) 
add_library(camera src/camera.cpp) 
add_library(entity src/entity.cpp) 
add_library(aabb src/aabb.cpp) 
add_library(fragment src/fragment.cpp) 
add_library(ray src/ray.cpp) 
add_library(scene src/scene.cpp) 

add_executable(RayTracer src/main.cpp)
target_link_libraries(RayTracer vec3 render camera entity fragment ray scene aabb ${OpenCV_LIBS})

# Enable debugging in Visual Studio Code
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

