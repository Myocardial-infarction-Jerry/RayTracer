CUDA_PATH     ?= /usr/local/cuda
HOST_COMPILER  = g++
NVCC           = $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# select one of these for Debug vs. Release
NVCC_DBG       = -g -G
# NVCC_DBG       =

NVCCFLAGS      = $(NVCC_DBG) -m64 -std=c++17
GENCODE_FLAGS  = -gencode arch=compute_86,code=sm_86

SRCS = main.cu
INCS = vec3.h ray.h hittable.h hittable_list.h sphere.h camera.h material.h utils.h interval.h aabb.h bvh.h texture.h perlin.h quad.h constant_medium.h

RayTracer: main.o
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o RayTracer main.o

main.o: $(SRCS) $(INCS)
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o main.o -c main.cu

run: RayTracer
	rm -f image.ppm
	./RayTracer > image.ppm
	convert image.ppm image.png
	
debug: RayTracer
	lldb ./RayTracer

clean:
	rm -f RayTracer main.o image.ppm image.png
