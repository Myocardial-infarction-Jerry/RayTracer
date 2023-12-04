CUDA_PATH     ?= /usr/local/cuda
HOST_COMPILER  = g++
NVCC           = $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# select one of these for Debug vs. Release
NVCC_DBG       = -g -G
# NVCC_DBG       =

NVCCFLAGS      = $(NVCC_DBG) -m64
GENCODE_FLAGS  = -gencode arch=compute_86,code=sm_86

SRCS = main.cu
INCS = vec3.h ray.h hittable.h hittable_list.h sphere.h camera.h material.h utils.h interval.h aabb.h

RayTracer: RayTracer.o
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o RayTracer RayTracer.o

RayTracer.o: $(SRCS) $(INCS)
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o RayTracer.o -c main.cu

run: RayTracer
	rm -f out.ppm
	./RayTracer > out.ppm
	convert out.ppm out.png
	
debug: RayTracer
	lldb ./RayTracer

clean:
	rm -f RayTracer RayTracer.o out.ppm out.jpg
