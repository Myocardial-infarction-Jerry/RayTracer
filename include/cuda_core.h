#ifndef CUDA_CORE_H
#define CUDA_CORE_H

#include <cuda_runtime.h>

#include "vec3.h"
#include "volume.h"

__global__ void cuda_kernel(point3 *devO, vec3 *devD, point3 *devC, double *devR, int N, int M);

#endif