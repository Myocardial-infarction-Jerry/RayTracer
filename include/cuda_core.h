#ifndef CUDA_CORE_H
#define CUDA_CORE_H

#include <cuda_runtime.h>
#include <cmath>

__global__ void cuda_kernel(double *c, double *r, double *o, double *d);

#endif