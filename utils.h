#ifndef UTILS_H
#define UTILS_H

#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <iostream>
#include <ctime>
#include <chrono>
#include <stdio.h>

#define IMAGE_WIDTH 800
#define IMAGE_HEIGHT 800
#define SAMPLE_PER_PIXEL 10000
#define RAY_DEPTH 40
#define RAND_SEED 1145141919

__device__ int *dSeed;

#define RANDVEC3 vec3(curand_uniform(localRandState), curand_uniform(localRandState), curand_uniform(localRandState))

#define FLT_MAX __FLT_MAX__

void checkCuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (!result)
        return;

    std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";

    cudaDeviceReset();
    exit(99);
}

#define checkCudaErrors(val) checkCuda((val), #val, __FILE__, __LINE__)
#define RND (curand_uniform(&localRandState))

#define WORLD ((hittable_list *)(*dWorld))

#include "material.h"
#include "sphere.h"
#include "camera.h"
#include "hittable_list.h"
#include "bvh.h"
#include "texture.h"
#include "quad.h"
#include "constant_medium.h"

#endif