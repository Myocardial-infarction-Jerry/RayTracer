#ifndef UTILS_H
#define UTILS_H

#include <curand_kernel.h>
#include <cuda_runtime.h>

#define IMAGE_WIDTH 1920
#define IMAGE_HEIGHT 1080
#define SAMPLE_PER_PIXEL 1000
#define RAY_DEPTH 50
#define RAND_SEED 1984

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

#endif