#include <iostream>
#include <chrono>
#include <stdio.h>

#include "utils.h"
#include "material.h"
#include "sphere.h"
#include "camera.h"
#include "hittable_list.h"
#include "bvh.h"
#include "texture.h"

__global__ void randInit(curandState *randState) {
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    curand_init(RAND_SEED, 0, 0, randState);
}

__global__ void renderInit(int maxX, int maxY, curandState *randState) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= maxX || j >= maxY)
        return;

    int pixelIdx = j * maxX + i;
    curand_init(RAND_SEED + pixelIdx, 0, 0, &randState[pixelIdx]);
}

__device__ vec3 getColor(const ray &r, hittable **world, curandState *localRandState) {
    ray curRay = r;
    vec3 curAttenuation = vec3(1.0f, 1.0f, 1.0f);
    vec3 color = vec3(0.0f, 0.0f, 0.0f);
    for (int depth = 0; depth < RAY_DEPTH; ++depth) {
        hitRecord rec;
        if (!(*world)->hit(curRay, interval(0.001f, FLT_MAX), rec)) {
            // vec3 unitDirection = curRay.direction().unit();
            // float t = 0.5f * (unitDirection.y() + 1.0f);
            // vec3 c = (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
            // color += curAttenuation * c * 1;
            color += curAttenuation * ((hittable_list *)(*world))->background;
            return color;
        }

        ray scattered;
        vec3 attenuation = vec3(0.0f, 0.0f, 0.0f);
        vec3 colorFromEmission = rec.matPtr->emitted(0.0, 0.0, attenuation);
        color += colorFromEmission * curAttenuation;
        if (!rec.matPtr->scatter(curRay, rec, attenuation, scattered, localRandState))
            return color;

        curAttenuation *= attenuation;
        curRay = scattered;
    }

    return vec3(0.0f, 0.0f, 0.0f);
}

__global__ void render(vec3 *fb, int maxX, int maxY, int ns, camera **cam, hittable **world, curandState *randState) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= maxX || j >= maxY)
        return;

    int pixelIdx = j * maxX + i;
    curandState localRandState = randState[pixelIdx];
    vec3 color(0, 0, 0);
    for (int s = 0; s < ns; ++s) {
        ray r = (*cam)->getRay(float(i + RND) / float(maxX), float(j + RND) / float(maxY), &localRandState);
        color += getColor(r, world, &localRandState);
    }

    randState[pixelIdx] = localRandState;
    color /= float(ns);
    color[0] = sqrt(color[0]);
    color[1] = sqrt(color[1]);
    color[2] = sqrt(color[2]);
    fb[pixelIdx] = color;
}

__global__ void freeWorld(hittable **dWorld, camera **dCamera) {
    for (hittable *cur = (*dWorld)->nextObject; cur != nullptr; cur = cur->nextObject)
        delete cur;
    delete *dWorld;
    delete *dCamera;
}

__global__ void randomSphere(hittable **dWorld, camera **dCamera, int nx, int ny, curandState *randState) {
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    curandState localRandState = *randState;
    *dWorld = new hittable_list();
    ((hittable_list *)(*dWorld))->background = vec3(0.5f, 0.7f, 1.0f) * 0.2f;
    ((hittable_list *)(*dWorld))->add(new sphere(vec3(0.0f, -10000.0f, -1.0f), 10000.0f, new lambertian(new checkerTexture(0.32f, vec3(.2f, .3f, .1f), vec3(.9f, .9f, .9f)))));

    for (int i = -11; i < 11; ++i)
        for (int j = -11; j < 11; ++j) {
            float chooseMat = RND;
            vec3 center(i + RND, 0.2, j + RND);

            if (chooseMat < 0.8f)
                ((hittable_list *)(*dWorld))->add(new sphere(center, center + vec3(0, RND * 0.5, 0), 0.2, new lambertian(vec3(RND * RND, RND * RND, RND * RND))));
            else if (chooseMat < 0.95f)
                ((hittable_list *)(*dWorld))->add(new sphere(center, 0.2, new metal(vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND)));
            else
                ((hittable_list *)(*dWorld))->add(new sphere(center, 0.2, new dielectric(1.5)));
        }
    ((hittable_list *)(*dWorld))->add(new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5)));
    ((hittable_list *)(*dWorld))->add(new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1))));
    ((hittable_list *)(*dWorld))->add(new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0)));
    ((hittable_list *)(*dWorld))->add(new sphere(vec3(4, 8, 3), 3, new diffuseLight(vec3(1, .9, .6) * 10.0f)));
    *randState = localRandState;
    // (*dWorld) = (hittable_list *)(new bvhNode(*dWorld));

    vec3 lookFrom(0, 2, 14);
    vec3 lookAt(0, 0, 0);
    float focusLen = 10.0f;
    float aperture = 0.01f;
    *dCamera = new camera(
        lookFrom,
        lookAt,
        vec3(0, 1, 0),
        30.0,
        float(nx) / float(ny),
        aperture,
        focusLen
    );
}

__global__ void twoSphere(hittable **dWorld,camera **dCamera,int nx,int ny,curandState *randState){
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    curandState localRandState = *randState;
    *dWorld = new hittable_list();
    ((hittable_list *)(*dWorld))->background = vec3(0.5f, 0.7f, 1.0f);

    auto checker = new checkerTexture(0.8f, vec3(.2f, .3f, .1f), vec3(.9f, .9f, .9f));
    ((hittable_list *)(*dWorld))->add(new sphere(vec3(0, -10, 0), 10, new lambertian(checker)));
    ((hittable_list *)(*dWorld))->add(new sphere(vec3(0, 10, 0), 10, new lambertian(checker)));
    *randState = localRandState;

    vec3 lookFrom(13, 2, 3);
    vec3 lookAt(0, 0, 0);
    float focusLen = 10.0f;
    float aperture = 0.1f;
    *dCamera = new camera(
        lookFrom,
        lookAt,
        vec3(0, 1, 0),
        30.0,
        float(nx) / float(ny),
        aperture,
        focusLen
    );
}

int main(int argc, char const *argv[]) {
    int nx = IMAGE_WIDTH;
    int ny = IMAGE_HEIGHT;
    int ns = SAMPLE_PER_PIXEL;
    int tx = 16;
    int ty = 16;
    size_t stackSize = 2048;

    checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, stackSize));
    std::cerr << "CUDA Stack Size Limit: " << stackSize << " bytes\n";

    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int pixelNum = nx * ny;
    size_t fbSize = pixelNum * sizeof(vec3);

    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fbSize));

    curandState *dRandState;
    checkCudaErrors(cudaMalloc((void **)&dRandState, pixelNum * sizeof(curandState)));
    curandState *dRandState_;
    checkCudaErrors(cudaMalloc((void **)&dRandState_, 1 * sizeof(curandState)));

    randInit << <1, 1 >> > (dRandState_);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    hittable **dWorld;
    checkCudaErrors(cudaMalloc((void **)&dWorld, sizeof(hittable *)));
    camera **dCamera;
    checkCudaErrors(cudaMalloc((void **)&dCamera, sizeof(camera *)));
    switch (0) {
    case 0:
        randomSphere << <1, 1 >> > (dWorld, dCamera, nx, ny, dRandState_);
        break;
    case 1:
        twoSphere << <1, 1 >> > (dWorld, dCamera, nx, ny, dRandState_);
        break;
    }
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();

    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    renderInit << <blocks, threads >> > (nx, ny, dRandState);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render << <blocks, threads >> > (fb, nx, ny, ns, dCamera, dWorld, dRandState);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    std::cerr << "\ntook " << (float)(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) / 1000.0f << " s.\n";

    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixelIdx = j * nx + i;
            int ir = (int)(fmin(256, 255.99 * fb[pixelIdx].r()));
            int ig = (int)(fmin(256, 255.99 * fb[pixelIdx].g()));
            int ib = (int)(fmin(256, 255.99 * fb[pixelIdx].b()));
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    freeWorld << <1, 1 >> > (dWorld, dCamera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(dCamera));
    checkCudaErrors(cudaFree(dWorld));
    checkCudaErrors(cudaFree(dRandState));
    checkCudaErrors(cudaFree(dRandState_));
    checkCudaErrors(cudaFree(fb));
    cudaDeviceReset();

    return 0;
}
