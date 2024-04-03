#ifndef RENDER_CUH
#define RENDER_CUH

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "Camera.cuh"
#include "Scene.cuh"
#include "Vec3.cuh"

namespace Render {
    __host__ void render(const Camera &camera, const Scene &scene, const int &width, const int &height, const int &samples, Vec3 *pixels);
    __global__ void renderKernel(Camera *camera, Scene *scene, int width, int height, int samples, Vec3 *pixels);
    __host__ void saveImage(const char *filename, const int &width, const int &height, const Vec3 *pixels);
};

__host__ void Render::render(const Camera &camera, const Scene &scene, const int &width, const int &height, const int &samples, Vec3 *pixels) {
    Camera *d_camera;
    Scene *d_scene;
    Vec3 *d_pixels;

    cudaMalloc(&d_camera, sizeof(Camera));
    cudaMalloc(&d_scene, sizeof(Scene));
    cudaMalloc(&d_pixels, width * height * sizeof(Vec3));

    cudaMemcpy(d_camera, &camera, sizeof(Camera), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scene, &scene, sizeof(Scene), cudaMemcpyHostToDevice);

    dim3 blocks(width / 8, height / 8);
    dim3 threads(8, 8);

    Render::renderKernel << <blocks, threads >> > (d_camera, d_scene, width, height, samples, d_pixels);

    cudaMemcpy(pixels, d_pixels, width * height * sizeof(Vec3), cudaMemcpyDeviceToHost);
}

__global__ void Render::renderKernel(Camera *camera, Scene *scene, int width, int height, int samples, Vec3 *pixels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    pixels[y * width + x] = Vec3(x, y, 0).normalize();
}

__host__ void Render::saveImage(const char *filename, const int &width, const int &height, const Vec3 *pixels) {
    // Save image using OpenCV
    cv::Mat image(height, width, CV_8UC3);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            Vec3 pixel = pixels[y * width + x];
            cv::Vec3b color(pixel[0] * 255, pixel[1] * 255, pixel[2] * 255);
            image.at<cv::Vec3b>(y, x) = color;
        }
    }

    cv::imwrite(filename, image);
}

#endif