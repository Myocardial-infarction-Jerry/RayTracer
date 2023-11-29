#include <cuda_runtime.h>
#include <stdio.h>
#include <bits/stdc++.h>

__global__ void cuda_kernel(double *c, double *r, double *o, double *d, double *d_, int N, int M) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= N || j >= M) return;

    double co[3] = { o[j * 3 + 0] - c[i * 3 + 0],o[j * 3 + 1] - c[i * 3 + 1],o[j * 3 + 2] - c[i * 3 + 2] };

    double A = d[j * 3 + 0] * d[j * 3 + 0] + d[j * 3 + 1] * d[j * 3 + 1] + d[j * 3 + 2] * d[j * 3 + 2];
    double B_ = d[j * 3 + 0] * co[0] + d[j * 3 + 1] * co[1] + d[j * 3 + 2] * co[2];
    double C = co[0] * co[0] + co[1] * co[1] + co[2] * co[2] - r[i] * r[i];
    double t = (-B_ - std::sqrt(B_ * B_ - A * C)) / A;

    if (B_ * B_ < A * C || t < 0)
        return;

    double n[3] = { o[j * 3 + 0] + d[j * 3 + 0] * t - c[i * 3 + 0],o[j * 3 + 1] + d[j * 3 + 1] * t - c[i * 3 + 1],o[j * 3 + 2] + d[j * 3 + 2] * t - c[i * 3 + 2] };
    n[0] /= r[i], n[1] /= r[i], n[2] /= r[i];
    double len = n[0] * d[j * 3 + 0] + n[1] * d[j * 3 + 1] + n[2] * d[j * 3 + 2];
    d_[(i * N + j) * 3 + 0] = d[j * 3 + 0] * (1 + 2 * len);
    d_[(i * N + j) * 3 + 1] = d[j * 3 + 1] * (1 + 2 * len);
    d_[(i * N + j) * 3 + 2] = d[j * 3 + 2] * (1 + 2 * len);
}

int main(int argc, char const *argv[]) {
    //CUDA Compute
    double c[2 * 3] = { 1,0,0,-1,0,0 };
    double r[2] = { 1,1 };
    double o[2 * 3] = { 0,1,0,0,1,0 };
    double d[2 * 3] = { 1,-1,0,-1,-1,0 };
    double d_[2 * 2 * 3];

    double *d_c; cudaMalloc(&d_c, sizeof(double) * 2 * 3);
    double *d_r; cudaMalloc(&d_r, sizeof(double) * 2);
    double *d_o; cudaMalloc(&d_o, sizeof(double) * 2 * 3);
    double *d_d; cudaMalloc(&d_d, sizeof(double) * 2 * 3);
    double *d_d_; cudaMalloc(&d_d_, sizeof(double) * 2 * 2 * 3);

    cudaMemcpy(d_c, c, sizeof(double) * 2 * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, r, sizeof(double) * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_o, o, sizeof(double) * 2 * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, d, sizeof(double) * 2 * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_d_, d_, sizeof(double) * 2 * 2 * 3, cudaMemcpyHostToDevice);

    int N = 16;
    dim3 threads(N, N);
    dim3 blocks(1, 1);
    cuda_kernel << <blocks, threads >> > (d_c, d_r, d_o, d_d, d_d_, 2, 2);
    cudaDeviceSynchronize();

    cudaMemcpy(o, d_o, sizeof(double) * 2 * 3, cudaMemcpyDeviceToHost);
    cudaMemcpy(r, d_r, sizeof(double) * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(c, d_c, sizeof(double) * 2 * 3, cudaMemcpyDeviceToHost);
    cudaMemcpy(d, d_d, sizeof(double) * 2 * 3, cudaMemcpyDeviceToHost);
    cudaMemcpy(d_, d_d_, sizeof(double) * 2 * 2 * 3, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            printf("%lf %lf %lf\n", d_[(i * 2 + j) * 3 + 0], d_[(i * 2 + j) * 3 + 1], d_[(i * 2 + j) * 3 + 2]);

    return 0;
}
