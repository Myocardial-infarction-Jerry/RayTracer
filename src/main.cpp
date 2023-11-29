#include "utils.h"
#include "cuda_core.h"

extern std::ofstream fp;

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