#include "cuda_core.h"

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