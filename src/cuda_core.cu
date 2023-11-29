#include "cuda_core.h"

__global__ void cuda_kernel(point3 *devO, vec3 *devD, point3 *devC, double *devR, hitrec *devH, int N, int M) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= N || j >= M) return;

    vec3 oc = devO[i] - devC[j];
    double A = dot(devD[i], devD[i]);
    double B_ = dot(devD[i], oc);
    double C = oc.squared_length() - devR[j] * devR[j];
    if (B_ * B_ - A * C < 0) return;

    double t = (-B_ - std::sqrt(B_ * B_ - A * C)) / A;
}