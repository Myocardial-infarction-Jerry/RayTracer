#ifndef PERLIN_H
#define PERLIN_H

#include "utils.h"
#include "vec3.h" 

class perlin {
public:
    __device__ perlin() {
        curandState localRandState;
        curand_init(RAND_SEED, 10, 0, &localRandState);
        randFloat = new float[perlin::pointCount];
        for (int i = 0; i < perlin::pointCount; ++i)
            randFloat[i] = RND;

        permX = perlinGeneratePerm(localRandState);
        permY = perlinGeneratePerm(localRandState);
        permZ = perlinGeneratePerm(localRandState);
    }

    __device__ ~perlin() {
        delete[] randFloat;
        delete[] permX;
        delete[] permY;
        delete[] permZ;
    }

    __device__ float noise(const vec3 &p) const {
        auto i = static_cast<int>(4 * p.x()) & 255;
        auto j = static_cast<int>(4 * p.y()) & 255;
        auto k = static_cast<int>(4 * p.z()) & 255;

        return randFloat[permX[i] ^ permY[j] ^ permZ[k]];
    }

private:
    static const int pointCount = 256;
    float *randFloat;
    int *permX, *permY, *permZ;

    __device__ static int *perlinGeneratePerm(curandState &localRandState) {
        auto p = new int[perlin::pointCount];

        for (int i = 0; i < perlin::pointCount; ++i)
            p[i] = i;

        permute(p, perlin::pointCount, localRandState);

        return p;
    }

    __device__ static void permute(int *p, int n, curandState &localRandState) {
        for (int i = n - 1; i >= 0; --i) {
            int target = static_cast<int>(RND * (i + 1));
            int tmp = p[i];
            p[i] = p[target];
            p[target] = tmp;
        }
    }
};

#endif