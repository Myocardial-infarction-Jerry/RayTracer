#ifndef PERLIN_H
#define PERLIN_H

#include "utils.h"
#include "vec3.h" 

class perlin {
public:
    __device__ perlin(curandState *randState) {
        curandState localRandState = *randState;
        randVec = new vec3[perlin::pointCount];
        for (int i = 0; i < perlin::pointCount; ++i)
            randVec[i] = vec3(RND * 2 - 1, RND * 2 - 1, RND * 2 - 1);

        permX = perlinGeneratePerm(localRandState);
        permY = perlinGeneratePerm(localRandState);
        permZ = perlinGeneratePerm(localRandState);
        *randState = localRandState;
    }

    __device__ ~perlin() {
        delete[] randVec;
        delete[] permX;
        delete[] permY;
        delete[] permZ;
    }

    __device__ float noise(const vec3 &p) const {
        auto u = p.x() - floor(p.x());
        auto v = p.y() - floor(p.y());
        auto w = p.z() - floor(p.z());
        auto i = static_cast<int>(floor(p.x()));
        auto j = static_cast<int>(floor(p.y()));
        auto k = static_cast<int>(floor(p.z()));
        vec3 c[2][2][2];

        for (int di = 0; di < 2; di++)
            for (int dj = 0; dj < 2; dj++)
                for (int dk = 0; dk < 2; dk++)
                    c[di][dj][dk] = randVec[
                        permX[(i + di) & 255] ^
                            permY[(j + dj) & 255] ^
                            permZ[(k + dk) & 255]
                    ];

        return perlin_interp(c, u, v, w);
    }

    __device__ float turb(const vec3 &p, const int &depth = 7) const {
        auto accum = 0.0;
        auto tempP = p;
        auto weight = 1.0;

        for (int i = 0; i < depth; i++) {
            accum += weight * noise(tempP);
            weight *= 0.5;
            tempP *= 2;
        }

        return fabs(accum);
    }

private:
    static const int pointCount = 256;
    vec3 *randVec;
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

    __device__ static float perlin_interp(vec3 c[2][2][2], const float &u, const float &v, const float &w) {
        auto uu = u * u * (3 - 2 * u);
        auto vv = v * v * (3 - 2 * v);
        auto ww = w * w * (3 - 2 * w);
        auto accum = 0.0;

        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                for (int k = 0; k < 2; k++) {
                    vec3 weight_v(u - i, v - j, w - k);
                    accum += (i * uu + (1 - i) * (1 - uu))
                        * (j * vv + (1 - j) * (1 - vv))
                        * (k * ww + (1 - k) * (1 - ww))
                        * dot(c[i][j][k], weight_v);
                }

        return accum;
    }
};

#endif