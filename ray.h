#ifndef RAY_H
#define RAY_H

#include "utils.h"
#include "vec3.h"

class ray {
public:
    __device__ ray() { T = 0; }
    __device__ ray(const vec3 &origin, const vec3 &direction, const float &time) { ori = origin, dir = direction, T = time; }
    __device__ vec3 origin() const { return ori; }
    __device__ vec3 direction() const { return dir; }
    __device__ float time() const { return T; }
    __device__ vec3 at(float t) const { return ori + t * dir; }

    // private:
    vec3 ori, dir;
    float T;
};

__device__ bool refract(const vec3 &v, const vec3 &n, float niOverNt, vec3 &refracted) {
    vec3 uv = v.unit();
    float dt = dot(uv, n);
    float discriminant = 1.0f - niOverNt * niOverNt * (1 - dt * dt);
    if (discriminant > 0) {
        refracted = niOverNt * (uv - n * dt) - n * sqrt(discriminant);
        return true;
    }
    else
        return false;
}

__device__ vec3 reflect(const vec3 &v, const vec3 &n) { return v - 2.0f * dot(v, n) * n; }

#endif