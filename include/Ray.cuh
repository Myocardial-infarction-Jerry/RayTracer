#ifndef RAY_CUH
#define RAY_CUH

#include "Vec3.cuh"

class Ray {
public:
    Vec3 o, d;

    __device__ Ray() {}
    __device__ Ray(const Vec3 &origin, const Vec3 &direction);

    __device__ Vec3 at(const float &t) const;
};

// ! Ray implementation

__device__ Ray::Ray() : o(Vec3()), d(Vec3()) {}
__device__ Ray::Ray(const Vec3 &origin, const Vec3 &direction) : o(origin), d(direction) {}

__device__ Vec3 Ray::at(const float &t) const { return o + t * d; }

#endif

