#ifndef AABB_CUH
#define AABB_CUH

#include "Interval.cuh"
#include "Hittable.cuh"

class AABB : public Hittable {
public:
    Interval ival[3];

    __device__ AABB();
    __device__ AABB(const Interval &x, const Interval &y, const Interval &z);
    __device__ AABB   (const Vec3 &a, const Vec3 &b);
    __device__ AABB(const AABB &a, const AABB &b);

    __device__ Interval &operator[](int index);
    __device__ Interval operator[](int index) const;

    __device__ bool hit(const Ray &r, Interval rayT) const;
};

__device__ AABB::AABB() : ival{Interval(), Interval(), Interval()} {}
__device__ AABB::AABB(const Interval &x, const Interval &y, const Interval &z) : ival{x, y, z} {}
__device__ AABB::AABB(const Vec3 &a, const Vec3 &b) : ival{Interval(a[0], b[0]), Interval(a[1], b[1]), Interval(a[2], b[2])} {}
__device__ AABB::AABB(const AABB &a, const AABB &b) : ival{Interval(a[0], b[0]), Interval(a[1], b[1]), Interval(a[2], b[2])} {}

#endif