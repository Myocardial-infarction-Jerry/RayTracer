#ifndef HITTABLE_CUH
#define HITTABLE_CUH

#include "Ray.cuh"
#include "AABB.cuh"
#include "Material.cuh"
#include "Interval.cuh"

class HitRecord {
public:
    float u, v;
    Vec3 p, n;
    Material *mat;
};

class Hittable {
public:
    __device__ virtual bool hit(const Ray &r, const Interval &ival, HitRecord &rec) const = 0;
    __device__ virtual AABB boundingBox() const = 0;
};

#endif