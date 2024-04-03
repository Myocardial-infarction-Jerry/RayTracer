#ifndef MATERIAL_CUH
#define MATERIAL_CUH

#include "Ray.cuh"
#include "hittable.cuh"

class Material {
public:
    __device__ virtual bool scatter(const Ray &r_in, const HitRecord &rec, Vec3 &attenuation, Ray &scattered) const;
    __device__ virtual Vec3 emitted(const float &u, const float &v, const Vec3 &p) const;
};

#endif