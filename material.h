#ifndef MATERIAL_H
#define MATERIAL_H

class hitRecord;

#include "utils.h"
#include "ray.h"
#include "hittable.h"

__device__ float schlick(float cosine, float refIdx) {
    float r0 = (1.0f - refIdx) / (1.0f + refIdx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}

__device__ vec3 randomInUnitSphere(curandState *localRandState) {
    vec3 p;
    do {
        p = 2.0f * RANDVEC3 - vec3(1, 1, 1);
    } while (p.squared_length() >= 1.0f);
    return p;
}

class material {
public:
    __device__ virtual bool scatter(
        const ray &rIn,
        const hitRecord &rec,
        vec3 &attenuation,
        ray &scattered,
        curandState *localRandState
    ) const = 0;
    __device__ virtual vec3 emitted(double u, double v, const vec3 &p) const { return vec3(0.0f, 0.0f, 0.0f); }
};

class lambertian :public material {
public:
    __device__ lambertian(const vec3 &a) :albedo(a) {}
    __device__ virtual bool scatter(
        const ray &rIn,
        const hitRecord &rec,
        vec3 &attenuation,
        ray &scattered,
        curandState *localRandState
    ) const override {
        vec3 target = rec.p + rec.normal + randomInUnitSphere(localRandState);
        scattered = ray(rec.p, target - rec.p, rIn.time());
        attenuation = albedo;
        return true;
    }

    // private:
    vec3 albedo;
};

class metal :public material {
public:
    __device__ metal(const vec3 &a, float f) :albedo(a) { fuzz = (f < 1) ? f : 1.0f; }
    __device__ virtual bool scatter(
        const ray &rIn,
        const hitRecord &rec,
        vec3 &attenuation,
        ray &scattered,
        curandState *localRandState
    ) const override {
        vec3 reflected = reflect(rIn.direction().unit(), rec.normal);
        scattered = ray(rec.p, reflected + fuzz * randomInUnitSphere(localRandState), rIn.time());
        attenuation = albedo;
    }

    // private:
    vec3 albedo;
    float fuzz;
};

class dielectric :public material {
public:
    __device__ dielectric(float ri) :refIdx(ri) {}
    __device__ virtual bool scatter(
        const ray &rIn,
        const hitRecord &rec,
        vec3 &attenuation,
        ray &scattered,
        curandState *localRandState
    ) const override {
        vec3 outwardNormal;
        vec3 reflected = reflect(rIn.direction(), rec.normal);
        float niOverNt;
        attenuation = vec3(1.0f, 1.0f, 1.0f);
        vec3 refracted;
        float reflectProb;
        float cosine;

        if (dot(rIn.direction(), rec.normal) > 0.0f) {
            outwardNormal = -rec.normal;
            niOverNt = refIdx;
            cosine = dot(rIn.direction(), rec.normal) / rIn.direction().length();
            cosine = sqrt(1.0f - refIdx * refIdx * (1 - cosine * cosine));
        }
        else {
            outwardNormal = rec.normal;
            niOverNt = 1.0f / refIdx;
            cosine = -dot(rIn.direction(), rec.normal) / rIn.direction().length();
        }

        if (refract(rIn.direction(), outwardNormal, niOverNt, refracted))
            reflectProb = schlick(cosine, refIdx);
        else
            reflectProb = 1.0f;
        if (curand_uniform(localRandState) < reflectProb)
            scattered = ray(rec.p, reflected, rIn.time());
        else
            scattered = ray(rec.p, refracted, rIn.time());
        return true;
    }

    // private:
    float refIdx;
};

class diffuseLight :public material {
public:
    __device__ diffuseLight(vec3 color) :albedo(color) {}

    __device__ virtual bool scatter(
        const ray &rIn,
        const hitRecord &rec,
        vec3 &attenuation,
        ray &scattered,
        curandState *localRandState
    ) const override {
        return false;
    }

    __device__ vec3 emitted(double u, double v, const vec3 &p) const override { return albedo; }

    // private:
    vec3 albedo;
};

#endif