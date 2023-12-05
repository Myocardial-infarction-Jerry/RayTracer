#ifndef MATERIAL_H
#define MATERIAL_H

class hitRecord;

#include "utils.h"
#include "ray.h"
#include "hittable.h"
#include "texture.h"

__device__ float schlick(const float &cosine, const float &refIdx) {
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
    __device__ virtual vec3 emitted(const float &u, const float &v, const vec3 &p) const { return vec3(0.0f, 0.0f, 0.0f); }
};

class lambertian :public material {
public:
    __device__ lambertian(texture *a) :albedo(a) {}
    __device__ lambertian(const vec3 &a) : albedo(new solidColor(a)) {}

    __device__ virtual bool scatter(
        const ray &rIn,
        const hitRecord &rec,
        vec3 &attenuation,
        ray &scattered,
        curandState *localRandState
    ) const override {
        vec3 target = rec.p + rec.normal + randomInUnitSphere(localRandState);
        scattered = ray(rec.p, target - rec.p, rIn.time());
        attenuation = albedo->value(rec.u, rec.v, rec.p);
        return true;
    }

    // private:
    texture *albedo;
};

class metal :public material {
public:
    __device__ metal(const vec3 &a, const float &f) :albedo(a) { fuzz = (f < 1) ? f : 1.0f; }
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
        return (dot(scattered.direction(), rec.normal) > 0);
    }

    // private:
    vec3 albedo;
    float fuzz;
};

class dielectric :public material {
public:
    __device__ dielectric(const float &ri) :refIdx(ri) {}
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
    __device__ diffuseLight(texture *t) :emit(t) {}
    __device__ diffuseLight(const vec3 &color) : emit(new solidColor(color)) {}

    __device__ virtual bool scatter(
        const ray &rIn,
        const hitRecord &rec,
        vec3 &attenuation,
        ray &scattered,
        curandState *localRandState
    ) const override {
        return false;
    }

    __device__ vec3 emitted(const float &u, const float &v, const vec3 &p) const override { return emit->value(u, v, p); }

    // private:
    texture *emit;
};

class isotropic : public material {
public:
    __device__ isotropic(texture *t) :albedo(t) {}
    __device__ isotropic(const vec3 &color) : albedo(new solidColor(color)) {}

    __device__ virtual bool scatter(
        const ray &rIn,
        const hitRecord &rec,
        vec3 &attenuation,
        ray &scattered,
        curandState *localRandState
    ) const override {
        scattered = ray(rec.p, RANDVEC3.unit(), rIn.time());
        attenuation = albedo->value(rec.u, rec.v, rec.p);
        return true;
    }

    texture *albedo;
};

#endif