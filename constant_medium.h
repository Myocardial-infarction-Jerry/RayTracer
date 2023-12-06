#ifndef CONSTANT_MEDIUM_H
#define CONSTANT_MEDIUM_H

#include <curand_kernel.h>

#include "utils.h"
#include "hittable.h"
#include "material.h"
#include "texture.h"
#include "interval.h"

class constant_medium :public hittable {
public:
    __device__ constant_medium(hittable *b, const float &d, texture *a, curandState *randState)
        : boundary(b), negInvDensity(-1.0f / d), phaseFunction(new isotropic(a)), localRandState(randState) {}

    __device__ constant_medium(hittable *b, const float &d, const vec3 &c, curandState *randState)
        : boundary(b), negInvDensity(-1.0f / d), phaseFunction(new isotropic(c)), localRandState(randState) {}

    __device__ virtual bool hit(const ray &r, const interval &rayT, hitRecord &rec) const override {
        hitRecord rec1, rec2;

        if (!boundary->hit(r, interval(-FLT_MAX, FLT_MAX), rec1))
            return false;

        if (!boundary->hit(r, interval(rec1.T + 0.0001f, FLT_MAX), rec2))
            return false;

        if (rec1.T < rayT.min) rec1.T = rayT.min;
        if (rec2.T > rayT.max) rec2.T = rayT.max;

        if (rec1.T >= rec2.T)
            return false;

        if (rec1.T < 0)
            rec1.T = 0;

        auto rayLength = r.direction().length();
        auto distanceInsideBoundary = (rec2.T - rec1.T) * rayLength;
        auto hitDistance = negInvDensity * log(curand_uniform(localRandState));

        if (hitDistance > distanceInsideBoundary)
            return false;

        rec.T = rec1.T + hitDistance / rayLength;
        rec.p = r.at(rec.T);

        rec.normal = vec3(1, 0, 0);
        rec.matPtr = phaseFunction;

        return true;
    }

    __device__ virtual aabb boundingBox() const override { return boundary->boundingBox(); }

private:
    hittable *boundary;
    float negInvDensity;
    material *phaseFunction;
    curandState *localRandState;
};

#endif