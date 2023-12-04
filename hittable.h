#ifndef HITTABLE_H
#define HITTABLE_H

#include "ray.h"
#include "aabb.h"
#include "interval.h"

class material;

class hitRecord {
public:
    float T;
    vec3 p;
    vec3 normal;
    material *matPtr;
};

class hittable {
public:
    __device__ hittable() :nextObject(nullptr) {}

    __device__ virtual bool hit(const ray &r, const interval &rayT, hitRecord &rec) const = 0;
    __device__ virtual aabb boundingBox() const = 0;
    // __device__ virtual void add(hittable *object) {}

    hittable *nextObject;
};

#endif