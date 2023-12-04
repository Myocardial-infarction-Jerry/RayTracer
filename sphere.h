#ifndef SPHERE_H
#define SPHERE_H

#include "utils.h"
#include "aabb.h"
#include "hittable.h"

class sphere :public hittable {
public:
    __device__ sphere() {}
    __device__ sphere(vec3 center, float r, material *m) :center0(center), radius(r), matPtr(m), isMoving(false) {
        vec3 rvec(r, r, r);
        bbox = aabb(center - rvec, center + rvec);
    }
    __device__ sphere(vec3 u, vec3 v, float r, material *m) : center0(u), radius(r), matPtr(m), isMoving(true) {
        centerVec = v - u;
        vec3 rvec(r, r, r);
        aabb bbox1(u - rvec, u + rvec);
        aabb bbox2(v - rvec, v + rvec);
        bbox = aabb(bbox1, bbox2);
    }

    __device__ virtual bool hit(const ray &r, const interval &rayT, hitRecord &rec) const;
    __device__ virtual aabb boundingBox() const { return bbox; }
    __device__ vec3 sphereCenter(double time) const { return center0 + time * centerVec; }

    // private:
    vec3 center0;
    bool isMoving;
    vec3 centerVec;
    float radius;
    material *matPtr;
    aabb bbox;
};

__device__ bool sphere::hit(const ray &r, const interval &rayT, hitRecord &rec) const {
    vec3 center = isMoving ? sphereCenter(r.time()) : center0;
    vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - a * c;

    if (discriminant <= 0)
        return false;

    float temp = (-b - sqrt(discriminant)) / a;
    if (temp < rayT.max && temp > rayT.min) {
        rec.T = temp;
        rec.p = r.at(rec.T);
        rec.normal = (rec.p - center) / radius;
        rec.matPtr = matPtr;
        return true;
    }

    temp = (-b + sqrt(discriminant)) / a;
    if (temp < rayT.max && temp > rayT.min) {
        rec.T = temp;
        rec.p = r.at(rec.T);
        rec.normal = (rec.p - center) / radius;
        rec.matPtr = matPtr;
        return true;
    }

    return false;
}

#endif