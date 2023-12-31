#ifndef SPHERE_H
#define SPHERE_H

#include "utils.h"
#include "aabb.h"
#include "hittable.h"

class sphere :public hittable {
public:
    __device__ sphere() {}
    __device__ sphere(const vec3 &center, const float &r, material *m) :center0(center), radius(r), matPtr(m), isMoving(false) {
        vec3 rvec(r, r, r);
        bbox = aabb(center - rvec, center + rvec);
    }
    __device__ sphere(const vec3 &u, const vec3 &v, const float &r, material *m) : center0(u), radius(r), matPtr(m), isMoving(true) {
        centerVec = v - u;
        vec3 rvec(r, r, r);
        aabb bbox1(u - rvec, u + rvec);
        aabb bbox2(v - rvec, v + rvec);
        bbox = aabb(bbox1, bbox2);
    }

    __device__ virtual bool hit(const ray &r, const interval &rayT, hitRecord &rec) const override {
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
            getSphereUV(rec.normal, rec.u, rec.v);
            rec.matPtr = matPtr;
            return true;
        }

        temp = (-b + sqrt(discriminant)) / a;
        if (temp < rayT.max && temp > rayT.min) {
            rec.T = temp;
            rec.p = r.at(rec.T);
            rec.normal = (rec.p - center) / radius;
            getSphereUV(rec.normal, rec.u, rec.v);
            rec.matPtr = matPtr;
            return true;
        }

        return false;
    }
    __device__ virtual aabb boundingBox() const { return bbox; }
    __device__ vec3 sphereCenter(const float &time) const { return center0 + time * centerVec; }

    __device__ static void getSphereUV(const vec3 &p, float &u, float &v) {
        float theta = acos(-p.y());
        float phi = atan2(-p.z(), p.x()) + M_PI;
        u = phi / (2 * M_PI);
        v = theta / M_PI;
    }

    // private:
    vec3 center0;
    bool isMoving;
    vec3 centerVec;
    float radius;
    material *matPtr;
    aabb bbox;
};

#endif