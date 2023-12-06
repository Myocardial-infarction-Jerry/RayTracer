#ifndef QUAD_H
#define QUAD_H

#include "utils.h"
#include "hittable.h"
#include "hittable_list.h"

class quad :public hittable {
public:
    __device__ quad(const vec3 &_Q, const vec3 &_u, const vec3 &_v, material *m) :Q(_Q), u(_u), v(_v), matPtr(m) {
        auto n = cross(u, v);
        normal = n.unit();
        D = dot(normal, Q);
        w = n / dot(n, n);
        setBoundingBox();
    }

    __device__ virtual void setBoundingBox() { bbox = aabb(Q, Q + u + v).pad(); }
    __device__ virtual aabb boundingBox() const override { return bbox; }
    __device__ virtual bool hit(const ray &r, const interval &rayT, hitRecord &rec) const override {
        auto denom = dot(normal, r.direction());

        if (fabs(denom) < 1E-8)
            return false;

        auto t = (D - dot(normal, r.origin())) / denom;
        if (!rayT.contains(t))
            return false;

        auto intersection = r.at(t);
        vec3 planarHitptVector = intersection - Q;
        auto alpha = dot(w, cross(planarHitptVector, v));
        auto beta = dot(w, cross(u, planarHitptVector));

        if (!isInterior(alpha, beta, rec))
            return false;

        rec.T = t;
        rec.p = intersection;
        rec.matPtr = matPtr;

        return true;
    }

    __device__ virtual bool isInterior(const float &a, const float &b, hitRecord &rec) const {
        if ((a < 0) || (1 < a) || (b < 0) || (1 < b))
            return false;

        rec.u = a;
        rec.v = b;
        return true;
    }

    vec3 Q;
    vec3 u, v, normal, w;
    float D;
    material *matPtr;
    aabb bbox;
};

__device__ hittable *box(const vec3 &a, const vec3 &b, material *mat) {
    hittable *sides = new hittable_list();

    auto min = point3(fmin(a.x(), b.x()), fmin(a.y(), b.y()), fmin(a.z(), b.z()));
    auto max = point3(fmax(a.x(), b.x()), fmax(a.y(), b.y()), fmax(a.z(), b.z()));

    auto dx = vec3(max.x() - min.x(), 0, 0);
    auto dy = vec3(0, max.y() - min.y(), 0);
    auto dz = vec3(0, 0, max.z() - min.z());

    ((hittable_list *)(sides))->add(new quad(vec3(min.x(), min.y(), max.z()), dx, dy, mat)); // front
    ((hittable_list *)(sides))->add(new quad(vec3(max.x(), min.y(), max.z()), -dz, dy, mat)); // right
    ((hittable_list *)(sides))->add(new quad(vec3(max.x(), min.y(), min.z()), -dx, dy, mat)); // back
    ((hittable_list *)(sides))->add(new quad(vec3(min.x(), min.y(), min.z()), dz, dy, mat)); // left
    ((hittable_list *)(sides))->add(new quad(vec3(min.x(), max.y(), max.z()), dx, -dz, mat)); // top
    ((hittable_list *)(sides))->add(new quad(vec3(min.x(), min.y(), min.z()), dx, dz, mat)); // bottom

    bvhNode::buildFromList(&sides);

    return sides->nextObject;
}

#endif