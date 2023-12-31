#ifndef AABB_H
#define AABB_H

#include "interval.h"
#include "vec3.h"
#include "ray.h"

class aabb {
public:
    interval x, y, z;

    __device__ aabb() {}
    __device__ aabb(const interval &ix, const interval &iy, const interval &iz) : x(ix), y(iy), z(iz) {}
    __device__ aabb(const vec3 &a, const vec3 &b) {
        x = interval(fmin(a[0], b[0]), fmax(a[0], b[0]));
        y = interval(fmin(a[1], b[1]), fmax(a[1], b[1]));
        z = interval(fmin(a[2], b[2]), fmax(a[2], b[2]));
    }
    __device__ aabb(const aabb &a, const aabb &b) {
        x = interval(a.x, b.x);
        y = interval(a.y, b.y);
        z = interval(a.z, b.z);
    }

    __device__ aabb pad() {
        float delta = 0.0001f;
        interval newX = (x.size() >= delta) ? x : x.expand(delta);
        interval newY = (y.size() >= delta) ? y : y.expand(delta);
        interval newZ = (z.size() >= delta) ? z : z.expand(delta);

        return aabb(newX, newY, newZ);
    }

    __device__ const interval &axis(int n) const {
        if (n == 0) return x;
        if (n == 1) return y;
        return z;
    }

    __device__ bool hit(const ray &r, interval rayT) const {
        for (int a = 0; a < 3; a++) {
            auto invD = 1 / r.direction()[a];
            auto orig = r.origin()[a];

            auto t0 = (axis(a).min - orig) * invD;
            auto t1 = (axis(a).max - orig) * invD;

            if (invD < 0) {
                auto t = t0;
                t0 = t1;
                t1 = t;
            }

            if (t0 > rayT.min) rayT.min = t0;
            if (t1 < rayT.max) rayT.max = t1;

            if (rayT.max <= rayT.min)
                return false;
        }
        return true;
    }
};

__device__ inline aabb operator+(const aabb &bbox, const vec3 &offset) { return aabb(bbox.x + offset.x(), bbox.y + offset.y(), bbox.z + offset.z()); }
__device__ inline aabb operator+(const vec3 &offset, const aabb &bbox) { return aabb(bbox.x + offset.x(), bbox.y + offset.y(), bbox.z + offset.z()); }

#endif