#ifndef HITTABLE_H
#define HITTABLE_H

#include "ray.h"
#include "aabb.h"
#include "interval.h"

class material;

class hitRecord {
public:
    __device__ void setFaceNormal(const ray &r, const vec3 &outwardNormal) {
        frontFace = dot(r.direction(), outwardNormal) < 0;
        normal = frontFace ? outwardNormal : -outwardNormal;
    }

    float T;
    float u, v;
    vec3 p;
    vec3 normal;
    material *matPtr;
    bool frontFace;
};

class hittable {
public:
    __device__ hittable() :nextObject(nullptr) {}

    __device__ virtual bool hit(const ray &r, const interval &rayT, hitRecord &rec) const = 0;
    __device__ virtual aabb boundingBox() const = 0;

    hittable *nextObject;
};

class translate :public hittable {
public:
    __device__ translate(hittable *p, const vec3 &displacement) :object(p), offset(displacement) { bbox = object->boundingBox() + offset; }

    __device__ virtual bool hit(const ray &r, const interval &rayT, hitRecord &rec) const override {
        ray offsetR(r.origin() - offset, r.direction(), r.time());

        if (!object->hit(offsetR, rayT, rec))
            return false;

        rec.p += offset;

        return true;
    }

    __device__ virtual aabb boundingBox() const override { return bbox; }

private:
    hittable *object;
    vec3 offset;
    aabb bbox;
};

class rotateY : public hittable {
public:
    __device__ rotateY(hittable *p, const float &angle) : object(p) {
        auto radians = angle * M_PI / 180.0f;
        sinTheta = sin(radians);
        cosTheta = cos(radians);
        bbox = object->boundingBox();

        vec3 min(FLT_MAX, FLT_MAX, FLT_MAX);
        vec3 max(-FLT_MAX, -FLT_MAX, -FLT_MAX);

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    auto x = i * bbox.x.max + (1 - i) * bbox.x.min;
                    auto y = j * bbox.y.max + (1 - j) * bbox.y.min;
                    auto z = k * bbox.z.max + (1 - k) * bbox.z.min;

                    auto newx = cosTheta * x + sinTheta * z;
                    auto newz = -sinTheta * x + cosTheta * z;

                    vec3 tester(newx, y, newz);

                    for (int c = 0; c < 3; c++) {
                        min[c] = fmin(min[c], tester[c]);
                        max[c] = fmax(max[c], tester[c]);
                    }
                }
            }
        }

        bbox = aabb(min, max);
    }


    __device__ virtual bool hit(const ray &r, const interval &rayT, hitRecord &rec) const override {
        auto origin = r.origin();
        auto direction = r.direction();

        origin[0] = cosTheta * r.origin()[0] - sinTheta * r.origin()[2];
        origin[2] = sinTheta * r.origin()[0] + cosTheta * r.origin()[2];

        direction[0] = cosTheta * r.direction()[0] - sinTheta * r.direction()[2];
        direction[2] = sinTheta * r.direction()[0] + cosTheta * r.direction()[2];

        ray rotatedR(origin, direction, r.time());

        if (!object->hit(rotatedR, rayT, rec))
            return false;

        auto p = rec.p;
        p[0] = cosTheta * rec.p[0] + sinTheta * rec.p[2];
        p[2] = -sinTheta * rec.p[0] + cosTheta * rec.p[2];

        auto normal = rec.normal;
        normal[0] = cosTheta * rec.normal[0] + sinTheta * rec.normal[2];
        normal[2] = -sinTheta * rec.normal[0] + cosTheta * rec.normal[2];

        rec.p = p;
        rec.normal = normal;

        return true;
    }

    __device__ virtual aabb boundingBox() const override { return bbox; }

private:
    hittable *object;
    double sinTheta;
    double cosTheta;
    aabb bbox;
};

#endif