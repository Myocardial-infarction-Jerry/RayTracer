#ifndef BVH_H
#define BVH_H

#include "utils.h"
#include "hittable.h"
#include "hittable_list.h"

class bvhNode :public hittable {
public:
    __device__ bvhNode() :left(nullptr), right(nullptr) {}
    __device__ bvhNode(hittable *objList) : left(nullptr), right(nullptr) {

    }

    __device__ virtual bool hit(const ray &r, const interval &rayT, hitRecord &rec) const override {
        if (!bbox.hit(r, rayT))
            return false;

        bool hitLeft = left->hit(r, rayT, rec);
        bool hitRight = right->hit(r, interval(rayT.min, hitLeft ? rec.T : rayT.max), rec);

        return hitLeft || hitRight;
    }

    __device__ virtual aabb boundingBox() const override { return bbox; }

    __device__ inline static bool boxCompare(hittable *a, hittable *b, int axisIdx) { return a->boundingBox().axis(axisIdx).min < b->boundingBox().axis(axisIdx).min; }
    __device__ inline static bool boxXCompare(hittable *a, hittable *b) { return boxCompare(a, b, 0); }
    __device__ inline static bool boxYCompare(hittable *a, hittable *b) { return boxCompare(a, b, 1); }
    __device__ inline static bool boxZCompare(hittable *a, hittable *b) { return boxCompare(a, b, 2); }

    __device__ static hittable *bvhSort(hittable *head, bool(*comparator)(hittable *, hittable *)) {

    }

    // private:
    hittable *left, *right;
    aabb bbox;
};

#endif