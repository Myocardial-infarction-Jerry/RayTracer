#ifndef BVH_H
#define BVH_H

#include "utils.h"
#include "hittable.h"
#include "hittable_list.h"

class bvhNode :public hittable {
public:
    __device__ bvhNode(hittable *_left = nullptr, hittable *_right = nullptr) :left(_left), right(_right) {
        if (left != nullptr) bbox = aabb(bbox, left->boundingBox());
        if (right != nullptr) bbox = aabb(bbox, right->boundingBox());
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
    // __device__ inline static bool boxXCompare(hittable *a, hittable *b) { return boxCompare(a, b, 0); }
    // __device__ inline static bool boxYCompare(hittable *a, hittable *b) { return boxCompare(a, b, 1); }
    // __device__ inline static bool boxZCompare(hittable *a, hittable *b) { return boxCompare(a, b, 2); }

    __device__ static void bvhSort(hittable **objList, const int &axisIdx) {
        int listLen = 0;
        auto cur = *objList;
        for (cur = cur->nextObject; cur != nullptr; cur = cur->nextObject)
            listLen++;

        for (int i = 0; i < listLen; ++i) {
            cur = *objList;
            for (int j = 0; j < listLen - 1; ++j) {
                auto p = cur->nextObject;
                auto q = p->nextObject;
                if (!boxCompare(p, q, axisIdx)) {
                    p->nextObject = q->nextObject;
                    q->nextObject = p;
                    cur->nextObject = q;
                }
                cur = cur->nextObject;
            }
        }
    }

    __device__ static void buildFromList(hittable **objList) {
        curandState localRandState;
        curand_init(RAND_SEED, 0, 0, &localRandState);
        int axis;
        hittable *cur, *p, *q;
        while (1) {
            axis = static_cast<int>(RND * 3);
            bvhSort(objList, axis);

            cur = *objList;
            p = cur->nextObject;
            q = p->nextObject;
            if (q == nullptr)
                break;
            while (1) {
                cur->nextObject = new bvhNode(p, q);
                cur = cur->nextObject;

                p = q->nextObject;
                if (p == nullptr)
                    break;
                q = p->nextObject;
                if (q == nullptr) {
                    cur->nextObject = p;
                    p->nextObject = nullptr;
                    break;
                }
            }
        }

        cur = (*objList)->nextObject;
        delete *objList;
        *objList = cur;
    }

    // private:
    hittable *left, *right;
    aabb bbox;
};

#endif