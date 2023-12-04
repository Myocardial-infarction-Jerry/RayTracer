#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.h"
#include "aabb.h"

class hittable_list :public hittable {
public:
    __device__ hittable_list() {}

    __device__ void add(hittable *object) {
        object->nextObject = nextObject;
        nextObject = object;
        bbox = aabb(bbox, object->boundingBox());
    }

    __device__ virtual bool hit(const ray &r, const interval &rayT, hitRecord &rec) const override {
        hitRecord tempRec;
        bool hitAnything = false;
        float closest = rayT.max;

        for (hittable *object = nextObject; object != nullptr; object = object->nextObject) {
            interval ival(rayT.min, closest);
            if (!object->hit(r, ival, tempRec))
                continue;

            hitAnything = true;
            closest = tempRec.T;
            rec = tempRec;
        }

        return hitAnything;
    }

    __device__ virtual aabb boundingBox() const override { return bbox; }

    // private:
    aabb bbox;
    vec3 background;
};

#endif