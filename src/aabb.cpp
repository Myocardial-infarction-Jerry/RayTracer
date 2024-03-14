#include "aabb.h"

AABB::AABB() {}
AABB::AABB(const Fragment &fragment) {
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            float val = fragment.vertices[j][i];
            ival[i] += val;
        }
}

Interval &AABB::operator[](int index) { return ival[index]; }
Interval AABB::operator[](int index) const { return ival[index]; }