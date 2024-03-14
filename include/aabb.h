#ifndef AABB_H
#define AABB_H

#include "fragment.h"
#include "interval.h"

class AABB {
public:
    Interval ival[3];

    Interval &operator[](int index);
    Interval operator[](int index) const;

    AABB();
    AABB(const Fragment &fragment);

    AABB operator+(const AABB &other);
    AABB &operator+=(const AABB &other);
};

#endif