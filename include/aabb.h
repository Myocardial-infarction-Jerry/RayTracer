#ifndef AABB_H
#define AABB_H

#include "fragment.h"

class Interval {
public:
    float mmin, mmax;

    Interval();
    Interval(float val);
    Interval(float min, float max);

    Interval operator+(const Interval &other);
    Interval &operator+=(const Interval &other);
};

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