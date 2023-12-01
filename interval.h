#ifndef INTERVAL_H
#define INTERVAL_H

#include <stdlib.h>
#include <math.h>

class interval {
public:
    __device__ interval() {}
    __device__ interval(const double &_min, const double &_max) :min(_min), max(_max) {}
    __device__ interval(const interval &a, const interval &b) : min(fmin(a.min, b.min)), max(fmax(a.max, b.max)) {}

    __device__ bool contains(double x) const { return min <= x && x <= max; }
    __device__ double size() const { return max - min; }
    __device__ interval expand(double delta) const {
        double padding = delta / 2;
        return interval(min - padding, max + padding);
    }

    double min, max;
};

const interval empty = interval(+INFINITY, -INFINITY);
const interval universe = interval(-INFINITY, +INFINITY);

__device__ interval operator+(const interval &ival, double displacement) { return interval(ival.min + displacement, ival.max + displacement); }
__device__ interval operator+(double displacement, const interval &ival) { return interval(ival.min + displacement, ival.max + displacement); }

#endif