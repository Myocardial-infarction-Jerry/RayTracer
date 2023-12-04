#ifndef INTERVAL_H
#define INTERVAL_H

class interval {
public:
    __device__ interval() {}
    __device__ interval(const float &_min, const float &_max) :min(_min), max(_max) {}
    __device__ interval(const interval &a, const interval &b) : min(fmin(a.min, b.min)), max(fmax(a.max, b.max)) {}

    __device__ bool contains(float x) const { return min <= x && x <= max; }
    __device__ float size() const { return max - min; }
    __device__ interval expand(float delta) const {
        float padding = delta / 2;
        return interval(min - padding, max + padding);
    }

    float min, max;

    static const interval empty, universe;
};

const interval interval::empty = interval(FLT_MAX, -FLT_MAX);
const interval interval::universe = interval(-FLT_MAX, FLT_MAX);

__device__ interval operator+(const interval &iVal, float displacement) { return interval(iVal.min + displacement, iVal.max + displacement); }
__device__ interval operator+(float displacement, const interval &iVal) { return interval(iVal.min + displacement, iVal.max + displacement); }

#endif