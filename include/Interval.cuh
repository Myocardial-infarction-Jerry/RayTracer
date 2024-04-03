#ifndef INTERVAL_CUH
#define INTERVAL_CUH

#include <cuda_runtime.h>

class Interval {
    float mmin, mmax;

    __device__ Interval();
    __device__ Interval(const float &min, const float &max);
    __device__ Interval(const Interval &ivalA, const Interval &ivalB);

    __device__ bool contains(const float &val) const;
    __deivce__ float size() const;
    __device__ Interval expand(const float &delta) const;
};

__device__ Interval operator+(const Interval &ival, const float &displacement);
__device__ Interval operator+(const float &displacement, const Interval &ival);

// ! Interval implementation

__device__ Interval::Interval() : mmin(0.0f), mmax(0.0f) {}
__device__ Interval::Interval(const float &min, const float &max) : mmin(min), mmax(max) {}
__device__ Interval::Interval(const Interval &ivalA, const Interval &ivalB) : mmin(fmin(ivalA.mmin, ivalB.mmin)), mmax(fmax(ivalA.mmax, ivalB.mmax)) {}

__device__ bool Interval::contains(const float &val) const { return (val >= mmin && val <= mmax); }
__device__ float Interval::size() const { return (mmax - mmin); }
__device__ Interval Interval::expand(const float &delta) const { return Interval(mmin - delta / 2, mmax + delta / 2); }

__device__ Interval operator+(const Interval &ival, const float &displacement) { return Interval(ival.mmin + displacement, ival.mmax + displacement); }
__device__ Interval operator+(const float &displacement, const Interval &ival) { return Interval(ival.mmin + displacement, ival.mmax + displacement); }

#endif

