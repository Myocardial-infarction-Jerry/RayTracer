#include "interval.h"

#include <cmath>

Interval::Interval() : mmin(std::numeric_limits<float>::max()), mmax(std::numeric_limits<float>::min()) {}
Interval::Interval(float val) : mmin(val), mmax(val) {}
Interval::Interval(float min, float max) : mmin(min), mmax(max) {}

bool Interval::contain(float val) { return val >= mmin && val <= mmax; }

Interval Interval::operator+(const Interval &other) { return Interval(std::fmin(mmin, other.mmin), std::fmax(mmax, other.mmax)); }
Interval &Interval::operator+=(const Interval &other) {
    mmin = std::fmin(mmin, other.mmin);
    mmax = std::fmax(mmax, other.mmax);
    return *this;
}
