#include "aabb.h"

Interval::Interval() : mmin(std::numeric_limits<float>::max()), mmax(std::numeric_limits<float>::min()) {}
Interval::Interval(float val) : mmin(val), mmax(val) {}
Interval::Interval(float min, float max) : mmin(min), mmax(max) {}

Interval Interval::operator+(const Interval &other) { return Interval(std::fmin(mmin, other.mmin), std::fmax(mmax, other.mmax)); }
Interval &Interval::operator+=(const Interval &other) {
    mmin = std::fmin(mmin, other.mmin);
    mmax = std::fmax(mmax, other.mmax);
    return *this;
}

Interval &AABB::operator[](int index) { return ival[index]; }
Interval AABB::operator[](int index) const { return ival[index]; }

AABB::AABB() {}
AABB::AABB(const Fragment &fragment) {
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            float val = fragment.vertices[j][i];
            ival[i] += val;
        }
}