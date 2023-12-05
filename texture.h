#ifndef TEXTURE_H
#define TEXTURE_H

#include "utils.h"
#include "vec3.h"

class texture {
public:
    __device__ virtual ~texture() = default;

    __device__ virtual vec3 value(const float &u, const float &v, const vec3 &p) const = 0;
};

class solidColor :public texture {
public:
    __device__ solidColor(const vec3 &c) :color(c) {}
    __device__ solidColor(const float &r, const float &g, const float &b) : color(vec3(r, g, b)) {}

    __device__ vec3 value(const float &u, const float &v, const vec3 &p) const override { return color; }

private:
    vec3 color;
};

class checkerTexture :public texture {
public:
    __device__ checkerTexture(const float &_scale, texture *_even, texture *_odd)
        :invScale(1.0f / _scale), even(_even), odd(_odd) {}
    __device__ checkerTexture(const float &_scale, const vec3 &a, const vec3 &b)
        : invScale(1.0f / _scale), even(new solidColor(a)), odd(new solidColor(b)) {}

    __device__ vec3 value(const float &u, const float &v, const vec3 &p) const override {
        auto xInteger = static_cast<int>(std::floor(invScale * p.x()));
        auto yInteger = static_cast<int>(std::floor(invScale * p.y()));
        auto zInteger = static_cast<int>(std::floor(invScale * p.z()));

        bool isEven = (xInteger + yInteger + zInteger) % 2 == 0;

        return isEven ? even->value(u, v, p) : odd->value(u, v, p);
    }

private:
    float invScale;
    texture *even, *odd;
};

#endif