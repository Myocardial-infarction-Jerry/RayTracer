#ifndef VEC3_H
#define VEC3_H

#include <cmath>

class Vec3 {
public:
    float x, y, z;

    Vec3();
    Vec3(float x, float y, float z);

    Vec3 operator+(const Vec3 &other) const;
    Vec3 operator-() const;
    Vec3 operator-(const Vec3 &other) const;
    Vec3 operator*(float scalar) const;
    friend Vec3 operator*(float scalar, const Vec3 &v);
    Vec3 operator/(float scalar) const;

    float dot(const Vec3 &other) const;
    Vec3 cross(const Vec3 &other) const;
    float length() const;
    Vec3 normalize() const;
};

float randomFloat();

#endif // Vec3_H
