#include "vec3.h"

Vec3::Vec3() : v{ 0,0,0 } {}
Vec3::Vec3(float x, float y, float z) : v{ x,y,z } {}

float &Vec3::operator[](int index) { return v[index]; }
float Vec3::operator[](int index) const { return v[index]; }

Vec3 Vec3::operator+(const Vec3 &other) const { return Vec3(v[0] + other.v[0], v[1] + other.v[1], v[2] + other.v[2]); }
Vec3 Vec3::operator-() const { return Vec3(-v[0], -v[1], -v[2]); }
Vec3 Vec3::operator-(const Vec3 &other) const { return Vec3(v[0] - other.v[0], v[1] - other.v[1], v[2] - other.v[2]); }
Vec3 Vec3::operator*(float scalar) const { return Vec3(v[0] * scalar, v[1] * scalar, v[2] * scalar); }
Vec3 operator*(float scalar, const Vec3 &v) { return Vec3(v.v[0] * scalar, v.v[1] * scalar, v.v[2] * scalar); }
Vec3 Vec3::operator/(float scalar) const { return Vec3(v[0] / scalar, v[1] / scalar, v[2] / scalar); }

float Vec3::dot(const Vec3 &other) const { return v[0] * other.v[0] + v[1] * other.v[1] + v[2] * other.v[2]; }
Vec3 Vec3::cross(const Vec3 &other) const {
    return Vec3(
        v[1] * other.v[2] - v[2] * other.v[1],
        v[2] * other.v[0] - v[0] * other.v[2],
        v[0] * other.v[1] - v[1] * other.v[0]
    );
}
float Vec3::length() const { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }
Vec3 Vec3::normalize() const { float len = length(); return Vec3(v[0] / len, v[1] / len, v[2] / len); }

float randomFloat() { return (float)rand() / RAND_MAX; }