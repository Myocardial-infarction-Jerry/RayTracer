#include "vec3.h"

Vec3::Vec3() : x(0), y(0), z(0) {}
Vec3::Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

Vec3 Vec3::operator+(const Vec3 &other) const {
    return Vec3(x + other.x, y + other.y, z + other.z);
}

Vec3 Vec3::operator-() const {
    return Vec3(-x, -y, -z);
}

Vec3 Vec3::operator-(const Vec3 &other) const {
    return Vec3(x - other.x, y - other.y, z - other.z);
}

Vec3 Vec3::operator*(float scalar) const {
    return Vec3(x * scalar, y * scalar, z * scalar);
}

Vec3 operator*(float scalar, const Vec3 &v) {
    return Vec3(v.x * scalar, v.y * scalar, v.z * scalar);
}

Vec3 Vec3::operator/(float scalar) const {
    return Vec3(x / scalar, y / scalar, z / scalar);
}

float Vec3::dot(const Vec3 &other) const {
    return x * other.x + y * other.y + z * other.z;
}

Vec3 Vec3::cross(const Vec3 &other) const {
    return Vec3(
        y * other.z - z * other.y,
        z * other.x - x * other.z,
        x * other.y - y * other.x
    );
}

float Vec3::length() const {
    return std::sqrt(x * x + y * y + z * z);
}

Vec3 Vec3::normalize() const {
    float len = length();
    return Vec3(x / len, y / len, z / len);
}

float randomFloat() { return (float)rand() / RAND_MAX; }