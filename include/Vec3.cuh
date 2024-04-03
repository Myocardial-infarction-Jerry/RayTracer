#ifndef VEC3_CUH
#define VEC3_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>

#define EPSILON 0.00001

class Vec3 {
public:
    float v[3];

    __host__ __device__ Vec3();
    __host__ __device__ Vec3(float x, float y, float z);

    __host__ __device__ float operator[](int index) const;
    __host__ __device__ float &operator[](int index);

    __host__ friend std::ostream &operator<<(std::ostream &o, const Vec3 &vec);

    __host__ __device__ Vec3 operator+(const Vec3 &vec) const;
    __host__ __device__ Vec3 operator-(const Vec3 &vec) const;
    __host__ __device__ Vec3 operator*(const Vec3 &vec) const;
    __host__ __device__ Vec3 operator/(const Vec3 &vec) const;

    __host__ __device__ Vec3 operator*(const float &val) const;
    __host__ __device__ Vec3 operator/(const float &val) const;
    __host__ __device__ friend Vec3 operator*(const float &val, const Vec3 &vec);

    __host__ __device__ bool operator==(const Vec3 &vec) const;
    __host__ __device__ bool operator!=(const Vec3 &vec) const;

    __host__ __device__ float length() const;
    __host__ __device__ float lengthSquared() const;
    __host__ __device__ Vec3 normalize() const;

    static __host__ Vec3 random();
    static __device__ Vec3 random(curandState *state);
};

__host__ __device__ float dot(const Vec3 &vec1, const Vec3 &vec2);
__host__ __device__ Vec3 cross(const Vec3 &vec1, const Vec3 &vec2);

// ! Vec3 implementation

__host__ __device__ Vec3::Vec3() : v{ 0, 0, 0 } {}
__host__ __device__ Vec3::Vec3(float x, float y, float z) : v{ x, y, z } {}

__host__ __device__ float Vec3::operator[](int index) const { return v[index]; }
__host__ __device__ float &Vec3::operator[](int index) { return v[index]; }

__host__ std::ostream &operator<<(std::ostream &os, const Vec3 &v) { os << "(" << v[0] << ", " << v[1] << ", " << v[2] << ")"; return os; }

__host__ __device__ Vec3 Vec3::operator+(const Vec3 &vec) const { return Vec3(v[0] + vec[0], v[1] + vec[1], v[2] + vec[2]); }
__host__ __device__ Vec3 Vec3::operator-(const Vec3 &vec) const { return Vec3(v[0] - vec[0], v[1] - vec[1], v[2] - vec[2]); }
__host__ __device__ Vec3 Vec3::operator*(const Vec3 &vec) const { return Vec3(v[0] * vec[0], v[1] * vec[1], v[2] * vec[2]); }
__host__ __device__ Vec3 Vec3::operator/(const Vec3 &vec) const { return Vec3(v[0] / vec[0], v[1] / vec[1], v[2] / vec[2]); }

__host__ __device__ Vec3 Vec3::operator*(const float &val) const { return Vec3(v[0] * val, v[1] * val, v[2] * val); }
__host__ __device__ Vec3 Vec3::operator/(const float &val) const { return Vec3(v[0] / val, v[1] / val, v[2] / val); }
__host__ __device__ Vec3 operator*(const float &val, const Vec3 &vec) { return Vec3(vec[0] * val, vec[1] * val, vec[2] * val); }

__host__ __device__ bool Vec3::operator==(const Vec3 &vec) const { return fabs(v[0] - vec[0]) < EPSILON && fabs(v[1] - vec[1]) < EPSILON && fabs(v[2] - vec[2]) < EPSILON; }
__host__ __device__ bool Vec3::operator!=(const Vec3 &vec) const { return !(*this == vec); }

__host__ __device__ float Vec3::length() const { return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }
__host__ __device__ float Vec3::lengthSquared() const { return v[0] * v[0] + v[1] * v[1] + v[2] * v[2]; }
__host__ __device__ Vec3 Vec3::normalize() const { return *this / length(); }

__host__ __device__ float dot(const Vec3 &vec1, const Vec3 &vec2) { return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]; }
__host__ __device__ Vec3 cross(const Vec3 &vec1, const Vec3 &vec2) { return Vec3(vec1[1] * vec2[2] - vec1[2] * vec2[1], vec1[2] * vec2[0] - vec1[0] * vec2[2], vec1[0] * vec2[1] - vec1[1] * vec2[0]); }

__host__ Vec3 Vec3::random() { return Vec3(rand() / (RAND_MAX + 1.0), rand() / (RAND_MAX + 1.0), rand() / (RAND_MAX + 1.0)); }
__device__ Vec3 Vec3::random(curandState *state) { return Vec3(curand_uniform(state), curand_uniform(state), curand_uniform(state)); }

#endif

