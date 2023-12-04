#ifndef CAMERA_H
#define CAMERA_H

#include "utils.h"
#include "vec3.h"
#include "ray.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__device__ vec3 randomInUnitDisk(curandState *localRandState) {
    vec3 p;
    do {
        p = 2.0f * vec3(curand_uniform(localRandState), curand_uniform(localRandState), 0) - vec3(1, 1, 0);
    } while (dot(p, p) >= 1.0f);
    return p;
}

class camera {
public:
    __device__ camera(
        vec3 lookFrom,
        vec3 lookAt,
        vec3 up,
        float fov,
        float aspect,
        float aperture,
        float focusLen
    ) {
        lensRadius = aperture / 2.0f;
        float theta = fov * (float)(M_PI) / 180.0f;
        float halfHeight = tan(theta / 2.0f);
        float halfWidth = aspect * halfHeight;

        origin = lookFrom;
        w = (lookFrom - lookAt).unit();
        u = cross(up, w).unit();
        v = cross(w, u);
        lowerLeftCorner =
            origin - halfWidth * focusLen * u
            - halfHeight * focusLen * v
            - focusLen * w;
        horizontal = 2.0f * halfWidth * focusLen * u;
        vertical = 2.0f * halfHeight * focusLen * v;
    }

    __device__ ray getRay(float s, float t, curandState *localRandState) {
        vec3 rd = lensRadius * randomInUnitDisk(localRandState);
        vec3 offset = u * rd.x() + v * rd.y();
        return ray(
            origin + offset,
            lowerLeftCorner + s * horizontal + t * vertical - origin - offset,
            curand_uniform(localRandState)
        );
    }

    // private:
    vec3 origin;
    vec3 lowerLeftCorner;
    vec3 horizontal, vertical;
    vec3 u, v, w;
    float lensRadius;
};

#endif