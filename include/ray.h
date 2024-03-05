#ifndef RAY_H
#define RAY_H

#include "vec3.h"

class Ray {
public:
    Ray();
    Ray(Vec3 origin, Vec3 direction);
    ~Ray();

    Vec3 origin;
    Vec3 direction;
    unsigned int source;
    Vec3 pointAtParameter(float t);
};

#endif