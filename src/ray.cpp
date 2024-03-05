#include "ray.h"

Ray::Ray() : origin(0, 0, 0), direction(0, 0, 0) {}
Ray::Ray(Vec3 origin, Vec3 direction) : origin(origin), direction(direction.normalize()) {}
Ray::~Ray() {}

Vec3 Ray::pointAtParameter(float t) {
    return origin + t * direction;
}