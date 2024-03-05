#ifndef CAMERA_H
#define CAMERA_H

#include <vector>

#include "vec3.h"
#include "ray.h"

class Camera {
public:
    Camera();
    ~Camera();

    std::vector<Ray> getRayList() const;
    std::vector<Vec3> getImage() const;

    Vec3 position, direction, up;
    float fov, lens;
    unsigned int width, height;
    unsigned int SPP;
};

#endif