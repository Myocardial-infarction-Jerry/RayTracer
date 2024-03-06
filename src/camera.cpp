#include "camera.h"

Camera::Camera() :position(0, 0, 0), direction(0, 0, 1), up(0, 1, 0), fov(60), lens(0), width(1920), height(1080), SPP(1) {}
Camera::~Camera() {}

std::vector<Ray> Camera::getRayList() const {
    std::vector<Ray> rayList(width * height * SPP);
    Vec3 horizontal = direction.cross(up).normalize() * 2 * tan(fov / 2);
    Vec3 vertical = up.normalize() * 2 * tan(fov / 2);
    Vec3 lowerLeftCorner = position - horizontal / 2 - vertical / 2 - direction;
    for (unsigned int i = 0; i < height; ++i) {
        for (unsigned int j = 0; j < width; ++j) {
            for (unsigned int k = 0; k < SPP; ++k) {
                float u = (j + (lens ? randomFloat() : 0)) / width;
                float v = (i + (lens ? randomFloat() : 0)) / height;
                rayList[i * width * SPP + j * SPP + k] = Ray(position, lowerLeftCorner + u * horizontal + v * vertical - position);
                rayList[i * width * SPP + j * SPP + k].source = i * width + j;
            }
        }
    }
    return rayList;
}

std::vector<Vec3> Camera::getImage() const {
    std::vector<Vec3> image(width * height, Vec3(0, 0, 0));
    return image;
}