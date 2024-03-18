#include "camera.h"

Camera::Camera() :position(0, 0, 0), direction(0, 0, 1), up(0, 1, 0), fov(M_PI / 3), lens(1), width(1920), height(1080), SPP(100) {}
Camera::~Camera() {}

std::vector<Ray> Camera::getRayList(unsigned long long start, int size) const {
    std::vector<Ray> rayList(size);
    Vec3 horizontal = direction.cross(up).normalize() * 2 * tan(fov / 2) * lens;
    Vec3 vertical = up.normalize() * horizontal.length() / width * height;
    Vec3 lowerLeftCorner = position - horizontal / 2 - vertical / 2 + direction.normalize() * lens;

    int i = start / width / SPP;
    int j = start % (width * SPP) / SPP;
    int k = start % SPP;

    for (int t = 0; t < size; ++t) {
        float u = (randomFloat() + j) / width;
        float v = (randomFloat() + i) / height;
        rayList[t] = Ray(position, lowerLeftCorner + u * horizontal + v * vertical - position);
        rayList[t].source = i * width + j;

        ++k;
        if (k == SPP)
            ++j, k = 0;
        if (j == width)
            ++i, j = 0;
        if (i == height)
            break;
    }

    return rayList;
}

std::vector<Vec3> Camera::getImage() const {
    std::vector<Vec3> image(width * height, Vec3(0, 0, 0));
    return image;
}