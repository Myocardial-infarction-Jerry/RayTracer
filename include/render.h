#ifndef RENDER_H
#define RENDER_H

#include "ray.h"
#include "scene.h"
#include "camera.h"

namespace Render {
    // int maxDepth = 50;

    void render(const Scene &scene, const Camera &camera, std::vector<Vec3> &image);
    void renderWorker(const Scene &scene, std::vector<Ray> rayList, std::vector<Vec3> &image);
    void renderKernel(const Scene &scene, const Ray &ray, Vec3 &color, int depth = 0);
} // namespace Render

#endif