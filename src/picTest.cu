#include "Camera.cuh"
#include "Scene.cuh"
#include "Render.cuh"

int main(int argc, char const *argv[]) {
    Camera camera;
    Scene scene;
    int width = 800;
    int height = 600;
    int samples = 100;
    Vec3 *pixels = new Vec3[width * height];

    Render::render(camera, scene, width, height, samples, pixels);
    Render::saveImage("image.png", width, height, pixels);

    return 0;
}
