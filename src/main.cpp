#include "scene.h"
#include "camera.h"
#include "render.h"

#include <iostream>
#include <vector>

int main(int argc, char const *argv[]) {
    Scene scene;
    scene.addEntity(Entity::cube(Vec3(-1, 0, -5), Vec3(2, 2, 2), nullptr));
    Camera camera;
    auto image = camera.getImage();
    Render::render(scene, camera, image);

    std::cout << "P3\n" << camera.width << " " << camera.height << "\n255\n";
    for (unsigned int i = 0; i < camera.height; ++i) {
        for (unsigned int j = 0; j < camera.width; ++j) {
            auto color = image[i * camera.width + j];
            std::cout << static_cast<int>(color.x * 255) << " " << static_cast<int>(color.y * 255) << " " << static_cast<int>(color.z * 255) << "\n";
        }
    }

    return 0;
}
