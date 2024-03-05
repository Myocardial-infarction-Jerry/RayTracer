#ifndef SCENE_H
#define SCENE_H

#include <vector>

#include "vec3.h"
#include "entity.h"

class Scene {
public:
    Scene();
    ~Scene();

    void addEntity(Entity entity);

    std::vector<Entity> entitiesList;
    Vec3 backgroundColor;
};

#endif