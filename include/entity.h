#ifndef ENTITY_H
#define ENTITY_H

#include <vector>
#include <string>

#include "vec3.h"
#include "fragment.h"
#include "material.h"

class Entity {
public:
    Entity();
    ~Entity();

    void load(std::string path);

    std::vector<Fragment> fragmentsList;
    std::string name;


    // static Entity ball(Vec3 position, float radius, unsigned int rings, unsigned int sectors, Material *_material);
    // static Entity rectangle(Vec3 position, Vec3 size, Material *_material);
    // static Entity cube(Vec3 position, Vec3 size, Material *_material);
};

#endif