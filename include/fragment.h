#ifndef FRAGMENT_H
#define FRAGMENT_H

#include "vec3.h"
#include "ray.h"
#include "material.h"

class hitRecord {
public:
    bool hit;
    float t;
    Vec3 p;
    Vec3 normal;
    std::shared_ptr<Material> material;
};

class Fragment {
public:
    Fragment();

    hitRecord hit(const Ray &ray) const;

    Vec3 vertices[3];
    Vec3 uvs[3];
    Vec3 normals[3];
    std::shared_ptr<Material> material;
};

#endif