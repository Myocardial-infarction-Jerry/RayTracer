#ifndef MATERIAL_H
#define MATERIAL_H

#include "tiny_obj_loader.h"
#include "vec3.h"

class Material : public tinyobj::material_t {
public:
    Material();
    Material(const tinyobj::material_t &material);
    Vec3 getColor(const Vec3 &p) const;
};

#endif