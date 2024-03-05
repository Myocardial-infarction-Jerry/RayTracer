#ifndef FRAGMENT_H
#define FRAGMENT_H

#include "vec3.h"
#include "material.h"

class Fragment {
public:
    Fragment();

    Vec3 vertices[3];
    Vec3 uvs[3];
    Vec3 normals;
    Material *material;
};

#endif