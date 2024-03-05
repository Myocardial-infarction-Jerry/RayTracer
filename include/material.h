#ifndef MATERIAL_H
#define MATERIAL_H

#include "vec3.h"

class Material {
public:
    Material();
    ~Material();

    float Ns, Ni, d;
    Vec3 Ka, Kd, Ks, Ke;
    float illum;
};

#endif