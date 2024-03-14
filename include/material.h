#ifndef MATERIAL_H
#define MATERIAL_H

#include "vec3.h"

#include <string>

class Material {
public:
    Material();
    ~Material();

    std::string name;
    float Ns, Ni, d;
    Vec3 Ka, Kd, Ks, Ke;
    float illum;
};

#endif