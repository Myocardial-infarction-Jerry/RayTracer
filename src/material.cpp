#include "material.h"

Material::Material() :tinyobj::material_t() {}
Material::Material(const tinyobj::material_t &material) :tinyobj::material_t(material) {}
