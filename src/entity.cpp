#include <fstream>

#include "vec3.h"
#include "entity.h"

Entity::Entity() :name("None") {}
Entity::~Entity() {}

Entity Entity::ball(Vec3 position, float radius, unsigned int rings, unsigned int sectors, Material *_material) { return Entity(); }

Entity Entity::rectangle(Vec3 position, Vec3 size, Material *_material) {
    std::vector<Vec3> vertices;
    Vec3 normal;
    if (size.z == 0) {
        vertices.push_back(position);
        vertices.push_back(position + Vec3(size.x, 0, 0));
        vertices.push_back(position + Vec3(size.x, size.y, 0));
        vertices.push_back(position + Vec3(0, size.y, 0));
        normal = Vec3(0, 0, 1);
    }
    else if (size.y == 0) {
        vertices.push_back(position);
        vertices.push_back(position + Vec3(size.x, 0, 0));
        vertices.push_back(position + Vec3(size.x, 0, size.z));
        vertices.push_back(position + Vec3(0, 0, size.z));
        normal = Vec3(0, 1, 0);
    }
    else if (size.x == 0) {
        vertices.push_back(position);
        vertices.push_back(position + Vec3(0, size.y, 0));
        vertices.push_back(position + Vec3(0, size.y, size.z));
        vertices.push_back(position + Vec3(0, 0, size.z));
        normal = Vec3(1, 0, 0);
    }

    std::vector<Fragment> fragmentsList(2);
    for (int i = 0; i < 2; ++i) {
        fragmentsList[i].material = _material;
        fragmentsList[i].normals = normal;
    }
    fragmentsList[0].vertices[0] = vertices[0]; fragmentsList[0].vertices[1] = vertices[1]; fragmentsList[0].vertices[2] = vertices[2];
    fragmentsList[1].vertices[0] = vertices[0]; fragmentsList[1].vertices[1] = vertices[2]; fragmentsList[1].vertices[2] = vertices[3];

    Entity entity;
    entity.fragmentsList = fragmentsList;
    entity.name = "Rectangle";
    return entity;
}

Entity Entity::cube(Vec3 position, Vec3 size, Material *_material) {
    std::vector<Vec3> vertices(8);
    for (int i = 0; i < 8; ++i) {
        vertices[i] = position;
        if (i & 1) vertices[i].x += size.x;
        if (i & 2) vertices[i].y += size.y;
        if (i & 4) vertices[i].z += size.z;
    }

    std::vector<Fragment> fragmentsList(12);
    for (int i = 0; i < 12; ++i)
        fragmentsList[i].material = _material;

    fragmentsList[0].vertices[0] = vertices[0]; fragmentsList[0].vertices[1] = vertices[1]; fragmentsList[0].vertices[2] = vertices[2]; fragmentsList[0].normals = Vec3(0, 0, 1);
    fragmentsList[1].vertices[0] = vertices[0]; fragmentsList[1].vertices[1] = vertices[2]; fragmentsList[1].vertices[2] = vertices[3]; fragmentsList[1].normals = Vec3(0, 0, 1);
    fragmentsList[2].vertices[0] = vertices[4]; fragmentsList[2].vertices[1] = vertices[5]; fragmentsList[2].vertices[2] = vertices[6]; fragmentsList[2].normals = Vec3(0, 0, -1);
    fragmentsList[3].vertices[0] = vertices[4]; fragmentsList[3].vertices[1] = vertices[6]; fragmentsList[3].vertices[2] = vertices[7]; fragmentsList[3].normals = Vec3(0, 0, -1);
    fragmentsList[4].vertices[0] = vertices[0]; fragmentsList[4].vertices[1] = vertices[1]; fragmentsList[4].vertices[2] = vertices[5]; fragmentsList[4].normals = Vec3(0, 1, 0);
    fragmentsList[5].vertices[0] = vertices[0]; fragmentsList[5].vertices[1] = vertices[5]; fragmentsList[5].vertices[2] = vertices[4]; fragmentsList[5].normals = Vec3(0, 1, 0);
    fragmentsList[6].vertices[0] = vertices[2]; fragmentsList[6].vertices[1] = vertices[3]; fragmentsList[6].vertices[2] = vertices[7]; fragmentsList[6].normals = Vec3(0, -1, 0);
    fragmentsList[7].vertices[0] = vertices[2]; fragmentsList[7].vertices[1] = vertices[7]; fragmentsList[7].vertices[2] = vertices[6]; fragmentsList[7].normals = Vec3(0, -1, 0);
    fragmentsList[8].vertices[0] = vertices[0]; fragmentsList[8].vertices[1] = vertices[4]; fragmentsList[8].vertices[2] = vertices[7]; fragmentsList[8].normals = Vec3(1, 0, 0);
    fragmentsList[9].vertices[0] = vertices[0]; fragmentsList[9].vertices[1] = vertices[7]; fragmentsList[9].vertices[2] = vertices[3]; fragmentsList[9].normals = Vec3(1, 0, 0);
    fragmentsList[10].vertices[0] = vertices[1]; fragmentsList[10].vertices[1] = vertices[5]; fragmentsList[10].vertices[2] = vertices[6]; fragmentsList[10].normals = Vec3(-1, 0, 0);
    fragmentsList[11].vertices[0] = vertices[1]; fragmentsList[11].vertices[1] = vertices[6]; fragmentsList[11].vertices[2] = vertices[2]; fragmentsList[11].normals = Vec3(-1, 0, 0);

    Entity entity;
    entity.fragmentsList = fragmentsList;
    entity.name = "Cube";
    return entity;
}


void Entity::load(std::string path) {}
