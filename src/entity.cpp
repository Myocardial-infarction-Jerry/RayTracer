#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "entity.h"
#include "vec3.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

Entity::Entity() :name("None") {}
Entity::~Entity() {}

void Entity::load(const char *inputFile) {
    tinyobj::ObjReaderConfig readerConfig; readerConfig.triangulate = true;
    readerConfig.mtl_search_path = std::__fs::filesystem::path(inputFile).parent_path().string();

    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(inputFile, readerConfig)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader: " << reader.Error();
        }
        exit(1);
    }

    if (!reader.Warning().empty()) {
        std::cout << "TinyObjReader: " << reader.Warning();
    }

    auto &attrib = reader.GetAttrib();
    auto &shapes = reader.GetShapes();
    std::vector<std::shared_ptr<Material>> materials;
    for (auto &material : reader.GetMaterials())
        materials.push_back(std::make_shared<Material>(material));

    // Loop over shapes
    for (size_t s = 0; s < shapes.size(); s++) {
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);
            Fragment fragment;

            // Loop over vertices in the face.
            for (size_t v = 0; v < fv; v++) {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];
                fragment.vertices[v] = Vec3(vx, vy, vz);

                // Check if `normal_index` is zero or positive. negative = no normal data
                if (idx.normal_index >= 0) {
                    tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
                    tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
                    tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];
                    fragment.normals[v] = Vec3(nx, ny, nz);
                }

                // Check if `texcoord_index` is zero or positive. negative = no texcoord data
                if (idx.texcoord_index >= 0) {
                    tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
                    tinyobj::real_t ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
                    fragment.uvs[v] = Vec3(tx, ty, 0);
                }
            }
            index_offset += fv;

            fragment.material = materials[shapes[s].mesh.material_ids[f]];

            fragmentsList.push_back(fragment);
        }
    }

    std::cerr << "Load model " << inputFile << std::endl;
}
