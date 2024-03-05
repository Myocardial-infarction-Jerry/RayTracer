#include "scene.h"

Scene::Scene() :backgroundColor(0, 0, 0) {}
Scene::~Scene() {}

void Scene::addEntity(Entity entity) { entitiesList.push_back(entity); }