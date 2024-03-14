#include "render.h"

#include "interval.h"

#include <thread>
#include <vector>
#include <iostream>

void Render::render(const Scene &scene, const Camera &camera, std::vector<Vec3> &image) {
    // int numWorkers = std::thread::hardware_concurrency(); // 获取可用的硬件并发线程数
    int numWorkers = 1; // 将并发线程数设置为 4

    std::vector<std::thread> workers; // 存储所有worker线程的向量
    std::vector<Ray> rayList = camera.getRayList(); // 存储所有ray的向量

    // 为每个worker线程分配任务
    for (int i = 0; i < numWorkers; ++i) {
        std::vector<Ray> raySubList(rayList.begin() + i * rayList.size() / numWorkers, rayList.begin() + (i + 1) * rayList.size() / numWorkers);
        workers.push_back(std::thread(Render::renderWorker, std::ref(scene), std::ref(raySubList), std::ref(image)));
    }

    // 等待所有worker线程完成
    for (auto &worker : workers) {
        worker.join();
    }
}

void Render::renderWorker(const Scene &scene, std::vector<Ray> rayList, std::vector<Vec3> &image) {
    for (unsigned int i = 0; i < rayList.size(); ++i) {
        Vec3 color;
        renderKernel(scene, rayList[i], color, 0);
        image[rayList[i].source] = image[rayList[i].source] + color;
    }
}

void Render::renderKernel(const Scene &scene, const Ray &ray, Vec3 &color, int depth) {
    // std::cerr << ray.origin << std::endl;
    // std::cerr << ray.direction << std::endl;
    for (auto &entity : scene.entitiesList) {
        for (auto &fragment : entity.fragmentsList) {
            hitRecord record = fragment.hit(ray);
            if (record.hit)
                color = color + Vec3(0.7, 0.7, 0.7);
        }
    }
}