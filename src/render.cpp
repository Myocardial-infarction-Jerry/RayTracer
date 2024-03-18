#include "render.h"

#include "interval.h"

#include <thread>
#include <vector>
#include <iostream>

void Render::render(const Scene &scene, const Camera &camera, std::vector<Vec3> &image) {
    int numWorkers = std::thread::hardware_concurrency(); // 获取可用的硬件并发线程数

    std::cerr << "Rendering with " << numWorkers << " processes" << std::endl;

    std::vector<std::thread> workers; // 存储所有worker线程的向量
    unsigned long long rayCount = camera.width * camera.height * camera.SPP;

    // 为每个worker线程分配任务
    int threads = (rayCount + rayPerWorker - 1) / rayPerWorker;
    for (int i = 0; i < threads; ++i) {
        std::cerr << "\rCurrent: " << i << "/" << threads << " ";

        if (i % numWorkers == 0) {
            for (auto &worker : workers)
                worker.join();
            workers.clear();
        }

        workers.push_back(std::thread(Render::renderWorker, std::ref(scene), std::ref(camera), i * Render::rayPerWorker, std::ref(image)));
    }

    for (auto &worker : workers)
        worker.join();

    std::cerr << "\rDealing with SPP...          ";

    for (auto &pixel : image)
        pixel = pixel / camera.SPP;

    std::cerr << "\rDone!                        " << std::endl;
}

void Render::renderWorker(const Scene &scene, const Camera &camera, unsigned long long start, std::vector<Vec3> &image) {
    std::vector<Ray> rayList = camera.getRayList(start, Render::rayPerWorker);

    for (unsigned int i = 0; i < rayList.size(); ++i) {
        Vec3 color;
        renderKernel(scene, rayList[i], color, 0);
        image[rayList[i].source] = image[rayList[i].source] + color;
    }
}

void Render::renderKernel(const Scene &scene, const Ray &ray, Vec3 &color, int depth) {
    for (auto &entity : scene.entitiesList) {
        for (auto &fragment : entity.fragmentsList) {
            hitRecord record = fragment.hit(ray);
            if (record.hit)
                color = color + Vec3(0.7, 0.7, 0.7);
        }
    }
}