#include "scene.h"
#include "camera.h"
#include "render.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <chrono> // Include the chrono library for time measurement

int main(int argc, char const *argv[]) {
    int renderMode = 1;

    Scene scene;
    Camera camera;
    camera.position = Vec3(278, 278, -900);
    camera.direction = (Vec3(278, 278, 0) - camera.position).normalize();
    Entity entity; entity.load("models/cornell_box.obj");
    scene.addEntity(entity);

    auto image = camera.getImage();
    auto startTime = std::chrono::high_resolution_clock::now();
    Render::render(scene, camera, image);
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

    cv::Mat displayImage(camera.height, camera.width, CV_8UC3);
    for (unsigned int i = 0; i < camera.height; ++i) {
        for (unsigned int j = 0; j < camera.width; ++j) {
            auto color = image[i * camera.width + j];
            displayImage.at<cv::Vec3b>(i, j) = cv::Vec3b(static_cast<int>(color[0] * 255), static_cast<int>(color[1] * 255), static_cast<int>(color[2] * 255));
        }
    }
    cv::putText(displayImage, "Render Time: " + std::to_string(duration) + " ms", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    cv::imwrite("image.png", displayImage);

    if (renderMode == 1) {
        cv::imshow("Ray Tracer", displayImage);
        cv::waitKey(0);
    }

    return 0;
}
