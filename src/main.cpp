#include "scene.h"
#include "camera.h"
#include "render.h"

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <chrono> // Include the chrono library for time measurement

int main(int argc, char const *argv[]) {
    // Scene scene;
    // Camera camera;
    // auto image = camera.getImage();

    // // Measure the start time
    // auto startTime = std::chrono::high_resolution_clock::now();

    // Render::render(scene, camera, image);

    // // Measure the end time
    // auto endTime = std::chrono::high_resolution_clock::now();

    // // Calculate the duration in milliseconds
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

    // cv::Mat displayImage(camera.height, camera.width, CV_8UC3);
    // for (unsigned int i = 0; i < camera.height; ++i) {
    //     for (unsigned int j = 0; j < camera.width; ++j) {
    //         auto color = image[i * camera.width + j];
    //         displayImage.at<cv::Vec3b>(i, j) = cv::Vec3b(static_cast<int>(color.x * 255), static_cast<int>(color.y * 255), static_cast<int>(color.z * 255));
    //     }
    // }

    // // Display the render time in the top-left corner of the image
    // cv::putText(displayImage, "Render Time: " + std::to_string(duration) + " ms", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);

    // cv::imwrite("image.png", displayImage); // Save the image as image.png
    // cv::imshow("Ray Tracer", displayImage);
    // cv::waitKey(0);

    Entity cube;
    cube.load("models/cube.obj");

    return 0;
}
