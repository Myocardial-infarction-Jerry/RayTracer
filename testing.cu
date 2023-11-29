#include <cuda_runtime.h>
#include <stdio.h>

// 定义结构体
struct MyStruct {
    int a;
    float b;
    // 其他成员变量
};

__global__ void test(MyStruct *devData) {
    printf("a = %d, b = %f\n", devData->a, devData->b);
    devData->a++;
    printf("a = %d, b = %f\n", devData->a, devData->b);
}

int main() {
    // 创建并填充结构体实例
    MyStruct myData;
    myData.a = 10;
    myData.b = 3.14f;
    // 填充其他成员变量

    // 在设备上分配内存
    MyStruct *devData;
    cudaMalloc((void **)&devData, sizeof(MyStruct));

    // 将结构体从主机内存复制到设备内存
    cudaMemcpy(devData, &myData, sizeof(MyStruct), cudaMemcpyHostToDevice);

    // 在设备上使用复制的结构体数据
    test << <1, 1 >> > (devData);
    cudaDeviceSynchronize();

    // 释放设备内存
    cudaFree(devData);

    return 0;
}