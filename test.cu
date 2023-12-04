#include <iostream>
#include <cuda_runtime.h>

class DynamicArray {
public:
    __host__ __device__ DynamicArray() : size(0), capacity(0), data(nullptr) {}

    __host__ __device__ ~DynamicArray() {
        if (data) {
            delete[] data;
        }
    }

    __host__ __device__ void push_back(int value) {
        if (size == capacity) {
            expand();
        }
        data[size++] = value;
    }

    __host__ __device__ int operator[](int index) const {
        return data[index];
    }

    __host__ __device__ int &operator[](int index) {
        return data[index];
    }

    __host__ __device__ int getSize() const {
        return size;
    }

private:
    __host__ __device__ void expand() {
        int newCapacity = capacity == 0 ? 1 : capacity * 2;
        int *newData = new int[newCapacity];
        if (data) {
            memcpy(newData, data, size * sizeof(int));
            delete[] data;
        }
        data = newData;
        capacity = newCapacity;
    }

    int size;
    int capacity;
    int *data;
};

__global__ void kernel(DynamicArray *arr) {
    arr->push_back(1);
    arr->push_back(2);
    arr->push_back(3);
}

int main() {
    DynamicArray *arr;
    cudaMalloc((void **)&arr, sizeof(DynamicArray));

    kernel << <1, 1 >> > (arr);

    DynamicArray hostArr;
    cudaMemcpy(&hostArr, arr, sizeof(DynamicArray), cudaMemcpyDeviceToHost);

    for (int i = 0; i < hostArr.getSize(); i++) {
        std::cout << hostArr[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(arr);

    return 0;
}