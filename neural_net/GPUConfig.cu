#include <cuda_runtime.h>

#include <iostream>

#include "GPUConfig.h"

int THREADS_PER_BLOCK = 256;  // Default value, will be updated

void initGPUConfig() {
    cudaDeviceProp prop;
    int device;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    // Set THREADS_PER_BLOCK dynamically
    THREADS_PER_BLOCK = std::max(prop.maxThreadsPerBlock, 256);  // Choose a safe default (256)

    std::cout << "GPU Detected: " << prop.name << std::endl;
    std::cout << "Max Threads Per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Using THREADS_PER_BLOCK: " << THREADS_PER_BLOCK << std::endl;
}
