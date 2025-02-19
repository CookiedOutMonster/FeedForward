#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <iostream>

#include "../neural_net/LayerGPU.h"

__global__ void testMemoryKernel(const float* expected, const float* actual, int* results, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        results[idx] = (expected[idx] == actual[idx]) ? 1 : 0;
    }
}

void testMemory(const float* h_expected, const float* h_actual, size_t size) {
    // Allocate device memory
    float* d_expected;
    float* d_actual;
    int* d_results;
    int* h_results = new int[size];

    cudaMalloc((void**)&d_expected, size * sizeof(float));
    cudaMalloc((void**)&d_actual, size * sizeof(float));
    cudaMalloc((void**)&d_results, size * sizeof(int));

    // Copy host data to device
    cudaMemcpy(d_expected, h_expected, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_actual, h_actual, size * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel launch configuration
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    testMemoryKernel<<<blocks, threadsPerBlock>>>(d_expected, d_actual, d_results, size);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_results, d_results, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Validate results
    for (size_t i = 0; i < size; i++) {
        EXPECT_EQ(h_results[i], 1) << "Mismatch at index " << i;
    }

    // Cleanup
    delete[] h_results;
    cudaFree(d_expected);
    cudaFree(d_actual);
    cudaFree(d_results);
}

TEST(LayerGPUTest, BasicAllocation) {
    LayerGPU* layer = createLayer(2, 3);

    ASSERT_NE(layer, nullptr);
    ASSERT_NE(layer->d_neuronActivations, nullptr);
    ASSERT_NE(layer->d_weights, nullptr);
    ASSERT_NE(layer->d_biases, nullptr);

    freeLayerGPU(layer);
}

TEST(LayerGPUTest, LargeLayerAllocation) {
    LayerGPU* layer = createLayer(10000, 10000);

    ASSERT_NE(layer, nullptr);
    ASSERT_NE(layer->d_neuronActivations, nullptr);
    ASSERT_NE(layer->d_weights, nullptr);
    ASSERT_NE(layer->d_biases, nullptr);

    freeLayerGPU(layer);
}

TEST(LayerGPUTest, UpdatingLayerTest) {
    LayerGPU* layer = createLayer(5, 0);

    float activations[] = {1, 2, 3, 4, 5};
    size_t numOfActivations = 5;

    updateLayerActivations(layer, activations, numOfActivations);

    // Verify the layer's neuronActivations match expected activations
    testMemory(activations, layer->d_neuronActivations, numOfActivations);

    freeLayerGPU(layer);
}

TEST(LayerGPUTest, UpdateLayerThrowsOnSizeMismatch) {
    LayerGPU* layer = createLayer(5, 0);  // Layer expects 5 neurons

    float activations[3] = {1, 2, 3};  // Only 3 activations given (mismatch)
    size_t numOfActivations = 3;

    EXPECT_THROW(updateLayerActivations(layer, activations, numOfActivations),
                 std::out_of_range  // Expected exception type
    );

    freeLayerGPU(layer);
}