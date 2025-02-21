#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstring>  // For memset

#include "GPUConfig.h"
#include "MatrixOperations.h"

void MatrixVectorMultiply(const float *weights, const float *prev_activations, float *output, int curr_neurons,
                          int prev_neurons) {
    // Initialize output to zero
    std::memset(output, 0, curr_neurons * sizeof(float));

    for (int row = 0; row < curr_neurons; row++) {
        for (int col = 0; col < prev_neurons; col++) {
            output[row] += weights[row * prev_neurons + col] * prev_activations[col];
        }
    }
}

int calculateBlocks(int size) {
    return (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
}

TEST(DotProduct, DotProduct_5vector_x_5vector_CPU) {
    // Dot product of two 5x1 vectors, output should be 70

    int expectedValue = 70;

    double vector_1[5] = {1, 2, 3, 4, 5};
    double vector_2[5] = {2, 3, 4, 5, 6};
    int size = 5;

    int actualValue = DotProduct(vector_1, vector_2, size);

    ASSERT_EQ(expectedValue, actualValue);
}

TEST(DotProduct, DotProduct_5vector_x_5vector_GPU) {
    int expectedValue = 70;

    double vector_1[5] = {1, 2, 3, 4, 5};
    double vector_2[5] = {2, 3, 4, 5, 6};
    int size = 5;

    // Allocate memory on the device
    double *d_vector_1, *d_vector_2;
    int *d_result;
    cudaMalloc(&d_vector_1, size * sizeof(double));
    cudaMalloc(&d_vector_2, size * sizeof(double));
    cudaMalloc(&d_result, sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_vector_1, vector_1, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector_2, vector_2, size * sizeof(double), cudaMemcpyHostToDevice);

    // Launch the kernel (using 1 thread block with 1 thread)
    DotProductKernel<<<1, 1>>>(d_vector_1, d_vector_2, d_result, size);

    // Copy the result from device to host
    int actualValue;
    cudaMemcpy(&actualValue, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    // Free the device memory
    cudaFree(d_vector_1);
    cudaFree(d_vector_2);
    cudaFree(d_result);

    // Assert the result is as expected
    ASSERT_EQ(expectedValue, actualValue);
}

// Test adding two matrixes together
TEST(MatrixAddition, TestAddingTwoVectorsTogether) {
    std::vector<int> test_sizes = {10, 100, 500, 1000, 5000, 10000};  // Array sizes to test

    for (int size : test_sizes) {
        printf("\nTesting array size: %d\n", size);

        float *h_arr_a = new float[size];
        float *h_arr_b = new float[size];
        float *h_result = new float[size];

        // Initialize test data
        for (int i = 0; i < size; i++) {
            h_arr_a[i] = static_cast<float>(i);
            h_arr_b[i] = static_cast<float>(i * 2);
        }

        float *d_arr_a, *d_arr_b, *d_result;
        cudaMalloc(&d_arr_a, size * sizeof(float));
        cudaMalloc(&d_arr_b, size * sizeof(float));
        cudaMalloc(&d_result, size * sizeof(float));

        cudaMemcpy(d_arr_a, h_arr_a, size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_arr_b, h_arr_b, size * sizeof(float), cudaMemcpyHostToDevice);

        int blocks = calculateBlocks(size);

        // Launch kernel
        MatrixAddition<<<blocks, THREADS_PER_BLOCK>>>(d_arr_a, d_arr_b, d_result, size);
        cudaDeviceSynchronize();

        // Copy result back
        cudaMemcpy(h_result, d_result, size * sizeof(float), cudaMemcpyDeviceToHost);

        // Verify results
        for (int i = 0; i < size; i++) {
            EXPECT_DOUBLE_EQ(h_result[i], h_arr_a[i] + h_arr_b[i]) << "Mismatch at index " << i;
        }

        // Cleanup
        cudaFree(d_arr_a);
        cudaFree(d_arr_b);
        cudaFree(d_result);
        delete[] h_arr_a;
        delete[] h_arr_b;
        delete[] h_result;
    }
}

TEST(MatrixVectorMultiplyTest, ComputesCorrectOutput) {
    int numNeurons = 3;
    int numPrevNeurons = 5;

    // Define test input
    float weights[numNeurons * numPrevNeurons] = {0.1,  0.87, 0.12, 0.55, 0.01, 0.23, 0.76, 0.44,
                                                  0.98, 0.33, 0.67, 0.15, 0.89, 0.42, 0.57};  // 3 rows, 5 columns

    float prev_activations[numPrevNeurons] = {2.5, 1.212, 7.5, 0.5, 3.1};  // 5 elements
    float exp_output[numNeurons] = {0.0f};                                 // Expected output

    // Compute expected output on CPU
    MatrixVectorMultiply(weights, prev_activations, exp_output, numNeurons, numPrevNeurons);

    // Allocate CUDA memory
    float *d_weights, *d_prev_activations, *d_result;
    cudaMalloc(&d_weights, (numNeurons * numPrevNeurons) * sizeof(float));
    cudaMalloc(&d_prev_activations, numPrevNeurons * sizeof(float));
    cudaMalloc(&d_result, numNeurons * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_weights, weights, (numNeurons * numPrevNeurons) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prev_activations, prev_activations, numPrevNeurons * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, numNeurons * sizeof(float));  // Initialize to zero

    // Setup CUDA kernel
    int numThreads = min(THREADS_PER_BLOCK, numNeurons);
    int numBlocks = calculateBlocks(numNeurons);

    // Execute CUDA kernel
    MatrixMultiplication<<<numBlocks, numThreads>>>(d_weights, d_prev_activations, d_result, numPrevNeurons);
    cudaDeviceSynchronize();

    // Copy result back to host
    float h_result[numNeurons] = {0.0f};  // Host output array
    cudaMemcpy(h_result, d_result, numNeurons * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < numNeurons; i++) {
        EXPECT_FLOAT_EQ(h_result[i], exp_output[i]) << "Mismatch at index " << i;
    }

    // Cleanup
    cudaFree(d_weights);
    cudaFree(d_prev_activations);
    cudaFree(d_result);
}