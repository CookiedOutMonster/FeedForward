#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "GPUConfig.h"
#include "MatrixOperations.h"

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

        double *h_arr_a = new double[size];
        double *h_arr_b = new double[size];
        double *h_result = new double[size];

        // Initialize test data
        for (int i = 0; i < size; i++) {
            h_arr_a[i] = static_cast<double>(i);
            h_arr_b[i] = static_cast<double>(i * 2);
        }

        double *d_arr_a, *d_arr_b, *d_result;
        cudaMalloc(&d_arr_a, size * sizeof(double));
        cudaMalloc(&d_arr_b, size * sizeof(double));
        cudaMalloc(&d_result, size * sizeof(double));

        cudaMemcpy(d_arr_a, h_arr_a, size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_arr_b, h_arr_b, size * sizeof(double), cudaMemcpyHostToDevice);

        int blocks = calculateBlocks(size);

        // Launch kernel
        MatrixAddition<<<blocks, THREADS_PER_BLOCK>>>(d_arr_a, d_arr_b, d_result, size);
        cudaDeviceSynchronize();

        // Copy result back
        cudaMemcpy(h_result, d_result, size * sizeof(double), cudaMemcpyDeviceToHost);

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
