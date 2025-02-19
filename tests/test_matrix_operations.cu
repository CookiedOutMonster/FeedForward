#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "MatrixOperations.h"

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
