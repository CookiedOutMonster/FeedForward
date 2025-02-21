#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstring>  // For memset

#include "GPUConfig.h"
#include "MatrixOperations.h"

float cpuSigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

double cpuReLU(double x) {
    return fmax(0.0, x);
}

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

TEST(ActivationFunctions, SigmoidActivation_CPU_vs_GPU) {
    std::vector<int> test_sizes = {10, 100, 500, 1000};

    for (int size : test_sizes) {
        printf("\nTesting Sigmoid activation with array size: %d\n", size);

        // Initialize host arrays
        float *h_input = new float[size];
        float *h_expected = new float[size];
        float *h_result = new float[size];

        // Initialize test data with a range of values (negative, zero, positive)
        for (int i = 0; i < size; i++) {
            h_input[i] = -5.0f + (10.0f * i / size);  // Range from -5 to 5
            h_expected[i] = cpuSigmoid(h_input[i]);   // Calculate expected results on CPU
        }

        // Allocate device memory
        float *d_neurons;
        cudaMalloc(&d_neurons, size * sizeof(float));

        // Copy input to device
        cudaMemcpy(d_neurons, h_input, size * sizeof(float), cudaMemcpyHostToDevice);

        // Launch kernel
        int numThreads = THREADS_PER_BLOCK;
        int numBlocks = calculateBlocks(size);
        sigmoidActivationKernel<<<numBlocks, numThreads>>>(d_neurons, size);
        cudaDeviceSynchronize();

        // Check for kernel execution errors
        cudaError_t err = cudaGetLastError();
        ASSERT_EQ(cudaSuccess, err) << "Kernel execution failed: " << cudaGetErrorString(err);

        // Copy result back to host
        cudaMemcpy(h_result, d_neurons, size * sizeof(float), cudaMemcpyDeviceToHost);

        // Verify results with epsilon for floating point comparison
        const float epsilon = 1e-5f;
        for (int i = 0; i < size; i++) {
            EXPECT_NEAR(h_expected[i], h_result[i], epsilon)
                << "Sigmoid mismatch at index " << i << " for input " << h_input[i];
        }

        // Cleanup
        cudaFree(d_neurons);
        delete[] h_input;
        delete[] h_expected;
        delete[] h_result;
    }
}

TEST(ActivationFunctions, ReluActivation_CPU_vs_GPU) {
    std::vector<int> test_sizes = {10, 100, 500, 1000};

    for (int size : test_sizes) {
        printf("\nTesting ReLU activation with array size: %d\n", size);

        // Initialize host arrays
        double *h_input = new double[size];
        double *h_expected = new double[size];
        double *h_result = new double[size];

        // Initialize test data with a range of values (negative, zero, positive)
        for (int i = 0; i < size; i++) {
            h_input[i] = -5.0 + (10.0 * i / size);  // Range from -5 to 5
            h_expected[i] = cpuReLU(h_input[i]);    // Calculate expected results on CPU
        }

        // Allocate device memory
        double *d_neurons;
        cudaMalloc(&d_neurons, size * sizeof(double));

        // Copy input to device
        cudaMemcpy(d_neurons, h_input, size * sizeof(double), cudaMemcpyHostToDevice);

        // Launch kernel
        int numThreads = THREADS_PER_BLOCK;
        int numBlocks = calculateBlocks(size);
        reluActivationKernel<<<numBlocks, numThreads>>>(d_neurons, size);
        cudaDeviceSynchronize();

        // Check for kernel execution errors
        cudaError_t err = cudaGetLastError();
        ASSERT_EQ(cudaSuccess, err) << "Kernel execution failed: " << cudaGetErrorString(err);

        // Copy result back to host
        cudaMemcpy(h_result, d_neurons, size * sizeof(double), cudaMemcpyDeviceToHost);

        // Verify results
        const double epsilon = 1e-10;
        for (int i = 0; i < size; i++) {
            EXPECT_NEAR(h_expected[i], h_result[i], epsilon)
                << "ReLU mismatch at index " << i << " for input " << h_input[i];
        }

        // Cleanup
        cudaFree(d_neurons);
        delete[] h_input;
        delete[] h_expected;
        delete[] h_result;
    }
}

// Test specific edge cases for Sigmoid
TEST(ActivationFunctions, SigmoidEdgeCases) {
    const int size = 5;
    float h_input[size] = {-100.0f, -10.0f, 0.0f, 10.0f, 100.0f};
    float h_expected[size];
    float h_result[size];

    // Calculate expected values
    for (int i = 0; i < size; i++) {
        h_expected[i] = cpuSigmoid(h_input[i]);
    }

    // Allocate device memory
    float *d_neurons;
    cudaMalloc(&d_neurons, size * sizeof(float));
    cudaMemcpy(d_neurons, h_input, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    sigmoidActivationKernel<<<1, size>>>(d_neurons, size);
    cudaDeviceSynchronize();

    // Copy results back
    cudaMemcpy(h_result, d_neurons, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify results
    const float epsilon = 1e-5f;
    for (int i = 0; i < size; i++) {
        EXPECT_NEAR(h_expected[i], h_result[i], epsilon) << "Sigmoid edge case mismatch for input " << h_input[i];
    }

    cudaFree(d_neurons);
}

// Test specific edge cases for ReLU
TEST(ActivationFunctions, ReluEdgeCases) {
    const int size = 5;
    double h_input[size] = {-100.0, -0.001, 0.0, 0.001, 100.0};
    double h_expected[size];
    double h_result[size];

    // Calculate expected values
    for (int i = 0; i < size; i++) {
        h_expected[i] = cpuReLU(h_input[i]);
    }

    // Allocate device memory
    double *d_neurons;
    cudaMalloc(&d_neurons, size * sizeof(double));
    cudaMemcpy(d_neurons, h_input, size * sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel
    reluActivationKernel<<<1, size>>>(d_neurons, size);
    cudaDeviceSynchronize();

    // Copy results back
    cudaMemcpy(h_result, d_neurons, size * sizeof(double), cudaMemcpyDeviceToHost);

    // Verify results
    const double epsilon = 1e-10;
    for (int i = 0; i < size; i++) {
        EXPECT_NEAR(h_expected[i], h_result[i], epsilon) << "ReLU edge case mismatch for input " << h_input[i];
    }

    cudaFree(d_neurons);
}

// Performance test for large arrays
TEST(ActivationFunctions, SigmoidPerformance) {
    const int size = 10000000;  // 10M elements

    // Allocate host memory
    float *h_input = new float[size];

    // Initialize with random values between -5 and 5
    for (int i = 0; i < size; i++) {
        h_input[i] = -5.0f + (rand() / (float)RAND_MAX) * 10.0f;
    }

    // Allocate device memory
    float *d_neurons;
    cudaMalloc(&d_neurons, size * sizeof(float));
    cudaMemcpy(d_neurons, h_input, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel and time it
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int numThreads = THREADS_PER_BLOCK;
    int numBlocks = calculateBlocks(size);

    cudaEventRecord(start);
    sigmoidActivationKernel<<<numBlocks, numThreads>>>(d_neurons, size);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Sigmoid activation on %d elements took %.3f ms\n", size, milliseconds);

    // Clean up
    cudaFree(d_neurons);
    delete[] h_input;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // No assertions, this is just a performance measurement
    SUCCEED();
}