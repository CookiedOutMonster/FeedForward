#include <cuda_runtime.h>

#include <iostream>

#include "MatrixOperations.h"

// TODO this might create a data race...
__global__ void MatrixAddition(float *d_arr_a, float *d_arr_b, float *d_result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // be wary of potential warp divergence...
    if (idx < size) {
        // three global memory access...
        d_result[idx] = d_arr_a[idx] + d_arr_b[idx];
    }
}

// TODO create for matrix multiplication
// need to solve how this looks like for contigous 1D memory
// the idea I had last night is that i just need to know the length of the array (width)
__global__ void MatrixMultiplication(float *d_W, float *d_A, float *d_O, int prevNumNeurons) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;  // Unique thread ID

    if (row < gridDim.x * blockDim.x) {  // Bounds check
        float sum = 0.0f;
        for (int col = 0; col < prevNumNeurons; col++) {
            sum += d_W[row * prevNumNeurons + col] * d_A[col];
        }
        printf("sum = %f\n", sum);
        d_O[row] = sum;
    }
}

// this will need to change to handle
__host__ __device__ int DotProduct(double *leftHandSideVector, double *rightHandSideVector, int vectorSize) {
    int scalar = 0;
    for (int i = 0; i < vectorSize; i++) {
        scalar += leftHandSideVector[i] * rightHandSideVector[i];
    }

    return scalar;
}

__global__ void DotProductKernel(double *leftHandSideVector, double *rightHandSideVector, int *result, int size) {
    *result = DotProduct(leftHandSideVector, rightHandSideVector, size);
}

// Sigmoid activation kernel
__global__ void sigmoidActivationKernel(float *neurons, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        // Call the static activation function directly
        neurons[idx] = 1.0 / (1.0 + exp(-neurons[idx]));
    }
}

// ReLU activation kernel
__global__ void reluActivationKernel(double *neurons, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        // Call the static activation function directly
        neurons[idx] = max(0.0, neurons[idx]);
    }
}