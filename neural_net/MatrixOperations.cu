#include <cuda_runtime.h>

#include <iostream>

#include "MatrixOperations.h"

// TODO create a method for matrix addition
__global__ void MatrixAddition(double *d_arr_a, double *d_arr_b, double *d_result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // be wary of potential warp divergence...
    if (idx < size) {
        // three global memory access...
        d_result[idx] = d_arr_a[idx] + d_arr_b[idx];
    }
}

// TODO create for matrix multiplication

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
