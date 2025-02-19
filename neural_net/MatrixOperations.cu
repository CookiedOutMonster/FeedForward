#include <cuda_runtime.h>

#include <iostream>

#include "MatrixOperations.h"

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
