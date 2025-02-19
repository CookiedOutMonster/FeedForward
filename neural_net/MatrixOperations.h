#include <cuda_runtime.h>
__host__ __device__ int DotProduct(double *leftHandSideVector, double *rightHandSideVector, int size);
__global__ void DotProductKernel(double *leftHandSideVector, double *rightHandSideVector, int *result, int size);