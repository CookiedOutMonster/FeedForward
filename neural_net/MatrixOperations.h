#include <cuda_runtime.h>

__global__ void MatrixAddition(double *d_arr_a, double *d_arr_b, double *d_result, int size);

__host__ __device__ int DotProduct(double *leftHandSideVector, double *rightHandSideVector, int size);
__global__ void DotProductKernel(double *leftHandSideVector, double *rightHandSideVector, int *result, int size);