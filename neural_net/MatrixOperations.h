#include <cuda_runtime.h>

__global__ void MatrixAddition(float *d_arr_a, float *d_arr_b, float *d_result, int size);
__global__ void MatrixMultiplication(float *d_W, float *d_A, float *d_O, int prevNumNeurons);
__host__ __device__ int DotProduct(double *leftHandSideVector, double *rightHandSideVector, int size);
__global__ void DotProductKernel(double *leftHandSideVector, double *rightHandSideVector, int *result, int size);
__global__ void sigmoidActivationKernel(float *neurons, int size);
__global__ void reluActivationKernel(double *neurons, int size);
