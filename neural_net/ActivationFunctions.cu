#include "ActivationFunctions.h"

__host__ __device__ double Sigmoid::activate(double weightedSum) {
    return 1.0 / (1.0 + exp(-weightedSum));
}

std::unique_ptr<ActivationFunction> Sigmoid::clone() const {
    return std::make_unique<Sigmoid>();
}

__host__ __device__ double ReLU::activate(double weightedSum) {
    return std::max(0.0, weightedSum);  // ReLU activation: max(0, x)
}

std::unique_ptr<ActivationFunction> ReLU::clone() const {
    return std::make_unique<ReLU>();
}