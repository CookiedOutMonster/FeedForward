#include <random>

#include "LayerGPU.h"

using namespace std;

LayerGPU* createLayer(int numNeurons, int prevLayerNeurons, std::mt19937 rng) {
    LayerGPU* layer = new LayerGPU();

    layer->numNeurons = numNeurons;
    layer->prevLayerNeurons = prevLayerNeurons;
    layer->numWeights = numNeurons * prevLayerNeurons;
    layer->numBiases = numNeurons;

    // Allocate device memory
    checkCuda(cudaMalloc((void**)&layer->d_neuronActivations, numNeurons * sizeof(float)),
              "Failed to allocate neuronActivations");

    if (prevLayerNeurons != 0) {
        checkCuda(cudaMalloc((void**)&layer->d_weights, layer->numWeights * sizeof(float)),
                  "Failed to allocate weights");

        checkCuda(cudaMalloc((void**)&layer->d_biases, numNeurons * sizeof(float)), "Failed to allocate biases");
    }

    // Allocate host memory
    layer->h_curr_neuronActivations = (float*)malloc(numNeurons * sizeof(float));
    layer->h_prev_weights = (float*)malloc(layer->numWeights * sizeof(float));
    layer->h__curr_biases = (float*)malloc(numNeurons * sizeof(float));

    if (!layer->h_curr_neuronActivations || !layer->h_prev_weights || !layer->h__curr_biases) {
        delete layer;
        throw std::runtime_error("Failed to allocate host memory");
    }

    // Initialize device memory
    checkCuda(cudaMemset(layer->d_neuronActivations, 0, numNeurons * sizeof(float)),
              "Failed to initialize neuronActivations");

    // generalizeable to initalize it to random
    if (prevLayerNeurons != 0) {
        initializeRandomWeightsAndBiases(layer, rng, 0.0f, 1.0f);
    }

    return layer;
}

void initializeRandomWeightsAndBiases(LayerGPU* layer, mt19937 rng, float min, float max) {
    if (layer->prevLayerNeurons == 0) return;  // Skip input layer

    std::uniform_real_distribution<float> dist(min, max);

    // Initialize weights
    for (int i = 0; i < layer->numWeights; i++) {
        layer->h_prev_weights[i] = dist(rng);
    }

    // Initialize biases
    for (int i = 0; i < layer->numBiases; i++) {
        layer->h__curr_biases[i] = dist(rng);
    }

    // Copy to device
    checkCuda(
        cudaMemcpy(layer->d_weights, layer->h_prev_weights, layer->numWeights * sizeof(float), cudaMemcpyHostToDevice),
        "Failed to copy weights to device");

    checkCuda(
        cudaMemcpy(layer->d_biases, layer->h__curr_biases, layer->numBiases * sizeof(float), cudaMemcpyHostToDevice),
        "Failed to copy biases to device");
}

cudaError_t freeLayerGPU(LayerGPU* layer) {
    cudaError_t cudaStatus = cudaSuccess;

    if (layer->d_neuronActivations) {
        cudaStatus = cudaFree(layer->d_neuronActivations);
        if (cudaStatus != cudaSuccess) return cudaStatus;
    }

    if (layer->d_weights) {
        cudaStatus = cudaFree(layer->d_weights);
        if (cudaStatus != cudaSuccess) return cudaStatus;
    }

    if (layer->d_biases) {
        cudaStatus = cudaFree(layer->d_biases);
        if (cudaStatus != cudaSuccess) return cudaStatus;
    }

    // Free host memory
    free(layer->h_curr_neuronActivations);
    free(layer->h_prev_weights);
    free(layer->h__curr_biases);

    return cudaSuccess;
}

/*


Consider writing tests for these but eh I already have lots


*/

void updateLayerData(float* deviceData, const float* hostData, size_t numElements, const char* errorMsg) {
    size_t size = numElements * sizeof(float);
    checkCuda(cudaMemcpy(deviceData, hostData, size, cudaMemcpyHostToDevice), errorMsg);
}

void updateLayerActivations(const LayerGPU* layer, const float* newActivations, size_t numOfActivations) {
    if (numOfActivations != layer->numNeurons) {
        throw std::out_of_range("Number of neurons in layer exceeds the given data.");
    }
    updateLayerData(layer->d_neuronActivations, newActivations, numOfActivations, "Error copying activations to GPU");
}

void updateLayerWeights(LayerGPU* layer, const float* newWeights, size_t numOfWeights) {
    if (numOfWeights != layer->numWeights) {
        throw std::out_of_range("Number of weights in layer exceeds the given data.");
    }
    updateLayerData(layer->d_weights, newWeights, numOfWeights, "Error copying weights to GPU");
}

void updateLayerBiases(LayerGPU* layer, const float* newBiases, size_t numOfBiases) {
    if (numOfBiases != layer->numBiases) {
        throw std::out_of_range("Number of biases in layer exceeds the given data.");
    }
    updateLayerData(layer->d_biases, newBiases, numOfBiases, "Error copying biases to GPU");
}

void checkCuda(cudaError_t status, const char* msg) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + " | CUDA Error: " + cudaGetErrorString(status));
    }
}
