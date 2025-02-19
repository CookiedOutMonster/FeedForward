#ifndef LAYERGPU_H
#define LAYERGPU_H

#include <cuda_runtime.h>
#include <stdio.h>

#include <random>
#include <stdexcept>

typedef struct {
    // device
    float *d_neuronActivations;
    float *d_weights;
    float *d_biases;

    // Host
    float *h_neuronActivations;
    float *h_weights;
    float *h_biases;

    int numNeurons;
    int numWeights;
    int numBiases;
    int prevLayerNeurons;
} LayerGPU;

// Function declarations
LayerGPU *createLayer(int numNeurons, int prevLayerNeurons, std::mt19937 rng = std::mt19937(std::random_device{}()));
cudaError_t freeLayerGPU(LayerGPU *layer);
void updateLayerActivations(LayerGPU *layer, const float *newActivations, size_t numOfActivations);
void checkCuda(cudaError_t status, const char *msg);

void initializeRandomWeightsAndBiases(LayerGPU *layer, std::mt19937 rng, float min, float max);

// Generalized update functions for weights and biases
void updateWeights(LayerGPU *layer, const float *newWeights, size_t numOfWeights);
void updateBiases(LayerGPU *layer, const float *newBiases, size_t numOfBiases);

#endif  // LAYERGPU_H