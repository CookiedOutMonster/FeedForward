#ifndef LAYERGPU_H
#define LAYERGPU_H

#include <cuda_runtime.h>
#include <stdio.h>

#include <random>
#include <stdexcept>

/*
    - Struct contains 1D arrays for information required to activate at the current layer
    - This means that it contains the weights of the previous layer but the biases and the activations of the curent
    - Layer is initialized using random values but a seed for testing
*/
typedef struct {
    // device
    float *d_neuronActivations;
    float *d_weights;
    float *d_biases;

    // Host
    float *h_curr_neuronActivations;
    float *h_prev_weights;
    float *h__curr_biases;

    // @TODO consider removing "prevLayerNeurons"
    int numNeurons;
    int numWeights;
    int numBiases;
    int prevLayerNeurons;
} LayerGPU;

// Function declarations
LayerGPU *createLayer(int numNeurons, int prevLayerNeurons, std::mt19937 rng = std::mt19937(std::random_device{}()));
cudaError_t freeLayerGPU(LayerGPU *layer);
void updateLayerActivations(const LayerGPU *layer, const float *newActivations, size_t numOfActivations);

// consider moving this fction into GPUConfig
void checkCuda(cudaError_t status, const char *msg);

void initializeRandomWeightsAndBiases(LayerGPU *layer, std::mt19937 rng, float min, float max);

// Generalized update functions for weights and biases
void updateWeights(LayerGPU *layer, const float *newWeights, size_t numOfWeights);
void updateBiases(LayerGPU *layer, const float *newBiases, size_t numOfBiases);
void transferActivationsHostToDevice(LayerGPU *layer);

#endif  // LAYERGPU_H