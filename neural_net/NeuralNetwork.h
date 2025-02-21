#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <omp.h>

#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>  // For std::out_of_range
#include <string>
#include <vector>

#include "GPUConfig.h"
#include "Layer.h"
#include "LayerGPU.h"
#include "MatrixOperations.h"

using namespace std;

// TODO: Potentially make a NueralNetwork Manager so we do not have like overlapping layers objects. But I think it's
// fine...
/// Could also maybe fix by using a Template...
class NeuralNetwork {
   private:
    int numLayers;
    const LayerGPU* firstLayerGPU;
    const LayerGPU* lastLayerGPU;

    ///////////////////////////////////////////////////////
    ////               Neural Network Layers           ////
    ///////////////////////////////////////////////////////

    unique_ptr<unique_ptr<Layer>[]> layersCPU;
    std::vector<LayerGPU*> layersGPU;

    ///////////////////////////////////////////////////////
    ////               Random Generator                ////
    ///////////////////////////////////////////////////////

    mt19937 rng;
    std::uniform_real_distribution<double> dist{0.0, 1.0};

    ///////////////////////////////////////////////////////
    ////                 Init/Cleanup Layers           ////
    ///////////////////////////////////////////////////////

    void initLayersCPU(const std::vector<int>& layerConfig);
    void initLayersCUDA(const std::vector<int>& layerConfig);

    void freeLayersGPU();

    ///////////////////////////////////////////////////////
    ////               Helper Methods                  ////
    ///////////////////////////////////////////////////////

    void validateLayerIndex(int i);
    void inputCUDA(vector<float>& inputData);

   public:
    ///////////////////////////////////////////////////////
    ////               Constructors                    ////
    ///////////////////////////////////////////////////////

    NeuralNetwork(const vector<int>& layerConfig, const std::string& activationType);
    NeuralNetwork(const vector<int>& layerConfig, const std::string& activationType, bool isCudaAccelerated);

    ///////////////////////////////////////////////////////
    ////               Getters and Setters             ////
    ///////////////////////////////////////////////////////
    int getNumLayers();

    Layer* getLayerCPU(int i);
    LayerGPU* getLayerGPU(int i);

    /*
                       For Testing:
    */
    NeuralNetwork(const vector<int>& layerConfig, const std::string& activationType, int seed);

    ///////////////////////////////////////////////////////
    ////               Forward Propagation             ////
    ///////////////////////////////////////////////////////

    vector<float> feedForwardCPU(const vector<float> inputData);
    vector<float> feedForwardCUDA(vector<float>& inputData);

    ///////////////////////////////////////////////////////
    ////               Testing Utilities               ////
    ///////////////////////////////////////////////////////

    void setSeed(unsigned int seed);
};

#endif  // NEURALNETWORK_H
