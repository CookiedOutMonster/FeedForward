#include "NeuralNetwork.h"

///////////////////////////////////////////////////////
////                 Constructor                   ////
///////////////////////////////////////////////////////

const bool DEBUG_INFO = false;
static const int FIRST_LAYER = 0;
static const int SECOND_LAYER = 1;

NeuralNetwork::NeuralNetwork(const vector<int>& layerConfig, const std::string& activationType) {
    initLayersCPU(layerConfig);
}

NeuralNetwork::NeuralNetwork(const vector<int>& layerConfig, const std::string& activationType,
                             bool isCudaAccelerated) {
    if (isCudaAccelerated) {
        initLayersCUDA(layerConfig);
        initGPUConfig();
    } else {
        initLayersCPU(layerConfig);
    }
}

/*
        For Testing: ie has a seed
*/
NeuralNetwork::NeuralNetwork(const vector<int>& layerConfig, const std::string& activationType, int seed) {
    rng = std::mt19937(std::random_device{}());
    setSeed(seed);
    initLayersCPU(layerConfig);
}

///////////////////////////////////////////////////////
////                 Init/Cleanup Layers           ////
///////////////////////////////////////////////////////

void NeuralNetwork::initLayersCPU(const std::vector<int>& layerConfig) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    unique_ptr<ActivationFunction> sigmoid = createActivationFunction("sigmoid");

    this->numLayers = layerConfig.size();
    this->layersCPU = std::make_unique<std::unique_ptr<Layer>[]>(this->numLayers);

    // Initialize the first layer
    this->layersCPU[0] = std::make_unique<Layer>(layerConfig[0], sigmoid);

    for (int i = 1; i < this->numLayers; i++) {
        this->layersCPU[i] = std::make_unique<Layer>(layerConfig[i], sigmoid);

        auto& prevLayer = this->layersCPU[i - 1];
        auto& currLayer = this->layersCPU[i];

        for (int prevNeuronIndex = 0; prevNeuronIndex < prevLayer->getNumberOfNeurons(); prevNeuronIndex++) {
            const shared_ptr<ActivationNeuron>& prevNeuron = prevLayer->getNeuron(prevNeuronIndex);

            for (int currNeuronIndex = 0; currNeuronIndex < currLayer->getNumberOfNeurons(); currNeuronIndex++) {
                const shared_ptr<ActivationNeuron>& currNeuron = currLayer->getNeuron(currNeuronIndex);

                double weight = dist(rng);
                double bias = dist(rng);

                std::shared_ptr<ActivationNeuron> sharedPrev(prevNeuron.get(), [](ActivationNeuron*) {});
                std::shared_ptr<ActivationNeuron> sharedCurr(currNeuron.get(), [](ActivationNeuron*) {});

                auto edge = std::make_shared<Edge>(sharedPrev, sharedCurr, weight, bias);

                prevNeuron->addOutgoingEdge(edge);
                currNeuron->addIncomingEdge(edge);

                if (DEBUG_INFO) {
                    std::cout << "Edge created: Layer " << (i - 1) << " Neuron " << prevNeuronIndex << " → Layer " << i
                              << " Neuron " << currNeuronIndex << " | Weight: " << weight << " | Bias: " << bias
                              << "current neuron value " << currNeuron->getWeightedSum() << "\n";
                }
            }
        }
    }
}

void NeuralNetwork::initLayersCUDA(const std::vector<int>& layerConfig) {
    this->numLayers = layerConfig.size();
    layersGPU.resize(this->numLayers);

    for (size_t i = 0; i < layerConfig.size(); ++i) {
        int numNeurons = layerConfig[i];
        int prevLayerNeurons = (i == FIRST_LAYER) ? 0 : layerConfig[i - 1];

        layersGPU[i] = createLayer(numNeurons, prevLayerNeurons);

        if (i == FIRST_LAYER) {
            layersGPU[i]->numWeights = 0;
            layersGPU[i]->numBiases = 0;
        }
    }

    // @TODO potentially turn this into a function
    this->firstLayerGPU = getLayerGPU(FIRST_LAYER);
    this->lastLayerGPU = getLayerGPU(numLayers - 1);
}

void NeuralNetwork::freeLayersGPU() {
    for (LayerGPU* layer : layersGPU) {
        freeLayerGPU(layer);
    }
    layersGPU.clear();
}

///////////////////////////////////////////////////////
////                Getters and Setters            ////
///////////////////////////////////////////////////////

// the '&' here is because we are passing by reference instead of making a copy of the layer
Layer* NeuralNetwork::getLayerCPU(int i) {
    validateLayerIndex(i);
    return layersCPU[i].get();  // Get raw pointer from unique_ptr
}

LayerGPU* NeuralNetwork::getLayerGPU(int i) {
    validateLayerIndex(i);
    return layersGPU[i];
}

///////////////////////////////////////////////////////
////          Forward Propigation CPU              ////
///////////////////////////////////////////////////////

// TODO: eventually consolidate this into multiple functions no idea why that didn't work
// it's also kinda gross to read.
vector<float> NeuralNetwork::feedForwardCPU(vector<float> inputData) {
    // Get and validate first layer
    Layer* firstLayer = getLayerCPU(0);
    int numNeuronsFirstLayer = firstLayer->getNumberOfNeurons();

    if (numNeuronsFirstLayer != inputData.size()) {
        throw std::invalid_argument("Mismatch: first layer has " + std::to_string(numNeuronsFirstLayer) +
                                    " neurons, but input data has " + std::to_string(inputData.size()) + " elements.");
    }

    // Initialize the output vector based on the size of the last layer
    Layer* lastLayer = getLayerCPU(numLayers - 1);
    int outputSize = lastLayer->getNumberOfNeurons();
    vector<float> output(outputSize, 0.0f);

// Set input layer values - no activation
#pragma omp parallel for
    for (int i = 0; i < numNeuronsFirstLayer; i++) {
        ActivationNeuron& neuron = *firstLayer->getNeuron(i).get();
        neuron.setWeightedSum(inputData[i]);
        // No activation for input layer - just pass through the raw values
    }

    // Process hidden layers and output layer
    for (int i = SECOND_LAYER; i < numLayers; i++) {
        Layer* currentLayer = getLayerCPU(i);
        int layerSize = currentLayer->getNumberOfNeurons();

        // #pragma omp parallel for
        for (int j = 0; j < layerSize; j++) {
            ActivationNeuron& neuron = *currentLayer->getNeuron(j);
            const auto& incomingEdges = neuron.getIncomingEdges();

            if (DEBUG_INFO) {
#pragma omp critical
                {
                    int thread_id = omp_get_thread_num();
                    std::cout << "Thread " << thread_id << " processing Neuron " << j << " in Layer " << i << " with "
                              << incomingEdges.size() << " incoming edges.\n";
                }
            }

            float weightedSum = 0.0f;
            int numEdges = incomingEdges.size();

            for (int k = 0; k < numEdges; k++) {
                const shared_ptr<Edge>& edge = incomingEdges[k];
                ActivationNeuron& baseNeuron = *edge->getBaseNeuron();
                // For input layer neurons, getWeightedSum() will return the raw input
                // For other layers, getActivationValue() returns the activated value
                float baseValue = (i == SECOND_LAYER) ? baseNeuron.getWeightedSum() : baseNeuron.getActivationValue();
                weightedSum += baseValue * edge->getWeight();
            }

            neuron.setWeightedSum(weightedSum);
            neuron.activate();

            // Store output if it's the last layer
            if (i == numLayers - 1) {
                output[j] = neuron.getActivationValue();
            }
        }
    }

    return output;
}

///////////////////////////////////////////////////////
////          Forward Propigation GPU              ////
///////////////////////////////////////////////////////

// it's okay if this is not perfect!
vector<float> NeuralNetwork ::feedForwardCUDA(vector<float>& inputData) {
    // take input data and assign it to the first layer.
    // void inputCUDA(const vector<float>& inputData);

    // @TODO make this a function
    int numberOfOutPutNuerons = this->lastLayerGPU->numNeurons;
    vector<float> output(numberOfOutPutNuerons, 0.0f);

    // take input data and place it inside input layers
    inputCUDA(inputData);

    // so the layer of i's activations is computed as:
    // the activations of the previous layer?
    // * the weights of the previous layer?
    // plus the bias of the current layer?

    LayerGPU* prevLayer = const_cast<LayerGPU*>(firstLayerGPU);

    // this means: start on the second layer
    for (int i = SECOND_LAYER; i < this->numLayers; i++) {
        // current layer
        LayerGPU* currLayer = layersGPU[i];
        float* curr_device_bias = currLayer->d_biases;

        // get the value of the previous layer
        // holy memory access batman
        float* prev_device_activation = prevLayer->d_neuronActivations;
        float* prev_device_weights = prevLayer->d_weights;  // does the input layer have weights?

        // kernal call

        // problem: decide if the Layer's weights and biases are for the next layer or they are for the previous layer
        // lol

        // update prev layer
        prevLayer = currLayer;
    }

    return output;
}

void NeuralNetwork ::inputCUDA(vector<float>& inputData) {
    // TODO test producing an error as well :-)
    float* convertedInputData = inputData.data();
    size_t size = inputData.size();

    updateLayerActivations(this->firstLayerGPU, convertedInputData, size);
}

///////////////////////////////////////////////////////
////               Helper Methods                  ////
///////////////////////////////////////////////////////
void NeuralNetwork::validateLayerIndex(int i) {
    if (i < 0 || i >= this->numLayers) {
        throw std::out_of_range("Invalid layer index");
    }
}

///////////////////////////////////////////////////////
////               Testing Utilities               ////
///////////////////////////////////////////////////////
void NeuralNetwork::setSeed(unsigned int seed) {
    rng.seed(seed);
}
