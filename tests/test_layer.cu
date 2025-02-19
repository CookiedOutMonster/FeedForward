#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "ActivationFunctions.h"
#include "Layer.h"

using namespace std;

///////////////////////////////////////////////////////
////            Test Setup Section                 ////
///////////////////////////////////////////////////////

class TestLayer : public ::testing::Test {
   protected:
    void SetUp() override {}

    void createLayer(int numberOfNeurons, const unique_ptr<ActivationFunction>& activationFunction) {
        // int numberOfNeurons = 5;
        // activationFunction = make_unique<Sigmoid>();
        layer = std::make_unique<Layer>(numberOfNeurons, activationFunction);
    }

    unique_ptr<ActivationFunction> activationFunction;
    unique_ptr<Layer> layer;
};

///////////////////////////////////////////////////////
////                Layer Tests                    ////
///////////////////////////////////////////////////////

// Test that getting and setting number of neurons works as desired in a Layer Object
TEST_F(TestLayer, GetSetNumberOfNeurons) {
    int expectedNumberOfNeurons = 5;

    unique_ptr<ActivationFunction> sigmoid = make_unique<Sigmoid>();

    createLayer(5, sigmoid);

    int actualNumberOfNeurons = layer->getNumberOfNeurons();

    ASSERT_EQ(expectedNumberOfNeurons, actualNumberOfNeurons);
}

// Test that iterating through, setting activation layer, and activating works as expected on CPU
TEST_F(TestLayer, IterateThroughLayer) {
    int expectedNumberOfNeurons = 5;

    unique_ptr<ActivationFunction> sigmoid = make_unique<Sigmoid>();

    createLayer(5, sigmoid);

    // const auto& neurons = layer->getNeuronsInLayer();
    const vector<shared_ptr<ActivationNeuron>>& neurons = layer->getNeuronsInLayer();

    size_t vectorLen = neurons.size();

    // test that the neurons vector is not bigger than the expected size
    ASSERT_EQ(vectorLen, expectedNumberOfNeurons);

    double weightedSumArray[5] = {0.5, 0.3, 0.21, 0.7, 0.9};

    for (int i = 0; i < vectorLen; i++) {
        auto& neuron = neurons[i];
        double weightedSum = weightedSumArray[i];
        neuron->setWeightedSum(weightedSum);
    }

    double expectedActivationsArray[5] = {0.622459, 0.574442, 0.552321, 0.668187, 0.710949};

    for (int i = 0; i < vectorLen; i++) {
        auto& neuron = neurons[i];
        double actualNeuronActivationValue = neuron->activate();
        double expectedNeuronActivationValue = expectedActivationsArray[i];
        EXPECT_NEAR(actualNeuronActivationValue, expectedNeuronActivationValue, 0.0001);
    }
}

// Make a test for reading in binary image for input layer!