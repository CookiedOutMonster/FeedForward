#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "ActivationFunctions.h"
#include "Neuron.h"

///////////////////////////////////////////////////////
////        Test Setup Section                     ////
///////////////////////////////////////////////////////

class InputNeuronTest : public ::testing::Test {
   protected:
    InputNeuron inputNeuron;

    // You can set up test data here if needed
    void SetUp() override {
        // Optionally set up test data for each test
    }

    void TearDown() override {
        // Optionally clean up after each test
    }
};

class ActivationNeuronTest : public ::testing::Test {
   protected:
    std::unique_ptr<ActivationNeuron> activationNeuron;

    // Set up test data here
    void SetUp() override {
        std::unique_ptr<ActivationFunction> sigmoid = std::make_unique<Sigmoid>();
        activationNeuron = std::make_unique<ActivationNeuron>(std::move(sigmoid));
    }

    void TearDown() override {
        // Optionally clean up after each test, if needed
    }
};

///////////////////////////////////////////////////////
////               Helper Functions                ////
///////////////////////////////////////////////////////

std::vector<std::unique_ptr<ActivationNeuron>> createNeurons(int amountOfNeurons) {
    std::vector<std::unique_ptr<ActivationNeuron>> vectorOfNeurons;

    for (int i = 0; i < amountOfNeurons; i++) {
        std::unique_ptr<ActivationFunction> sigmoid = std::make_unique<Sigmoid>();
        vectorOfNeurons.push_back(std::make_unique<ActivationNeuron>(std::move(sigmoid)));
    }

    return vectorOfNeurons;
}

///////////////////////////////////////////////////////
////              Input Neuron Tests               ////
///////////////////////////////////////////////////////

TEST_F(InputNeuronTest, GetSetWeightedSumSingle) {
    double expectedValue = 42.0;

    inputNeuron.setWeightedSum(expectedValue);

    double actualValue = inputNeuron.getWeightedSum();

    EXPECT_EQ(expectedValue, actualValue);
}

///////////////////////////////////////////////////////
////             Activation Neuron Tests           ////
///////////////////////////////////////////////////////

TEST_F(ActivationNeuronTest, GetSetWeightedSumSingleActivation_CPU) {
    double weightedSum = 42.0;

    activationNeuron->setWeightedSum(weightedSum);

    double result = activationNeuron->activate();

    std::cout << "Activated result: " << result << std::endl;

    EXPECT_EQ(result, 1);
}

TEST(ActivationNeuronMulitple, GetSetMultipleActivations_CPU) {
    // goal here is to test encapsulation

    int numberOfNeurons = 5;

    double weightedSumList[numberOfNeurons] = {2.0, 3.5, 0.8, 25.0, 69.0};

    double actualActivationValues[numberOfNeurons];
    double expectedActivationValues[numberOfNeurons] = {0.880797, 0.970687, 0.689974, 1, 1};

    std::vector<std::unique_ptr<ActivationNeuron>> vectorOfNeurons = createNeurons(numberOfNeurons);

    int index = 0;
    for (const auto& neuron : vectorOfNeurons) {
        double weightedSumForEachNeuron = weightedSumList[index];
        neuron->setWeightedSum(weightedSumForEachNeuron);
        double activationOfNeuronAtIndex = neuron->activate();
        actualActivationValues[index] = activationOfNeuronAtIndex;
        index++;
    }

    // check
    for (int i = 0; i < numberOfNeurons; i++) {
        double actualValue = actualActivationValues[i];
        double expectedValue = expectedActivationValues[i];
        EXPECT_NEAR(actualValue, expectedValue, 0.0001);
    }
}
