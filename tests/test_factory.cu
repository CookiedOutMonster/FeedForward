#include <gtest/gtest.h>

#include <memory>

#include "ActivationFunctions.h"  // Using Sigmoid activation function
#include "Factory.h"

using namespace std;

///////////////////////////////////////////////////////
////               Test Setup Section              ////
///////////////////////////////////////////////////////

// Test fixture to manage common setup and teardown
class NeuronFactoryTest : public ::testing::Test {
   protected:
    void SetUp() override {
        numberOfNeurons = 5;
        activationFunction = std::make_unique<Sigmoid>();
    }

    int numberOfNeurons;
    unique_ptr<ActivationFunction> activationFunction;
};

///////////////////////////////////////////////////////
////               Test Neuron Factory             ////
///////////////////////////////////////////////////////

// Test that the createNeurons method creates the correct number of neurons
TEST_F(NeuronFactoryTest, CreateCorrectNumberOfNeurons) {
    auto neurons = NeuronFactory::createNeurons(numberOfNeurons, std::move(activationFunction));

    // Verify that the number of neurons matches the expected number
    EXPECT_EQ(neurons.size(), 5);  // Check the size of the vector

    // Check if the neurons array is properly allocated (should not be empty)
    EXPECT_FALSE(neurons.empty());
}

// Test that createNeurons properly handles zero neurons
TEST_F(NeuronFactoryTest, ZeroNeurons) {
    auto neurons = NeuronFactory::createNeurons(0, std::make_unique<Sigmoid>());

    // Verify that the result is an empty vector
    EXPECT_TRUE(neurons.empty());  // Check if the vector is empty
}

// Test that createNeurons properly assigns a Sigmoid function to each neuron
TEST_F(NeuronFactoryTest, AllNeuronsHaveSigmoid) {
    // Create neurons with Sigmoid activation function
    auto neurons = NeuronFactory::createNeurons(numberOfNeurons, std::move(activationFunction));

    // First pass: Assign a unique weighted sum to each neuron
    for (int i = 0; i < numberOfNeurons; ++i) {
        double inputValue = static_cast<double>(i) / numberOfNeurons;  // Discrete values from 0 to 1
        neurons[i]->setWeightedSum(inputValue);                        // Assigning the weighted sum to each neuron
    }

    // Second pass: Call the activate function and check if the result matches Sigmoid's expected output
    for (int i = 0; i < numberOfNeurons; ++i) {
        double inputValue = static_cast<double>(i) / numberOfNeurons;
        double expectedOutput = 1.0 / (1.0 + exp(-inputValue));     // Sigmoid activation function formula
        EXPECT_NEAR(neurons[i]->activate(), expectedOutput, 1e-5);  // Compare the actual result to the expected output
    }
}

///////////////////////////////////////////////////////
////             Test Activation Factory           ////
///////////////////////////////////////////////////////

TEST(TestActivationFactory, GetSigmoidActivationFromString) {
    unique_ptr<ActivationFunction> sigmoid = createActivationFunction("sigmoid");

    double sigmaResult = sigmoid->activate(1.5);

    double expected = 0.8175744761937223;

    EXPECT_NEAR(sigmaResult, expected, 0.0001);
}

TEST(TestActivationFactory, GetActivationFunctionThrowsOnInvalidType) {
    EXPECT_THROW({ auto activationFunction = createActivationFunction("eskeddit"); }, std::invalid_argument);
}

// test for relu

// test unsupported
