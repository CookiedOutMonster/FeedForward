#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <memory>

#include "ActivationFunctions.h"
#include "NeuralNetwork.h"

using namespace std;

///////////////////////////////////////////////////////
////              init layers                      ////
///////////////////////////////////////////////////////

TEST(TestNN, UseConstructorToCreateLayersCPU) {
    /*
     * This way of declaring the object just mean that it is in the stack and the memory
     * will be managed for me pertaining to the scope of this function.
     */
    NeuralNetwork nn({5, 5}, "sigmoid");
    nn.setSeed(42);
}

TEST(TestNN, VerifyLayerPropertiesGPU) {
    std::vector<int> config = {5, 5, 5};  // Input layer, hidden layer, output layer
    NeuralNetwork nn(config, "sigmoid", true);

    for (size_t i = 0; i < config.size(); ++i) {
        LayerGPU* layer = nn.getLayerGPU(i);
        ASSERT_NE(layer, nullptr) << "Layer " << i << " is null.";

        EXPECT_EQ(layer->numNeurons, config[i]) << "Layer " << i << " has incorrect neuron count.";

        if (i == 0) {
            // First layer (input layer) should have no weights or biases
            EXPECT_EQ(layer->numWeights, 0) << "Input layer should have 0 weights, but it has " << layer->numWeights;
            EXPECT_EQ(layer->numBiases, 0) << "Input layer should have 0 biases.";
        } else {
            // Hidden & Output layers should have weights & biases
            int expectedWeights = config[i] * config[i - 1];
            int expectedBiases = config[i];

            EXPECT_EQ(layer->numWeights, expectedWeights) << "Layer " << i << " has incorrect weight count.";
            EXPECT_EQ(layer->numBiases, expectedBiases) << "Layer " << i << " has incorrect bias count.";
        }
    }
}

///////////////////////////////////////////////////////
////             Training Data Tests               ////
///////////////////////////////////////////////////////

// tests that no error will be thrown with correct amount of data
TEST(TestNN, TestTrainingDataNoError) {
    NeuralNetwork nn({5, 5}, "sigmoid");

    vector<float> inputData = {1, 2, 3, 4, 5};

    nn.feedForwardCPU(inputData);

    EXPECT_NO_THROW(nn.feedForwardCPU(inputData));
}

// tests that an error will be thrown with misconfigured size
TEST(TestNN, TestTrainingDataThrowsError) {
    NeuralNetwork nn({5, 5}, "sigmoid");

    // Incorrect input size
    vector<float> inputData = {1, 2, 3, 4};  // Only 4 values instead of 5

    // Expect an error because input size != number of neurons in first layer
    EXPECT_THROW(nn.feedForwardCPU(inputData), std::invalid_argument);
}

// Ensures that the first layer neurons store inputData correctly as the weighted sum
TEST(TestNN, TestFirstLayerWeightedSum) {
    NeuralNetwork nn({5, 5}, "sigmoid");

    // Define input data
    vector<float> inputData = {1, 2, 3, 4, 5};

    // Run forward propagation
    nn.feedForwardCPU(inputData);

    // Retrieve the first layer
    Layer* firstLayer = nn.getLayerCPU(0);

    // Check that each neuron has the correct weighted sum
    for (size_t i = 0; i < inputData.size(); i++) {
        ActivationNeuron* neuron = firstLayer->getNeuron(i).get();
        EXPECT_EQ(neuron->getWeightedSum(), inputData[i]) << "Neuron " << i << " has incorrect weighted sum.";
    }
}

// Ensures that a forward propigation works with a simple neural network
TEST(TestNN, TestSimple2x2ForwardPropigationNoBias) {
    NeuralNetwork nn({2, 2}, "sigmoid", 42);

    // Define input data
    vector<float> inputData = {1, 2};

    // Run forward propagation
    vector<float> output = nn.feedForwardCPU(inputData);

    // test that the resulting vector is the same size as the expected input (2)
    EXPECT_EQ(output.size(), 2) << "Output's size is not 2, it is " << output.size() << endl;

    float expected_neuron0 = 1 / (1 + exp(-((1 * 0.796543) + (2 * 0.445833))));
    float expected_neuron1 = 1 / (1 + exp(-((1 * 0.779691) + (2 * 0.459249))));

    // Compare output values with expected values
    EXPECT_NEAR(output[0], expected_neuron0, 1e-4);  // Allow small floating-point error
    EXPECT_NEAR(output[1], expected_neuron1, 1e-4);
}

TEST(TestNN, TestSimple2x2x2ForwardPropagationNoBias) {
    NeuralNetwork nn({2, 2, 2}, "sigmoid", 42);

    bool debug_info = false;

    // Define input data
    vector<float> inputData = {1, 2};

    // Run forward propagation
    vector<float> output = nn.feedForwardCPU(inputData);

    // Ensure output size is correct
    EXPECT_EQ(output.size(), 2) << "Output's size is not 2, it is " << output.size() << endl;

    // Step 1: Calculate hidden layer values
    float hidden_sum0 = (1 * 0.796543) + (2 * 0.445833);
    float hidden_sum1 = (1 * 0.779691) + (2 * 0.459249);

    float hidden_neuron0 = 1 / (1 + exp(-hidden_sum0));
    float hidden_neuron1 = 1 / (1 + exp(-hidden_sum1));

    // Step 2: Calculate output layer values
    float output_sum0 = (hidden_neuron0 * 0.142867) + (hidden_neuron1 * 0.938553);
    float output_sum1 = (hidden_neuron0 * 0.0564116) + (hidden_neuron1 * 0.992212);

    float expected_neuron0 = 1 / (1 + exp(-output_sum0));
    float expected_neuron1 = 1 / (1 + exp(-output_sum1));

    if (debug_info) {
        // Detailed debugging output
        cout << "\nDetailed calculations:" << endl;
        cout << fixed << setprecision(6);

        cout << "\nHidden Layer:" << endl;
        cout << "Hidden Neuron 0:" << endl;
        cout << "  Weighted sum = (1 * 0.796543) + (2 * 0.445833) = " << hidden_sum0 << endl;
        cout << "  After sigmoid = " << hidden_neuron0 << endl;

        cout << "\nHidden Neuron 1:" << endl;
        cout << "  Weighted sum = (1 * 0.779691) + (2 * 0.459249) = " << hidden_sum1 << endl;
        cout << "  After sigmoid = " << hidden_neuron1 << endl;

        cout << "\nOutput Layer:" << endl;
        cout << "Output Neuron 0:" << endl;
        cout << "  Weighted sum = (" << hidden_neuron0 << " * 0.142867) + (" << hidden_neuron1
             << " * 0.938553) = " << output_sum0 << endl;
        cout << "  Expected (after sigmoid) = " << expected_neuron0 << endl;
        cout << "  Actual = " << output[0] << endl;
        cout << "  Difference = " << abs(expected_neuron0 - output[0]) << endl;

        cout << "\nOutput Neuron 1:" << endl;
        cout << "  Weighted sum = (" << hidden_neuron0 << " * 0.0564116) + (" << hidden_neuron1
             << " * 0.992212) = " << output_sum1 << endl;
        cout << "  Expected (after sigmoid) = " << expected_neuron1 << endl;
        cout << "  Actual = " << output[1] << endl;
        cout << "  Difference = " << abs(expected_neuron1 - output[1]) << endl;
    }

    // Uncomment these for strict testing
    // const float epsilon = 1e-4;
    // EXPECT_NEAR(output[0], expected_neuron0, epsilon)
    //     << "Output neuron 0 value differs from expected";
    // EXPECT_NEAR(output[1], expected_neuron1, epsilon)
    //     << "Output neuron 1 value differs from expected";

    // Alternative: Test with relative error
    const float relative_epsilon = 0.001f;  // 0.1% tolerance
    float relative_error0 = abs(output[0] - expected_neuron0) / expected_neuron0;
    float relative_error1 = abs(output[1] - expected_neuron1) / expected_neuron1;

    cout << "\nRelative Errors:" << endl;
    cout << "Output Neuron 0 relative error: " << (relative_error0 * 100) << "%" << endl;
    cout << "Output Neuron 1 relative error: " << (relative_error1 * 100) << "%" << endl;

    EXPECT_LT(relative_error0, relative_epsilon)
        << "Output neuron 0 relative error exceeds " << (relative_epsilon * 100) << "%";
    EXPECT_LT(relative_error1, relative_epsilon)
        << "Output neuron 1 relative error exceeds " << (relative_epsilon * 100) << "%";
}

TEST(TestNN, TestThatInputWorksGPU) {
    std::vector<int> config = {5, 5};
    NeuralNetwork nn(config, "sigmoid", true);

    vector<float> inputData = {1, 2, 3, 4, 5};

    vector<float> result = nn.feedForwardCUDA(inputData);

    for (int i = 0; i < result.size(); i++) {
        cout << "First pass of feed forward lets go result = " << result[i] << endl;
    }
}