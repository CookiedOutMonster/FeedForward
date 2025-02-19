#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <typeinfo>

#include "ActivationFunctions.h"
#include "Edge.h"
#include "Factory.h"
#include "Layer.h"

using namespace std;

///////////////////////////////////////////////////////
////               Test Setup Section              ////
///////////////////////////////////////////////////////

class SimpleEdge : public ::testing::Test {
   protected:
    InputNeuron inputNeuron;

    void SetUp() override {
        sigmoid = make_unique<Sigmoid>();
        neurons = NeuronFactory::createNeurons(2, sigmoid);
    }

    void TearDown() override {
        // Optionally clean up after each test
    }

   public:
    unique_ptr<ActivationFunction> sigmoid;
    vector<shared_ptr<ActivationNeuron>> neurons;
};

class MultiEdge : public ::testing::Test {
   protected:
    InputNeuron inputNeuron;

    void SetUp() override {
        sigmoid = make_unique<Sigmoid>();
    }

    void TearDown() override {
        deleteNeurons();
    }

    void createNeurons(int numOfNeurons) {
        neurons.clear();
        neurons = NeuronFactory::createNeurons(numOfNeurons, sigmoid);
    }

    void deleteNeurons() {
        neurons.clear();
    }

   public:
    unique_ptr<ActivationFunction> sigmoid;
    vector<shared_ptr<ActivationNeuron>> neurons;
};

///////////////////////////////////////////////////////
////               Simple Edge Tests               ////
///////////////////////////////////////////////////////

// test that the edge is connecting two neurons and ensure that the neurons have the same memory address
TEST_F(SimpleEdge, TestConnection) {
    bool printErrors = false;

    // Create two neurons with the Sigmoid activation function
    // unique_ptr<ActivationFunction> sigmoid = make_unique<Sigmoid>();
    // vector<unique_ptr<ActivationNeuron>> neurons = NeuronFactory::createNeurons(2, sigmoid);

    // Move the first and second neuron from the vector to shared_ptrs
    shared_ptr<ActivationNeuron> baseNeuron = move(neurons[0]);
    shared_ptr<ActivationNeuron> connectedNeuron = move(neurons[1]);

    // Get the raw memory address of the neurons
    ActivationNeuron* baseNeuronMemoryAddy = baseNeuron.get();
    ActivationNeuron* connectedNeuronMemoryAddy = connectedNeuron.get();

    // Create an Edge object linking the two neurons with a weight and bias
    Edge edge(baseNeuron, connectedNeuron, 0.3, 1);

    // Get the 'to' neuron from the edge
    shared_ptr<ActivationNeuron> edgeBaseNeuron = edge.getBaseNeuron();
    shared_ptr<ActivationNeuron> edgeConnectedNeuron = edge.getConnectedNeuron();

    if (printErrors) {
        // Print memory addresses of the connected neurons
        cout << "Memory address of 'base' neuron: " << baseNeuronMemoryAddy << endl;
        cout << "Memory address of 'connected' neuron: " << connectedNeuronMemoryAddy << endl;
        cout << "Memory address of 'base' neuron from edge: " << edgeBaseNeuron.get() << endl;
    }

    ASSERT_EQ(baseNeuronMemoryAddy, edgeBaseNeuron.get());
    ASSERT_EQ(connectedNeuronMemoryAddy, edgeConnectedNeuron.get());
}

// write a test to compute the activation of a neuron
TEST_F(SimpleEdge, TestActivationComputation) {
    shared_ptr<ActivationNeuron> baseNeuron = move(neurons[0]);
    shared_ptr<ActivationNeuron> connectedNeuron = move(neurons[1]);

    // set a value for base neuron's activation to calculate the activation of the connectedNeuron
    double weightedSum = 0.03;
    baseNeuron->setWeightedSum(0.03);
    baseNeuron->activate();

    // create an edge
    Edge edge(baseNeuron, connectedNeuron, 0.3, 1);

    double connectedNeuronWeightedSum = (baseNeuron->getActivationValue() * edge.getWeight()) + edge.getBias();

    double expectedValue = 1.1522498373150509;

    EXPECT_NEAR(expectedValue, connectedNeuronWeightedSum, 0.0001);

    // baseNeuron->getActivationValue();

    // perform the calculation of the weighted sum on the connected neuron
}

///////////////////////////////////////////////////////
////               Constructor Edge Tests          ////
///////////////////////////////////////////////////////

//  Test for Null baseNeuron
TEST(EdgeTest, TestNullBaseNeuron) {
    unique_ptr<ActivationFunction> sigmoid = make_unique<Sigmoid>();
    vector<shared_ptr<ActivationNeuron>> neurons = NeuronFactory::createNeurons(1, sigmoid);

    shared_ptr<ActivationNeuron> connectedNeuron = move(neurons[0]);

    // Expect an exception if the base neuron is null
    EXPECT_THROW(Edge edge(nullptr, connectedNeuron, 0.3f, 1.0f), std::invalid_argument);
}

// Test for Null connectedNeuron
TEST(EdgeTest, TestNullConnectedNeuron) {
    unique_ptr<ActivationFunction> sigmoid = make_unique<Sigmoid>();
    vector<shared_ptr<ActivationNeuron>> neurons = NeuronFactory::createNeurons(1, sigmoid);

    shared_ptr<ActivationNeuron> baseNeuron = move(neurons[0]);

    // Expect an exception if the connected neuron is null
    EXPECT_THROW(Edge edge(baseNeuron, nullptr, 0.3f, 1.0f), std::invalid_argument);
}

// Test for Both Neurons and Values Being Invalid
TEST(EdgeTest, TestInvalidNeuronsAndValues) {
    // Expect an exception if both neurons are null
    EXPECT_THROW(Edge edge(nullptr, nullptr, 0.0f, 0.0f), std::invalid_argument);
}

// write a test to ensure that we cannot have an edge connected to itself
TEST(CyclicEdge, TestCannotConnectItself) {
    unique_ptr<ActivationFunction> sigmoid = make_unique<Sigmoid>();
    vector<shared_ptr<ActivationNeuron>> neurons = NeuronFactory::createNeurons(1, sigmoid);

    shared_ptr<ActivationNeuron> baseNeuron = move(neurons[0]);

    // create an edge that connects itself
    EXPECT_THROW(Edge edge(baseNeuron, baseNeuron, 0.3, 1), std::invalid_argument);
}

///////////////////////////////////////////////////////
////               1:N Edge Tests                  ////
///////////////////////////////////////////////////////

// write a test to ensure that we can have a single home neuron and multiple connected neurons
TEST_F(MultiEdge, TestOneNeuronManyConnections) {
    // Set the maximum number of neurons you want to test
    int maxNumOfNeurons = 100;

    for (int numOfNeurons = 2; numOfNeurons <= maxNumOfNeurons; ++numOfNeurons) {
        createNeurons(numOfNeurons);

        shared_ptr<ActivationNeuron> baseNeuron = move(neurons[0]);

        vector<Edge> edges;

        for (int i = 1; i < numOfNeurons; ++i) {
            shared_ptr<ActivationNeuron> connectedNeuron = move(neurons[i]);

            // Ensure that the connected neuron is not the base neuron
            EXPECT_NE(baseNeuron.get(), connectedNeuron.get());

            if (numOfNeurons > 2 && i > 2) {
                bool uniqueNeuron = true;

                // Check that each connected neuron is unique
                for (const auto& edge : edges) {
                    if (edge.getConnectedNeuron() == connectedNeuron) {
                        uniqueNeuron = false;
                        break;
                    }
                }

                // If the connectedNeuron is not unique, print an error and fail the test
                if (!uniqueNeuron) {
                    cout << "Duplicate neuron found: " << connectedNeuron.get() << endl;
                }

                // Ensure that each connected neuron is unique
                EXPECT_TRUE(uniqueNeuron);
            }

            edges.emplace_back(baseNeuron, connectedNeuron, 0.3f, 1.0f);
        }

        // Ensure baseNeuron is correctly set for all edges
        for (const auto& edge : edges) {
            ASSERT_EQ(baseNeuron, edge.getBaseNeuron());
        }
    }
}
