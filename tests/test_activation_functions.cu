#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <iostream>

#include "ActivationFunctions.h"

// Test case for the ActivationFunction (with polymorphism)
class ActivationTest : public ::testing::Test {
   protected:
    std::unique_ptr<ActivationFunction> activationFunction;  // Base class pointer to ActivationFunction

    // Optionally set up test data for each test
    void SetUp() override {
        // Create a Sigmoid instance and assign it to the base class pointer
        activationFunction = std::make_unique<Sigmoid>();
    }

    void TearDown() override {
        // Clean up if needed (though unique_ptr automatically handles this)
    }
};

TEST_F(ActivationTest, TestSigmoidOnZero) {
    double weightedSum = 0.0;

    activationFunction = std::make_unique<Sigmoid>();

    double actualValue = activationFunction->activate(weightedSum);
    double expectedValue = 0.5;

    EXPECT_EQ(expectedValue, actualValue);
}

TEST_F(ActivationTest, TestSigmoidOn5point3) {
    double weightedSum = 5.3;

    activationFunction = std::make_unique<Sigmoid>();

    double expectedValue = 0.995033;
    double actualValue = activationFunction->activate(weightedSum);

    EXPECT_NEAR(actualValue, expectedValue, 0.0001);
}