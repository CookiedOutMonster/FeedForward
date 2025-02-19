#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <memory>

class ActivationFunction {
   public:
    virtual double activate(double weightedSum) = 0;
    virtual std::unique_ptr<ActivationFunction> clone() const = 0;  // Add the clone function
    virtual ~ActivationFunction() = default;                        // Ensure proper cleanup of derived classes
};

class Sigmoid : public ActivationFunction {
   public:
    double activate(double weightedSum) override;
    std::unique_ptr<ActivationFunction> clone() const override;
};

class ReLU : public ActivationFunction {
   public:
    double activate(double weightedSum) override;
    std::unique_ptr<ActivationFunction> clone() const override;
};

#endif