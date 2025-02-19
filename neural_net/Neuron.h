#include <memory>

#include "ActivationFunctions.h"
#include "Edge.h"

#ifndef NEURON_H
#define NEURON_H

// this follow's the L in SOLID
class NeuronManager {
   private:
    double weightedSum;
    double activationValue;

    size_t numberOfEdges;
    unique_ptr<unique_ptr<Edge>[]> edges;

   public:
    double getWeightedSum();
    void setWeightedSum(double incomingWeightedSum);

    double getActivationValue();
    void setActivationValue(double incomingActivationValue);

    size_t getNumberOfEdges();
    void setNumberOfEdges(size_t numOfEdges);

    const unique_ptr<Edge>& getEdge(size_t index) const;
    void setEdge(size_t index, unique_ptr<Edge> edge);
};

class InputNeuron {
   private:
    NeuronManager neuronManager;

   public:
    double getWeightedSum();

    void setWeightedSum(double incomingWeightedSum);
};

class ActivationNeuron {
   private:
    NeuronManager neuronManager;
    std::unique_ptr<ActivationFunction> activationFunction;

    std::vector<std::shared_ptr<Edge>> incomingEdges;
    std::vector<std::shared_ptr<Edge>> outgoingEdges;

   public:
    ActivationNeuron(std::unique_ptr<ActivationFunction> activation);

    double getWeightedSum();
    void setWeightedSum(double incomingWeightedSum);

    double getActivationValue();
    void setActivationValue(double incomingActivationValue);

    double activate();

    void addIncomingEdge(const std::shared_ptr<Edge>& edge);
    void addOutgoingEdge(const std::shared_ptr<Edge>& edge);

    const std::vector<std::shared_ptr<Edge>>& getIncomingEdges() const;
    const std::vector<std::shared_ptr<Edge>>& getOutgoingEdges() const;
};

#endif  // NEURON_H

/*
- use a destructor to "clean up" memory allocations in a class

- use a destructor when using system resources like file handling or sockets

- or when I want to do something

*/

// the idea is to do

// NeuralNetwork neural net([nodesPerLayer...], activationFunction)