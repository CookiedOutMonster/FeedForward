#include <stdexcept>

#include "Neuron.h"

///////////////////////////////////////////////////////
////               Nueron Manager                  ////
///////////////////////////////////////////////////////

// Getter for the weighted sum
double NeuronManager ::getWeightedSum() {
    return this->weightedSum;
}

// Setter for the weighted sum
void NeuronManager ::setWeightedSum(double incomingWeightedSum) {
    this->weightedSum = incomingWeightedSum;
}

// Getter for the weighted sum
double NeuronManager ::getActivationValue() {
    return this->activationValue;
}

// Setter for the weighted sum
void NeuronManager ::setActivationValue(double incomingActivationValue) {
    this->activationValue = incomingActivationValue;
}

const unique_ptr<Edge>& NeuronManager::getEdge(size_t index) const {
    if (index >= this->numberOfEdges) {
        throw out_of_range("Edge index out of bounds");
    }
    return edges[index];
}

void NeuronManager ::setEdge(size_t index, unique_ptr<Edge> edge) {
    if (index >= this->numberOfEdges) {
        throw out_of_range("Edge index out of bounds");
    }
    edges[index] = std::move(edge);
}

///////////////////////////////////////////////////////
////               Input Neuron                    ////
///////////////////////////////////////////////////////

double InputNeuron ::getWeightedSum() {
    return neuronManager.getWeightedSum();
}

void InputNeuron ::setWeightedSum(double incomingWeightedSum) {
    neuronManager.setWeightedSum(incomingWeightedSum);
}

///////////////////////////////////////////////////////
////               Activation Neuronn              ////
///////////////////////////////////////////////////////

ActivationNeuron::ActivationNeuron(std::unique_ptr<ActivationFunction> activation)
    : activationFunction(std::move(activation)) {
    setWeightedSum(0.0f);
}

double ActivationNeuron ::getWeightedSum() {
    return neuronManager.getWeightedSum();
}

void ActivationNeuron ::setWeightedSum(double incomingWeightedSum) {
    neuronManager.setWeightedSum(incomingWeightedSum);
}

// Getter for the weighted sum
double ActivationNeuron ::getActivationValue() {
    return neuronManager.getActivationValue();
}

// Setter for the weighted sum
void ActivationNeuron ::setActivationValue(double incomingActivationValue) {
    neuronManager.setActivationValue(incomingActivationValue);
}

double ActivationNeuron::activate() {
    double activationValue = activationFunction->activate(neuronManager.getWeightedSum());
    neuronManager.setActivationValue(activationValue);
    return activationValue;
}

void ActivationNeuron::addIncomingEdge(const std::shared_ptr<Edge>& edge) {
    incomingEdges.push_back(edge);
}

void ActivationNeuron::addOutgoingEdge(const std::shared_ptr<Edge>& edge) {
    outgoingEdges.push_back(edge);
}

const std::vector<std::shared_ptr<Edge>>& ActivationNeuron::getIncomingEdges() const {
    return incomingEdges;
}

const std::vector<std::shared_ptr<Edge>>& ActivationNeuron::getOutgoingEdges() const {
    return outgoingEdges;
}
