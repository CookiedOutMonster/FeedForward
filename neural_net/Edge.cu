#include <iostream>

#include "Edge.h"

Edge::Edge(const std::shared_ptr<ActivationNeuron>& baseNeuron,
           const std::shared_ptr<ActivationNeuron>& connectedNeuron, float weight, float bias)
    : baseNeuron(baseNeuron), connectedNeuron(connectedNeuron), weight(weight), bias(bias) {
    // Check for null pointers
    if (!baseNeuron) {
        throw std::invalid_argument("Base neuron cannot be null.");
    }
    if (!connectedNeuron) {
        throw std::invalid_argument("Connected neuron cannot be null.");
    }

    auto baseNeuronMemAddy = baseNeuron.get();
    auto connectedNeuronMemAddy = connectedNeuron.get();

    if (baseNeuronMemAddy == connectedNeuronMemAddy) {
        throw std::invalid_argument("Base neuron and connected neuron cannot be the same neuron.");
    }

    if (weight < 0.0f) {
        std::cerr << "Warning: Weight is negative, which may affect training.\n";
    }
    if (bias < 0.0f) {
        std::cerr << "Warning: Bias is negative, which may affect training.\n";
    }
}

double Edge::getWeight() {
    return this->weight;
}

void Edge::setWeight(double weight) {
    this->weight = weight;
}

double Edge::getBias() {
    return this->bias;
}

void Edge::setBias(double bias) {
    this->bias = bias;
}

std::shared_ptr<ActivationNeuron> Edge::getBaseNeuron() const {
    return this->baseNeuron;
}

std::shared_ptr<ActivationNeuron> Edge::getConnectedNeuron() const {
    return this->connectedNeuron;
}

void Edge::setBaseNeuron(std::shared_ptr<ActivationNeuron> baseNeuron) {
    this->baseNeuron = baseNeuron;
}

void Edge::setConnectedNeuron(std::shared_ptr<ActivationNeuron> connectedNeuron) {
    this->connectedNeuron = connectedNeuron;
}
