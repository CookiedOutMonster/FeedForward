#include "Layer.h"

using namespace std;

Layer::Layer(int numberOfNeurons, const unique_ptr<ActivationFunction>& activationPrototype) {
    this->numberOfNeurons = numberOfNeurons;
    neurons = NeuronFactory::createNeurons(numberOfNeurons, activationPrototype);
}

const vector<shared_ptr<ActivationNeuron>>& Layer::getNeuronsInLayer() const {
    return this->neurons;
}

int Layer::getNumberOfNeurons() const {
    return this->numberOfNeurons;
}

shared_ptr<ActivationNeuron> Layer::getNeuron(int index) const {
    if (index < 0 || index >= numberOfNeurons) {
        throw out_of_range("Neuron index out of bounds");
    }
    return neurons[index];
}
