#include <stdexcept>
#include <vector>

#include "Factory.h"
#include "Neuron.h"

using namespace std;

///////////////////////////////////////////////////////
//               Neuron Factory
///////////////////////////////////////////////////////

vector<std::shared_ptr<ActivationNeuron>> NeuronFactory::createNeurons(
    int numberOfNeurons, const unique_ptr<ActivationFunction>& activation) {
    vector<std::shared_ptr<ActivationNeuron>> neurons;
    neurons.reserve(numberOfNeurons);

    for (int i = 0; i < numberOfNeurons; ++i) {
        neurons.push_back(createNeuron(activation));
    }

    return neurons;
}

std::shared_ptr<ActivationNeuron> NeuronFactory::createNeuron(const std::unique_ptr<ActivationFunction>& activation) {
    return make_shared<ActivationNeuron>(activation->clone());
}

///////////////////////////////////////////////////////
//               Activation Factory
///////////////////////////////////////////////////////

std::unique_ptr<ActivationFunction> createActivationFunction(const std::string& activationType) {
    if (activationType == "sigmoid") {
        return make_unique<Sigmoid>();
    } else if (activationType == "relu") {
        return make_unique<ReLU>();
    } else {
        throw invalid_argument("Activaton function not supported: " + activationType);
    }
}