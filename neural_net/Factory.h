#ifndef FACTORY_H
#define FACTORY_H

#include <memory>
#include <string>
#include <vector>

#include "ActivationFunctions.h"
#include "Neuron.h"

using namespace std;

///////////////////////////////////////////////////////
//               Neuron Factory
///////////////////////////////////////////////////////

class NeuronFactory {
   public:
    // Creates an array of neurons with the specified activation function
    static vector<std::shared_ptr<ActivationNeuron>> createNeurons(
        int numberOfNeurons, const std::unique_ptr<ActivationFunction>& activation);

   private:
    // Helper function to create a single neuron
    static std::shared_ptr<ActivationNeuron> createNeuron(const std::unique_ptr<ActivationFunction>& activation);
};

///////////////////////////////////////////////////////
//               Activation Factory
///////////////////////////////////////////////////////

std::unique_ptr<ActivationFunction> createActivationFunction(const std::string& activationType);

#endif  // FACTORY_H
