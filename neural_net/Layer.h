#include <memory>
#include <stdexcept>  // For std::out_of_range
#include <vector>

#include "ActivationFunctions.h"
#include "Factory.h"
#include "Neuron.h"

#ifndef LAYER_H
#define LAYER_H

using namespace std;

class Layer {
   private:
    int numberOfNeurons = 0;
    // @TODO Cuda does not directly support the vector data structure. I will need to copy the contents of the vector to
    // the device and operate on it normally
    vector<shared_ptr<ActivationNeuron>> neurons;

   public:
    Layer(int numberOfNeurons, const unique_ptr<ActivationFunction>& activationPrototype);

    int getNumberOfNeurons() const;
    // Return a const reference to prevent modification
    // const before function call means it returns a constant
    // const after means it' s like a read only kinda thing.
    const vector<shared_ptr<ActivationNeuron>>& getNeuronsInLayer() const;

    // New method to get a shared pointer to a specific neuron
    shared_ptr<ActivationNeuron> getNeuron(int index) const;
};

#endif