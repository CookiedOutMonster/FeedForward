#include <memory>
#include <vector>

class ActivationNeuron;

#ifndef EDGE_H
#define EDGE_H

using namespace std;

class Edge {
   private:
    double weight;
    double bias;
    shared_ptr<ActivationNeuron> connectedNeuron;
    shared_ptr<ActivationNeuron> baseNeuron;

   public:
    Edge(const std::shared_ptr<ActivationNeuron>& baseNeuron, const std::shared_ptr<ActivationNeuron>& connectedNeuron,
         float weight, float bias);

    double getWeight();
    void setWeight(double weight);

    double getBias();
    void setBias(double bias);

    std::shared_ptr<ActivationNeuron> getBaseNeuron() const;
    std::shared_ptr<ActivationNeuron> getConnectedNeuron() const;

    void setBaseNeuron(std::shared_ptr<ActivationNeuron> toNeuron);
    void setConnectedNeuron(std::shared_ptr<ActivationNeuron> fromNeuron);
};

#endif