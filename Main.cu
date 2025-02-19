#include <gtest/gtest.h>

#include <cmath>
#include <fstream>    // for std::ifstream
#include <iostream>   // for std::cout, std::cerr
#include <stdexcept>  // for std::runtime_error
#include <string>     // for std::string

#include "MNISTReader.h"
#include "neural_net/NeuralNetwork.h"

__global__ void what() {
    printf("Hello from GPU!\n");
}

void tryAndReadImage(std::string fullImageDirectory) {
    try {
        MNISTReader reader(fullImageDirectory);  // Adjust the path to your MNIST image file
        int a = reader.get_image_size();
        std::cout << a << std::endl;

        uchar* image = reader.get_image(0);
        for (int i = 0; i < 784; i++) {
            uchar pixel = image[i];
            std::cout << " " + std::to_string((int)pixel) + " ";

            if (i % 28 == 0) {
                std::cout << "\n";
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

int main() {
    NeuralNetwork nn({5, 5}, "sigmoid", true);

    std::vector<float> inputData = {1, 2, 3, 4, 5};

    nn.feedForwardCUDA(inputData);

    return 0;
}

/*
int main() {

    std::string DATA_SUB_FOLDER = "data";
    std::string TRAINING_IMAGES_FILE_NAME = "train-images.idx3-ubyte";
    int IMAGE_SIZE;

    constexpr double e = 2.718281828459045;

    // 4 nodes x 3 output

    float previousNeuronActivations[4] = {1, 2, 3, 2.5};

    float edgeWeightsPriorLayer[3][4] = {{2.3, 3.5, 0.7, 0.1}, {0.5, 1.7, 3.6, 6.9}, {1.2, 0.11, 0.23, 5}};

    float biases[3] = {3, 7, 1};

    int outputNeuronValues[3] = {0, 0, 0};

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            outputNeuronValues[i] += previousNeuronActivations[j] * edgeWeightsPriorLayer[i][j];
        }
        outputNeuronValues[i] += biases[i];
    }

    for (int i = 0; i < 3; i++) {
        std::cout << outputNeuronValues[i] << std::endl;
    }

    // Activation is add up inputs plus weights plus bias

    /*
    double result = 0;

    for (int i = 0; i < 4; i++) {
        result = result + inputs[i] + weights[i];
    }

    result = result + bias;
    result = -result;

    // and then you would need to put this into a sigmoid function

    std::cout << result << std::endl;

    // sigmoid
    // double pow(double base, double exponent);

    double resultingSigmoid = 1 / (1 + std::exp(result));  // Use exp for e^x
    std::cout << resultingSigmoid << std::endl;




    /*
    // Launch the CUDA kernel with 1 block and 1024 threads
    what<<<1, 1>>>();

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    std::string trainingImagesFullDirectory = DATA_SUB_FOLDER + "/" + TRAINING_IMAGES_FILE_NAME;
    std::cout << trainingImagesFullDirectory << std::endl;  // std::endl prints a newline after the string


    return 0;
}
*/