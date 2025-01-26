#include <iostream>
#include <vector>
#include <ctime>
#include "NeuralNetwork.h"  // Assuming your DLL's header is correctly linked.

int main()
{
    // Seed the random number generator for reproducibility
    srand(time(0));

    // Create a NeuralNetwork object with input layer size of 3, hidden layer size of 5, and output layer size of 2
    NeuralNetwork nn(3, 5, 2);

    // Example training data (XOR-like problem)
    std::vector<std::vector<double>> inputs = {
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 1.0},
        {0.0, 1.0, 0.0},
        {0.0, 1.0, 1.0},
        {1.0, 0.0, 0.0},
        {1.0, 0.0, 1.0},
        {1.0, 1.0, 0.0},
        {1.0, 1.0, 1.0}
    };

    std::vector<std::vector<double>> targets = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0},
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };

    // Train the neural network
    std::cout << "Training the neural network...\n";
    nn.Train(inputs, targets, 0.001, 20000);  // Learning rate = 0.1, Epochs = 10000

    // Now test the neural network with the same inputs (just to see how well it's learned)
    std::cout << "\nTesting the trained neural network...\n";
    for (size_t i = 0; i < inputs.size(); i++)
    {
        std::vector<double> output = nn.Predict(inputs[i]);
        std::cout << "Input: ";
        for (double val : inputs[i])
        {
            std::cout << val << " ";
        }
        std::cout << " -> Prediction: ";
        for (double val : output)
        {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}