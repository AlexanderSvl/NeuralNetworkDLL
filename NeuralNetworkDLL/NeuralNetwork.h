#pragma once

#include "pch.h"

#ifdef NEURALNETWORK_EXPORTS
#define NEURALNETWORK_API __declspec(dllexport)
#else
#define NEURALNETWORK_API __declspec(dllimport)
#endif

class NEURALNETWORK_API  NeuralNetwork
{
public:
    // Constructor to allocate memory for weights, biases, and neuron values
    NeuralNetwork(int input_layer_size, int hidden_layer_size, int output_layer_size, double learning_rate = 0.01);

    // Destructor
    ~NeuralNetwork();

    // Forward propagation for prediction
    std::vector<double> Predict(const std::vector<double>& input);

    // Training function for the neural network
    void Train(const std::vector<std::vector<double>>& inputs,
        const std::vector<std::vector<double>>& targets,
        double learning_rate,
        int epochs);

    // Testing function to evaluate the network on unseen data
    void Test(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets);

    // Save the network's weights and biases to a file
    void SaveModel(const std::string& file_path);

    // Load the network's weights and biases from a file
    void LoadModel(const std::string& file_path);

    // Print the network's structure and weights
    void PrintNetwork();

private:
    // Randomly initializes weights and biases for the network connections
    void InitializeWeights();

    // Forward propagation
    std::vector<double> Forward(const std::vector<double>& input);

    // Backward propagation
    void Backward(const std::vector<double>& target);

    // Error calculation between predicted output and target output
    double CalculateLoss(const std::vector<double>& predicted_output, const std::vector<double>& target_output);

    // Activation function (Sigmoid)
    double Activation(double x) const;

    // Derivative of the activation function
    double ActivationDerivative(double sigmoidOutput) const;

    // Layer sizes
    int inputSize;
    int hiddenSize;
    int outputSize;

    // Weights and biases
    std::vector<std::vector<double>> weightsInputHidden; // Weights from input to hidden
    std::vector<std::vector<double>> weightsHiddenOutput; // Weights from hidden to output
    std::vector<double> biasesHidden; // Biases for hidden layer
    std::vector<double> biasesOutput; // Biases for output layer

    // Neuron values (activations)
    std::vector<double> inputLayer; // Input layer activations
    std::vector<double> hiddenLayer; // Hidden layer activations
    std::vector<double> outputLayer; // Output layer activations

    // Gradients for backpropagation
    std::vector<std::vector<double>> gradientsInputHidden; // Gradients for input-hidden weights
    std::vector<std::vector<double>> gradientsHiddenOutput; // Gradients for hidden-output weights
    std::vector<double> gradientsBiasesHidden; // Gradients for hidden biases
    std::vector<double> gradientsBiasesOutput; // Gradients for output biases

    // Training parameter
    double learningRate; // Learning rate for gradient descent
};