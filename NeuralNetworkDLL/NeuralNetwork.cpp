#include "pch.h"
#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(int input_layer_size, int hidden_layer_size, int output_layer_size)
	: inputSize(input_layer_size),
	  hiddenSize(hidden_layer_size),
	  outputSize(output_layer_size)
{
	// Memory allocation for the weight matrices
	weightsInputHidden.resize(hiddenSize, std::vector<double>(inputSize));
	weightsHiddenOutput.resize(outputSize, std::vector<double>(hiddenSize));

	// Memory allocation for the biases
	biasesHidden.resize(hiddenSize);
	biasesOutput.resize(outputSize);

	// Memory allocation for neurons	
	inputLayer.resize(inputSize);
	hiddenLayer.resize(hiddenSize);
	outputLayer.resize(outputSize);

	// Memory allocation for gradients (used during training)
	gradientsInputHidden.resize(hiddenSize, std::vector<double>(inputSize));
	gradientsHiddenOutput.resize(outputSize, std::vector<double>(hiddenSize));
	gradientsBiasesHidden.resize(hiddenSize);
	gradientsBiasesOutput.resize(outputSize);
}

void NeuralNetwork::InitializeWeights()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    // Xavier initialization bounds for input-hidden weights
    double boundInputHidden = sqrt(6.0 / (inputSize + hiddenSize));
    std::uniform_real_distribution<double> distInputHidden(-boundInputHidden, boundInputHidden);

    // Xavier initialization bounds for hidden-output weights
    double boundHiddenOutput = sqrt(6.0 / (hiddenSize + outputSize));
    std::uniform_real_distribution<double> distHiddenOutput(-boundHiddenOutput, boundHiddenOutput);

    // Initialize weightsInputHidden
    for (size_t i = 0; i < hiddenSize; ++i)
    {
        for (size_t j = 0; j < inputSize; ++j)
        {
            weightsInputHidden[i][j] = distInputHidden(gen);
        }
    }

    // Initialize weightsHiddenOutput
    for (size_t i = 0; i < outputSize; ++i)
    {
        for (size_t j = 0; j < hiddenSize; ++j)
        {
            weightsHiddenOutput[i][j] = distHiddenOutput(gen);
        }
    }

    // Initialize biases with a small constant (0.1)
    for (size_t i = 0; i < hiddenSize; ++i)
    {
        biasesHidden[i] = 0.1;
    }

    for (size_t i = 0; i < outputSize; ++i)
    {
        biasesOutput[i] = 0.1;
    }
}

std::vector<double> NeuralNetwork::Forward(const std::vector<double>& input)
{
    // Step 1: Calculate the activations for the hidden layer
    for (size_t i = 0; i < hiddenSize; ++i)
    {
        double sum = 0.0;

        // Calculate the weighted sum for each hidden neuron
        for (size_t j = 0; j < inputSize; ++j)
        {
            sum += input[j] * weightsInputHidden[i][j]; // Input multiplied by weights
        }

        sum += biasesHidden[i];             // Add bias for the hidden neuron
        hiddenLayer[i] = Activation(sum);   // Apply activation function (sigmoid)
    }

    // Step 2: Calculate the activations for the output layer
    for (size_t i = 0; i < outputSize; ++i)
    {
        double sum = 0.0;

        // Calculate the weighted sum for each output neuron
        for (size_t j = 0; j < hiddenSize; ++j)
        {
            sum += hiddenLayer[j] * weightsHiddenOutput[i][j]; // Hidden layer output multiplied by weights
        }

        sum += biasesOutput[i];             // Add bias for the output neuron
        outputLayer[i] = Activation(sum);   // Apply activation function (sigmoid)
    }

    return outputLayer;
}
    
double NeuralNetwork::Activation(double x) const
{
    return 1 / (1 + std::exp(-x));
}

double NeuralNetwork::ActivationDerivative(double sigmoidOutput) const
{
    return sigmoidOutput * (1 - sigmoidOutput);
}
