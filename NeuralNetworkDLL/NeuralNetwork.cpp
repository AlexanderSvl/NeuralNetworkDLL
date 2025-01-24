#include "pch.h"
#include "NeuralNetwork.h"
#include <iostream>

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

NeuralNetwork::~NeuralNetwork()
{
}

std::vector<double> NeuralNetwork::Predict(const std::vector<double>& input)
{
    return Forward(input);
}

void NeuralNetwork::Train(const std::vector<std::vector<double>>& inputs,
                          const std::vector<std::vector<double>>& targets,
                          double learning_rate,
                          int epochs)
{
    // Initialize the weights and biases randomly
    InitializeWeights();

    // Set the learning rate
    learningRate = learning_rate;

    // Iterate through epochs (iterations of training)
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        double total_loss = 0.0;

        // Loop through each training example
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            // Get the input and target for the current example
            const std::vector<double>& input = inputs[i];
            const std::vector<double>& target = targets[i];

            // Perform the forward pass
            outputLayer = Forward(input);

            // Calculate the loss for the current example
            total_loss += CalculateLoss(outputLayer, target);

            // Perform the backward pass to calculate gradients
            Backward(target);

            // Update weights and biases using the gradients and learning rate
            // Update weights from input to hidden
            for (size_t j = 0; j < hiddenSize; ++j)
            {
                for (size_t k = 0; k < inputSize; ++k)
                {
                    weightsInputHidden[j][k] -= learningRate * gradientsInputHidden[j][k];
                }
            }

            // Update weights from hidden to output
            for (size_t j = 0; j < outputSize; ++j)
            {
                for (size_t k = 0; k < hiddenSize; ++k)
                {
                    weightsHiddenOutput[j][k] -= learningRate * gradientsHiddenOutput[j][k];
                }
            }

            // Update biases for hidden layer
            for (size_t j = 0; j < hiddenSize; ++j)
            {
                biasesHidden[j] -= learningRate * gradientsBiasesHidden[j];
            }

            // Update biases for output layer
            for (size_t j = 0; j < outputSize; ++j)
            {
                biasesOutput[j] -= learningRate * gradientsBiasesOutput[j];
            }
        }

        // Print the loss for the current epoch (optional)
        total_loss /= inputs.size();
        std::cout << "Epoch " << epoch + 1 << "/" << epochs << " - Loss: " << total_loss << std::endl;
    }
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

void NeuralNetwork::Backward(const std::vector<double>& target)
{
    std::vector<double> output_layer_error = std::vector<double>(outputSize);

    // Calculation of outpu layer error
    for (size_t i = 0; i < outputSize; i++)
    {
        output_layer_error[i] = outputLayer[i] - target[i];
    }

    std::vector<double> hidden_layer_error = std::vector<double>(hiddenSize);

    // Calculation of hidden layer error
    for (size_t i = 0; i < hiddenSize; ++i)
    {
        double error = 0.0;

        // Calculate error for hidden neuron i
        for (size_t j = 0; j < outputSize; ++j)
        {
            error += output_layer_error[j] * weightsHiddenOutput[j][i];
        }

        // Apply the derivative of the activation function for hidden layer
        hidden_layer_error[i] = error * ActivationDerivative(hiddenLayer[i]);
    }

    // Calculation of the gradients
    for (size_t i = 0; i < outputSize; ++i)
    {
        for (size_t j = 0; j < hiddenSize; ++j)
        {
            gradientsHiddenOutput[i][j] = output_layer_error[i] * hiddenLayer[j];
        }
    }

    // Calculation of the gradients
    for (size_t i = 0; i < inputSize; ++i)
    {
        for (size_t j = 0; j < hiddenSize; ++j)
        {
            gradientsInputHidden[i][j] = hidden_layer_error[j] * inputLayer[i];
        }
    }

    // Calculation of the gradients
    for (size_t i = 0; i < outputSize; ++i)
    {
        gradientsBiasesOutput[i] = output_layer_error[i];
    }

    // Calculation of the gradients
    for (size_t i = 0; i < hiddenSize; ++i)
    {
        gradientsBiasesHidden[i] = hidden_layer_error[i];
    }
}

double NeuralNetwork::CalculateLoss(const std::vector<double>& predicted_output, const std::vector<double>& target_output)
{
    double total_loss = 0.0;

    for (size_t i = 0; i < outputSize; i++)
    {
        total_loss += std::pow((predicted_output[i] - target_output[i]), 2);
    }

    total_loss /= static_cast<double>(outputSize);

    return total_loss;
}
    
double NeuralNetwork::Activation(double x) const
{
    return 1 / (1 + std::exp(-x));
}

double NeuralNetwork::ActivationDerivative(double sigmoidOutput) const
{
    return sigmoidOutput * (1 - sigmoidOutput);
}
