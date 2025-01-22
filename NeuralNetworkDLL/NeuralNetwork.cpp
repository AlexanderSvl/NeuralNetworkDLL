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