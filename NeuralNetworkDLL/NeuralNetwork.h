#pragma once

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <string>

class NeuralNetwork
{
public:
	// Allocates memory for weights, biases and neuron values.
	NeuralNetwork(int input_layer_size, int hidden_layer_size, int output_layer_size);

	~NeuralNetwork();

	// Forward propagation for prediction.
	std::vector<double> Predict(const std::vector<double>& input);

	// Training function for the neural network.
	void Train(const std::vector<double>& inputs,
		const std::vector<double>& targets,
		double learning_rate,
		int epochs);

	// Utility functions
	void PrintNetwork();
	void SaveModel(const std::string& file_path);
	void LoadModel(const std::string& file_path);

private:
	// Randomly initializes weights and biases for the network connections.
	void InitializeWeights();

	// Forward propagation
	std::vector<double> Forward(const std::vector<double>& input);

	// Backward propagation
	void Backward(const std::vector<double>& target);

	// Error calculation between predicted output and target output. 
	double CalculateLoss(const std::vector<double>& predicted_output,
		const std::vector<double>& targed_output);

	// Applies the activation function (e.g., Sigmoid, ReLU) to a given value x.
	double Activation(double x) const;
	// Computes the derivative of the activation function for backpropagation.
	double ActivationDerivative(double x) const;

	// Layer sizes
	int inputSize;
	int hiddenSize;
	int outputSize;

	// Weights and biases
	std::vector<std::vector<double>> weightsInputHidden;	// Weights from input to hidden
	std::vector<std::vector<double>> weightsHiddenOutput;	// Weights from hidden to output
	std::vector<double> biasesHidden;						// Biases for hidden layer
	std::vector<double> biasesOutput;						// Biases for output layer

	// Neuron values (activations)
	std::vector<double> inputLayer;							// Input layer activations
	std::vector<double> hiddenLayer;						// Hidden layer activations
	std::vector<double> outputLayer;						// Output layer activations

	// Gradients for backpropagation
	std::vector<std::vector<double>> gradientsInputHidden;  // Gradients for input-hidden weights
	std::vector<std::vector<double>> gradientsHiddenOutput; // Gradients for hidden-output weights
	std::vector<double> gradientsBiasesHidden;              // Gradients for hidden biases
	std::vector<double> gradientsBiasesOutput;              // Gradients for output biases

	// Training parameter
	double learningRate;									// Learning rate for gradient descent
};

#endif // NEURAL_NETWORK_H