#pragma once

#ifndef NEURAL_NETWORK_H
#define NEUTRAL_NETWORK_H

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
	void Activation(double x);

	// Computes the derivative of the activation function for backpropagation.
	void ActivationDerivative(double x);
};

#endif // NEUTRAL_NETWORK_H