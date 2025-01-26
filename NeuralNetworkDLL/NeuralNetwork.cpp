#include "pch.h"
#include "NeuralNetwork.h"

// Constructor: Allocate memory for weights, biases, and neuron activations
NeuralNetwork::NeuralNetwork(int input_layer_size, int hidden_layer_size, int output_layer_size, double learning_rate)
    : inputSize(input_layer_size),
    hiddenSize(hidden_layer_size),
    outputSize(output_layer_size),
    learningRate(learning_rate)
{
    // Initialize weights and biases
    weightsInputHidden.resize(hiddenSize, std::vector<double>(inputSize));
    weightsHiddenOutput.resize(outputSize, std::vector<double>(hiddenSize));
    biasesHidden.resize(hiddenSize);
    biasesOutput.resize(outputSize);

    // Initialize neuron layers
    inputLayer.resize(inputSize);
    hiddenLayer.resize(hiddenSize);
    outputLayer.resize(outputSize);

    // Initialize gradients
    gradientsInputHidden.resize(hiddenSize, std::vector<double>(inputSize));
    gradientsHiddenOutput.resize(outputSize, std::vector<double>(hiddenSize));
    gradientsBiasesHidden.resize(hiddenSize);
    gradientsBiasesOutput.resize(outputSize);
}

// Destructor
NeuralNetwork::~NeuralNetwork() {}

// Predict function
std::vector<double> NeuralNetwork::Predict(const std::vector<double>& input)
{
    return Forward(input);
}

// Train the neural network
void NeuralNetwork::Train(const std::vector<std::vector<double>>& inputs,
    const std::vector<std::vector<double>>& targets,
    double learning_rate,
    int epochs)
{
    InitializeWeights();
    learningRate = learning_rate;

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        double total_loss = 0.0;

        for (size_t i = 0; i < inputs.size(); ++i)
        {
            const std::vector<double>& input = inputs[i];
            const std::vector<double>& target = targets[i];

            outputLayer = Forward(input);
            total_loss += CalculateLoss(outputLayer, target);

            Backward(target);

            // Update weights and biases
            for (size_t j = 0; j < hiddenSize; ++j)
            {
                for (size_t k = 0; k < inputSize; ++k)
                {
                    weightsInputHidden[j][k] -= learningRate * gradientsInputHidden[j][k];
                }
            }

            for (size_t j = 0; j < outputSize; ++j)
            {
                for (size_t k = 0; k < hiddenSize; ++k)
                {
                    weightsHiddenOutput[j][k] -= learningRate * gradientsHiddenOutput[j][k];
                }
            }

            for (size_t j = 0; j < hiddenSize; ++j)
            {
                biasesHidden[j] -= learningRate * gradientsBiasesHidden[j];
            }

            for (size_t j = 0; j < outputSize; ++j)
            {
                biasesOutput[j] -= learningRate * gradientsBiasesOutput[j];
            }
        }

        total_loss /= inputs.size();
        std::cout << "Epoch " << epoch + 1 << "/" << epochs << " - Loss: " << total_loss << std::endl;
    }
}

// Initialize weights using Xavier initialization
void NeuralNetwork::InitializeWeights()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    double boundInputHidden = sqrt(6.0 / (inputSize + hiddenSize));
    std::uniform_real_distribution<double> distInputHidden(-boundInputHidden, boundInputHidden);

    double boundHiddenOutput = sqrt(6.0 / (hiddenSize + outputSize));
    std::uniform_real_distribution<double> distHiddenOutput(-boundHiddenOutput, boundHiddenOutput);

    for (size_t i = 0; i < hiddenSize; ++i)
    {
        for (size_t j = 0; j < inputSize; ++j)
        {
            weightsInputHidden[i][j] = distInputHidden(gen);
        }
    }

    for (size_t i = 0; i < outputSize; ++i)
    {
        for (size_t j = 0; j < hiddenSize; ++j)
        {
            weightsHiddenOutput[i][j] = distHiddenOutput(gen);
        }
    }

    for (size_t i = 0; i < hiddenSize; ++i)
    {
        biasesHidden[i] = 0.1;
    }

    for (size_t i = 0; i < outputSize; ++i)
    {
        biasesOutput[i] = 0.1;
    }
}

// Forward pass
std::vector<double> NeuralNetwork::Forward(const std::vector<double>& input)
{
    for (size_t i = 0; i < hiddenSize; ++i)
    {
        double sum = 0.0;
        for (size_t j = 0; j < inputSize; ++j)
        {
            sum += input[j] * weightsInputHidden[i][j];
        }
        sum += biasesHidden[i];
        hiddenLayer[i] = Activation(sum);
    }

    for (size_t i = 0; i < outputSize; ++i)
    {
        double sum = 0.0;
        for (size_t j = 0; j < hiddenSize; ++j)
        {
            sum += hiddenLayer[j] * weightsHiddenOutput[i][j];
        }
        sum += biasesOutput[i];
        outputLayer[i] = Activation(sum);
    }

    return outputLayer;
}

// Backward pass
void NeuralNetwork::Backward(const std::vector<double>& target)
{
    std::vector<double> output_layer_error(outputSize);

    for (size_t i = 0; i < outputSize; ++i)
    {
        output_layer_error[i] = outputLayer[i] - target[i];
    }

    std::vector<double> hidden_layer_error(hiddenSize);

    for (size_t i = 0; i < hiddenSize; ++i)
    {
        double error = 0.0;
        for (size_t j = 0; j < outputSize; ++j)
        {
            error += output_layer_error[j] * weightsHiddenOutput[j][i];
        }
        hidden_layer_error[i] = error * ActivationDerivative(hiddenLayer[i]);
    }

    for (size_t i = 0; i < outputSize; ++i)
    {
        for (size_t j = 0; j < hiddenSize; ++j)
        {
            gradientsHiddenOutput[i][j] = output_layer_error[i] * hiddenLayer[j];
        }
    }

    for (size_t i = 0; i < hiddenSize; ++i)
    {
        for (size_t j = 0; j < inputSize; ++j)
        {
            gradientsInputHidden[i][j] = hidden_layer_error[i] * inputLayer[j];
        }
    }

    for (size_t i = 0; i < outputSize; ++i)
    {
        gradientsBiasesOutput[i] = output_layer_error[i];
    }

    for (size_t i = 0; i < hiddenSize; ++i)
    {
        gradientsBiasesHidden[i] = hidden_layer_error[i];
    }
}

// Calculate loss (MSE)
double NeuralNetwork::CalculateLoss(const std::vector<double>& predicted_output, const std::vector<double>& target_output)
{
    double total_loss = 0.0;
    for (size_t i = 0; i < outputSize; ++i)
    {
        total_loss += std::pow(predicted_output[i] - target_output[i], 2);
    }
    return total_loss / outputSize;
}

// Test the network
void NeuralNetwork::Test(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets)
{
    double total_loss = 0.0;

    for (size_t i = 0; i < inputs.size(); ++i)
    {
        std::vector<double> prediction = Forward(inputs[i]);
        total_loss += CalculateLoss(prediction, targets[i]);
    }

    std::cout << "Test Loss: " << total_loss / inputs.size() << std::endl;
}

// Save model
void NeuralNetwork::SaveModel(const std::string& file_path)
{
    std::ofstream file(file_path, std::ios::binary);

    if (!file.is_open())
    {
        std::cerr << "Error opening file for saving model!" << std::endl;
        return;
    }

    file.write(reinterpret_cast<char*>(&inputSize), sizeof(inputSize));
    file.write(reinterpret_cast<char*>(&hiddenSize), sizeof(hiddenSize));
    file.write(reinterpret_cast<char*>(&outputSize), sizeof(outputSize));

    for (const auto& row : weightsInputHidden)
    {
        file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
    }

    for (const auto& row : weightsHiddenOutput)
    {
        file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
    }

    file.write(reinterpret_cast<const char*>(biasesHidden.data()), biasesHidden.size() * sizeof(double));
    file.write(reinterpret_cast<const char*>(biasesOutput.data()), biasesOutput.size() * sizeof(double));

    file.close();
}

// Load model
void NeuralNetwork::LoadModel(const std::string& file_path)
{
    std::ifstream file(file_path, std::ios::binary);

    if (!file.is_open())
    {
        std::cerr << "Error opening file for loading model!" << std::endl;
        return;
    }

    file.read(reinterpret_cast<char*>(&inputSize), sizeof(inputSize));
    file.read(reinterpret_cast<char*>(&hiddenSize), sizeof(hiddenSize));
    file.read(reinterpret_cast<char*>(&outputSize), sizeof(outputSize));

    weightsInputHidden.resize(hiddenSize, std::vector<double>(inputSize));
    weightsHiddenOutput.resize(outputSize, std::vector<double>(hiddenSize));
    biasesHidden.resize(hiddenSize);
    biasesOutput.resize(outputSize);

    for (auto& row : weightsInputHidden)
    {
        file.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
    }

    for (auto& row : weightsHiddenOutput)
    {
        file.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
    }

    file.read(reinterpret_cast<char*>(biasesHidden.data()), biasesHidden.size() * sizeof(double));
    file.read(reinterpret_cast<char*>(biasesOutput.data()), biasesOutput.size() * sizeof(double));

    file.close();
}

// Activation function (Sigmoid)
double NeuralNetwork::Activation(double x) const
{
    return 1.0 / (1.0 + std::exp(-x));
}

// Derivative of the activation function (Sigmoid)
double NeuralNetwork::ActivationDerivative(double sigmoidOutput) const
{
    return sigmoidOutput * (1.0 - sigmoidOutput);
}

// Print network structure and weights
void NeuralNetwork::PrintNetwork()
{
    std::cout << "Input Layer Size: " << inputSize << std::endl;
    std::cout << "Hidden Layer Size: " << hiddenSize << std::endl;
    std::cout << "Output Layer Size: " << outputSize << std::endl;

    std::cout << "Weights (Input -> Hidden):" << std::endl;
    for (const auto& row : weightsInputHidden)
    {
        for (double w : row)
        {
            std::cout << w << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Weights (Hidden -> Output):" << std::endl;
    for (const auto& row : weightsHiddenOutput)
    {
        for (double w : row)
        {
            std::cout << w << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Biases (Hidden): ";
    for (double b : biasesHidden)
    {
        std::cout << b << " ";
    }
    std::cout << std::endl;

    std::cout << "Biases (Output): ";
    for (double b : biasesOutput)
    {
        std::cout << b << " ";
    }
    std::cout << std::endl;
}