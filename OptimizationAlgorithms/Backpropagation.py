__author__ = 'Ahmed Hani Ibrahim'
from LearningAlgorithm import *


class Backpropagation(LearningAlgorithm):
    def learn(self, learningRate, input, output, network):
        """
        :param learningRate: double
        :param input: list
        :param output: list
        :param network: [[Neuron]]
        :return: [[Neuron]]

        Training the network with Backpropagation algorithm, it does the following
            1- Calculate the error signal for each neuron on each layer
            2- Update the weights of each neuron according to its update formula
            3- Return the new weights of the whole network
        """

        for i in range(len(network) - 1, 0, -1):
            for j in range(0, len(network[i])):
                currentNeuron = network[i][j]

                if i == len(network) - 1:
                    currentNeuron.SignalError = (output[j] - currentNeuron.Output) * \
                                                currentNeuron.ActivationFunction.derivative(currentNeuron.Net)
                else:
                    summation = 0.0

                    for k in range(0, len(network[i + 1])):
                        nextNeuron = network[i + 1][k]
                        summation += (nextNeuron.Weights[j] * nextNeuron.SignalError)

                    currentNeuron.SignalError = summation * currentNeuron.ActivationFunction.derivative(
                        currentNeuron.Net)

                network[i][j] = currentNeuron

        for i in range(0, len(network)):
            for j in range(0, len(network[i])):
                currentWeights = network[i][j].Weights
                currentBias = network[i][j].Bias

                for k in range(0, len(currentWeights)):
                    if i == 0:
                        currentWeights[k] += learningRate * network[i][j].SignalError * input[k]
                    else:
                        currentWeights[k] += learningRate * network[i][j].SignalError * network[i - 1][j].Output

                currentBias += learningRate * network[i][j].SignalError
                network[i][j].update(currentWeights, currentBias)

        return network