__author__ = 'Ahmed Hani Ibrahim'
from Neuron import *

class FeedforwardNeuralNetwork(object):
    __numberOfLayers = 0
    __numberOfInput = 0
    __network = [[Neuron]]
    __numberOfNeuronsPerLayer = 0

    def __init__(self, numberOfLayers):
        if numberOfLayers < 2:
            raise Exception("Can't Initiate Network with lower than 2 layers")

        self.__numberOfLayers = numberOfLayers
        self.__network = [[Neuron]]

    def setNetwork(self, numberOfNeuronsPerLayer):
        if self.__numberOfLayers != len(numberOfNeuronsPerLayer):
            raise Exception("Wrong List size for numOfNeuronsPerLayer")

        self.__numberOfNeuronsPerLayer = numberOfNeuronsPerLayer
        self.__numberOfInput = numberOfNeuronsPerLayer[0]

        self.__network = [Neuron for j in range(1, self.__numberOfLayers).append([Neuron])]

    def setLayer(self, layerIndex, activationFunction):
        if layerIndex == 0:
           raise Exception("Can't set Input Layer")

        for i in range(0, self.__numberOfNeuronsPerLayer[layerIndex]):
            neuron = Neuron(self.__numberOfNeuronsPerLayer[layerIndex - 1], activationFunction)
            self.__network[layerIndex - 1].append(neuron)

    def setNeuron(self, layerIndex, neuronIndex, weights, bias):
        if len(weights) != self.__numberOfNeuronsPerLayer[layerIndex - 1]:
            raise Exception("Invalid Weights Size!")

        self.__network[layerIndex - 1][neuronIndex].update(weights, bias)

    def feedforward(self, input):
        if len(input) != self.__numberOfInput:
            raise Exception("Invalid Input Size!")

        currentInput = input
        nextInput = []

        for i in range(1, self.__numberOfLayers):
            for j in range(0, self.__numberOfNeuronsPerLayer[j]):
                nextInput.append(self.__network[i - 1][j].feedforward(currentInput))

            currentInput = nextInput
            nextInput = []

        output = currentInput

        return output

    def train(self, trainingSamples, trainingLabels, learningRate, learningAlgorithm):
        for i in range(0, len(trainingSamples)):
            output = self.feedforward(trainingSamples[i])
            self.__network = learningAlgorithm.learn(learningRate, trainingSamples[i], trainingLabels[i], self.__network)



