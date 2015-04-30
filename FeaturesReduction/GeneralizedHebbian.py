__author__ = 'Ahmed Hani Ibrahim'

import random


class GeneralizedHebbian(object):
    __input = []
    __numberOfFeatures = 0
    __output = []
    __weights = [[]]
    __learningRate = 0.0

    @property
    def Weights(self):
        pass
    @Weights.getter
    def Weights(self):
        return self.__weights

    def __init__(self, input, numberOfFeatures, learningRate):
        self.__input = input
        self.__numberOfFeatures = numberOfFeatures
        self.__learningRate = learningRate
        self.__weights = [[random.random() for j in range(0, len(input))] for i in range(0, self.__numberOfFeatures)]

    def train(self, epochs, trainingSamples):
        for iter in range(0, epochs):
            for i in range(0, len(trainingSamples)):
                self.__output = self.featuresReduction(trainingSamples[i])
                self.__update(trainingSamples[i])

    def featuresReduction(self, features):
        output = [0.0 for i in range(0, self.__numberOfFeatures)]

        for i in range(0, self.__numberOfFeatures):
            for j in range(0, len(features)):
                output[i] += features[j] * self.__weights[i][j]

        return output

    @classmethod
    def __update(cls, features):
        for i in range(0, len(cls.__weights)):
            for j in range(0, len(cls.__weights[i])):
                cls.__weights[i][j] += cls.__learningRate * cls.__output[i] * (features[j] - cls.__computeOutput(i, j))

    @classmethod
    def __computeOutput(cls, outputIndex, inputIndex):
        sum = 0.0
        sum += [cls.__output[i] * cls.__weights[i][inputIndex] for i in range(0, outputIndex)]

        return sum




