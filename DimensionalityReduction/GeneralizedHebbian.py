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
        """
        :param input: Number of features to be reduced --[double]
        :param numberOfFeatures: The number of resultant features after reduction --int
        :param learningRate: --double
        """
        self.__input = input
        self.__numberOfFeatures = numberOfFeatures
        self.__learningRate = learningRate
        self.__weights = [[random.random() for j in range(0, len(input))] for i in range(0, self.__numberOfFeatures)]

    def train(self, epochs, trainingSamples):
        """
        :param epochs: Number of iterations --int
        :param trainingSamples: The training data --[[double]]
        """
        for iter in range(0, epochs):
            for i in range(0, len(trainingSamples)):
                self.__output = self.featuresReduction(trainingSamples[i])
                self.__update(trainingSamples[i])

    def featuresReduction(self, features):
        """
        :param features: The features that to be reduced --[double]
        :return: The new features after reduction [double]
        """
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




