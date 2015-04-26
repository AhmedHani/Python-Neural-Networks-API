__author__ = 'ENG.AHMED HANI'
from ActivationFunctions.MathFunction import *
import random


class Neuron(object):
    __weights = []
    __bias = 0.0
    __input = []
    __net = 0.0
    __output = 0.0
    __activationFunction = MathFunction
    __signalError = 0.0

    @property
    def Input(self):
        return self.__input

    @property
    def Weights(self):
        return self.__weights

    @property
    def SignalError(self):
        return self.__signalError

    @property
    def SignalError(self):
        pass

    @SignalError.getter
    def SignalError(self):
        return self.__signalError

    @SignalError.setter
    def SignalError(self, value):
        self.__signalError = value

    @property
    def Bias(self):
        return self.__bias

    @property
    def Net(self):
        pass

    @Net.getter
    def Net(self):
        return self.__net

    @property
    def Output(self):
        pass

    @Output.getter
    def Output(self):
        return self.__output

    @property
    def ActivationFunction(self):
        pass

    @ActivationFunction.getter
    def ActivationFunction(self):
        return self.__activationFunction

    def __init__(self, activationFunction, numberOfInput=None, weights=None, bias=None):
        """
        :param activationFunction: ActivationFunction
        :param numberOfInput: int
        :param weights: list
        :param bias: double

        Initialize the class with 3 different ways
        1: Given the number of input for the Neuron
        2: Given the weights for each input and the activation function
        3: Given the weights for each input, bias and activation function
        """

        if numberOfInput is not None:
            self.__weights = [round(random.random(), 3) for i in range(numberOfInput)]
            self.__bias = round(random.random(), 2)

        elif weights is not None and activationFunction is not None:
            self.__weights = weights
            self.__bias = round(random.random(), 2)
            numberOfInput = len(weights)

        elif weights is not None and bias is not None and activationFunction is not None:
            self.__weights = weights
            self.__bias = bias
            numberOfInput = len(weights)

        else:
            raise Exception("Invalid Arguments!")

        self.__activationFunction = activationFunction
        self.__input = [0.0 for i in range(numberOfInput)]
        self.__output = 0.0
        self.__signalError = 0.0

    @classmethod
    def __linearCalculation(self):
        size = len(self.__weights)
        result = 0.0

        for i in range(0, size):
            result += (self.__input[i] * self.__weights[i])

        result += self.__bias

        return result

    @classmethod
    def feedforward(self, input):
        """
        :param input: list
        :return: list

        Compute the output of the Neuron given the input
        """
        self.__input = input

        self.__net = self.__linearCalculation()
        self.__output = self.__activationFunction.function(MathFunction(), self.__net)

        return self.__output

    def update(self, weights, bias):
        """
        :param weights: list
        :param bias: double

        Set the new values of weights and bias after training
        """
        self.__weights = weights
        self.__bias = bias
