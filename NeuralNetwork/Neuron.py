__author__ = 'ENG.AHMED HANI'
from ActivationFunctions.MathFunction import *


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
    def Bias(self):
        return self.__bias

    @property
    def Net(self):
        return self.__net

    @property
    def Output(self):
        return self.__output

    @property
    def ActivationFunction(self):
        return self.__activationFunction

    def __init__(self, activationFunction, numberOfInput=None, weights=None, bias=None):
        if numberOfInput is not None:
            self.__weights = [0.0 for i in range(numberOfInput)]
            self.__bias = 0.0

        elif weights is not None and activationFunction is not None:
            self.__activationFunction = activationFunction
            self.__weights = weights
            self.__bias = 0.0
            numberOfInput = len(weights)

        elif weights is not None and bias is not None and activationFunction is not None:
            self.__weights = weights
            self.__bias = bias
            self.__activationFunction = activationFunction
            numberOfInput = len(weights)

        else:
            raise Exception("Invalid Arguments!")

        self.__activationFunction = activationFunction
        self.__input = [0.0 for i in range(numberOfInput)]
        self.__output = 0.0
        self.__signalError = 0.0

    @classmethod
    def __init__(cls, weights, bias, activationFunction):
        cls.__weights = weights
        cls.__init__(len(weights))
        cls.__bias = bias
        cls.__activationFunction = activationFunction

    def __linearCalculation(self):
        size = len(self.__weights)
        result = 0.0

        for i in range(0, size):
            result += (self.__input[i] * self.__weights[i])

        result += self.__bias

        return result

    def feedforward(self, input):
        self.__input = input

        self.__net = self.__linearCalculation()
        self.__output = self.__activationFunction.function(self.__net)

        return self.__output

    def update(self, weights, bias):
        self.__weights = weights
        self.__bias = bias
