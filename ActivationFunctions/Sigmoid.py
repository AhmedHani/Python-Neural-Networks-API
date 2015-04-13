__author__ = 'ENG.AHMED HANI'
from MathFunction import *
import math


class Sigmoid(MathFunction):

    def function(self, input):
        """
        :param input: double
        :return: double

        Compute the sigmoid of the given input
        """
        sigmoid = (1 / (1 + math.exp(-input)))

        return sigmoid

    def derivative(self, input):
        """
        :param input: double
        :return: double

        Compute the derivative of the Sigmoid value
        """
        sigmoid = self.function(input)

        return sigmoid * (1 - sigmoid)
