__author__ = 'ENG.AHMED HANI'
from MathFunction import *
import math


class Sigmoid(MathFunction):

    def function(self, input):
        sigmoid = (1 / (1 + math.exp(-input)))

        return sigmoid

    def derivative(self, input):
        sigmoid = self.function(input)

        return sigmoid * (1 - sigmoid)
