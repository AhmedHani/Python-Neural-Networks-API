__author__ = 'Ahmed Hani Ibrahim'
import abc


class MathFunction(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def function(self, input):
        return

    @abc.abstractmethod
    def derivative(self, input):
        return
