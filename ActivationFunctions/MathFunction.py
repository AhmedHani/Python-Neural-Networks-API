__author__ = 'Ahmed Hani Ibrahim'
import abc


class MathFunction(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def function(self, input):
        """
        :param input: double
        :return: double
        """
        return

    @abc.abstractmethod
    def derivative(self, input):
        """
        :param input: double
        :return: double
        """
        return
