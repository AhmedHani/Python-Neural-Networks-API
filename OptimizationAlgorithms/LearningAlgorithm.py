__author__ = 'Ahmed Hani Ibrahim'
import abc


class LearningAlgorithm(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def learn(self, learningRate, input, output, network):
        return
