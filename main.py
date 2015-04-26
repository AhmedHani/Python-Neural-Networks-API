from NeuralNetwork.Neuron import *

__author__ = 'ENG.AHMED HANI'
import numpy as np
from NeuralNetwork.FeedforwardNeuralNetwork import *
from ActivationFunctions.Sigmoid import *
import random
from OptimizationAlgorithms.Backpropagation import *

network = FeedforwardNeuralNetwork(4)
s = Sigmoid()
network.setNetwork([2, 5, 2, 2])
trainingSample = [[0 for i in range(2)] for j in range(10)]
labels = [[0 for j in range(3)] for i in range(0, 50)]

for i in range(0, 20):
    labels[i] = [1, 0]
for i in range(20, 40):
    labels[i] = [0, 1]
for i in range(40, 50):
    labels[i] = [1, 1]

for i in range(0, 10):
    for j in range(0, 2):
        trainingSample[i][j] = round(random.random(), 2)

x = 0
z = 0

while x < 100:
    x += 1
    z = network.train(trainingSample, labels, 0.1, Backpropagation())

z = network.computOutput([0.2, 0.4])




