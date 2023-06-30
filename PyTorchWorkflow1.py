import torch
from torch import nn
#import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt

RANDOMSEED = 42

# Version info
print("Torch Version: " + str(torch.__version__))
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on Device: " + device)

torch.manual_seed(RANDOMSEED)

# Linear Regression with known parameters
intercept = 0.7
slope = 0.3

start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = slope * X + intercept

# Establish training and testing sets
trainSplit = int(0.8*len(X))
X_train, y_train = X[:trainSplit], y[:trainSplit]
X_test, y_test = X[trainSplit:], y[trainSplit:]

def plotPredictions(trainData=X_train,
                    trainLabels=y_train,
                    testData=X_test,
                    testLabels=y_test,
                    predictions=None):
    """
    Plots training data, test data, and compares predictions
    """
    plt.figure(figsize=(5,4))
    #Plot training data in blue
    plt.scatter(trainData, trainLabels, c="b", s=4, label="Training Data")
    #Plot training data in green
    plt.scatter(testData, testLabels, c="g", s=4, label="Testing Data")
    #Are there predicitons?
    if predictions is not None:
        plt.scatter(testData, predictions, c="r", s=4, label="Predicted Data")
    #Legend
    plt.legend(prop={"size":14})
    plt.show()

plotPredictions()
