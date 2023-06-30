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
intercept = 0.3
slope = 0.7

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
class LinearRegressionModel(nn.Module): # <- almost everything in PyTorch inherits from nn.Module
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
    #Forward Method, defines computation

    def forward(self, x: torch.Tensor) -> torch.Tensor: # "x" is the input data
        return self.weights * x + self.bias



model0 = LinearRegressionModel()

# Loss Function
lossFn = nn.L1Loss()
# Optimizer
optimizer = torch.optim.SGD(params=model0.parameters(), lr=0.01)

#print(list(model0.parameters()))

## Predictions using torch.inference_mode
with torch.inference_mode():
    y_preds = model0(X_test)



print(y_preds)

plotPredictions(predictions=y_preds)