import torch
from torch import nn
#import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt

RANDOMSEED = 40

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
y = slope * X * X + intercept

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
        self.p1 = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.p2 = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.p3 = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
    #Forward Method, defines computation

    def forward(self, x: torch.Tensor) -> torch.Tensor: # "x" is the input data
        return self.p1 * x * x + self.p2 * x + self.p3



model0 = LinearRegressionModel()

# Loss Function
lossFn = nn.L1Loss()
# Optimizer
optimizer = torch.optim.SGD(params=model0.parameters(), lr=0.01)

#print(list(model0.parameters()))

## Predictions using torch.inference_mode, FORWARD PASS
#with torch.inference_mode():
#    y_preds = model0(X_test)

torch.manual_seed(RANDOMSEED)

X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)
model0.to(device)

lossplot = []

epochs = 1000
#Loops Through Data
for epoch in range(epochs):
    # Set model to training mode
    model0.train() #Train mode in Pytorch sets all parameters that require gradients to require gradients
    
    # 1. Forward Pass
    y_preds = model0(X_train)

    # 2. Loss Function
    loss = lossFn(y_preds, y_train)
    lossplot.append(loss.cpu().item())

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Perform backprop on the loss with respect to parameters
    loss.backward()

    # 5. Step the optimizer
    optimizer.step() # By default how the optimizer changes will accumulate through the loop, so they have to be zerod in step 3


    model0.eval() # turn of gradient tracking

    with torch.inference_mode():
        test_pred = model0(X_test)
        test_loss = lossFn(test_pred, y_test)
    
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")

model0.eval()
with torch.inference_mode():
    y_pred = model0(X_test)
print(model0.state_dict())
plt.scatter(torch.arange(0, len(lossplot), 1).unsqueeze(dim=1), lossplot, s=2)
plt.show()
plotPredictions(predictions=y_pred.cpu())