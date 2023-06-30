import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RANDOMSEED = 42

# Version info
print("Torch Version: " + str(torch.__version__))
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on Device: " + device)

torch.manual_seed(RANDOMSEED)

tens = torch.tensor([1, 2 , 3], dtype=None, device=device)
numpyten = tens.cpu().numpy()
print(numpyten)