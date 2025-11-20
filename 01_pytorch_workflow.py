# 00_pytorch_fundamentals
print("#################################################################34")

# PyTorch Workflow
## Let's explore an example PyTorch end-to-end workflow.
"""
What we're covering
1. Get data ready (into tensors)
2. Build or pick a pretrained model
3. Fitting the model to data (training)
4. Making predictions and evaluate a model (inference)
5. Saving and loading a model
6. Put it all together
"""

import torch
from torch import nn # contain's all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt

# Check PyTorch version
print(torch.__version__)

print("#################################################################35")
# 1. Data (preparing and loading)

"""
Data can be almost anything in machine learning.
 - Excel spreadsheet
 - Images of any kind
 - Videos
 - Audio
 - DNA
 - Text

Machine learning is a game of two parts:
 1. Get data into a numerical representation
 2. Build a model to learn patterns in that numerical representation

To showcase this, let's create some known data using the linear regression formula!
We'll use a linear regression formula to make a straight line with known parameters.
"""

## Create known parameters
weight = 0.7
bias = 0.3

## Create some data
start = 0 
end = 1
step = 0.02
X = torch.arange(start,end,step).unsqueeze(dim=1) # X is usually a matrix/tensor so it's capitalized
y = weight * X + bias

print(X[:10])
print(y[:10])

print(len(X), len(y))
