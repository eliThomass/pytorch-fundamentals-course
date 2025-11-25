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
X = torch.arange(start,end,step).unsqueeze(dim=1) # X is usually a matrix/tensor so it's capitalized, unsqueeze adds extra dim
y = weight * X + bias # y = wX + b

print(X[:10])
print(y[:10])

print(len(X), len(y)) # What might be a better way to visualize it? Right now just numbers on a page

print("#################################################################36")
# Creating train and test sets

"""
Splitting data into training and test sets. 
ONE OF THE MOST IMPORTANT concepts in ML in general.
Check 4:38 in video for images or website.

Training set always (60-80% of data), testing set always (10-20%), validation set often (10-20%)

Let's create a training and test set with our data!
"""

## Create a train/test split

train_split = int(0.8 * len(X)) # 80% of data
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

print(len(X_train), len(y_train), len(X_test), len(y_test))
## We have now split our data
## Let's visualize this. MATPLOTLIB!!
## "Visualize, visualize, visualize"

def plot_predictions(train_data = X_train, 
                     train_labels = y_train, 
                     test_data = X_test, 
                     test_labels = y_test, 
                     predictions = None):
    """
    Plots training data, test data and compares predictions.
    """
    plt.figure(figsize=(10,7))

    # Plot training data in blue
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html 
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    # Are there predictions??
    if predictions:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})

plot_predictions()
plt.show()
## We are going to build a model to learn the pattern of the blue dots.

print("#################################################################38")
# Building our first PyTorch model

"""
What our model does:
 * Start with random values (weight & bias)
 * Look at training data and adjust the random values to get closer to the ideal values we used to create the data

How does it do so?:
 1. Gradient descent
 2. Backpropagation
"""

## Create linear regression model class
class LinearRegulationModel(nn.Module): # <- almost everything in PyTorch relates to nn.Module
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, # start with a random weight value
                                                requires_grad=True, # it can update via gradient descent
                                                dtype=torch.float)) # PyTorch loves f32
        self.bias = nn.Parameter(torch.randn(1, # random value again
                                             requires_grad=True, # can update
                                             dtype=torch.float))
        
        # Forward method to define the computation in the model
        def forward(self, x: torch.Tensor) -> torch.Tensor: # x is the input data
            return self.weights * x + self.bias # this is the linear regression formula from line 52

print("#################################################################40")
# PyTorch model building essentials
"""
* torch.nn - contains all of the buildings for computational graphs
* torch.nn.Parameter - what parameters should our model try and learn, often a PyTorch layer from torch.nn will set these for us
* torch.nn.Module - the base class for all neural network modules, if you subclass it, you should overwrite forward
* torch.optim - this is where the optimizers in PyTorch live, they will help with gradient descent
* def forward() - all nn.Module subclasses require you to overwrite forward, this method defines what happens in the forward computation
"""
# https://www.learnpytorch.io/pytorch_cheatsheet/


