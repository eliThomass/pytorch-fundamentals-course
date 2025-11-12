# 00_pytorch_fundamentals
######################################

# 13. Introduction to Tensors

import torch
import torchvision
#import pandas
import numpy as np
import matplotlib.pyplot as plt

print(torch.__version__)
print(torch.cuda.is_available())

## Creating tensors & their types

### scalar

scalar = torch.tensor(7)
print(scalar)
print(scalar.item())

### Vector

vector = torch.tensor([7, 7])
print(vector)
print(vector.ndim) # 1 dim, not 0. Number of square brackets = ndim
print(vector.shape) # 2 x 1 elements

### MATRIX

MATRIX = torch.tensor([[7, 8],
                       [9,10]])
print(MATRIX)
print(MATRIX.ndim)
print(MATRIX[1])
print(MATRIX.shape) # 2x2

### TENSOR
TENSOR = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5]]])
print(TENSOR)
print(TENSOR.ndim)
print(TENSOR.shape) # we have 1 3x3 shape matrix
print(TENSOR[0])

# PRACTICE

TENSORR = torch.tensor([[[1, 6],
                         [4, 9],
                         [5, 1],
                         [6, 2]]])
print(TENSORR.shape)


#########################################################################

# 14. Creating tensors in PyTorch

## Scalar and Vector usually get lowercase variable names,
## Matrix and Tensor usually get uppercase (as in prev examples)

## Random Tensors
"""
Random Tensors are important because the way many neural networks learn
is that they START with tensors full of random numbers and then ADJUST
those random numbers to better represent the data (3Blue1Brown videos)

Start w/ random numbers -> look at data -> update -> look -> update -> ...
"""
### Create random tensor of size (3, 4)
### https://docs.pytorch.org/docs/stable/generated/torch.rand.html
random_tensor = torch.rand(3, 4)
print(random_tensor)
print(random_tensor.ndim)

### Create a random tensor w/ similar shape to an image tensor
random_img_size_tensor = torch.rand(size=(224, 224, 3)) # height, width, RGB
print(random_img_size_tensor.shape, random_img_size_tensor.ndim)
print(random_img_size_tensor)


## Zeros and Ones

### Create a tensor of all zeros
zeros = torch.zeros(size=(3, 4))
print(zeros)
print(zeros * random_tensor)

### Create a tensor of all ones
ones = torch.ones(size=(3, 4))
print(ones.dtype) # Default data type (always float32 unless defined)
print(ones)


## Creating a range of tensors and tensors-like

### Use torch.arange() (.range() depricated)
zerotonine = torch.arange(0, 10)
print(zerotonine)
step77 = torch.arange(start=1,end=1000,step=77)
print(step77)

### Creating tensors like (zeros, rand, ones, etc.)
ninezeros = torch.zeros_like(input=zerotonine)
print(ninezeros)

##################################################################

# 17. Dealing with tensor datatypes

