# 00_pytorch_fundamentals
print("#################################################################13")

# 13. Introduction to Tensors

import torch
import torchvision
import pandas
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


print("#################################################################14")

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

print("#################################################################17")

# 17. Dealing with tensor datatypes

""" 3 common errors in pytorch
1. Tensors not right datatype
2. Tensors not right shape
3. Tensors not on the right device
"""
# THREE MOST IMPORTANT PARAMETERS ARE: dtype, device, requires_grad

f32_tensor = torch.tensor([3.0, 6.0, 9.0], 
                          dtype=None, ## Datatype of tensor
                          device=None, ## Device the tensor is on
                          requires_grad=False) ## Whether or not to track gradients
print(f32_tensor) ### why? default dtype, even if None, is specified to float32

f16_tensor = f32_tensor.type(torch.float16) # can use this to fix issue #1
print(f16_tensor) 
print(f16_tensor * f32_tensor) # sometimes operations can auto convert

print("#################################################################18")
# 18. Tensor attributes

"""
Getting information from tensors
1. to get datatype, can use tensor.dtype
2. to get shape: can use tensor.shape
3. to get device, can use tensor.device
"""

some_tensor = torch.rand([3,4], dtype=torch.float16, device="cuda")
print(some_tensor)

## Details on above ^

print(f"Datatype of tensor: {some_tensor.dtype}")
print(f"Shape of tensor: {some_tensor.shape} OR {some_tensor.size()}")
print(f"Device of tensor: {some_tensor.device}")


print("#################################################################19")
# Manipulating Tensors (tensor operations)

"""
Tensor operations include:
* Addition, Subtraction, Division, Multiplication (element-wise)
* Matrix multiplication
"""

tensor = torch.tensor([1, 2, 3])
print(tensor)
tensor = tensor + 10 ## Ten to each cell
print(tensor)

tensor = tensor * 10 ## Multiply each cell by ten
print(tensor)

tensor = tensor // 10 ## Each cell divided (notice dtype changes! could also use //)
print(tensor)

tensor = tensor - 10 ## Back to original (with different dtype)
print(tensor)

print("#################################################################20, 21, 22")
# Matrix Multiplication

"""
Two main ways of performing multiplication in neural networks and DL

1. Element-wise multiplication (multiplying each element by a number)
2. Matrix multiplication (dot product)
 - multiply rows by columns on the two tables, go into new table 
 - https://www.mathsisfun.com/algebra/matrix-multiplying.html
"""

## Element-wise
print(tensor * tensor)
## Matrix multiplication
print(torch.matmul(tensor, tensor)) # Notice how size changes

###  Matrix mult by hand for tensor([1, 2, 3])
#### 1*1 + 2*2 + 3*3 = 14

"""
One of the most common errors in machine learning: shape errors
* Two main rules that matrix mult needs to satisfy
 1. The inner dimensions must match (m x n) * (n x p) = ?
 2. Resulting matrix has shape of outer dimensions ? = (m x p)
"""

## Shape for matrix mult
tensor_A = torch.tensor([[1,2],
                         [3,4],
                         [5,6]])
tensor_B = torch.tensor([[7, 10],
                         [8, 11],
                         [9, 12]])
try:
    torch.mm(tensor_A, tensor_B) # torch.mm = torch.matmal
except:
    print("3x2 and 3x2 mult produces exception")

"""
To fix our shape issues, we can manipulate the shape of one of our tensors
by using a transpose, which switches axis / dimensions of given tensor
"""

print(tensor_B, tensor_B.shape)
tensor_B = tensor_B.T # transposed
print(tensor_B, tensor_B.shape) # now 2x3
print(torch.mm(tensor_A, tensor_B)) # notice it is now 3x3 (main rule 2)

print("#################################################################23")
# Finding min, max, mean, sum, etc. (tensor aggregation)

x = torch.arange(1, 100, 10)
print(x)

## Finding min (either one)
torch.min(x)
print(x.min())

## max
torch.max(x)
print(x.max())

## Find mean (CANT ON LONG INTEGER DTYPE)
print(torch.mean(x.type(torch.float32)))
x.type(torch.float32).mean()

## Find sum
torch.sum(x)
print(x.sum())

# Finding the POSITIONAL min and max

print(x.argmin()) # tensor at index _ is our min value
print(x.argmax()) # tensor at index _ is our max value


print("#################################################################25")
# Reshaping, views, and stacking tensors

"""
* Reshaping - reshapes an input tensor to a defined shape
* View - return a view of an input tensor of certain shape but keep the same memory as original
* Stacking - combine multiple tensors on top of each other
"""

x = torch.arange(1, 11)
print(x, x.shape)

# Add a extra dimension
x_reshaped = x.reshape(1, 10) # dimensions must be compatable with original
print(x_reshaped, x_reshaped.shape)
x_reshaped = x.reshape(10, 1) # dimensions must be compatable with original
print(x_reshaped, x_reshaped.shape)
x_reshaped = x.reshape(5, 2) # dimensions must be compatable with original
print(x_reshaped, x_reshaped.shape)

# Change the view
z = x.view(1, 10) # similar to reshape (but it shares the same memory as x)
print(z) # Changing z will change x as they share memory

# Stack tensors on top of each other
x_stacked = torch.stack([x, x, x, x])
print(x_stacked)
x_stacked = torch.stack([x, x, x, x], dim=1)
print(x_stacked)

print("#################################################################26")
# Squeezing, unsqueezing, and permuting tensors
"""
* Squeeze - removes all single dimensions from a tensor
* Unsqueeze - adds a single dimension to target tensor at a specific dim
* Permute - return a view of the input with dimensions permuted (swapped in a certain way)
"""

# Squeeze & Unsqueeze
x_reshaped = x.reshape(1, 10)
print(x_reshaped, x_reshaped.shape)
x_squeezed = x_reshaped.squeeze()
print(x_squeezed, x_squeezed.shape) # notice it got rid of the '1' dimension
x_unsqueezed = x_squeezed.unsqueeze(dim=0)
print(x_unsqueezed, x_unsqueezed.shape)
x_unsqueezed = x_squeezed.unsqueeze(dim=1)
print(x_unsqueezed, x_unsqueezed.shape)

# torch.permute - rearranges dimensions of a target tensor in a specified order (shares memory)
# often used with image tensors
x_original = torch.rand(size=(224, 224, 3)) # [height, width, RGB]

## permute the original tensor to rearrange the axis (or dim) order
x_permuted = x_original.permute(2, 0, 1) # shifts dims 0->1, 1->2, 2->0
print(x_original.shape)
print(x_permuted.shape) # changing this will also change original

print("#################################################################27")
# Selecting data from tensors (indexing)

x = torch.arange(1, 10).reshape(1, 3, 3)
print(x, x.shape)

print(x[0]) # Let's index on dim 0
print(x[0][0]) # dim 1
print(x[0][0][0]) # dim 2

print(x[:, 0]) # use : to select "all" of a target dimension

# get all values of 0th and 1st dimensions, but only index 1 of 2nd dimension
print(x[:, :, 1])

print("#################################################################28")
## PyTorch tensors & NumPy

"""
NumPy is a popular computing library, and because of this,
PyTorch has functionality to interact with it.

* Data in NumPy, want in PyTorch tensor -> torch.from_numpy(ndarray)
* PyTorch tensor, ant in numpy -> torch.Tensor.numpy()
"""

array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array)
print(array, tensor) 
# ^ why is tensor torch.float64? this is because NumPy default if f64, where PyTorch is f32

## Tensor to NumPy
tensor = torch.ones(7)
numpy_tensor = tensor.numpy()
print(tensor, numpy_tensor)

print("#################################################################29")
# Reproductability (trying to take random out of random)

"""
In short how a neural network learns:

start with random nums -> tensor operations -> update rand nums to try and make them
better representations of the data -> again -> again...

So far, each time we create a random tensor, the numbers are totally different each execution.

To reduce the randomness in NNs and PyTorch comes the concept of a random seed.
Essentially what the random seed does is flavor the randomness.
"""

random_tensor_A = torch.rand(3, 4)
random_tensor_B = torch.rand(3, 4)

print(random_tensor_A)
print(random_tensor_B)
print(random_tensor_A == random_tensor_B) # Highly likely won't be True in any tensor

# Now let's make random but reproductable tensors
# Let's set the random seed
RANDOM_SEED = 1234

torch.manual_seed(RANDOM_SEED)
random_tensor_C = torch.rand(3, 4)

# need to call manual_seed for each subsequent method
torch.manual_seed(RANDOM_SEED)
random_tensor_D = torch.rand(3, 4)

print(random_tensor_C)
print(random_tensor_D)
print(random_tensor_C == random_tensor_D)
# https://docs.pytorch.org/docs/stable/notes/randomness.html

print("#################################################################30")
# Running tensors & PyTorch objects on GPUs for faster computation

"""
GPUs = faster computation on numbers due to CUDA + NVIDIA hardware + PyTorch

"""

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
print(torch.cuda.device_count())

## Putting tensors and models on the GPU

### Tensor default on CPU
tensor = torch.tensor([1, 2, 3])
print(tensor, tensor.device)

###  Move tensor to GPU (if available)
tensor_on_gpu = tensor.to(device)
print(tensor_on_gpu, tensor_on_gpu.device)

### If tensor on GPU, can't transform it to NumPy, do this
back_to_cpu = tensor_on_gpu.cpu().numpy()
print(back_to_cpu)