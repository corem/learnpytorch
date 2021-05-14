import torch
import numpy as np

# Initializing a tensor directly from data
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
print(x_data)

# Initializing a tensor from a NumPy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)

# Initializing a tensor from another tensor
x_ones = torch.ones_like(x_data)
x_rand = torch.rand_like(x_data, dtype=torch.float)
print(x_ones)
print(x_rand)

# Initializing a tensor with a shape tuple
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(rand_tensor)
print(ones_tensor)
print(zeros_tensor)

# Tensor attributes
tensor = torch.rand(3,4)
print(tensor.shape)
print(tensor.dtype)
print(tensor.device)

# Tensor operations
if torch.cuda.is_available():
  tensor = tensor.to('cuda')
print(tensor.device)

# Indexing and slicing
tensor = torch.ones(4, 4)
print('First row: ',tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., -1])
tensor[:,1] = 0
print(tensor)

# Joining tensors
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# Arithmetic operations
# Computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)

# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# Single-element tensors
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# In-place operations
print(tensor, "\n")
tensor.add_(5)
print(tensor)



