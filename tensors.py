import torch
import numpy as np

# Initializing a tensor directly from data
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

# Initializing a tensor from a NumPy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)