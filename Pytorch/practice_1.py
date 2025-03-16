import torch
import numpy as np
print(torch.__version__)

# create a tensor
t = torch.tensor([1, 2])
print(t)

# create a 2*3 matrix
x = torch.rand(2, 3)
print(x)

# create a matrix with values = 0
x1 = torch.zeros(2, 3)
print(x1)

# random
x = torch.rand(2, 2)
y = torch.rand(2, 2)
print(x)
print(y)
print(x+y)
print(torch.add(x,y))


# Transfer
a = np.array((1, 2))
t1 = torch.tensor(a)
print(t1)