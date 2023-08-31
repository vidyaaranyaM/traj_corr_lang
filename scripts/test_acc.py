# import numpy as np

# first = np.array([[10, 2],
#                  [-9, 0],
#                  [-1, 4]])
# second = np.array([[1, 0],
#                  [0, 0],
#                  [1, 0]])

# # print(np.all(first == second, axis=1).sum())
# second = (first == first.max(axis=1)[:,None]).astype(int)
# print(second)

import torch
import torch.nn as nn
import torch.nn.functional as F

input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5).softmax(dim=1)

# print("input: ", input.shape)
# print("target: ", target.shape)
# loss = nn.CrossEntropyLoss()

# output = loss(input, target)
# print("output: ", output)

print("target: ", target)
mask = torch.where(target == target.max(dim=1, keepdim=True)[0], 1, 0)
print("mask: ", mask)
