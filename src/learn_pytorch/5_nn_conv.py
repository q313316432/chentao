import torch
import torch.nn as nn
import torch.nn.functional as F


input_data = torch.tensor([
    [1, 2, 0, 1, 1],
    [2, 2, 0, 5, 4],
    [6, 2, 1, 2, 1],
    [5, 2, 0, 3, 3],
    [1, 2, 0, 3, 1]
])

kernel = torch.tensor([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
])

input_data = torch.reshape(input_data, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

output = F.conv2d(input_data, kernel, stride=1)
print(output)

# class Model(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 1, 3)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         return F.softmax(x, dim=0)
#
#
# fei_yang = Model()
# y = fei_yang.forward(torch.Tensor([[-1, 1, 2, -2], [-1, 1, 2, -2]]))
# print(y)
