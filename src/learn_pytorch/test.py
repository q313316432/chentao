import torch

input = torch.tensor([1], dtype=torch.float32)

gelu = torch.nn.GELU()

torch.nn.Ba

output = gelu(input)

print(output)
