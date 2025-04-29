import torch

input = torch.randn([120, 5, 5, 5])

flattern_layer = torch.nn.Flatten(start_dim=2)

linear_layer = torch.nn.Linear(25, 3, True)

input = flattern_layer(input)

print(input.shape)

output = linear_layer(input)

print(output.shape)