import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader

input = torch.tensor([0, 1, 0], dtype=torch.float32)
target = torch.tensor([1, 0, 0], dtype=torch.float32)

input = torch.reshape(input, (1, 3))
target = torch.reshape(target, [1, 3])


# L1Loss
loss_l1 = nn.L1Loss(reduction='sum')
result = loss_l1(input, target)
# print(result)

# MSE
loss_mse = nn.MSELoss()
result = loss_mse(input, target)
# print(result)

# crossEntropy
loss_crossEntropy = nn.CrossEntropyLoss()
y = torch.tensor([1])
result = loss_crossEntropy(input, y)
# print(result)



class Yang(nn.Module):

    def __init__(self,):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)

dataset = torchvision.datasets.CIFAR10("../dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=10)

yang = Yang()

# imgs, targets = next(iter(dataloader))
# outputs = yang(imgs)
# print(imgs.shape)
# print(targets.shape)
# print(outputs.shape)
# result_loss = loss_crossEntropy(outputs, targets)
# print(result_loss)
# result_loss.backward() # 会在module中产生梯度

optim = torch.optim.SGD(yang.parameters(), lr=0.01)
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = yang(imgs)
        result_loss = loss_crossEntropy(outputs, targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        running_loss += result_loss
    print(running_loss)

