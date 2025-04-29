import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader

train_sets = torchvision.datasets.CIFAR10('../data/', train=False, transform=torchvision.transforms.ToTensor()
                                          ,download=True)

test_sets = torchvision.datasets.CIFAR10('./data', train=False, transform=torchvision.transforms.ToTensor()
                                         ,download=True)

train_dataloader = DataLoader(train_sets, batch_size=64)
test_dataloader = DataLoader(test_sets, batch_size=64)

class Yang(nn.Module):

    def __init__(self):
        super(Yang, self).__init__()

        self.Conv1 = nn.Conv2d(3, 32, 5)
        self.Maxpool1 = nn.MaxPool2d(2)
        self.Conv2 = nn.Conv2d(32, 32, 5, )


    def forward(self, x):
        return x

if __name__ == '__main__':
