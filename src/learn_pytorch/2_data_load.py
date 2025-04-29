import torchvision.transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision.datasets import CIFAR10

test_set = CIFAR10("../dataset", False, torchvision.transforms.ToTensor(), download=True)

test_load = DataLoader(test_set, 10, shuffle=True)

# img, target = test_set[0]
# print(img.shape)
# print(target)

writer = SummaryWriter("../logs")

data = next(iter(test_load))
imgs, targets = data
writer.add_images("cifar10", imgs, 0)

writer.close()

