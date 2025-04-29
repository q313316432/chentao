from torchvision import datasets
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

dataset_transform = transforms.Compose([
    transforms.ToTensor()
])


train_set = datasets.CIFAR10(root="./dataset", train=True, transform= dataset_transform, download=True)
test_set = datasets.CIFAR10(root="./dataset", train=False, transform= dataset_transform, download=True)

writer = SummaryWriter("logs")
for i in range(10):
    img, target = train_set[i]
    writer.add_image("p10", img, i)

writer.close()
