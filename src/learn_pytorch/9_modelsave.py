import torch.nn
import torchvision.transforms
from torchvision import datasets

# imagetnet_train = datasets.ImageNet("../dataset/imagetnet", split="train"
#                                     ,transform=torchvision.transforms.ToTensor()
#                                     ,)

vgg16 = torchvision.models.vgg16()
print(vgg16)

vgg16.classifier.add_module('7', torch.nn.Linear(1000, 10, True))
print(vgg16)

torch.save(vgg16, 'vgg16.pth')
torch.save(vgg16.state_dict(), 'vgg16_stat.pth')

stat = torch.load('vgg16_stat.pth')

vgg16_2 = torchvision.models.vgg16()
vgg16_2.load_state_dict(stat)

print(vgg16_2)

