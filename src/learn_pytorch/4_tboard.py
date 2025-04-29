from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
for i in range(101):
    writer.add_scalar("my scalar1", i, i)



writer.close()


