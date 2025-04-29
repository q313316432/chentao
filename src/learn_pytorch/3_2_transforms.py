from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")

img_path = 'data/hymenoptera_data/train/ants/0013035.jpg'

img = Image.open(img_path)
print(img)

# numpy方式画图
img_arr = np.array(img)
writer.add_image("image", img_arr, 1, dataformats='HWC')

# transform的方式
to_tensor = transforms.ToTensor()
img_tensor = to_tensor(img)
writer.add_image("image", img_tensor, 2)

# opencv方式
import cv2
cv_img = cv2.imread(img_path) #读取的是ndarray

trans_norm = transforms.Normalize([1, 3, 5], [3, 2, 1])
img_norm = trans_norm(img_tensor)
writer.add_image("image", img_norm, 3)

trans_resize = transforms.Resize([300, 200])
trans_norm_totensor = transforms.Compose([trans_resize, to_tensor]) # compose的组合用法
img_norm = trans_norm_totensor(img_tensor)

writer.add_image("image", img_norm, 4)

writer.close()


