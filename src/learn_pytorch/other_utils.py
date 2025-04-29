import torch
import numpy as np



# onehot <-> label
target = torch.tensor([2, 1, 1])

# labels --> one-hot
one_hot = torch.nn.functional.one_hot(target)
print(one_hot)
# one-hot --> labels
labels_again = torch.argmax(one_hot, dim=1)
print(labels_again)