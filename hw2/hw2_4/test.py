import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

# Load the pretrained model

import torch
 
print(dir(models))

alexnet = models.alexnet(pretrained=True)

print(alexnet)