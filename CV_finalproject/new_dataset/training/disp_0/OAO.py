import sys
import os
import glob
import torch.nn.functional
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage
import PIL.Image as Image
import pdb
a = glob.glob('*.png')

cut_h = 375
cut_w = 630


for i in a:
    #pdb.set_trace()
    resized_img = Image.open(i)
    resized_img = ToTensor()(resized_img)

    ch, h, w = resized_img.shape

    out = F.interpolate(resized_img.unsqueeze(0), scale_factor = [cut_h/h, cut_w/w])
    out = out.resize(out.shape[1], out.shape[2], out.shape[3])
    ToPILImage()(out).save(i, mode = 'png')



