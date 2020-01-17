import sys
import os
import glob
import torch.nn.functional
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage
import PIL.Image as Image

imgL_list = glob.glob('./new_dataset/vision.middlebury.edu/stereo/data/scenes2014/zip/*/im0.png')

imgR_list = glob.glob('./new_dataset/vision.middlebury.edu/stereo/data/scenes2014/zip/*/im1.png')

# have to convert to png file
#disp_L_list = glob.glob('./new_dataset/vision.middlebury.edu/stereo/data/scenes2014/zip/*/disp0.pfm')

imgR_E = glob.glob('./new_dataset/vision.middlebury.edu/stereo/data/scenes2014/zip/*/im1E,png')
imgR_L = glob.glob('./new_dataset/vision.middlebury.edu/stereo/data/scenes2014/zip/*/im1L.png')

# check this size!
cut_h = 540
cut_w = 800


def resize(img_list, path):
    save_img = 0
    for img in img_list:
        resized_img = Image.open(img)
        resized_img = ToTensor()(resized_img)

        ch, h, w = resized_img.shape

        # Downsample by interpolate
        out = F.interpolate(resized_img.unsqueeze(0), scale_factor = [cut_h/h, cut_w/w])
        out = out.resize(out.shape[1], out.shape[2], out.shape[3])
 
        ToPILImage()(out).save('./new_dataset/training/'+path+'000'+str(save_img)+'_10.png', mode = 'png')
        save_img += 1 
resize(glob.glob('./new_dataset/training/disp_0/*.png'), 'disp_0/')
#resize(imgL_list, 'img_2/')
#resize(imgR_list, 'img_3/')

'''
## pfm to png
import util
from util import readPFM
import cv2

save_dis = 0
for disp in disp_L_list:
    aaa = readPFM(disp)
    cv2.imwrite('./new_dataset/training/disp_0/000'+str(save_dis)+'_10.png', aaa)
    #aaa = ToTensor()(aaa.copy())
    #ch, h, w = aaa.shape
    #out = F.interpolate(aaa.unsqueeze(0), scale_factor = [cut_h/h, cut_w/w])
    #out = out.resize(out.shape[1], out.shape[2], out.shape[3])
    #ToPILImage()(out).save('./new_dataset/training/disp_0/000'+str(save_dis)+'_10.png', mode = 'png')
    save_dis += 1
'''
