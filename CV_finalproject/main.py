
from __future__ import print_function

import numpy as np
import argparse
import cv2
import time
from util import writePFM
from matplotlib import pyplot as plt

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
from PSMNet_utils import preprocess 
from models import *
import cv2.ximgproc as cv2_x
import pdb
from classify import is_gray_img

parser = argparse.ArgumentParser(description='Disparity Estimation')
parser.add_argument('--input-left', default='./data/Synthetic/TL0.png', type=str, help='input left image')
parser.add_argument('--input-right', default='./data/Synthetic/TR0.png', type=str, help='input right image')
parser.add_argument('--output', default='./result1/TL0.pfm', type=str, help='left disparity map')
parser.add_argument('--loadmodel', default='./pretrained_model_KITTI2015.tar', help='loading model')
parser.add_argument('--model', default='stackhourglass', help='select model')
parser.add_argument('--maxdisp', type=int, default=192, help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
    print('stackhourglass')
elif args.model == 'basic':
    model = basic(args.maxdisp)
    print('basic')
else:
    print('no model')

is_real = is_gray_img(args.input_left)

model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if is_real:
    state_dict = torch.load('final_real.tar')
    model.load_state_dict(state_dict['state_dict'])
else:
    state_dict = torch.load('final_synthetic.tar')
    model.load_state_dict(state_dict['state_dict'])

def test(imgL,imgR):
    model.eval()

    if args.cuda:
       imgL = torch.FloatTensor(imgL).cuda()
       imgR = torch.FloatTensor(imgR).cuda()     

    imgL, imgR= Variable(imgL), Variable(imgR)

    with torch.no_grad():
        disp = model(imgL,imgR)
    
    disp = torch.squeeze(disp[1])
    pred_disp = disp.data.cpu().numpy()

    return pred_disp

def computeDisp(imgL_o, imgR_o):

    processed = preprocess.get_transform(augment=False)
        
    imgL = processed(imgL_o).numpy()
    imgR = processed(imgR_o).numpy()
    imgL = np.reshape(imgL,[1,3,imgL.shape[1],imgL.shape[2]])
    imgR = np.reshape(imgR,[1,3,imgR.shape[1],imgR.shape[2]])

    # pad to width and hight to 16 times
    if imgL.shape[2] % 16 != 0:
        times = imgL.shape[2]//16       
        top_pad = (times+1)*16 -imgL.shape[2]
    else:
        top_pad = 0
    if imgL.shape[3] % 16 != 0:
        times = imgL.shape[3]//16                       
        left_pad = (times+1)*16-imgL.shape[3]
    else:
        left_pad = 0     
    imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
    imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)


    start_time = time.time()
    pred_disp = test(imgL,imgR)

    ####################################  Original Disp ###
    print('time = %.2f' %(time.time() - start_time))
    if top_pad !=0 and left_pad != 0:
        img = pred_disp[top_pad:,:-left_pad]

    elif top_pad != 0 and left_pad == 0:
        img = pred_disp[top_pad:, :]

    elif top_pad == 0 and left_pad != 0:
        img = pred_disp[:, :-left_pad]

    elif top_pad == 0 and left_pad == 0:
        img = pred_disp

    img = refinement(imgL_o, img)
    if is_real:
        return np.subtract(192, img)
    
    return img


       
def refinement(Original, disp):

    disp = cv2_x.weightedMedianFilter(Original.astype('uint8'), disp.astype('uint8'), 15, 5, cv2_x.WMF_JAC)
    disp = cv2_x.weightedMedianFilter(Original.astype('uint8'), disp.astype('uint8'), 15, 5, cv2_x.WMF_JAC)

    return disp



def main():
    
    print('output file path:', args.output)
    print('Compute disparity for %s' % args.input_left)
    
    isgray = is_gray_img(args.input_left)

    if isgray:
       imgL_o = cv2.cvtColor(cv2.imread(args.input_left, 0), cv2.COLOR_GRAY2RGB)
       imgR_o = cv2.cvtColor(cv2.imread(args.input_right, 0), cv2.COLOR_GRAY2RGB)
    else:
       imgL_o = cv2.imread(args.input_left)
       imgR_o = cv2.imread(args.input_right)

    imgL_o = cv2.cvtColor(imgL_o, cv2.COLOR_BGR2GRAY)
    imgR_o = cv2.cvtColor(imgR_o, cv2.COLOR_BGR2GRAY)

    imgL_o = cv2.cvtColor(imgL_o, cv2.COLOR_GRAY2RGB)
    imgR_o = cv2.cvtColor(imgR_o, cv2.COLOR_GRAY2RGB)

    tic = time.time()
    disp = np.float32(computeDisp(imgL_o, imgR_o))
    toc = time.time()
    
    print('dispmap: {}'.format(disp))
    writePFM(args.output, disp)
    print('Elapsed time: %f sec.' % (toc - tic))


if __name__ == '__main__':
    main()
