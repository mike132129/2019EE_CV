import numpy as np
import re
import sys
import cv2

def readPFM(file):
    file = open(file, 'rb')

    header = file.readline().rstrip()
    header = header.decode('utf-8')
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)

#    print(data)
    return data

import glob
disp_list = glob.glob('*.pfm')

for i in disp_list:
    disp = readPFM(i)
    name = i.split('.')[0]
    
    cv2.imwrite('./'+name+'.png', disp)


#path = sys.argv[1]
#disp = readPFM(path)

#cv2.imwrite('new_dataset/disp/bicycle_R.png', disp)

