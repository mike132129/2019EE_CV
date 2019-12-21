import os
from util import writePFM
import cv2
import cv2.ximgproc as cv2_x
for i in range(0, 10):
	os.system('python3 score.py data/Synthetic/TLD'+str(i)+'.pfm result1/new_'+str(i)+'.pfm')

# 	os.system('python3 test.py --input-left=./data/Real/TL'+str(i)+'.bmp --input-right=./data/Synthetic/TR'+str(i)+'.png --output=./result2/TL'+str(i)+'.pfm')
# 	os.system('python3 visualize.py ./result2/TL'+str(i)+'.pfm')
# img = cv2.imread('./result1/TLD5.png').astype('float32')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(img)

# Il = cv2.imread('./data/Synthetic/TL5.png')
# print(Il)

# disp = cv2_x.weightedMedianFilter(Il.astype('uint8'), img.astype('uint8'), 15, 5, cv2_x.WMF_JAC)
# disp = cv2_x.weightedMedianFilter(Il.astype('uint8'), disp.astype('uint8'), 15, 5, cv2_x.WMF_JAC)

# writePFM('./result1/new_5.pfm', disp.astype('float32'))

'''

'''