import numpy as np
import cv2

from joint_bilateral_filter import Joint_bilateral_filter
import sys 
np.set_printoptions(threshold = sys.maxsize)

img = cv2.imread('1a.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
guidance = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
weight = [0.3, 0.3, 0.4]
gray = np.dot(img_rgb, weight).astype(np.uint8)
print(gray.shape)

#cv2.imwrite('1a_gray.png', gray)
imga_gray = cv2.imread('1a_gray.png')
imga_gray = cv2.cvtColor(imga_gray, cv2.COLOR_BGR2GRAY)

print(imga_gray.shape)

JBF = Joint_bilateral_filter(1, 0.05, border_type='reflect')


jbf_3 = JBF.joint_bilateral_filter(img_rgb, imga_gray).astype(np.uint8)
jbf_1 = JBF.joint_bilateral_filter(img_rgb, gray).astype(np.uint8)

error = (jbf_3-jbf_1)

print(error)

