import numpy as np
import cv2 as cv

img = cv.imread('./hw2-3_data/1_2.png')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = np.array(img)
img = img.flatten()
print(img)




