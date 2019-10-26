import numpy as np
import cv2
import argparse
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

TRAIN_TOTAL = 280
TEST_TOTAL = 120
OUTPUT_PATH = './kmeans'

img_gray_3d = cv2.imread('./hw2-3_data/1_1.png')
width = img_gray_3d.shape[0]
height = img_gray_3d.shape[1]
print(width, height)

def main():
	parser = argparse.ArgumentParser(description='k-means')
	parser.add_argument('--input_path', default = './hw2-3_data', help='path of input image')
	parser.add_argument('--output_path', default = './kmeans', help = 'path of output image')

	args = parser.parse_args()

	#parsing training data
	training_data = np.ndarray(shape = (TRAIN_TOTAL, height*width), dtype = np.float64)
	print(training_data.shape)
	for i in range(0, 40):
		for j in range(0, 7):
			img_gray_3d = cv2.imread(args.input_path+'/'+str(i+1)+'_'+str(j+1)+'.png')
			img_gray_2d = cv2.cvtColor(img_gray_3d, cv2.COLOR_BGR2GRAY)
			img_gray_1d = np.array(img_gray_2d, dtype = np.float64).flatten()
			training_data[i*7+j,:] = img_gray_1d

	#parsing testing data
	testing_data = np.ndarray(shape = (TEST_TOTAL, height*width), dtype = np.float64)
	print(testing_data.shape)
	for i in range(0, 40):
		for j in range(0, 3):
			img_gray_3d = cv2.imread(args.input_path+'/'+str(i+1)+'_'+str(j+8)+'.png')
			img_gray_2d = cv2.cvtColor(img_gray_3d, cv2.COLOR_BGR2GRAY)
			img_gray_1d = np.array(img_gray_2d, dtype = np.float64).flatten()
			testing_data[i*3+j,:] = img_gray_1d


