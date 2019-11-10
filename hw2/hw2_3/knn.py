import numpy as np
import cv2
import argparse
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from pca import mean_face, eigen_face


TRAIN_TOTAL = 280
TEST_TOTAL = 120
OUTPUT_PATH = './knn_output'

img_gray_3d = cv2.imread('./hw2-3_data/1_1.png')
width = img_gray_3d.shape[0]
height = img_gray_3d.shape[1]


def main():
	parser = argparse.ArgumentParser(description='knn')
	parser.add_argument('--input_path', default = './hw2-3_data', help='path of input image')
	parser.add_argument('--output_path', default = './knn_output', help = 'path of output image')

	args = parser.parse_args()

	#parsing training data
	print('parsing data...')
	training_data = np.ndarray(shape = (TRAIN_TOTAL, height*width), dtype = np.float64)
	for i in range(0, 40):
		for j in range(0, 7):
			img_gray_3d = cv2.imread(args.input_path+'/'+str(i+1)+'_'+str(j+1)+'.png')
			img_gray_2d = cv2.cvtColor(img_gray_3d, cv2.COLOR_BGR2GRAY)
			img_gray_1d = np.array(img_gray_2d, dtype = np.float64).flatten()
			training_data[i*7+j,:] = img_gray_1d

	#parsing testing data

	testing_data = np.ndarray(shape = (TEST_TOTAL, height*width), dtype = np.float64)
	for i in range(0, 40):
		for j in range(0, 3):
			img_gray_3d = cv2.imread(args.input_path+'/'+str(i+1)+'_'+str(j+8)+'.png')
			img_gray_2d = cv2.cvtColor(img_gray_3d, cv2.COLOR_BGR2GRAY)
			img_gray_1d = np.array(img_gray_2d, dtype = np.float64).flatten()
			testing_data[i*3+j,:] = img_gray_1d


	training_data_x = training_data
	training_data_y = []
	for i in range(1, 41):
		for j in range(7):
			training_data_y.append(i)

	testing_data_x = testing_data
	testing_data_y = []
	for i in range(1, 41):
		for j in range(3):
			testing_data_y.append(i)

	k = [1, 3, 5]
	n = [3, 10, 39]
	cv_scores = {}
	mean_face_img = mean_face(training_data)
	eigenface = eigen_face(training_data)
	normalized_train_img = np.subtract(training_data_x, mean_face_img.flatten())
	normalized_test_img = np.subtract(testing_data_x, mean_face_img.flatten())

	for p in k:
		for q in n:
			train_weight = np.dot(normalized_train_img, eigenface.transpose())
			train_weight = train_weight[:, :q]
	
			knn = KNeighborsClassifier(n_neighbors = p)
			scores = cross_val_score(knn, train_weight, training_data_y, cv = 3, scoring = 'accuracy')
			cv_scores[str(p)+','+str(q)] = scores.mean()

	highest_acc = 0
	best_param = ''

	for i, j in cv_scores.items():
		print("(k, n) = ({}), mean accuracy of three fold = {}".format(i, j))
		if j > highest_acc:
			highest_acc = j
			best_param = i

	print("best hyperparameter : (k, n) = ({})".format(best_param))

	best_param = best_param.split(',')

	best_k = int(best_param[0])
	best_n = int(best_param[1])
	
	# apply the best hyperparameter to knn
	train_weight = np.dot(normalized_train_img, eigenface.transpose())
	train_weight = train_weight[:, :best_n]

	test_weight = np.dot(normalized_test_img, eigenface.transpose())
	test_weight = test_weight[:, :best_n]

	knn = KNeighborsClassifier(n_neighbors = best_k)
	knn.fit(train_weight, training_data_y)
	pred = knn.predict(test_weight)

	accuracy = accuracy_score(pred, testing_data_y)
	print("accuracy on test data: {}".format(accuracy))


if __name__ == '__main__':
    main()


