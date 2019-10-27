import numpy as np
import cv2
import argparse
import os
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import sys

# np.set_printoptions(threshold=sys.maxsize)

TRAIN_TOTAL = 280
TEST_TOTAL = 120
OUTPUT_PATH = './pca_output'

img_gray_3d = cv2.imread('./hw2-3_data/1_1.png')
width = img_gray_3d.shape[0]
height = img_gray_3d.shape[1]

def mean_face(training_data):
	mean_face_pic = np.zeros((1, height*width))

	for i in training_data:
		mean_face_pic = np.add(mean_face_pic, i)

	mean_face_pic = np.divide(mean_face_pic, np.float64(TRAIN_TOTAL))

	mean_face_img_2d = mean_face_pic.reshape(width, height)

	return mean_face_img_2d

def eigen_face(training_data):

	normalized_training_data = np.ndarray(shape = (TRAIN_TOTAL, height*width))
	mean_face_img = mean_face(training_data)

	for i in range(TRAIN_TOTAL):
		normalized_training_data[i] = np.subtract(training_data[i], mean_face_img.flatten())

	cov_matrix = np.matmul(normalized_training_data, normalized_training_data.transpose())/np.float(TRAIN_TOTAL)

	eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

	idx = eigenvalues.argsort()[::-1]   
	eigenvalues = eigenvalues[idx]
	eigenvectors = eigenvectors[:,idx]

	eigenface = eigenvectors.transpose()
	eigenface = np.dot(eigenface, normalized_training_data)

	eigenface = normalize(eigenface, axis = 1, norm = 'l2')


	return eigenface

def t_SNE_visualizing(x):
	tsne = TSNE(perplexity = 80)
	x_2d = tsne.fit_transform(x)
	plt.figure(figsize = (6, 6))
	colors = 10*['r', 'g', 'b', 'c', 'm', 'y', 'k', 'gray', 'orange', 'purple']
	for i in range(40):
		for j in range(3):
			plt.scatter(x_2d[i*3+j, 0], x_2d[i*3+j, 1], c = colors[i])
	plt.title('t-SNE visualization')
	plt.show()




def main():
	parser = argparse.ArgumentParser(description='pca')
	parser.add_argument('--input_path', default = './hw2-3_data', help='path of input image')
	parser.add_argument('--output_path', default = './pca_output', help = 'path of output image')

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


	#find mean face in training data
	mean_face_img = mean_face(training_data)
	eigenface = eigen_face(training_data)

	#plot mean face
	cv2.imwrite(args.output_path+'/mean_face.png', mean_face_img.astype(np.uint8))
	plt.figure(figsize = (4, 4))
	plt.imshow(mean_face_img, cmap = 'gray')
	plt.title('mean face')
	plt.show()

	# plot eigenface
	fig, ax = plt.subplots(1,5)
	for i in range(5):
		img = eigenface[i].reshape(width,height)
		ax[i].imshow(img,cmap='gray')
		ax[i].set_title('eigenface-'+str(i+1))
		plt.imsave('./pca_output/eigenface_'+str(i+1)+'.png', img, cmap = 'gray')
	plt.show()
	
	img_test = cv2.imread(args.input_path+'/8_6.png')
	img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
	img_test_1d = np.array(img_test, dtype = np.float64).flatten()

	normalized_testing_img = np.subtract(img_test_1d, mean_face_img.flatten())

	
	weight = np.dot(eigenface, normalized_testing_img)

	n = [5, 50, 150, 279]
	#plot mse
	fig, ax = plt.subplots(1,4)
	for i in range(4):
		face_recons = np.zeros((1, width*height))
		for j in range(n[i]):
			face_recons += weight[j]*eigenface[j]
		face_recons += mean_face_img.flatten()
		error = np.subtract(face_recons, img_test_1d)
		mse_error = np.dot(error.transpose(), error)
		mse_error = np.sum(mse_error).astype(np.uint)
		img = face_recons.reshape(width, height)
		ax[i].imshow(img, cmap = 'gray')
		ax[i].set_title('n = {}\nM.S.E = {}'.format(n[i], mse_error))
	plt.show()


	#test data projection
	test_weight = np.zeros(shape = (TEST_TOTAL, TRAIN_TOTAL), dtype = np.float64)
	normalized_test_dataset = np.subtract(testing_data, mean_face_img.flatten())
	test_weight = np.dot(normalized_test_dataset, eigenface.transpose())

	test_weight = test_weight[:, :100]
	print
	print(test_weight.shape)

	t_SNE_visualizing(test_weight)

if __name__ == '__main__':
    main()


