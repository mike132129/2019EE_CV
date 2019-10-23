import numpy as np
import cv2
import argparse
import os

TRAIN_TOTAL = 280
TEST_TOTAL = 120
OUTPUT_PATH = './pca_output'

img_gray_3d = cv2.imread('./hw2-3_data/1_1.png')
width = img_gray_3d.shape[0]
height = img_gray_3d.shape[1]
print(width, height)

#proj_data = eigenspace

def mean_face(training_data):
	mean_face_pic = np.zeros((1, height*width))

	for i in training_data:
		mean_face_pic = np.add(mean_face_pic, i)

	mean_face_pic = np.divide(mean_face_pic, float(TRAIN_TOTAL))

	mean_face_img_2d = mean_face_pic.reshape(width, height)

	return mean_face_img_2d

def eigen_face(training_data, number):
	normalized_training_data = np.ndarray(shape = (TRAIN_TOTAL, height*width))
	mean_face_img = mean_face(training_data)

	for i in range(TRAIN_TOTAL):
		normalized_training_data[i] = np.subtract(training_data[i], mean_face_img.flatten())

	cov_matrix = np.cov(normalized_training_data)
	cov_matrix = np.divide(cov_matrix, TRAIN_TOTAL)
	

	eigenvalues, eigenvectors, = np.linalg.eig(cov_matrix)
	eig_pairs = [(eigenvalues[index], eigenvectors[:,index]) for index in range(len(eigenvalues))]


	# Sort the eigen pairs in descending order:
	eig_pairs.sort(reverse=True)
	eigvalues_sort  = [eig_pairs[index][0] for index in range(len(eigenvalues))]
	eigvectors_sort = [eig_pairs[index][1] for index in range(len(eigenvalues))]


	reduced_data = np.array(eigvectors_sort[:number]).transpose()
	eigenspace = np.dot(training_data.transpose(),reduced_data)
	eigenspace = eigenspace.transpose()

	for i in range(eigenspace.shape[0]):
		eigenface = eigenspace[i].reshape(width, height).astype(np.uint8)
		# cv2.imwrite(OUTPUT_PATH+'/eigenface_'+str(i+1)+'.png', eigenface)

	print("vvvv", reduced_data.shape)
	return eigvalues_sort, eigvectors_sort, normalized_training_data, eigenspace



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
	#cv2.imwrite(args.output_path+'/mean_face.png', mean_face_img.astype(np.uint8))

	#project photo onto eigenspace
	photo_be_projected = cv2.imread(args.input_path+'/8_6.png')
	photo_be_projected = cv2.cvtColor(photo_be_projected, cv2.COLOR_BGR2GRAY).flatten()
	normalized_projected_photo = np.subtract(photo_be_projected, mean_face_img.flatten())
	print(normalized_projected_photo)

	

	n = [5, 280]
	for i in n:
		print(i)
		eigenvalues, eigenvectors, normalized_training_data, eigenspace = eigen_face(training_data, i)
		eigenvectors = np.array(eigenvectors)
		eigenvalues = np.array(eigenvalues)
		print(eigenvectors.shape)
		print(eigenvalues.shape)
		print(eigenspace.shape)

		recons_photo = np.zeros((1, height*width))
		for j in range(i):
			recons_photo = np.add(recons_photo, np.dot(normalized_projected_photo, eigenspace[j])*eigenspace[j])
			# print("add", recons_photo)
		



		#photo_recons = 

	





if __name__ == '__main__':
    main()


