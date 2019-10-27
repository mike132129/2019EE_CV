from sklearn.decomposition import PCA
import numpy as np
import cv2
import argparse
import os
from sklearn.preprocessing import StandardScaler
import random as rd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

TRAIN_TOTAL = 70
TEST_TOTAL = 120
OUTPUT_PATH = './pca_output'

img_gray_3d = cv2.imread('./hw2-3_data/1_1.png')
width = img_gray_3d.shape[0]
height = img_gray_3d.shape[1]
Eu_weight = [0.6, 0.4, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
colors = 10*["g","r","c","b","k"]
markers = 10*['x', '+']
rnd = 7*np.array(list(range(0, 10)))
print(rnd)

def do_pca(training_data):

	x_train_std = StandardScaler().fit_transform(training_data)
	pca = PCA(n_components=10).fit(x_train_std)
	principalComponents = pca.transform(x_train_std)

	return principalComponents

class K_Means:
	def __init__(self, k=10, tol=0.00001, max_iter=200):
		self.k = k
		self.tol = tol
		self.max_iter = max_iter

	def fit(self,data):
		self.centroids = {}

		for i in range(self.k):
			rand = rd.randint(0, 70)
			self.centroids[i] = data[rnd[i]]
	
		for i in range(self.max_iter):
			self.classifications = {}

			for j in range(self.k):
				self.classifications[j] = []

			for featureset in data:
				distances = [np.sqrt(np.dot(np.multiply((featureset - self.centroids[centroid]), Eu_weight), featureset - self.centroids[centroid])) for centroid in self.centroids]
				classification = distances.index(min(distances))
				self.classifications[classification].append(featureset)

			plt.figure(i+1)
			plt.title('iter = '+str(i+1))

			for classification in self.classifications:
				color = colors[classification]
				marker = markers[classification]
				for featureset in self.classifications[classification]:
					plt.scatter(featureset[0], featureset[1], marker=marker, color=color, s=150, linewidths=5)


			for centroid in self.centroids:
				plt.scatter(self.centroids[centroid][0], self.centroids[centroid][1],
				marker="o", color="y", s=150, linewidths=5)
			legend_elements = [Line2D([0], [0], marker='o', color='w', label='centroid', markerfacecolor = 'y', markersize = 10)]
			plt.legend(handles = legend_elements, loc = 'best')
			plt.savefig('./output_kmeans/iter_' + str(i+1) + '.png')
			plt.show()

			prev_centroids = dict(self.centroids)

			for classification in self.classifications:
				self.centroids[classification] = np.average(self.classifications[classification],axis=0)

			optimized = True

			for c in self.centroids:
				original_centroid = prev_centroids[c]
				current_centroid = self.centroids[c]
				if np.sqrt(np.dot(np.multiply((current_centroid - original_centroid), Eu_weight), current_centroid - original_centroid)) > self.tol:
					optimized = False

			if optimized:
				print("total iter", i)
				break

	def predict(self,data):
		distances = [np.sqrt(np.dot(np.multiply((data - self.centroids[centroid]), Eu_weight), data - self.centroids[centroid])) for centroid in self.centroids]
		classification = distances.index(min(distances))
		return classification







def main():
	parser = argparse.ArgumentParser(description='kmeans')
	parser.add_argument('--input_path', default = './hw2-3_data', help='path of input image')
	parser.add_argument('--output_path', default = './output_kmeans', help = 'path of output image')

	args = parser.parse_args()

	training_data = np.ndarray(shape = (TRAIN_TOTAL, height*width), dtype = np.float64)
	print(training_data.shape)
	for i in range(0, 10):
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

	


	pca_vector = do_pca(training_data)

	clf = K_Means(k = 10, max_iter = 100)
	clf.fit(pca_vector)


	for i in range(10):
		print()
		for j in range(7):
			classi = clf.predict(pca_vector[i*7+j])
			print(classi, end = " ")






if __name__ == '__main__':
    main()

