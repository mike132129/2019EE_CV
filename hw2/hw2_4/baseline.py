###################################################################################
## Problem 4(b):                                                                 ##
## You should extract image features using pytorch pretrained alexnet and train  ##
## a KNN classifier to perform face recognition as your baseline in this file.   ##
###################################################################################

import sys
import glob, os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import alexnet
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

FEATURES_EXTRACTED = 256
n_neighbors = 1

def do_pca(training_data):

	x_train_std = StandardScaler().fit_transform(training_data)
	pca = PCA(n_components=25).fit(x_train_std)
	principalComponents = pca.transform(x_train_std)

	return principalComponents


def get_dataloader(folder, batch_size=32):
	# Data preprocessing
	normalize = transforms.Normalize(mean=[3*0.485, 3*0.456, 3*0.406], std=[0.229, 0.224, 0.225])
	trans = transforms.Compose([transforms.ToTensor(), normalize])
	train_path, val_path = os.path.join(folder,'train'), os.path.join(folder,'valid')
	# Get dataset using pytorch functions
	train_set = ImageFolder(train_path, transform=trans)
	val_set =  ImageFolder(val_path,  transform=trans)
	train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False)
	val_loader  = torch.utils.data.DataLoader(dataset=val_set,  batch_size=batch_size, shuffle=False)
	print ('==>>> total trainning batch number: {}'.format(len(train_loader)))
	print ('==>>> total validation batch number: {}'.format(len(val_loader)))
	return train_loader, val_loader


def main():

	folder, image_path = sys.argv[1], sys.argv[2]
	train_loader, val_loader = get_dataloader(folder, batch_size = 1)
	#draw_train_loader, draw_val_loader = get_dataloader10(folder, batch_size = 1)


	if torch.cuda.is_available():
		extractor = alexnet(pretrained = True).features.cuda()
	else:
		extractor = alexnet(pretrained = True).features

	

	feats_train = []
	train_y = []

	for batch, (x, label) in enumerate(train_loader,1):
		
		if torch.cuda.is_available():
			x = x.cuda()
			label = label.cuda()
			
		extractor.eval()
		feat = extractor(x).view(x.size(0), FEATURES_EXTRACTED, -1)
		feat = torch.mean(feat, 2)
		feat = feat.cpu().detach().numpy()
		feats_train.append(feat)
		train_y.append(label.item())

	feats_train = np.array(feats_train)
	feats_train = feats_train.reshape(6987, FEATURES_EXTRACTED)
	train_y = np.array(train_y)


	knn = KNeighborsClassifier(n_neighbors = n_neighbors)
	knn.fit(feats_train, train_y)


	# parsing valid data
	valid_y = []
	feats_valid = []
	for batch, (x, label) in enumerate(val_loader, 1):
		if torch.cuda.is_available():
			x = x.cuda()
			label = label.cuda()

		extractor.eval()
		feat = extractor(x).view(x.size(0), FEATURES_EXTRACTED, -1)
		feat = torch.mean(feat, 2)
		feat = feat.cpu().detach().numpy()
		feats_valid.append(feat)
		valid_y.append(label.item())

	feats_valid = np.array(feats_valid)
	feats_valid = feats_valid.reshape(1526, FEATURES_EXTRACTED)


	train_pred = knn.predict(feats_train)
	val_pred = knn.predict(feats_valid)

	train_accuracy = accuracy_score(train_pred, train_y)
	val_accuracy = accuracy_score(val_pred, valid_y)

	print("accuracy on training data: {}".format(train_accuracy))
	print("accuracy on validation data: {}".format(val_accuracy))	

	# Get first 10 train identities
	train10 = []
	number = np.zeros(10)
	for batch, (x, label) in enumerate(train_loader, 1):
		if torch.cuda.is_available():
			x = x.cuda()
			label = label.cuda()

		if label.item() < 10:
			feat = extractor(x).view(x.size(0), FEATURES_EXTRACTED, -1)
			feat = torch.mean(feat, 2)
			feat = feat.cpu().detach().numpy()
			train10.append(feat)
			number[label.item()] += 1
	train10 = np.array(train10)
	train10 = train10.reshape(train10.shape[0], train10.shape[2])


	# plot training TSNE
	tsne = TSNE(perplexity = 30)
	x_2d = tsne.fit_transform(train10)
	plt.figure()
	plt.subplot(1,2,1)
	plt.gca().set_title('t-SNE training data visualization')
	colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'gray', 'orange', 'purple']
	t = 0
	c = 0
	for i in number:
		for j in range(int(i)):
			plt.scatter(x_2d[t, 0], x_2d[t, 1], c = colors[c])
			t+=1
		c+=1



	# Get first 10 valid identities
	valid10 = []
	number = np.zeros(10)
	for batch, (x, label) in enumerate(val_loader, 1):
		if torch.cuda.is_available():
			x = x.cuda()
			label = label.cuda()

		if label.item() < 10:
			feat = extractor(x).view(x.size(0), FEATURES_EXTRACTED, -1)
			feat = torch.mean(feat, 2)
			feat = feat.cpu().detach().numpy()
			valid10.append(feat)
			number[label.item()] += 1
	valid10 = np.array(valid10)
	valid10 = valid10.reshape(valid10.shape[0], valid10.shape[2])


	# plot training TSNE
	tsne = TSNE(perplexity = 30)
	x_2d = tsne.fit_transform(train10)
	plt.subplot(1,2,2)
	plt.gca().set_title('t-SNE valid data visualization')
	colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'gray', 'orange', 'purple']
	t = 0
	c = 0
	for i in number:
		for j in range(int(i)):
			plt.scatter(x_2d[t, 0], x_2d[t, 1], c = colors[c])
			t+=1
		c+=1
	plt.savefig(image_path)
	plt.show()


if __name__ == "__main__":
	main()
    # TODO
    #folder, output_img = sys.argv[1], sys.argv[2]
