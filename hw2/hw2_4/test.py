import sys
import numpy as np
import torchvision.transforms as transforms
import torch
import glob, os
from torchvision.datasets import ImageFolder
# from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from pandas import DataFrame
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import torchvision.models as models
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def load_split_train_test(datadir, valid_size = .2):
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
	train_transforms = transforms.Compose([#transforms.CenterCrop(224),
											#transforms.Resize(224),
											#transforms.RandomHorizontalFlip(),
											transforms.ToTensor(),
											normalize,])


	train_data = ImageFolder(datadir+'/train', transform=train_transforms)
	test_data = ImageFolder(datadir+'/valid', transform=train_transforms)

	trainloader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle = True)
	testloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle = False)
	print ('==>>> total trainning batch number: {}'.format(len(trainloader)))
	print ('==>>> total validation batch number: {}'.format(len(testloader)))
	return trainloader, testloader

def get_dataloader(folder, batch_size=32):
	# Data preprocessing
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
	folder = sys.argv[1]

	# Get data loaders of training set and validation set
	train_loader, val_loader = get_dataloader(folder, batch_size = 1)


	model = models.resnet18(pretrained = True)
	model.eval()
	extractor1 = model.conv1
	extractor2 = model.bn1
	extractor3 = model.relu
	extractor4 = model.maxpool
	extractor5 = model.layer1


		# Get first 10 train identities

	train10 = []
	number = np.zeros(10)
	for batch, (x, label) in enumerate(tqdm(train_loader), 1):
		if torch.cuda.is_available():
			x = x.cuda()
			label = label.cuda()

		if label.item() < 10:
			feat = extractor1(x)
			feat = extractor2(feat)
			feat = extractor3(feat)
			feat = extractor4(feat)
			feat = extractor5(feat).view(x.size(0), 256, -1)
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
	for batch, (x, label) in enumerate(tqdm(val_loader), 1):
		if torch.cuda.is_available():
			x = x.cuda()
			label = label.cuda()

		if label.item() < 10:
			feat = extractor1(x)
			feat = extractor2(feat)
			feat = extractor3(feat)
			feat = extractor4(feat)
			feat = extractor5(feat).view(feat.size(0), 256, -1)
			feat = torch.mean(feat, 2)
			feat = feat.cpu().detach().numpy()
			valid10.append(feat)
			number[label.item()] += 1
	valid10 = np.array(valid10)
	valid10 = valid10.reshape(valid10.shape[0], valid10.shape[2])

	# plot training TSNE
	tsne = TSNE(perplexity = 30)
	x_2d = tsne.fit_transform(valid10)
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
	plt.show()


	# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
	# criterion = nn.CrossEntropyLoss()
	
	# use_cuda = torch.cuda.is_available()
	# if use_cuda:
	# 	model.cuda()

	# print(model)

	# ep = 15
	
	# training_accuracy = []
	# training_loss = []
	# validation_accuracy = []
	# validation_loss = []

	# for epoch in range(ep):
	# 	print('Epoch:', epoch)

	# 	#training 
	# 	correct_cnt, total_loss, total_cnt, ave_loss = 0, 0, 0, 0

	# 	for batch, (x, label) in enumerate(tqdm(train_loader), 1):
	# 		optimizer.zero_grad()

	# 		if use_cuda:
	# 			x, label = x.cuda(), label.cuda()

	# 		out = model(x)
	# 		loss = criterion(out, label)
	# 		loss.backward()
	# 		optimizer.step()

	# 		total_loss += loss.item()
	# 		_, pred_label = torch.max(out, 1)
	# 		total_cnt += x.size(0)
	# 		correct_cnt += (pred_label == label).sum().item()

	# 		if batch % 100 == 0 or batch == len(train_loader):
	# 			acc = correct_cnt / total_cnt
	# 			ave_loss = total_loss / batch           
	# 			print ('Training batch index: {}, train loss: {:.6f}, acc: {:.3f}'.format(
	# 				batch, ave_loss, acc))

	# 	training_accuracy.append(correct_cnt/total_cnt)
	# 	training_loss.append(ave_loss)

	# 	model.eval()

	# 	val_loss, val_acc = 0, 0
	# 	correct_cnt, total_loss, total_cnt, ave_loss = 0, 0, 0, 0
	# 	for batch, (x, label) in enumerate(val_loader,1):
	# 		if use_cuda:
	# 			x, label = x.cuda(), label.cuda()

	# 		pred_val = model(x)
	# 		loss = criterion(pred_val, label)

	# 		val_loss += loss.item()
	# 		_, pred_label = torch.max(pred_val, 1)
	# 		total_cnt += x.size(0)
	# 		correct_cnt += (pred_label == label).sum().item()

	# 		if batch == len(val_loader):
	# 			val_loss = val_loss / batch
	# 			val_acc = correct_cnt / total_cnt
	# 			print('Validation batch index: {}, val_loss: {:.6f}, acc" {:.3f}'.format(
	# 				batch, val_loss, val_acc))

	# 	validation_accuracy.append(val_acc)
	# 	validation_loss.append(val_loss)


	# 	model.train()

	# torch.save(model.state_dict(), './checkpoint/resnet18.pth')

	# print(model)

	# epoch = list(range(ep))

	# fig1 = plt.figure()
	# plt.plot(epoch, training_accuracy, label = 'training accuracy')
	# plt.legend(loc = 'best')
	# plt.xlabel('Epoch')
	# plt.ylabel('Accuracy')
	# plt.title('Epoch-training accuracy')
	# plt.savefig('./data/resnet_training_accuracy.png')

	# fig2 = plt.figure()
	# plt.plot(epoch, validation_accuracy, label = 'validation accurac')
	# plt.legend(loc = 'best')
	# plt.xlabel('Epoch')
	# plt.ylabel('Accuracy')
	# plt.title('Epoch-validation accuracy')
	# plt.savefig('./data/resnet_validation_accuracy.png')

	# fig3 = plt.figure()
	# plt.plot(epoch, training_loss, label = 'training loss')
	# plt.legend(loc = 'best')
	# plt.xlabel('Epoch')
	# plt.ylabel('Loss')
	# plt.title('Epoch-training loss')
	# plt.savefig('./data/resnet_training_loss.png')

	# fig4 = plt.figure()
	# plt.plot(epoch, validation_loss, label = 'validation loss')
	# plt.legend(loc = 'best')
	# plt.xlabel('Epoch')
	# plt.ylabel('Loss')
	# plt.title('Epoch-validation loss')
	# plt.savefig('./data/resnet_validation_loss.png')
	# print(validation_accuracy)


if __name__ == "__main__":
	main()

