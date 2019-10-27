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

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(46656, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 100)
        # TODO

    def forward(self, x):

        x = torch.nn.functional.relu(self.conv1(x))  
        x = self.max_pool_1(x)
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.max_pool_2(x)
        x = x.view(-1, 46656)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

    def name(self):
        return "ConvNet"

def get_dataloader(folder,batch_size=10):
	# Data preprocessing
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	trans = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), normalize])
	train_path, val_path = os.path.join(folder,'train'), os.path.join(folder,'valid')
	# Get dataset using pytorch functions
	train_set = ImageFolder(train_path, transform=trans)
	val_set =  ImageFolder(val_path,  transform=trans)
	train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
	val_loader  = torch.utils.data.DataLoader(dataset=val_set,  batch_size=batch_size, shuffle=False)
	print ('==>>> total trainning batch number: {}'.format(len(train_loader)))
	print ('==>>> total validation batch number: {}'.format(len(val_loader)))
	return train_loader, val_loader

def load_split_train_test(datadir, valid_size = .2):
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
	train_transforms = transforms.Compose([transforms.CenterCrop(224),
											transforms.Resize(224),
											transforms.RandomHorizontalFlip(),
											transforms.ToTensor(),
											normalize,])


	train_data = ImageFolder(datadir+'/train', transform=train_transforms)
	test_data = ImageFolder(datadir+'/valid', transform=train_transforms)

	trainloader = torch.utils.data.DataLoader(train_data, batch_size=10, shuffle = True)
	testloader = torch.utils.data.DataLoader(test_data, batch_size=10, shuffle = False)
	print ('==>>> total trainning batch number: {}'.format(len(trainloader)))
	print ('==>>> total validation batch number: {}'.format(len(testloader)))
	return trainloader, testloader

def main():
	folder = sys.argv[1]

	# Get data loaders of training set and validation set
	train_loader, testloader = load_split_train_test(folder)
	print(train_loader.dataset.classes)


	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	model = models.resnet50(pretrained=True)
	model.fc = nn.Sequential(nn.Linear(2048, 512),
							nn.ReLU(),
							nn.Dropout(0.2),
							nn.Linear(512, 100),
							nn.LogSoftmax(dim=1))
	optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
	criterion = nn.CrossEntropyLoss()
	
	use_cuda = torch.cuda.is_available()
	if use_cuda:
		model.cuda()

	epochs = 10
	
	training_accuracy = []
	training_loss = []
	validation_accuracy = []
	validation_loss = []

	for epoch in range(epochs):
		print('Epoch:', epoch)

		#training 
		correct_cnt, total_loss, total_cnt, ave_loss = 0, 0, 0, 0

		for batch, (x, label) in enumerate(tqdm(train_loader), 1):
			optimizer.zero_grad()

			if use_cuda:
				x, label = x.cuda(), label.cuda()

			out = model(x)
			loss = criterion(out, label)
			loss.backward()
			optimizer.step()

			total_loss += loss.item()
			_, pred_label = torch.max(out, 1)
			total_cnt += x.size(0)
			correct_cnt += (pred_label == label).sum().item()

			if batch % 2 == 0 or batch == len(train_loader):
				acc = correct_cnt / total_cnt
				ave_loss = total_loss / batch           
				print ('Training batch index: {}, train loss: {:.6f}, acc: {:.3f}'.format(
					batch, ave_loss, acc))
		training_accuracy.append(correct_cnt/total_cnt)
		training_loss.append(ave_loss)

		model.eval()
		model.train()

	torch.save(model.state_dict(), './checkpoint/%sbaseline.pth' % model.name())




if __name__ == "__main__":
	main()

