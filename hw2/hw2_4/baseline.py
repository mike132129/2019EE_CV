###################################################################################
## Problem 4(b):                                                                 ##
## You should extract image features using pytorch pretrained alexnet and train  ##
## a KNN classifier to perform face recognition as your baseline in this file.   ##
###################################################################################

import sys
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torchvision.models import alexnet
import torchvision.transforms as transforms
import torch
import glob, os
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from pandas import DataFrame
import pandas as pd


def get_dataloader(folder,batch_size=32):
	# Data preprocessing
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	trans = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])
	train_path, val_path = os.path.join(folder,'train'), os.path.join(folder,'valid')
	# Get dataset using pytorch functions
	train_set = ImageFolder(train_path, transform=trans)
	val_set =  ImageFolder(val_path,  transform=trans)
	train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
	val_loader  = torch.utils.data.DataLoader(dataset=val_set,  batch_size=batch_size, shuffle=False)
	print ('==>>> total trainning batch number: {}'.format(len(train_loader)))
	print ('==>>> total validation batch number: {}'.format(len(val_loader)))
	return train_loader, val_loader

def main():

	folder = sys.argv[1]
	train_loader, val_loader = get_dataloader(folder, batch_size = 1)
	alex = alexnet(pretrained = True)
	feats = []
	for batch, (x, label) in enumerate(tqdm(train_loader),1):
		
		if torch.cuda.is_available():
			x = x.cuda()
			label = label.cuda()
			alex = alex.cuda()
		extractor = alexnet(pretrained = True).features
		extractor.eval()
		feat = extractor(x).view(x.size(0), 256, -1)
		feat = torch.mean(feat, 2)
		feat = feat.cpu().detach().numpy()
		feats.append(feat)
	
	df = DataFrame(feats)
	export_csv = df.to_csv(r'./data/export_feature.csv', index = None, header = False)


	print(df)



		




	



if __name__ == "__main__":
	main()
    # TODO
    #folder, output_img = sys.argv[1], sys.argv[2]

    

