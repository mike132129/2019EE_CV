import os, sys, glob
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

class TestDataset(Dataset):
    """Test dataset."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = glob.glob(root_dir+'*.png')
        self.images.sort()
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_name.split('/')[-1]



if __name__ == "__main__":

	data_path = sys.argv[1]


	model = models.resnet18(pretrained = False)
	model.load_state_dict(torch.load('./checkpoint/resnet18.pth', map_location=torch.device('cpu')))


	use_cuda = torch.cuda.is_available()
	if use_cuda:
		model.cuda()

	print(model)

	model.eval()

	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	trans = transforms.Compose([transforms.ToTensor(), normalize])
	test_set = TestDataset(data_path, transform = trans)
	print('Length of Testing set:', len(test_set))
	test_loader = DataLoader(dataset = test_set, batch_size = 1, shuffle = False)
	prediction = []
	with torch.no_grad():
		for batch, (x, name) in enumerate(test_loader):
			if use_cuda:
				x = x.cuda()
			out = model(x)
			_, pred_label = torch.max(out,1)
			prediction.append((name[0][:-4], pred_label.item()))
	print(prediction)

