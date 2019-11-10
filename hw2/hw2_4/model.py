import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)
        
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):

        x = torch.nn.functional.relu(self.conv1(x))  
        x = self.max_pool_1(x)
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.max_pool_2(x)
        x = x.view(-1, 16*5*5)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

    def name(self):
        return "ConvNet"

class Fully(nn.Module):
    def __init__(self):
        super(Fully, self).__init__()
        # TODO
        # self.fc1 = nn.Linear(784, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)
        self.fc1 = nn.Sequential(nn.Linear(784, 120),
                                nn.ReLU(),
                                nn.Linear(120, 84),
                                nn.Dropout(0.3),
                                nn.ReLU(), 
                                nn.Linear(84, 10))


    def forward(self, x):
        x = x.view(x.size(0),-1) # flatten input tensor
        # TODO

        x = self.fc1(x)
        return x

    def name(self):
        return "Fully"

