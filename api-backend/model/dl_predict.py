import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms

import pickle
from PIL import Image

class neuralNet(nn.Module):
    def __init__(self):
        super(neuralNet,self).__init__()

        self.conv1= nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5)
        self.conv2= nn.Conv2d(in_channels=6,out_channels=12,kernel_size=5)
        self.conv3= nn.Conv2d(in_channels=12,out_channels=24,kernel_size=5)
        self.conv4= nn.Conv2d(in_channels=24,out_channels=48,kernel_size=5)
        
        self.fc1 = nn.Linear(in_features=48*12*12,out_features=240)
        self.fc2 = nn.Linear(in_features=240,out_features=120)
        self.out = nn.Linear(in_features=120,out_features=17)
        
        
    def forward(self,t):
        t = t
        
        t=self.conv1(t)
        t=F.relu(t)
        t=F.max_pool2d(t,kernel_size = 2, stride = 2)
        
        
        t=self.conv2(t)
        t=F.relu(t)
        t=F.max_pool2d(t,kernel_size = 2, stride = 2)

        t=self.conv3(t)
        t=F.relu(t)
        t=F.max_pool2d(t,kernel_size = 2, stride = 2)

        t=self.conv4(t)
        t=F.relu(t)
        t=F.max_pool2d(t,kernel_size = 2, stride = 2)
        
        t=t.reshape(-1,48*12*12)
        t=self.fc1(t)
        t=F.relu(t)
        
        
        t=self.fc2(t)
        t=F.relu(t)
        
        t=self.out(t)
        
        
        return t



def predict_disease(model, image):
	with open('model/labels.json', 'rb') as labels:
		labels = pickle.load(labels)

	model = model
	model.load_state_dict(torch.load("model/model.pth"))
	model.eval

	image = Image.open(io.BytesIO(image))
	resize = transforms.Compose([transforms.Resize((256, 256))])
	image = ToTensor()(image)

	y_result = model(resize(image).unsqueeze(0))
	result_idx = y_result.argmax(dim=1)

	for key,value in labels.items():
		if(value == result_idx):
			disease = key
			break

	
	return disease

