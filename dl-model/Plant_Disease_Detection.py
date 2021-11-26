#!/usr/bin/env python
# coding: utf-8

# # Github repo url
# 
# [https://github.com/SkyDocs/Plant-Disease-Detection](https://github.com/SkyDocs/Plant-Disease-Detection)

# In[86]:


import os

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle


# In[67]:


torch.cuda.is_available()


# In[68]:


#!unzip dataset.zip


# dataset classification

# In[69]:


class trainData():
    def __init__(self):
        self.labels,self.images = self.load_data()
    
    def load_data(self):
        labels={}
        images = {}
        count = 0
        
        resize = transforms.Compose([transforms.Resize((256,256))])
        main_dir = os.listdir(os.path.join("dataset","train"))
        reference = {}
        
        for i,dir in enumerate(main_dir):
            reference[dir]=i
            images_list = os.listdir(os.path.join("dataset","train",dir))
            local_cnt = 0
            
            for img in images_list:
                if local_cnt<500:
                    labels[count] = i
                    img_path = os.path.join("dataset","train",dir,img)
                    image = Image.open(img_path)
                    image = ToTensor()(image)
                    images[count] = resize(image)
                    count+=1
                    local_cnt+=1
                else:
                    break

        print(reference)
        return labels,images
      
    def __len__(self):
        return len(self.labels)
    

    # To return x,y values in each iteration over dataloader as batches.
    def __getitem__(self, idx):
        return (
            self.images[idx],
            self.labels[idx],
        )


# In[70]:


train_data = trainData()


# In[71]:


class validationData(trainData):

    def load_data(self):
          labels={}
          images = {}
          count = 0
          resize = transforms.Compose([transforms.Resize((256,256))])
          main_dir = os.listdir(os.path.join("dataset","valid"))
          for i,dir in enumerate(main_dir):
              print(i,dir)
              images_list = os.listdir(os.path.join("dataset","valid",dir))
              local_cnt = 0
              for img in images_list:
                  if(local_cnt<100):
                      labels[count] = i
                      img_path = os.path.join("dataset","valid",dir,img)
                      image = Image.open(img_path)
                      image = ToTensor()(image)
                      images[count] = resize(image)
                      count+=1
                      local_cnt+=1
                  else:
                      break

          return labels,images


# In[72]:


valid_data = validationData()


# In[73]:


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


# model

# In[74]:


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


# In[75]:


model = neuralNet()


# training

# In[84]:


def train(train_data,valid_data, model):
    model.train()

    
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=64,shuffle=True)
    valdataloader = torch.utils.data.DataLoader(valid_data, batch_size=32,shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_of_epochs = 5
    epochs = []
    losses = []
    for epoch in range(num_of_epochs):
        cnt = 0
        tot_loss = 0
        tot_correct = 0
        for batch, (x, y) in enumerate(dataloader):
            
            optimizer.zero_grad()
            y_pred = model(x)
            
            loss = F.cross_entropy(y_pred,y)

            loss.backward()
            optimizer.step()

        for batch, (x, y) in enumerate(valdataloader):
            optimizer.zero_grad()
            with torch.no_grad():
                y_pred = model(x)
                loss = F.cross_entropy(y_pred,y)

            tot_loss += loss.item()
            tot_correct += get_num_correct(y_pred,y)
        epochs.append(epoch)
        losses.append(tot_loss)
        print("Epoch",epoch,"total_correct",tot_correct,"loss:",tot_loss)
        torch.save(model.state_dict(), "model002_ep"+str(epoch+1)+".pth")
        
        plt.plot(epochs, losses, color='green', linewidth = 3, marker='o', markerfacecolor='blue', markersize=8) 
        plt.xlabel('epochs ---->',color='m',fontsize='x-large' ) 
        plt.ylabel('loss ------>',color='m',fontsize='x-large') 
        axes = plt.gca()
        axes.set_facecolor('c')
        axes.tick_params(axis='y', which='both', colors='tomato')
        axes.tick_params(axis='x', which='both', colors='#20ff14')
        plt.title("Val Loss vs Epoch",color='m',fontsize='x-large')


# In[85]:


train(train_data, valid_data, model)


# In[ ]:





# In[88]:


# Saving labels to label value as a json
main_dir = os.listdir(os.path.join("dataset","train"))
reference = {}
for i,dir in enumerate(main_dir):
    reference[dir]=i
with open('labels.json', 'wb') as iw:
    pickle.dump(reference, iw)


# In[89]:


torch.save(model.state_dict(), "model.pth")


# predict

# In[90]:


def predict(img_path):
    image = Image.open(img_path)
    image = ToTensor()(image)
    resize = transforms.Compose([transforms.Resize((256,256))])
    y_result = model(resize(image).unsqueeze(0))
    result_idx = y_result.argmax(dim=1)
    for key,value in reference.items():
        if(value==result_idx):
            print(key)
            break


# In[92]:


predict("/home/harshit/Downloads/Potato.jpg")


# In[ ]:




