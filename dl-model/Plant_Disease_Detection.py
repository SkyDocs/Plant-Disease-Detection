#!/usr/bin/env python
# coding: utf-8

# # Github repo url
# 
# [https://github.com/SkyDocs/Plant-Disease-Detection](https://github.com/SkyDocs/Plant-Disease-Detection)

# In[1]:


import os

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import torch.optim as optim


# In[2]:


torch.cuda.is_available()


# In[3]:


#!unzip dataset.zip


# dataset classification

# In[4]:


class trainData():
    
    def __init__(self):
        self.labels,self.images = self.load_data()

    def load_data(self):
        labels = {}
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
    
    
#     def load_data(self):
#         resize = transforms.Compose([transforms.Resize((256, 256))])
#         train_dir = os.listdir(os.path.join("dataset", "train"))
        
#         ref = {}
#         labels = {}
#         image_list = {}
#         global_count = 0
         
#         for i,dir in enumerate(train_dir):
#             ref[dir] = i
#             images = os.listdir(os.path.join("dataset", "train", dir))
#             count = 0
#             for img in images:
#                 if count < 500:
#                     labels[global_count] = i
#                     img = os.path.join("dataset", "train", dir, img)
#                     image = Image.open(img)
#                     image = ToTensor()(image)
#                     image_list[count] = resize(image)
#                     count += 1
#                     global_count += 1
#                 else:
#                     pirnt("Taken 500 images for training")
#                     break
        
#         print(ref)
#         return labels,image_list
    
    def __len__(self):
        return len(self.labels)
    
    # To return x,y values in each iteration over dataloader as batches.
    def __getitem__(self, idx):
        return(self.images[idx], self.labels[idx])


# In[5]:


train_data = trainData()


# In[6]:


class validationData(trainData):
    
    def load_data(self):
        resize = transforms.Compose([transforms.Resize((256,256))])
        valid_dir = os.listdir(os.path.join("dataset", "valid"))
        
        labels={}
        images = {}
        global_count = 0
         
        for i,dir in enumerate(valid_dir):
            print(i,dir)
            image_path = os.listdir(os.path.join("dataset", "valid", dir))
            count = 0
            for img in image_path:
                if(count<100):
                    labels[global_count] = i
                    img = os.path.join("dataset", "valid", dir, img)
                    image = Image.open(img)
                    image = ToTensor()(image)
                    images[count] = resize(image)
                    global_count+=1
                    count+=1
                else:
                    break
                    
        return labels,images


# In[7]:


valid_data = validationData()


# In[8]:


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


# model

# In[9]:


class neuralNet(nn.Module):
    def __init__(self):
        super(neuralNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5)
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=5)
        
        self.fc1 = nn.Linear(in_features=48*12*12,out_features=240)
        self.fc2 = nn.Linear(in_features=240,out_features=120)
        self.out = nn.Linear(in_features=120,out_features=17)
        
    def forward(self, x):
        x = x
        
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = x.reshape(-1, 48*12*12)
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        
        x = self.out(x)
        
        return x


# In[10]:


model = neuralNet()


# training

# In[11]:


def train(train_data, valid_data, model):
    model.train()
    
    trainData = torch.utils.data.DataLoader(train_data, batch_size=64,shuffle=True)
    validData = torch.utils.data.DataLoader(valid_data, batch_size=32,shuffle=True)
    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_of_epochs = 5
    epochs = []
    losses = []
    for epoch in range(num_of_epochs):
        cnt = 0
        tot_loss = 0
        tot_correct = 0
        for batch, (x, y) in enumerate(trainData):
            # Sets the gradients of all optimized tensors to zero
            optimizer.zero_grad()
            y_pred = model(x)
            # Compute loss (here CrossEntropyLoss)
            loss = F.cross_entropy(y_pred,y)

            loss.backward()
            optimizer.step()

        for batch, (x, y) in enumerate(validData):
            # Sets the gradients of all optimized tensors to zero
            optimizer.zero_grad()
            with torch.no_grad():
                y_pred = model(x)
                # Compute loss (here CrossEntropyLoss)
                loss = F.cross_entropy(y_pred,y)

            tot_loss += loss.item()
            tot_correct += get_num_correct(y_pred,y)
        epochs.append(epoch)
        losses.append(tot_loss)
        print("Epoch",epoch,"total_correct",tot_correct,"loss:",tot_loss)
        torch.save(model.state_dict(), "model002_ep"+str(epoch+1)+".pth")


# In[12]:


train(train_data, valid_data, model)


# In[ ]:





# In[ ]:




