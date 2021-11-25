#!/usr/bin/env python
# coding: utf-8

# # Github repo url
# 
# [https://github.com/SkyDocs/Plant-Disease-Detection](https://github.com/SkyDocs/Plant-Disease-Detection)

# In[3]:


import os

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor


# In[2]:


# get_ipython().system('unzip dataset.zip')


# dataset classification

# In[30]:


class train_Data():
    def __init__(self):
        self.labels,self.images = self.load_data()
        
    def load_data(self):
        resize = transforms.Compose([transforms.Resize(256, 256)])
        train_dir = os.listdir(os.path.join("dataset", "train"))
        
        ref = {}
        lables = {}
        image_list = {}
        global_count = 0
         
        for i,dir in enumerate(train_dir):
            ref[dir] = i
            images = os.listdir(os.path.join("dataset", "train", dir))
            count = 0
            for img in images:
                if count < 500:
                    lables[count] = i
                    img = os.path.join("dataset", "train", dir, img)
                    image = Image.open(img)
                    image = ToTensor()(image)
                    image_list[count] = resize(image)
                    count += 1
                    global_count += 1
                    print(i)
                else:
                    pirnt("Taken 500 images for training")
                    break
        
        print(reference)
        return lables,image_list
    
    def __len__(self):
        return len(self.labels)


# In[31]:


train_data = train_Data()


# In[32]:


# class validationData(trainData):
   
#     def load_data(self):
#         resize = transforms.Compose([transforms.Resize(256, 256)])
#         train_dir = os.listdir(os.path.join("dataset", "valid"))
        
#         ref = {}
#         lables = {}
#         image_list = {}
#         global_count = 0
         
#         for i,dir in enumerate(train_dir):
#             ref[dir] = i
#             images = os.listdir(os.path.join("dataset", "valid", dir))
#             count = 0
#             for img in images:
#                 if count < 100:
#                     lables[count] = i
#                     img = os.path.join("dataset", "valid", dir, img)
#                     image = Image.open(img)
#                     image = ToTensor()(image)
#                     image_list[count] = resize(image)
#                     count += 1
#                     global_count += 1
#                     print(i)
#                 else:
#                     pirnt("Taken 100 images for validation")
#                     break
        
#         print(reference)
#         return lables,image_list


# # In[33]:


# test_data = validationData()


# # model

# # In[34]:


# class neuralNet(nn.Module):
#     def __init__(self):
#         super(neuralNet, self).__init__()
        
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
#         self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
#         self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5)
#         self.conv4 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=5)
        
#         self.fc1 = nn.Linear(in_features=48*12*12,out_features=240)
#         self.fc2 = nn.Linear(in_features=240,out_features=120)
#         self.out = nn.Linear(in_features=120,out_features=17)
        
#     def forward(self, x):
#         x = x
        
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, kernel_size=2, stride=2)
        
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, kernel_size=2, stride=2)
        
#         x = self.conv3(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, kernel_size=2, stride=2)
        
#         x = self.conv4(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, kernel_size=2, stride=2)
        
#         x = x.reshape(-1, 48*12*12)
#         x = x.fc1(x)
#         x = F.relu(x)
        
#         x = self.fc2(x)
#         x = F.relu(x)
        
#         x = self.out(x)
        
#         return x


# # In[35]:


# model = neuralNet()


# # training

# # In[36]:


# def train(train_data, test_data, model):
#     model.train()
    
#     trainData = torch.utils.data.DataLoader(train_data, batch_size=64,shuffle=True)
#     testData = torch.utils.data.DataLoader(test_data, batch_size=32,shuffle=True)
    
    
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     num_of_epochs = 20
#     epochs = []
#     losses = []
#     for epoch in range(num_of_epochs):
#         cnt = 0
#         tot_loss = 0
#         tot_correct = 0
#         for batch, (x, y) in enumerate(trainData):
#             # Sets the gradients of all optimized tensors to zero
#             optimizer.zero_grad()
#             y_pred = model(x)
#             # Compute loss (here CrossEntropyLoss)
#             loss = F.cross_entropy(y_pred,y)

#             loss.backward()
#             optimizer.step()

#         for batch, (x, y) in enumerate(testData):
#             # Sets the gradients of all optimized tensors to zero
#             optimizer.zero_grad()
#             with torch.no_grad():
#                 y_pred = model(x)
#                 # Compute loss (here CrossEntropyLoss)
#                 loss = F.cross_entropy(y_pred,y)

#             tot_loss+=loss.item()
# #             tot_correct +=get_num_correct(y_pred,y)
#         epochs.append(epoch)
#         losses.append(tot_loss)
# #         print("Epoch",epoch,"total_correct",tot_correct,"loss:",tot_loss)
#         torch.save(model.state_dict(), "model002_ep"+str(epoch+1)+".pth")


# # In[37]:


# train(train_data,test_data, model)


# # In[ ]:





# # In[ ]:




