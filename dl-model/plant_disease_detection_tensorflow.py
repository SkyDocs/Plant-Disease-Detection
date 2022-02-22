#!/usr/bin/env python
# coding: utf-8

# Plant Disease Detection

# github repo: https://github.com/SkyDocs/Plant-Disease-Detection

# In[1]:


import tensorflow

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

import numpy as np


# In[2]:


train_ds = image_dataset_from_directory(directory="/content/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train",image_size=(256, 256))
valid_ds = image_dataset_from_directory(directory="/content/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid",image_size=(256, 256))

class_names = train_ds.class_names


rescale = Rescaling(scale=1.0/255)
train_ds = train_ds.map(lambda image,label:(rescale(image),label))
valid_ds  = valid_ds.map(lambda image,label:(rescale(image),label))


# In[3]:


model = keras.Sequential()


# In[4]:


model.add(keras.layers.Conv2D(32,(3,3),activation="relu",padding="same",input_shape=(256,256,3)))
model.add(keras.layers.Conv2D(32,(3,3),activation="relu",padding="same"))
model.add(keras.layers.MaxPooling2D(3,3))

model.add(keras.layers.Conv2D(64,(3,3),activation="relu",padding="same"))
model.add(keras.layers.Conv2D(64,(3,3),activation="relu",padding="same"))
model.add(keras.layers.MaxPooling2D(3,3))

model.add(keras.layers.Conv2D(128,(3,3),activation="relu",padding="same"))
model.add(keras.layers.Conv2D(128,(3,3),activation="relu",padding="same"))
model.add(keras.layers.MaxPooling2D(3,3))

model.add(keras.layers.Conv2D(256,(3,3),activation="relu",padding="same"))
model.add(keras.layers.Conv2D(256,(3,3),activation="relu",padding="same"))

model.add(keras.layers.Conv2D(512,(5,5),activation="relu",padding="same"))
model.add(keras.layers.Conv2D(512,(5,5),activation="relu",padding="same"))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(1568,activation="relu"))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(18,activation="softmax"))


# In[5]:


model.summary()


# In[6]:


model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),loss="sparse_categorical_crossentropy",metrics=['accuracy'])


# In[7]:


history = model.fit(train_ds, validation_data = valid_ds,epochs = 10, batch_size=1)


# In[8]:


model.save('model04.h5')


# In[9]:


result = model.evaluate(valid_ds)


# In[10]:


import pandas as pd
pd.DataFrame(history.history).plot(figsize=(10, 7));


# In[ ]:





# In[11]:


class_names


# In[ ]:




