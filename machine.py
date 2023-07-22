#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image, ImageOps
from sklearn.metrics import classification_report
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D,MaxPool2D,Dense,Flatten,Dropout


# In[2]:


data = []
categories = []
mainDir = "C:\\Users\\0Marwan Ashraf\\.0MyStuff\\.0college\\4th Semester\\Machine Learning\\Z Project\\"
path = mainDir + "images"
meta = pd.read_csv(mainDir + "annotations.csv")
print(meta.head())

# Check for missing values
if meta.isnull().sum().sum() > 0:
    print('There are missing values in the dataset.')
else: 
    print('there are not missing values')

print(meta.dtypes)


# Check for duplicates in rows
if meta.duplicated().sum()>0:
    print('There are duplicate rows in the dataset.')
else:
    print('there are not duplicate rows in the dataset ')

# check for duplicate values in the 'Filename' column
duplicates = meta[meta.duplicated(['file_name'], keep=False)]

# print the duplicate values
print(f"There are {len(duplicates)} in the dataset")

meta = meta.drop_duplicates(subset=['file_name'], keep='first')


# In[3]:


images = os.listdir(path)

for x in images:
    image = Image.open(path + '\\' + x)
    x1 = meta.x1[meta.file_name == x].iloc[0]
    x2 = meta.x2[meta.file_name == x].iloc[0]
    y1 = meta.y1[meta.file_name == x].iloc[0]
    y2 = meta.y2[meta.file_name == x].iloc[0]
    image = image.crop((x1,y1,x2,y2))
    image = image.resize((30,30))
    image = np.array(image)
    data.append(image)
    categories.append(meta.category[meta.file_name == x].iloc[0])
        
data = np.array(data)
categories = np.array(categories)

x_train,x_test,y_train,y_test = train_test_split(data,categories,test_size = 0.3,random_state = 0)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# converts a class vector (integers) to one-hot class matrix
y_train = to_categorical(y_train,58)
y_test = to_categorical(y_test,58)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


# In[4]:


CNN = mainDir + "CNN.h5"
ANN = mainDir + "ANN.h5"


# In[5]:


if(os.path.exists(CNN)):
    CNNmodel = load_model(CNN)
else:
    CNNmodel = Sequential()

    # the input shape determines the number of inputs to this layer
    # which is 30 x 30 x 3 
    # the kernel size is a list of 2 integers specifying the height and width of the 2d
    # convolution window

    CNNmodel.add(Conv2D(filters=32,kernel_size=(5,5),activation='relu',input_shape=x_train.shape[1:]))
    CNNmodel.add(Conv2D(filters=32,kernel_size=(5,5),activation='relu'))

    # users max pooling with pool size of 2x2 and stepping of 2 columns for each step

    CNNmodel.add(MaxPool2D(pool_size=(2,2)))

    # Dropout is easily implemented by randomly selecting nodes to be dropped out with a given probability

    CNNmodel.add(Dropout(rate = 0.25))
    CNNmodel.add(Flatten())
    CNNmodel.add(Dense(256,activation='relu'))

    # This is the output layer made of 58 neourons in because we have 58 classes in our dataset

    CNNmodel.add(Dropout(rate = 0.5))

    # it is a non-linear activation function which produces
    # outputs between zero and one but are restricted to have
    # sum of one while sigmoid function gives outputs independently 
    # from zero to one

    CNNmodel.add(Dense(58,activation='softmax'))

    # 1- categorical crossentropy is a probabilistic loss function which calculates the 
    # cross-entropy loss between true labesl and predicted ones
    # used when there is two or more label classes and labels are integers
    # 2- adam optimizer is a stochastic gradient descent method that is based on adaptive estimation
    # of first-order and second-order moments it's default learning rate is 0.001
    # 3- accuracy metrics calculates how often predictions equal labels

    CNNmodel.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    # trains the model for a fixed number of iterations(epochs)
    # number of samples per training iteration with default of 32

    CNNmodel.fit(x_train,y_train,batch_size=32,epochs=8,validation_data=(x_test,y_test))
    
    CNNmodel.save(mainDir+"CNN.h5")


# In[6]:


if(os.path.exists(ANN)):
    ANNmodel = load_model(ANN)
else:
    ANNmodel = Sequential()

    # makes an input layer with shape of 30(length) x 30(width) x 3(The three colors RGB) input neurons which results in 2700 neurons

    ANNmodel.add(Flatten(input_shape=(30,30,3)))

    # makes a hidden layer with 128 input neurons the dense type means that every neuron in the previous 
    # layer is connected to each neuron in this layer (rectified linear unit function)

    ANNmodel.add(Dense(128, activation='relu'))

    # makes a hidden layer with 64 input neurons the dense type means that every neuron in the previous 
    # layer is connected to each neuron in this layer (rectified linear unit function)

    ANNmodel.add(Dense(64, activation='relu'))

    # This is the output layer made of 58 neourons in because we have 58 classes in our dataset (softmax function)
    
    ANNmodel.add(Dense(58, activation='softmax'))

    ANNmodel.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    ANNmodel.fit(x_train, y_train, epochs=25,validation_data=(x_test,y_test))
    
    ANNmodel.save(mainDir+"ANN.h5")


# In[7]:


print(CNNmodel.summary())
print(ANNmodel.summary())


# In[8]:

# axis 1 means that the argmax method works on the horizontal line of the 2D array
# while axis 0 means that the argmax method works on the vertical line of the 2D array
y_actual = np.argmax(y_test,axis=1)
CNN_y_pred = CNNmodel.predict(x_test)
CNN_y_pred = np.argmax(CNN_y_pred,axis=1)

ANN_y_pred = ANNmodel.predict(x_test)
ANN_y_pred = np.argmax(ANN_y_pred,axis=1)


# In[9]:


CNN_CM = metrics.confusion_matrix(y_actual,CNN_y_pred)
figure,ax = plt.subplots(figsize=(13,13))
CNN_CMplot = metrics.ConfusionMatrixDisplay(CNN_CM)
CNN_CMplot.plot(ax=ax)
plt.title('CNN model')
plt.show()


# In[10]:


ANN_CM = metrics.confusion_matrix(y_actual,ANN_y_pred)
figure,ax = plt.subplots(figsize=(13,13))
ANN_CMplot = metrics.ConfusionMatrixDisplay(ANN_CM)
ANN_CMplot.plot(ax=ax)
plt.title('ANN model')
plt.show()


# In[11]:


print("CNN evaluation","-"*50)
print(classification_report(y_actual,CNN_y_pred))


# In[12]:


print("ANN evaluation","-"*50)
print(classification_report(y_actual,ANN_y_pred))


# In[14]:

# The code part to test the model

# img = Image.open(r"C:\Users\DELL\Desktop\png-transparent-bicycle-cycling-segregated-cycle-facilities-traffic-sign-motorcycle-traffic-signs-blue-text-trademark-thumbnail.png")
# img = img.convert("RGB")
# img = img.resize((30,30))
# img = np.array(img)
# p = []
# p.append(img)
# p = np.array(p)
# CNNout = np.argmax(CNNmodel.predict([p]))
# ANNout = np.argmax(ANNmodel.predict([p]))
# print(f'CNN: {CNNout}')
# print(f'ANN: {ANNout}')
# CNNimageName = meta.file_name[meta.category == CNNout].iloc[0]
# ANNimageName = meta.file_name[meta.category == ANNout].iloc[0]
# Image.open(f"{path}\\{CNNimageName}").show()
# Image.open(f"{path}\\{ANNimageName}").show()

