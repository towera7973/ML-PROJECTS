import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
#load the dateset in the form of numpy array 
(xtrain,ytrain),(xtest,ytest)=datasets.cifar10.load_data()
#show the shape of the data
print(xtrain.shape)
print(ytrain.shape)
print(xtest.shape)
#show the first image
plt.imshow(xtrain[0])
#lets see what in ytrain (suppose to be the names corresponding to the classes of images ofcourse)
print(ytrain[:7])
#Let change it to 1D Arrayy by reshaping it and see corresponding numbers for first 20 images 
ytrain=ytrain.reshape(-1,)
print(ytrain[:20])
#looks like y has images from 1 to 10 ,which we can give names based on the dataset description
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print(ytrain[11]) #we are assuming we have taken an image with number 11 assigned  in the dataset
print(classes[ytrain[11]])  
#lets plot some few images by creating a class to do that 

def plot_images(x,y,index)
    plt_figure(figsize=(20,3))
    plt_imshow(x[index])
    plt.ylabel(classes[y(index)])

plot_images(xtrain,ytrain,4)
plot_images(xtrain,ytrain,19)