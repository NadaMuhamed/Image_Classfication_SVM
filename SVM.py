# importing necessary libraries
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
import imageio as iio
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.image as mpimg
import os, sys
from PIL import Image

x1=[]#image cat
x1_resize=[]
x2=[]#image dog
x2_resize=[]
pathe_cats=[]#pathe cat image
pathe_dogs=[]#pathe dog image
hog_cat=[]
fdc=[]
hog_dog=[]
fdd=[]
i=0
for i in range(1000):
    pathe_cats.append("dataset\cat."+str(i)+".jpg")
for i in range(1000):
    pathe_dogs.append("dataset\dog."+str(i)+".jpg")

for i in range(1000):
    x1.append(mpimg.imread(pathe_cats[i]))
    x1_resize.append(resize(x1[i], (128, 64)))
    hog_cat.append(np.append(hog(x1_resize[i], orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=False, multichannel=True),0))
for i in range(1000):
    x2.append(mpimg.imread(pathe_dogs[i]))
    x2_resize.append(resize(x2[i], (128, 64)))
    hog_dog.append(np.append(hog(x2_resize[i], orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=False, multichannel=True),1))
data=np.append(hog_cat,hog_dog,axis=0)
print(data.shape)
np.random.shuffle(data)
x = data[:, :-1]
y = data[:, -1]

svm=SVC()
svm.fit(x,y)
print("training Model")
print(svm.score(x,y))
