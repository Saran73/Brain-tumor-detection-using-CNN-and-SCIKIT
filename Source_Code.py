#LOAD MODULES
!pip install scikit-learn
!pip install numpy
!pip install pandas
!pip install matplotlib
!pip install opencv-python



#COLLECTION OF DATA
import sklearn as sk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


import numpy as пр
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import os
path = os.listdir('C:\\Users\\ASUS\\Desktop\\MIni\\brain tumour\\training')
classes = {'no_tumor':0, 'pituitary_tumor':1}

import cv2
X = []
Y = []
for cls in classes:
    path = 'C:\\Users\\ASUS\\Desktop\\MIni\\brain tumour\\training\\' +cls
    for j in os.listdir(path):
        img = cv2.imread (path+'/'+j, 0)
        img = cv2.resize (img, (200,200))
        X.append (img)
        Y.append (classes[cls])


np.unique(Y)

X=np.array(X)
Y=np.array(Y)

#REPRESENTS IMAGES WITH OR WITHOUT TUMOUR
pd.Series(Y) .value_counts()

#TOTAL IMAGES LOADED 
X.shape

#DATA VISUALIZATION
plt.imshow(X[0],cmap='gray')

#PREPARE DATA FOR 2-D SETS
X_updated = X.reshape(len(X),-1)
X_updated.shape

#SPLIT OPERATION
xtrain, xtest, ytrain, ytest = train_test_split(X_updated, Y, random_state=10,
                                               test_size=.20)

#FOR TRAINING AND TESTING 
xtrain.shape, xtest.shape

#FEATURE SCALING
print (xtrain.max(), xtrain. min())
print (xtest. max(), xtest.min())
xtrain = xtrain/255
xtest = xtest/255
print (xtrain. max(), xtrain. min())
print (xtest. max(), xtest.min())

#MODEL TRAINING:
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import warnings
warnings.filterwarnings('ignore')
lg = LogisticRegression(C=0.1)
lg.fit(xtrain, ytrain)


sv = SVC()
sv.fit(xtrain, ytrain)


#EVALUATION
print("Training Score:", lg.score(xtrain, ytrain))
print("Testing Score:", lg.score(xtest, ytest))

print("Training Score:", sv.score(xtrain, ytrain))
print("Testing Score:", sv.score(xtest, ytest))


#PREDICTIONS
pred = sv.predict(xtest)
misclassified=np.where(ytest!=pred)


print("Total Misclassified Samples: ",len(misclassified[0]))
print (pred[3],ytest[6])

pred[3]
ytest[6]

#TEST MODEL
dec = {0:'No Tumor' , 1:'Positive Tumor'}

plt.figure(figsize=(12,8))
p = os.listdir('C:\\Users\\ASUS\\Desktop\\MIni\\brain tumour\\testing')
c=1
for i in os.listdir('C:\\Users\\ASUS\\Desktop\\MIni\\brain tumour\\testing\\')[:15]:
	plt.subplot(4,4,c)

	img = cv2.imread('C:\\Users\\ASUS\\Desktop\\MIni\\brain tumour\\testing\\'+i,0)
	img1 = cv2.resize(img, (200,200))
	img1 = img1.reshape(1,-1)/255
	p = sv.predict(img1)
	plt.title(dec[p[0]])
	plt.imshow(img, cmap='gray')
	plt.axis('off')
	c+=1

plt.figure(figsize=(12,8))
p= os.listdir('C:\\Users\\ASUS\\Desktop\\MIni\\brain tumour\\testing')
c=1
for i in os. listdir('C:\\Users\\ASUS\\Desktop\\MIni\\brain tumour\\testing\\')[:16]:
	plt.subplot(4,4,c)
	img = cv2.imread('C:\\Users\\ASUS\\Desktop\\MIni\\brain tumour\\testing\\'+i,0)
	img1 = cv2.resize(img, (200, 200))
	img1 = img1.reshape(1,-1)/255
	p = sv.predict(img1)
	plt.title(dec[p[0]])
	plt.imshow(img, cmap='gray')
	plt.axis('off')
	c+=1


pip  install seaborn

#CONFUSION MATRIX
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(ytest,pred,labels = sv.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = cm,display_labels = sv.classes_)
disp.plot()
plt.show()
