# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 14:22:38 2018

@author: daizh
"""
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from itertools import cycle
import pandas as pd
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

# Import some data to play with


df = pd.read_csv("C:\\Users\\daizh\\Desktop\\Wilson\\Hourly and weather categorized2.csv")
#df = pd.read_csv("C:\\Users\\daizh\\Desktop\\Wilson\\TTI_resampled3.csv")
X = df[['TTI','Max TemperatureF','Mean TemperatureF','Min TemperatureF',' Min Humidity']].values
Y = df['TTI_Category'].as_matrix()
Y = Y.ravel()
print (type(Y))
print (Y)

#print (Y)
# Binarize the output
y = label_binarize(Y, classes=['Bad','Good'])

print(y)
#print(X)
#print(y)
# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                    random_state=0)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=True))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)
#print (y_train)
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
print (y_score)

fpr[0], tpr[0], _ = roc_curve(y_test, y_score)
roc_auc[0] = auc(fpr[0], tpr[0])
print(roc_auc)
# Compute micro-average ROC curve and ROC area
#print(y_score.ravel())
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
plt.figure()
lw = 1
plt.plot(fpr[0], tpr[0], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
