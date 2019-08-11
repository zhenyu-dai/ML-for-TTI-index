# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 15:09:07 2018

@author: zhenyu
"""
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import sys
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler
# Import some data to play with
df = pd.read_csv("C:\\Users\\daizh\\Desktop\\Wilson\\2\\Hourly and weather ml.csv")
# X and y are different columns of the input data. Input X as numpy array
X = df[['TTI','Max TemperatureF','Mean TemperatureF','Min TemperatureF',' Min Humidity']].values
# # Reshape X. Do this if X has only one value per data point. In this case, TTI.

# # Input y as normal list
y = df['TTI_Category'].as_matrix()



X_resampled, y_resampled = SMOTE().fit_sample(X, y)


###
y_resampled = label_binarize(y_resampled, classes=['Good','Bad','Ok'])

y = label_binarize(y, classes=['Good','Bad','Ok'])
n_classes = y.shape[1]

#print type(y_resampled)

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.9999,
                                                    random_state=0)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=50))
y_score=classifier.fit(X_resampled, y_resampled).predict_proba(X_test)
#print X_resampled
#print y_resampled
#print X_test
#b=X_test.shpe[0]
#c=X_test.shape[1]
#print b
#print c
#sys.exit()
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
#print y_test
#print'+++++'
#print y_score
#sys.exit()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())

roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, 
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()