import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
import numpy as np
from sklearn import tree
from collections import Counter
#import data
df = pd.read_csv("C:\\Users\\daizh\\Desktop\\Wilson\\Hourly and weather categorized2.csv")
# X and y are different columns of the input data. Input X as numpy array
X = df[['TTI','Max TemperatureF','Mean TemperatureF','Min TemperatureF',' Min Humidity']].values
# # Reshape X. Do this if X has only one value per data point. In this case, TTI.

# # Input y as normal list
y = df['TTI_Category'].as_matrix()
ros = ADASYN()
X_resampled, y_resampled = ros.fit_sample(X, y)
# print(sorted(Counter(y_resampled).items()))
X=X_resampled
y=y_resampled
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
clf = MultinomialNB()
clf.fit(X_train, y_train)
y_test_pred = clf.predict(X_test)
print (metrics.confusion_matrix(y_test, y_test_pred))
print (metrics.classification_report(y_test, y_test_pred))