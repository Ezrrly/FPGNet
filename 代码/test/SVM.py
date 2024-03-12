import joblib
import lightgbm as lgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import numpy as np
import time
import csv
import random
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score,accuracy_score
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier

iris = loadmat('data1.mat')
data = iris['data']
a = np.array([])
for i in range(1054):
    a = np.append(a,data[random.randint(0,6764)])
    a = np.append(a,data[i+6764])
data = a.reshape(2108, 168)
X, y = data[:, 0:167], data[:, 167]
# data = a.reshape(2108, 4097)
# X, y = data[:, 0:4096], data[:, 4096]
# data = a.reshape(2108, 335)
# X, y = data[:, 0:334], data[:, 334]
# data = a.reshape(2108,2049)
# X, y = data[:, 0:2048], data[:, 2048]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4
)

gbm = MultinomialNB()
gbm.fit(X_train,y_train)
# joblib.dump(gbm, 'loan_model.pkl')
# gbm = joblib.load('loan_model.pkl')
#
a=[]
y_pred = gbm.predict(X_test)
test =loadmat('Test Macc Fpr.mat')
test =test['data']
test_pred =gbm.predict(test)
print(test_pred)
for i in range(len(test_pred)):
    a.append(test_pred[i])
print(a)
f = open('D:\\work\\pyGAT-master\\data\\note\\svm test.csv', 'a', encoding='utf-8', newline='')
csv_write = csv.writer(f)
csv_write.writerow(
        [int(a[j]) for j in range(len(a))])
f.close()
print(gbm.score(X_train,y_train))
print(gbm.score(X_test,y_test))
print(roc_auc_score(y_test,y_pred))