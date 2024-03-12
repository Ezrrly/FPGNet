import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,roc_curve, auc
import time
import csv
import random
from scipy.io import loadmat
from sklearn.datasets._california_housing import fetch_california_housing

iris = loadmat('data1.mat')
data = iris['data']
# iris1 = loadmat('FEB add.mat')
# data1 = iris1['data']


a = np.array([])
for i in range(1054):
    a = np.append(a,data[random.randint(0,6764)])
    a = np.append(a,data[i+6764])

# data = a.reshape(2108,335)
# # Macc(X, y = data[:, 0:166], data[:, 167]  data = a.reshape(2108,168) ,Macc Concat X, y = data[:, 0:333], data[:, 334] data = a.reshape(2108,335)
# X, y = data[:, 0:334], data[:, 334]
# data = a.reshape(2108,2049)
# X, y = data[:, 0:2048], data[:, 2048]
data = a.reshape(2108,168)
X, y = data[:, 0:167], data[:, 167]
# data = a.reshape(2108, 4097)
# X, y = data[:, 0:4096], data[:, 4096]

# data = a.reshape(2108,168)
# X, y = data[:, 0:166], data[:, 167]
X_train, X_test, y_train, y_test = train_test_split(
    X, y,  random_state=13, test_size=0.2
)





dtr = tree.DecisionTreeRegressor(random_state=6)
dtr.fit(X_train, y_train)
# y_pred = dtr.predict(data1)
# print(y_pred)
y_pred = dtr.predict(X_test)
y_train_fix = dtr.predict(X_train)
y_test_fix =dtr.predict(X_test)
print(y_test_fix)
a = 0
b = 0
for i in range(y_test_fix.shape[0]):
    if y_test_fix[i]==y_test[i]:
        a +=1
for i in range(y_train_fix.shape[0]):
    if y_train_fix[i]==y_train[i]:
        b +=1
print(a/y_test.shape[0])
print(b/y_train.shape[0])

a=[]
test =loadmat('Test Macc Fpr.mat')
test =test['data']
test_pred =dtr.predict(test)
print(test_pred)
for i in range(len(test_pred)):
    a.append(test_pred[i])
print(a)
f = open('D:\\work\\pyGAT-master\\data\\note\\RF test.csv', 'a', encoding='utf-8', newline='')
csv_write = csv.writer(f)
csv_write.writerow(
        [int(a[j]) for j in range(len(a))])
f.close()

print(dtr.score(X_train, y_train))
print(dtr.score(X_test, y_test))
print(roc_auc_score(y_test,y_pred))

fpr, tpr, thresholds = roc_curve(y_test,y_pred)
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, '#9400D3',label=u'AUC = %0.3f'% roc_auc)

plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.grid(linestyle='-.')
plt.grid(True)
plt.show()
print(roc_auc)
