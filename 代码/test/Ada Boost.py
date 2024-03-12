import joblib
import lightgbm as lgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import time
import random
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score,accuracy_score
import matplotlib.pyplot as plt


iris = loadmat('Rdk Concat.mat')
data = iris['data']
a = np.array([])
for i in range(1054):
    a = np.append(a,data[random.randint(0,6764)])
    a = np.append(a,data[i+6764])
# data = a.reshape(2108, 168)
# X, y = data[:, 0:167], data[:, 167]
data = a.reshape(2108, 4097)
X, y = data[:, 0:4096], data[:, 4096]
# data = a.reshape(2108, 335)
# X, y = data[:, 0:334], data[:, 334]
# data = a.reshape(2108,2049)
# X, y = data[:, 0:2048], data[:, 2048]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

gbm = AdaBoostClassifier(learning_rate=0.01, n_estimators=50)
gbm.fit(X_train,y_train)
t2 = time.process_time()
# joblib.dump(gbm, 'loan_model.pkl')
# gbm = joblib.load('loan_model.pkl')
#
y_pred = gbm.predict(X_test)
print(gbm.score(X_train,y_train))
print(gbm.score(X_test,y_test))
print(roc_auc_score(y_test,y_pred))

