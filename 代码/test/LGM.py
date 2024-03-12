import lightgbm as lgb
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import time
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score,accuracy_score
import matplotlib.pyplot as plt

t1 = time.process_time()
iris = loadmat('data/Adult.mat')
# diabetes = datasets.load_diabetes()
data = iris['Adult']
X, y = data[:, 0:13], data[:, 14]
for i in range(len(y)):
    if y[i] !=1:
        y[i] = 0
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3
)

train_data = lgb.Dataset(X_train, label=y_train)
validation_data = lgb.Dataset(X_test, label=y_test)

params = {
    # "min_data_in_leaf": 10,
    "n_estimators": 500,
    "learning_rate": 0.1,
    "max_depth": 4,
    "nthread": -1,
    "lambda_l1": 0.1,
    "lambda_l2": 0.02,
    "boosting": "gbdt",
    "objective": "binary",

}

gbm = lgb.train(params, train_data, valid_sets=[validation_data])
t2 = time.process_time()
print(f"用的时间是{t2-t1}")
y_pred = gbm.predict(X_test)
Y_train =gbm.predict(X_train)
threshold = 0.5
for i in range(len(y_pred)):
    if y_pred[i] >threshold:
        y_pred[i] = 1
    else:
        y_pred[i] = 0

for i in range(len(Y_train)):
    if Y_train[i] > threshold:
            Y_train[i] = 1
    else:
            Y_train[i] = 0
print(accuracy_score(Y_train, y_train))
print(accuracy_score(y_test, y_pred))

