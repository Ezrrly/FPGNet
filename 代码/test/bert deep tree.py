from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import numpy as np
import random
from sklearn.metrics import roc_auc_score
import pandas as pd
from scipy.io import loadmat
from deepforest import CascadeForestClassifier
from deepforest import CascadeForestRegressor
import pandas as pd

data = pd.read_csv("encoddata.csv")
x_train = data['text'].replace('\n', '')
print(x_train)
y_train = data['label']
params = {
    # "min_data_in_leaf": 10,
    "n_estimators": 200,
    "learning_rate": 0.01,
    "max_depth": 3,
    "nthread": -1,
    "lambda_l1": 0.1,
    "lambda_l2": 0.02,
    "boosting": "gbdt",
    "objective": "binary",

}
model = CascadeForestClassifier(random_state=12, n_jobs=-1, n_estimators=2, use_predictor=True, predictor="lightgbm", n_trees=200, predictor_kwargs=params)
# model = CascadeForestClassifier(random_state=1, n_jobs=-1, n_estimators=2, n_trees=100, predictor_kwargs=params)
model.fit(x_train, y_train)
# y_pred1 = model.predict(data1)
x_train = float(x_train)
# x_train= np.array(x_train)
# y_pred2 = model.predict(data2)
# print(y_pred1)
# y_pred = model.predict(x_test)
ac = model.score(x_train,y_train)
print(ac)
# acc = accuracy_score(y_test, y_pred) * 100
# accc = model.score(X_test,y_test)
# y_test_fix =model.predict(X_test)
# a = 0
# b = 0
# for i in range(y_test_fix.shape[0]):
#     if y_test_fix[i]==y_test[i]:
#         a +=1
# print(a / y_test.shape[0])
#
# print(accc)
# print("\nTesting Accuracy: {:.3f} %".format(acc))
# print(roc_auc_score(y_test,y_pred))