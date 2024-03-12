from gcforest.gcforest import GCForest
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from scipy.io import loadmat
import argparse
import joblib
import numpy as np
import sys
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from gcforest.utils.config_utils import load_json
from deepforest import CascadeForestClassifier
import numpy as np

iris = loadmat('data/Adult.mat')
# diabetes = datasets.load_diabetes()
data = iris['Adult']
X, y = data[:, 0:13], data[:, 14]
for i in range(len(y)):
    if y[i] != 1:
        y[i] = 0
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=13, test_size=0.1
)

X_train = np.expand_dims(X_train, axis=-1)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

con = {
    "net": {
        "outputs": ["pool1/7x7/ets", "pool1/7x7/rf", "pool1/10x10/ets", "pool1/10x10/rf", "pool1/13x13/ets",
                    "pool1/13x13/rf"],
        "layers": [

            {
                "type": "FGWinLayer",
                "name": "win1/7x7",
                "bottoms": ["X", "y"],
                "tops": ["win1/7x7/ets", "win1/7x7/rf"],
                "n_classes": 10,
                "estimators": [
                    {"n_folds": 3, "type": "ExtraTreesClassifier", "n_estimators": 20, "max_depth": 10, "n_jobs": -1,
                     "min_samples_leaf": 10},
                    {"n_folds": 3, "type": "RandomForestClassifier", "n_estimators": 20, "max_depth": 10, "n_jobs": -1,
                     "min_samples_leaf": 10}
                ],
                "stride_x": 2,
                "stride_y": 2,
                "win_x": 7,
                "win_y": 7
            },
            {
                "type": "FGWinLayer",
                "name": "win1/10x10",
                "bottoms": ["X", "y"],
                "tops": ["win1/10x10/ets", "win1/10x10/rf"],
                "n_classes": 10,
                "estimators": [
                    {"n_folds": 3, "type": "ExtraTreesClassifier", "n_estimators": 20, "max_depth": 10, "n_jobs": -1,
                     "min_samples_leaf": 10},
                    {"n_folds": 3, "type": "RandomForestClassifier", "n_estimators": 20, "max_depth": 10, "n_jobs": -1,
                     "min_samples_leaf": 10}
                ],
                "stride_x": 2,
                "stride_y": 2,
                "win_x": 10,
                "win_y": 10
            },

            {
                "type": "FGWinLayer",
                "name": "win1/13x13",
                "bottoms": ["X", "y"],
                "tops": ["win1/13x13/ets", "win1/13x13/rf"],
                "n_classes": 10,
                "estimators": [
                    {"n_folds": 3, "type": "ExtraTreesClassifier", "n_estimators": 20, "max_depth": 10, "n_jobs": -1,
                     "min_samples_leaf": 10},
                    {"n_folds": 3, "type": "RandomForestClassifier", "n_estimators": 20, "max_depth": 10, "n_jobs": -1,
                     "min_samples_leaf": 10}
                ],
                "stride_x": 2,
                "stride_y": 2,
                "win_x": 13,
                "win_y": 13
            },

            {
                "type": "FGPoolLayer",
                "name": "pool1",
                "bottoms": ["win1/7x7/ets", "win1/7x7/rf", "win1/10x10/ets", "win1/10x10/rf", "win1/13x13/ets",
                            "win1/13x13/rf"],
                "tops": ["pool1/7x7/ets", "pool1/7x7/rf", "pool1/10x10/ets", "pool1/10x10/rf", "pool1/13x13/ets",
                         "pool1/13x13/rf"],
                "pool_method": "avg",
                "win_x": 2,
                "win_y": 2
            }
        ]

    },

    "cascade": {
        "random_state": 0,
        "max_layers": 100,
        "early_stopping_rounds": 3,
        "look_indexs_cycle": [
            [0, 1],
            [2, 3],
            [4, 5]
        ],
        "n_classes": 10,
        "estimators": [
            {"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 1000, "max_depth": 30, "n_jobs": -1},
            {"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 1000, "max_depth": 30, "n_jobs": -1}
        ]
    }
}

gc = GCForest(con)
X_train_enc = gc.fit_transform(X_train, y_train)
y_pred = gc.predict(X_test)
ac = accuracy_score(X_train,y_train) * 100
acc = accuracy_score(y_test, y_pred) * 100
print("\nTesting Accuracy: {:.3f} %".format(ac))
print("\nTesting Accuracy: {:.3f} %".format(acc))
