import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from scipy.io import loadmat
import time
from sklearn.model_selection import train_test_split

t1 = time.process_time()
iris = loadmat('data/Adult.mat')
# diabetes = datasets.load_diabetes()
data = iris['Adult']
X, y = data[:, 0:13], data[:, 14]
for i in range(len(y)):
    if y[i] != 1:
        y[i] = 0
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=13
)

ET = ExtraTreesClassifier(n_estimators=500, n_jobs=5, oob_score=True, bootstrap=True)
ET.fit(X_train,y_train)
t2 = time.process_time()
print(t2-t1)
print(ET.score(X_train,y_train))
print(ET.score(X_test,y_test))