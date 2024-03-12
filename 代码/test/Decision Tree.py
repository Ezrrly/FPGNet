import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import time
from scipy.io import loadmat
from sklearn.datasets._california_housing import fetch_california_housing

t1 = time.process_time()
iris = loadmat('data/Adult.mat')
# diabetes = datasets.load_diabetes()
data = iris['Adult']
print(data.shape)
X, y = data[:, 0:13], data[:, 14]
for i in range(len(y)):
    if y[i] != 1:
        y[i] = 0


# dtr = tree.DecisionTreeRegressor(max_depth=2)
# dtr.fit(housing.data[:,[6,7]],housing.target)
# dot_data = \
#     tree.export_graphviz(
#         dtr,
#         out_file=None,
#         feature_names=housing.feature_names[6:8],
#         filled=True,
#         impurity=False,
#         rounded=True
#     )


data_train, data_test, target_train, target_test = \
    train_test_split(X, y, test_size=0.1, random_state=13)
dtr = tree.DecisionTreeRegressor(random_state=13)
dtr.fit(data_train, target_train)
t2 = time.process_time()
print(t2-t1)
predict1 = dtr.predict(data_test)
a = 0
for i in range(predict1.shape[0]):
    if predict1[i]==target_test[i]:
        a +=1
print(a/predict1.shape[0])
print(dtr.score(data_train,target_train))
print(dtr.score(data_test, target_test))


