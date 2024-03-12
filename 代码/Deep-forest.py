from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from scipy.io import loadmat
from deepforest import CascadeForestClassifier
from deepforest import CascadeForestRegressor

t1 = time.process_time()
iris = loadmat('data/Adult.mat')
# diabetes = datasets.load_diabetes()
data = iris['Adult']
X, y = data[:, 0:13], data[:, 14]
for i in range(len(y)):
    if y[i] != 1:
        y[i] = 0
X_train, X_test, y_train, y_test = train_test_split(
    X, y,  random_state=13, test_size=0.1
)

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
model = CascadeForestRegressor(random_state=1, n_jobs=-1, n_estimators=2, use_predictor=True, predictor="lightgbm", n_trees=200,predictor_kwargs=params)
model.fit(X_train, y_train)
t2 = time.process_time()
y_pred = model.predict(X_test)
ac = model.score(X_train,y_train)

acc = accuracy_score(y_test, y_pred) * 100

print(t2-t1)
print(ac)
print("\nTesting Accuracy: {:.3f} %".format(acc))

