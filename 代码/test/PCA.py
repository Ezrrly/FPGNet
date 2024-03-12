import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import csv

df = pd.read_csv('C:/Users/Ezrrly/Documents/WeChat Files/wxid_gtpq398x3wbm21/FileStorage/File/2022-03/MNIST_train.csv')
pca = PCA(n_components='mle')
pca.fit(df)
print(pca.explained_variance_ratio_)
