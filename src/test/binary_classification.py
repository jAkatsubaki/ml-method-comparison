import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import make_classification

# X, Y = make_classification(random_state=12,
#                            n_features=2, 
#                            n_redundant=0, 
#                            n_informative=1,
#                            n_clusters_per_class=1,
#                            n_classes=2)

# fig = plt.figure()
# plt.figure(figsize=(8, 7))
# plt.title("make_classification : n_features=2  n_classes=2")
# plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y, s=25, edgecolor='k')
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.show()

xx = np.linspace(0, 2*np.pi, num=400)
print(xx.reshape(20, 20))