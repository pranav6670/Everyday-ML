import numpy as np

X = np.array([[1, 2], [3, 4], [5, 6]])
print("Input Matrix : \n", X)

mean_X = np.mean(X, axis=0)
print("Mean : \n", mean_X)
