import numpy as np

X = np.array([[1, 2], [3, 4], [5, 6]])
print("Input Matrix: \n", X)

mean_X = np.mean(X, axis=0)
print("Mean: \n", mean_X)

cov_mat = (X - mean_X).T.dot((X - mean_X)) / (X.shape[0]-1)
print("Covariance matrix: \n", cov_mat)

print("NumPy covariance matrix: \n", np.cov(X.T))

cov_mat = np.cov(X.T)

eig_values, eig_vectors = np.linalg.eig(cov_mat)

print("Eigen vectors: \n", eig_vectors)
print("\nEigen values \n", eig_values)

u, s, v = np.linalg.svd(X.T)
print("\nU:", u)
print("\nS:", s)
print("\nV:", v)


