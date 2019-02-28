import numpy as np

# Input
X = np.array([[1, 2], [3, 4], [5, 6]])
print("Input Matrix: \n", X)

# Mean of data
mean_X = np.mean(X.T, axis=1)
print("Mean: \n", mean_X)

# Get zero mean data
zero_mean = X - mean_X
print("Zero mean data : \n", zero_mean)

# Get covariance
cov_X = np.cov(zero_mean.T)
print("Covariance matrix: \n", cov_X)

# Eigen values and vectors
eig_values, eig_vectors = np.linalg.eig(cov_X)

print("Eigen vectors: \n", eig_vectors)
print("\nEigen values \n", eig_values)

# SVD
u, s, v = np.linalg.svd(X.T)
print("\nU:", u)
print("\nS:", s)
print("\nV:", v)

# Check if lengths of EVs are same
for ev in eig_vectors:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
print('Everything ok!')

# Make a list of (eigen value, eigen vector) tuples
eig_pairs = [(np.abs(eig_values[i]), eig_vectors[:, i]) for i in range(len(eig_values))]

# Sort the (eigen value, eigen vector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

# Visually confirm that the list is correctly sorted by decreasing eigen values
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

