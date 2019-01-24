import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Load the dataset
data = pd.read_csv('brain.csv')

print(data.shape)
print(data.head)

# Load our i/p and o/p

X = data['Head Size(cm^3)'].values
Y = data['Brain Weight(grams)'].values

X = X / np.max(X)
Y = Y / np. max(Y)

plt.scatter(X, Y, color='red', label='Data Point')
# plt.show()

W, b = -1, -1


def sigmoid(x):
    return 1 / 1 + np.exp(-x)


for i in range(1000):
    for j in range(1, len(Y)):

        x = X[j]
        y = Y[j]
        weighted_sum = W * x + b
        squish = sigmoid(weighted_sum)
        fx = squish
        output = squish
        diff_weights = (fx - y) * fx * (1 - fx) * x
        diff_bias = (fx - y) * fx * (1 - fx)

    W = W - 0.01 * np.mean(diff_weights)
    b = b - 0.01 * np.mean(diff_bias)
    error = 0.5 * np.square(np.mean(squish) - y)
    w = W
    err = error


plt.plot(W, error)
plt.show()

