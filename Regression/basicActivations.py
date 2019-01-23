import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.01)


def binarystep(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def tanh(x):
    numerator = np.exp(x) - np.exp(-x)
    denominator = np.exp(x) + np.exp(-x)
    return numerator / denominator



t = tanh(x)
b = binarystep(x)
s = sigmoid(x)
r = relu(x)

plt.figure(1)
plt.plot(x, r, label='ReLU', color='green', linestyle='-.')
plt.plot(x, t, label='Tanh', color='pink', lw=1, linestyle=None)
plt.legend()
plt.figure(2)
plt.plot(x, b, label='Binary Step', color='red', linestyle=None)
plt.plot(x, s, label='Sigmoid', color='blue', linestyle='--')
plt.legend()
plt.show()