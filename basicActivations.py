import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-10, 10, 0.2)


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

def softplus(x):
    return np.log(1 + np.exp(x))

t = tanh(x)
b = binarystep(x)
s = sigmoid(x)
r = relu(x)
sp = softplus(x)

plt.figure(1)
plt.plot(x, s, label='Sigmoid', color='blue', linestyle='--')
plt.plot(x, t, label='tanh', color='red', linestyle=None)
plt.legend()

plt.figure(2)
plt.plot(x, sp, label='Softplus', color='violet', linestyle='-')
plt.plot(x, r, label='ReLU', color='green', linestyle='-.')
plt.plot(x, b, label='Binary Step', color='red', linestyle=None)
plt.legend()

plt.show()