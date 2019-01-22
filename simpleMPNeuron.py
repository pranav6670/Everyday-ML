__author__ = "Pranav Natekar"

import numpy as np

# Inputs for gates
inp = [(0, 0), (1, 0), (0, 1), (1, 1)]
# Input for not gate
inp_not = [0, 1]

# Perceptron rule states that:
#  y' = 1  if (W*x + b) >= 0
#     = 0  if (W*x + b) <  0
#
# where,
# W = weights, x = input, b = bias, y' = predicted output


def AND(x1, x2):
    """
    Function to model the AND gate
    :param x1: 1st input to AND gate
    :param x2: 2nd input to AND gate
    :return: 0, 1 based on input
    """
    # x1 x2 y
    # 0  0  0
    # 0  1  0
    # 1  0  0
    # 1  1  1
    x = np.array([x1, x2])
    w = np.array([1, 1])
    y = np.sum(w*x)
    if y <= 0:
        return 0
    else:
        return 1


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([1, 1])
    y = np.sum(w*x)
    if y <= 0:
        return 0
    else:
        return 1


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-1, -1])
    y = np.sum(w*x)
    if y <= 0:
        return 0
    else:
        return 1


typeof = input("Enter the type of gate")
# print(type)


if typeof == 'AND':
    print("AND gate selected")
elif typeof == 'OR':
    print("OR gate selected")
elif typeof == 'NAND':
    print("NAND gate selected")
elif typeof == 'NOR':
    print("NOR gate selected ")
elif typeof == 'NOT':
    print("NOT gate selected ")
else:
    print("Don't come here")
