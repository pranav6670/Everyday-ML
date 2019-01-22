__author__ = "Pranav Natekar"

import numpy as np

# Inputs for gates
inp = [(0, 0), (0, 1), (1, 0), (1, 1)]
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
    Function to model the AND gate.
    :x1: 1st input to AND gate
    :x2: 2nd input to AND gate
    :returns: 0, 1 based on input
    """
    # Truth table
    #  _____________
    # | x1 | x2 | y |
    # |-------------|
    # | 0  | 0  | 0 |
    # | 0  | 1  | 0 |
    # | 1  | 0  | 0 |
    # | 1  | 1  | 1 |
    #  -------------

    # Let's initialize the weights to 1 and bias to -1 : x1*1 + x2*1 - 1.
    # We know, y' = W*x + b
    # For 1st row in the input:
    # 0*1 + 0*1 - 1 = -1
    # According to perceptron rule it's correct.
    # For 2nd row in the input:
    # 0*1 + 1*1 -1 = 0
    # This defies the perceptron rule as, if w*x + b >= 0, then y'=1
    # Adjusting bias to -1.5, will make the combination of
    # x1 = 0 and x2 = 1 to give y' = 0.
    # This bias is valid to all 4 inputs.

    x = np.array([1, x1, x2])
    b = -1.5
    w = np.array([b, 1, 1])
    y = np.sum(w * x)
    if y <= 0:
        return 0
    else:
        return 1


def OR(x1, x2):
    """
    Function to model the OR gate.
    :x1: 1st input to OR gate
    :x2: 2nd input to OR gate
    :returns: 0, 1 based on input
    """
    # Truth table
    #  _____________
    # | x1 | x2 | y |
    # |-------------|
    # | 0  | 0  | 0 |
    # | 0  | 1  | 1 |
    # | 1  | 0  | 1 |
    # | 1  | 1  | 1 |
    #  -------------
    # Let's initialize the weights to 1 and bias to -0.5 : x1*1 + x2*1 - 0.5.
    # Initialization with above parameters follows the perceptron rule.
    x = np.array([1, x1, x2])
    b = -0.5
    w = np.array([b, 1, 1])
    y = np.sum(w*x)
    if y <= 0:
        return 0
    else:
        return 1


def NAND(x1, x2):
    """
    Function to model the NAND gate.
    :x1: 1st input to NAND gate
    :x2: 2nd input to OR gate
    :returns: 0, 1 based on input
    """
    # Truth table
    #  _____________
    # | x1 | x2 | y |
    # |-------------|
    # | 0  | 0  | 1 |
    # | 0  | 1  | 1 |
    # | 1  | 0  | 1 |
    # | 1  | 1  | 0 |
    #  -------------
    # Let's initialize the weights to -1 and bias to 1.5 : x1*-1 + x2*-1 + 1.5.
    # Initialization with above parameters follows the perceptron rule.
    # Note that bias and weights for AND and NAND are complementary.
    x = np.array([1, x1, x2])
    b = 1.5
    w = np.array([b, -1, -1])
    y = np.sum(w*x)
    if y <= 0:
        return 0
    else:
        return 1


def NOR(x1, x2):
    """
    Function to model the NOR gate.
    :x1: 1st input to NOR gate
    :x2: 2nd input to NOR gate
    :returns: 0, 1 based on input
    """
    # Truth table
    #  _____________
    # | x1 | x2 | y |
    # |-------------|
    # | 0  | 0  | 1 |
    # | 0  | 1  | 0 |
    # | 1  | 0  | 0 |
    # | 1  | 1  | 0 |
    #  -------------
    # Let's initialize the weights to -1 and bias to 0.5 : x1*-1 + x2*-1 + 0.5.
    # Initialization with above parameters follows the perceptron rule.
    # Note that bias and weights for OR and NOR are complementary.
    x = np.array([1, x1, x2])
    b = 0.5
    w = np.array([b, -1, -1])
    y = np.sum(w * x)
    if y <= 0:
        return 0
    else:
        return 1

def NOT(x1):
    """
    Function to map NOT gate.
    :x1: Input to NOT gate.
    :returns: 0 or 1 based on input.
    """
    # Truth table
    #  _________
    # | x1 | y  |
    # |---------|
    # | 0  |  1 |
    # | 1  |  0 |
    #  ---------
    # Let's initialize the weights to -1 and bias to 0.5 : x1*-1 + 0.5.
    # Initialization with above parameters follows the perceptron rule.
    x = np.array([1, x1])
    b = 0.5
    w = np.array([b, -1])
    y = np.sum(w * x)
    if y <= 0:
        return 0
    else:
        return 1


# Take input from user.
typeof = input("Enter the type of gate: ")
if not typeof.isupper():
    print("You've entered lowercase, I'm converting it to upper!")
    typeof = typeof.upper()

if typeof == 'AND':
    print("AND gate selected")
    for x in inp:
        y = AND(x[0], x[1])
        print(str(x) + " -> " + str(y))

elif typeof == 'OR':
    print("OR gate selected")
    for x in inp:
        y = OR(x[0], x[1])
        print(str(x) + " -> " + str(y))

elif typeof == 'NAND':
    print("NAND gate selected")
    for x in inp:
        y = NAND(x[0], x[1])
        print(str(x) + " -> " + str(y))

elif typeof == 'NOR':
    print("NOR gate selected ")
    for x in inp:
        y = NOR(x[0], x[1])
        print(str(x) + " -> " + str(y))

elif typeof == 'NOT':
    print("NOT gate selected ")
    for x in inp_not:
        y = NOT(x)
        print(str(x) + " -> " + str(y))

else:
    print("Why are you here?-_-")
