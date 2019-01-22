import numpy as np
import matplotlib.pyplot as plt

A = np.array([0, 0, 1, 1])
B = np.array([0, 1, 0, 1])

w1, w2 = 1, 1

threshold = 0
type = input("Enter the type of gate")
print(type)

if type == 'AND':
    inverse = 0
    threshold = 2
elif type =='OR':
    inverse = 0
    threshold = 1
elif type == 'NAND':
    inverse = 1
    threshold = 2
elif type == 'NOT':
    inverse = 1
    threshold = 0
elif type == 'NOR':
    inverse = 1
    threshold = 1
else:
    print("Don't come here")

# X = np.transpose((np.transpose(A), np.transpose(B)))
# print(X)
sop = A * w1+ B * w2
print(sop)