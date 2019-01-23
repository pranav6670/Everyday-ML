__author__ = "Pranav Natekar"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('brain.csv')

# print(data.shape)
# print(data.head)

# Load our i/p and o/p

X = data['Head Size(cm^3)'].values
Y = data['Brain Weight(grams)'].values

# We know y = mx + c, where
# y = o/p, x = i/p, m = slope, c = intercept

# We are using the ordinary Least Mean Square Method
# for achieving a best fit line on the data.
# The goal is to draw the line of best fit
# between X and Y which estimates the relationship between X and Y.

# Error is given as:
#
# error  =  ∑ (r)^2
#

# n = total length
# ri = distance between line and ith point

# Squaring each of the distance’s because
# some points would be above the line and some below.

# Now we have:
# 		∑  (X-x_mean)*(Y-y_mean)
#
# m =  ――――――――――――――――――――――――――――――
# 	          ∑  (X-x_mean)^2
#


# c = y_mean - (m * x_mean)

# Get means of i/p and o/p columns and length of i/p data
x_mean = np.mean(X)
y_mean = np.mean(Y)
n = len(X)

# using the formula to calculate the c and m
numerator = 0
denominator = 0
for i in range(n):
    numerator += (X[i] - x_mean) * (Y[i] - y_mean)
    denominator += (X[i] - x_mean) ** 2

m = numerator / denominator
c = y_mean - (m * x_mean)

# printing the coefficients
print("Slope: ", m, "\nIntercept: ", c)

# Now we have
# Brain weights =  325.57342104944223 + 0.26342933948939945 * Head size

# Plotting values
x_max = np.max(X) + 100
x_min = np.min(X) - 100

# Calculating line values of x and y
x = np.linspace(x_min, x_max, 1000)
y = c + m * x

# Plotting line
plt.plot(x, y, color='green', label='Linear Regression')

# Plot the data point
plt.scatter(X, Y, color='red', label='Data Point')

# x-axis label
plt.xlabel('Head Size (cm^3)')

# y-axis label
plt.ylabel('Brain Weight (grams)')


# Now we have our prediction, but we must know how accurate it is.
# There are many methods to achieve this but we would implement
# Root mean squared error and coefficient of Determination (R² Score).

# RMSE:
#  ――――――――――――――――――――――――――
# √1/n*∑(Yj - y_predicted)^2
#      j

rmse = 0
for i in range(n):
    y_pred = c + m * X[i]
    rmse += (Y[i] - y_pred) ** 2

rmse = np.sqrt(rmse / n)

print("RMSE: ", rmse)

# R² Score

#       SSR	  ∑(y_predicted - y_mean)
# R²  = ――― = ――――――――――――――――――――――――――
#       SST       ∑(y_i/p - y_mean)

# SSR = Sum of residuals
# SST = Sum of squares

sumofsquares = 0
sumofresiduals = 0

for i in range(n):
    y_pred = c + m * X[i]
    sumofsquares += (Y[i] - y_mean) ** 2
    sumofresiduals += (Y[i] - y_pred) ** 2

score = 1 - (sumofresiduals / sumofsquares)

print("R² Score: ", score)

plt.legend()
plt.savefig('plot.png')
plt.show()
