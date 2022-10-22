# importing the necessary functions :
import numpy as np
import pandas as pd
import matplotlib.pyplot as mpl
import math

# reading the data of features of apple and pear
path = r'/home/harjeet_singh/Documents/book.csv'
data_l = pd.read_csv(path)
data = np.array(data_l)
m,n = data.shape
m = m
n = n
print(m,"X",n)
print ("size is ", m, "X", n)

# initializing random weights and biases
W_1 = np.random.rand(1)
W_2 = np.random.rand(1)
B_ = np.random.rand(1)

l_r = 0.15                                           # learning rate

# For reducing the underflow error, rounding off to some significant digits
W1 = np.round(W_1, 6)
W2 = np.round(W_2, 6)
B = np.round(B_, 6)

init_vals = "W1 = ", W1, "W2 = ", W2, "B = ", B
print (init_vals)

def activation_func(W1, W2, B, l_r):                 # this function is calculatng error and updating weights and bias for one input data containing 2 features
    error = 0                                 #initialize error for each new set of data
    print("used weights for this set of data", "W1 = ", W1, "W2 = ", W2, "B = ", B)     # this will print updated input weights and biases for each new data
    for j in range(100):
        Z = W1 * data[i,0] + W2 * data[i,1] + B
        Z = np.round(Z, 5)
        print("weighted Sum", Z)
        error = Z - data[i,2]
        error = np.round(error, 5)
        print ("error in itre", i, "is" ,error)
        if error > 0 or Z == 0.99999:
            print("-----------correctrd at iteration-------------", j)
            break
        else:
            W1 = W1 - l_r * error * data[i,0]
            W2 = W2 - l_r * error * data[i,1]
            B = B - l_r * error
    return W1, W2, B, Z         # returning updated values

# The code will run from here :(
t = np.linspace(-1,1,m)
i = 0
pl_z = np.array([])

for i in range(0, m-3):
    print(data[i])
    print("---------------------------------data "
          "", i,"--------------------------------")
    init_val = "initial parameters are : ", "W1 = ", W1, "W2 = ", W2, "B = ", B
    print("adjsuting weights")
    W1_up, W2_up, B_up, Z = activation_func(W1, W2, B, l_r)
    W1 = np.round(W1_up, 4)
    W2 = np.round(W2_up, 4)
    B = np.round(B_up, 4)
print(init_vals)
print("final weights are", "W1 = ", W1, "W2 = ", W2, "B = ", B, )   # printing final weights for whole dataset


print("*****************************************************************")
print("*****************************************************************")
print("********test the calculated weights on different data************")
print("*****************************************************************")
print("*****************************************************************")
for k in range(m-3,m):
    print("given data", data[k])
    print("testing the weights:")
    Z = W1 * data[k, 0] + W2 * data[k, 1] + B
    Z = np.round(Z, 5)
    error = Z - data[k, 2]
    print("weighted Sum", Z, "error:", error)
    if error > 0 :
        print ("corect prediction")
    else :
        print("bad parameters")

mpl.show()


