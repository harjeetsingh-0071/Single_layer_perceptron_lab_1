import math

import numpy as np
from math import e
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt

def dev(dat, mean):
    k = len(dat)
    print("k",k)
    x = np.empty(shape = (0,1))
    for i in range(k):
        temp = dat[i] - mean
        x = np.append(x, temp)
    return x


def var(dat):
    x = (1/len(dat)) * (sum(pow(dat,2)))
    return x


def pdf_func(mean, variance, val):
    k = len(val)
    x = np.empty(shape=(0, 1))
    for i in range(k):
        temp =(1/pow((2*math.pi*variance),0.5)) * pow(e,-(pow((val-mean),2)/2*variance))
    x = np.append(x, temp)
    return x

path = r'C:\Users\Harjeet\Downloads\Book3_ori.csv'
data_l = pd.read_csv(path)
data = np.array(data_l)
m, n = data.shape
print(m, "X",n)
X1 = np.empty(shape = (0,2))
X2 = np.empty(shape = (0,2))
j = 0
k = 0
i=0
for i in range(m):
    if data[i,2] == 1. :
        X1 = np.append(X1, [[data[i,0],data[i,1]]], axis = 0)
    else:
        X2 = np.append(X2, [[data[i, 0], data[i, 1]]], axis=0)
g,h = X1.shape
#print(X2)
r,t = X2.shape
mean_apple_f = X1.sum(axis = 0)/g
mean_pear_f = X2.sum(axis = 0)/r
apple_f1 = X1[:,0]
apple_f2 = X1[:,1]
pear_f1 = X2[:,0]
pear_f2 = X2[:,1]

#deviations of each feature values of each class
dev_apple_f1 = dev(apple_f1, mean_apple_f[0])
dev_apple_f2 = dev(apple_f2, mean_apple_f[1])
dev_pear_f1 = dev(pear_f1, mean_pear_f[0])
dev_pear_f2 = dev(pear_f2, mean_pear_f[1])

#variance of each class
var_apple_f1 = var(dev_apple_f1)
var_apple_f2 = var(dev_apple_f2)
var_pear_f1 = var(dev_pear_f1)
var_pear_f2 = var(dev_pear_f1)

#probablity distribution function values
pdf_apple_f1 = pdf_func(mean_apple_f[0], var_apple_f1, apple_f1)
pdf_apple_f2 = pdf_func(mean_apple_f[1], var_apple_f2, apple_f2)
pdf_pear_f1 = pdf_func(mean_pear_f[0], var_pear_f1, apple_f1)
pdf_pear_f2 = pdf_func(mean_pear_f[1], var_pear_f2, apple_f2)

qq = np.arange(0,0.9,0.001)
plt.plot(qq, norm.pdf(qq, mean_apple_f[0], var_apple_f1), label='apple, feature1')
plt.plot(qq, norm.pdf(qq, mean_pear_f[0], var_pear_f1), label='pear, feature1')
plt.plot(qq, norm.pdf(qq, mean_apple_f[1], var_apple_f1), label='apple, feature2')
plt.plot(qq, norm.pdf(qq, mean_pear_f[1], var_pear_f1), label='pear, feature2')
plt.legend()
plt.show()
















