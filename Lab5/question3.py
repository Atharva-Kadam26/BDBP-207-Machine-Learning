"""Compute the derivative of a sigmoid function and visualize it"""
import math
import matplotlib.pyplot as plt
#derivative_of_sig_func=g(z)(1-g(z))
def der_sig_func(z):
    g_z = 1 / (1 + math.exp(-z))
    der_g= g_z * (1 - g_z)
    return der_g

X=[]
y=[]
for i in range(-100,100):
    pt = i * 0.1
    X.append(pt)
    y.append(der_sig_func(pt))
# print(X)
# print(y)
plt.figure()
plt.xlabel('X')
plt.ylabel('der_sig_func(X)')
plt.plot(X,y)
plt.show()
