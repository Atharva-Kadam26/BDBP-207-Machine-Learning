"""Implement sigmoid function in python and visualize it"""
import math
import matplotlib.pyplot as plt
def sigmoid_func(z):
    sigma = 1 / (1 + math.exp(-z))
    return sigma

# plus_inf=math.inf
# minus_inf=-math.inf
X=[]
y=[]
for i in range(-100,100):
    pt =i*0.1
    X.append(pt)
    y.append(sigmoid_func(pt))
    i += 1

plt.figure()
plt.plot(X,y)
plt.xlabel("X")
plt.ylabel("Sigmoid Function")
plt.show()







