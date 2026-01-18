"""Implement Gaussian PDF - mean = 0, sigma = 15 in the range[start=-100, stop=100, num=100]"""
import matplotlib.pyplot as plt
import math

start=-100
stop=100
num=100

mean=0
sigma=15

step=(stop-start)/(num-1)

x1=[]
y=[]

for i in range(num):
    x1.append(start)
    g_pdf=(1/(sigma*math.sqrt (2* math.pi)))* math.exp(-((start - mean)**2)/ (2*(sigma ** 2)))
    y.append(g_pdf)
    start+=step
plt.plot(x1,y)
plt.xlabel("x1")
plt.ylabel("y")
plt.title("Gaussian PDF")
plt.show()
