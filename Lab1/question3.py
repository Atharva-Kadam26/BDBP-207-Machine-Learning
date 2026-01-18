"""Implement y = 2x12 + 3x1 + 4 and plot x1, y in the range [start=-10, stop=10, num=100]"""
import matplotlib.pyplot as plt

start=-10
stop=10
num=100

x1=[]
y=[]

step=(stop-start)/(num-1)
for i in range(num):
    x1.append(start)
    y.append(2*start*start+3*start+4)
    start+=step

plt.plot(x1,y)
plt.title("2x1^2 + 3x1 + 4")
plt.show()
