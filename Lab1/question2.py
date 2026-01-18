"""Implement y = 2x1 + 3 and plot x1, y [start=-100, stop=100, num=100]"""
import matplotlib.pyplot as plt


# for i in range(-100,101,2):

start=-100
stop=100
num=100

step=(stop-start)/(num-1)
x1=[]
point=start

for i in range(num):
    x1.append(point)
    point+=step

y=[]
for j in x1:
    y.append(2*j+3)
plt.plot(x1,y)
plt.xlabel("x1")
plt.ylabel("y")
plt.show()