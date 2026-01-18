"""Implement y = x1^2, plot x1, y in the range [start=--10, stop=10, num=100]. Compute the value of derivatives at these points, x1 = -5, -3, 0, 3, 5.  What is the value of x1 at which the function value (y) is zero. What do you infer from this?"""
from matplotlib import pyplot as plt

start=-10
stop=10
num=100

step=(stop-start)/(num-1)

x1=[]
y=[]

for i in range(num):
    x1.append(start)
    y.append(start*start)
    start+=step

points_x1=[-5,-3,0,3,5]

for j in points_x1:
    print("Derivative at",j,"=",2*j)

plt.plot(x1,y)
plt.xlabel("x1")
plt.ylabel("y")
plt.show()
print("As it is visible in the graph at x1=0 --> y=0")
