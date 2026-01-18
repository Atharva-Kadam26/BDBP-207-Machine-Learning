"""Compute the dot product of two vectors, x and y given below
x = [2  1  2]T and y = [1  2  2]T . What is the meaning of the dot product of two vectors? Illustrate that with your own example.
"""

def dot_product(x,y):
    d_prod=0
    for i in range(len(x)):
        d_prod+=x[i]*y[i]
    return d_prod

x=[2,1,2]
y=[1,2,2]
print(dot_product(x,y))

#Test Output
a=[6,7]
b=[1,2]
print(dot_product(a,b))



