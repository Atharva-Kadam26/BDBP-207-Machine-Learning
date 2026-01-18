"""Here is a linear model.
	y = 2x1 + 3x2 + 3x3 + 4
	The coefficients, represented as theta, is a vector given below
	theta=[2 3 3]
	There are 5 samples represented as a matrix, X given below
	X=[[1,0,2],
	[0,1,1],
	[2,1,0],
	[1,1,1],
	[0,2,1]]
	Compute X*theta
"""
theta=[2,3,3]
X=[
    [1,0,2],
	[0,1,1],
	[2,1,0],
    [1,1,1],
    [0,2,1]
]

product=[]
for i in X:
    y=0
    for j in range(len(theta)):
        y+=i[j]*theta[j]
    product.append(y)
print(product)
