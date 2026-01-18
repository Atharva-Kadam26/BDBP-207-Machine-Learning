"""Implement y = 2x1 + 3x2 + 3x3 + 4, where x1, x2 and x3 are three independent variables. Compute the gradient of y at a few points and print the values."""

#we will perform partial differentiation
#it gives a vector gradient of [2,3,3]
def calc_y(x1,x2,x3):
    return 2*x1 + 3*x2 + 3*x3 + 4

def gradient():
    dy_dx1=2
    dy_dx2=3
    dy_dx3=3
    return dy_dx1,dy_dx2,dy_dx3

ex_points=[(4,3,2), #y=2(4)+3(3)+3(2)+4
    (-3,-1,2),      #y=2(−3)+3(−1)+3(2)+4
    (0,0,0)]        #y=2(0)+3(0)+3(0)+4

for p in ex_points:
    x1,x2,x3=p
    y=2*x1 + 3*x2 + 3*x3 + 4

    print("point:",ex_points)
    print("y=",y)
    print("gradient =(2,3,3)")
    print()

