"""Implement ATA  -  A = [[1 2 3], [4 5 6]]"""
from fontTools.svgLib.path.parser import ARC_ARGUMENT_TYPES

A = [[1,2,3],
     [4,5,6]]
A_T=[]
for i in range(len(A[0])):
    row=[]
    for j in range(len(A)):
        row.append(A[j][i])
    A_T.append(row)
print(A_T)

ATA=[]
for i in range(len(A_T)):
    row=[]
    for j in range(len(A[0])):
        product=0
        for k in range(len(A)):
            product+=A_T[i][k]*A[k][j]
        row.append(product)
    ATA.append(row)
print(ATA)