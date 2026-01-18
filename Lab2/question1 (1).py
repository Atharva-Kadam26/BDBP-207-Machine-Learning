"""For a design or feature matrix,
X=[[1,0,2],
[0,1,1],
[2,1,0],
[1,1,1],
[0,2,1]]
Compute the covariance matrix using matrix multiplications. Verify your results by using numpy library operations
"""
X=[[1,0,2],
[0,1,1],
[2,1,0],
[1,1,1],
[0,2,1]]

rows=len(X)
cols=len(X[0])

mean=[]

for i in range(cols):
    column_sum=0
    for j in range(rows):
        column_sum += X[j][i]
    mean.append(column_sum/rows)
print(mean)
print(X)

mean_centered=[]
for i in range(rows):
    row=[]
    for j in range(cols):
        row.append(X[i][j]-mean[j])
    mean_centered.append(row)
print(mean_centered)

mean_centered_t=[]

mean_rows=len(mean_centered)
mean_cols=len(mean_centered[0])

for i in range(mean_cols):
    mean_row=[]
    for j in range(mean_rows):
        mean_row.append(mean_centered[j][i])
    mean_centered_t.append(mean_row)
print(mean_centered_t)

product=[]
for i in range(mean_cols):
    row=[]
    for j in range(mean_cols):
        total=0
        for k in range(mean_rows):
            total+=mean_centered_t[i][k]*mean_centered[k][j]
        row.append(total)
    product.append(row)
print(product)

#1/n-1=1/(5-1)=1/4     as 5 is the number of sample that is number of rows

covariance_matrix=[]
for i in range(mean_cols):
    row=[]
    for j in range(mean_cols):
        row.append(product[i][j]/(rows-1))
    covariance_matrix.append(row)
print(covariance_matrix)
