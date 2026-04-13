"""Consider the following dataset. Implement the RBF kernel. Check if RBF kernel
separates the data well and compare it with the Polynomial Kernel."""
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

def load_data():
    df=pd.read_csv('rbf_kernel_dataset.csv')
    x1 = df['x1'].values
    x2 = df['x2'].values
    Y = df['Label'].values
    y=np.where(Y=='Blue',0,1) # as the labels are as Blue or Red which is non numerical
    return x1,x2,y

def rbf_model(x1,x2,y):
    X=np.column_stack([x1,x2])
    rbf = SVC(kernel='rbf', gamma=0.5)   # the value of gamma decides the spread of influence of training points. A
    #A small gamma leads to smoother decision boundaries(underfitting), while a large gamma leads to complex boundary(overfitting)
    trained_rbf= rbf.fit(X,y)
    return trained_rbf,X

def polynomial_model(x1,x2,y):
    X=np.column_stack([x1,x2])
    poly_kernel = SVC(kernel='poly', degree=3)
    trained_poly = poly_kernel.fit(X,y)
    return trained_poly,X

def plot_svm(model,X,y,title):
    x_min,x_max = X[:,0].min() -1 ,X[:,0].max() + 1
    y_min,y_max = X[:,1].min() -1 ,X[:,1].max() + 1

    xx,yy = np.meshgrid(
        np.linspace(x_min,x_max,200),
        np.linspace(y_min,y_max,200)
    )

    grid_points = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    Z = model.predict(grid_points)

    #reshape back to grid
    Z = Z.reshape(xx.shape)  #converts to grid


    #Plot graph
    plt.contourf(xx,yy,Z,alpha=0.3)
    plt.scatter(X[:,0],X[:,1],c=y,edgecolors='k')

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    return plt.show()

def main():
    x1,x2,y=load_data()
    trained_rbf,X = rbf_model(x1,x2,y)
    trained_poly,X= polynomial_model(x1,x2,y)
    rbf_plot = plot_svm(trained_rbf,X,y,'RBF')
    poly_plot = plot_svm(trained_poly,X,y,'Poly')


if __name__ == '__main__':
    main()

#Inference from the graphs:
#RBF Kernel produces a non-linear decision boundary that better separates the dataset
#While the polynomial kernel produces a simpler boundary that may misclassify some points
