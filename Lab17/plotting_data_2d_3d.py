"""Implement a feature mapping function called Transform()
phi : R^2 -> R^3
phi(x1,x2)=(x1^2,sqrt(2x1x2),x2^2)
Use following set of samples
Plot these points. Then transform these points using your “Transform” function into 3-dim
space. Plot the points and manipulate the points so that you can see a separating plane in 3D."""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def transform(x1,x2):   #feature mapping function
        phi1 = x1**2
        phi2 = np.sqrt(2) * x1 * x2
        phi3 = x2**2
        return np.column_stack((phi1,phi2,phi3))

def load_data():
    df = pd.read_csv("kernel_dataset.csv")
    x1 = df["x1"].values
    x2 = df["x2"].values
    y = df["Label"]
    return x1,x2,y


#Plotting 2D data
def plot_data(x1,x2,y):
    plt.figure(figsize=(6,5))
    for label in np.unique(y):
        plt.scatter(x1[y==label],x2[y==label],label=label)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("original Data(2D)")
    plt.legend()
    plt.show()

#Plotting 3D transformed data
def plot_3d(transformed_data,y):
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111, projection='3d')  #111 = 1row,1column,1 plot
    for label in np.unique(y):
        ax.scatter(
            transformed_data[y==label,0],
            transformed_data[y==label,1],
            transformed_data[y==label,2],
            label=label
        )

    xx, yy = np.meshgrid(
     np.linspace(min(transformed_data[:, 0]), max(transformed_data[:, 0]), 10),
        np.linspace(min(transformed_data[:, 1]), max(transformed_data[:, 1]), 10)
    )

    zz = 300 - 0.5 * xx - 4.5 * yy

    ax.plot_surface(xx, yy, zz, alpha=0.3)

    ax.set_xlabel("x1^2")
    ax.set_ylabel("sqrt2 x1x2")
    ax.set_zlabel("x2^2")
    plt.legend()
    plt.show()



def main():
    x1,x2,y = load_data()
    transformed_data=transform(x1,x2)
    print(transformed_data)
    # plot_data(x1,x2,y)
    plot_3d(transformed_data,y)


if __name__ == "__main__":
    main()

