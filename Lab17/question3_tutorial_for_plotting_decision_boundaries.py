import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay

#creating dataset
X=np.array([
    [0.4, -0.7],
        [-1.5, -1.0],
        [-1.4, -0.9],
        [-1.3, -1.2],
        [-1.1, -0.2],
        [-1.2, -0.4],
        [-0.5, 1.2],
        [-1.5, 2.1],
        [1.0, 1.0],
        [1.3, 0.8],
        [1.2, 0.5],
        [0.2, -2.0],
        [0.5, -2.4],
        [0.2, -2.3],
        [0.0, -2.7],
        [1.3, 2.1],
    ])

y = np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])


#Visualization
fig,ax = plt.subplots(figsize=(4,3))
x_min,x_max,y_min,y_max = -3,3,-3,3

ax.set(xlim=(x_min,x_max),ylim=(y_min,y_max))  #setting the limit of x-axis(-3 to 3) and y-axis(-3,3)

scatter = ax.scatter(X[:,0],X[:,1],s=150,c=y,label=y, edgecolors="k")  #X[:,0],X[:,1]-> x1 and x2 values  s=150 (size of points) c=y -> color based on class

ax.legend(*scatter.legend_elements(),loc="upper left",title="Classes")

ax.set_title("Samples in two-dimensional feature space")

plt.show()


#Training SVC model and plotting decision boundaries

def plot_decision_boundaries(kernel,ax=None,support_vectors=True):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))

    clf = svm.SVC(kernel=kernel,gamma=2).fit(X,y)

    x_min,x_max,y_min,y_max = -3,3,-3,3
    ax.set(xlim=(x_min,x_max),ylim=(y_min,y_max))

    DecisionBoundaryDisplay.from_estimator(response_method="predict",plot_method="pcolormesh",alpha = 0.3,estimator=clf,X=X,ax=ax,) #predict is for 0,1 class pcolormesh is for filled colour regions ,alpha here is for making the background less or more lighter

    if support_vectors:
        ax.scatter(clf.support_vectors_[:,0],
                   clf.support_vectors_[:,1],
                   s=150,
                   facecolors='none',
                   edgecolors='k',
                   )

    # Plot samples by color and add legend

    scatter =ax.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolors="k")

    ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")

    ax.set_title(f"{kernel}kernel")

    plt.show()

def main():
    plot_decision_boundaries("linear")
    plot_decision_boundaries("poly")
    plot_decision_boundaries("rbf")
    plot_decision_boundaries("sigmoid")

if __name__ == "__main__":
    main()



