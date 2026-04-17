#Simulated data
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

np.random.seed(0)

X=np.random.standard_normal((50,2))
X[:25,0] += 3
X[:25,1] -= 4

#K=2 i.e clusters=2
kmeans = KMeans(n_clusters=2,n_init=20,random_state=999).fit(X)
kmeans.labels_

plt.scatter(X[:,0], X[:,1], c=kmeans.labels_)
plt.title("K-Means (K=2)")
plt.show()

#K=3 i.e. clusters=3
kmeans = KMeans(n_clusters=3,n_init=20,random_state=999).fit(X)
kmeans.labels_

plt.scatter(X[:,0], X[:,1], c=kmeans.labels_)
plt.title("K-Means (K=3)")
plt.show()

kmeans1 = KMeans(n_clusters=3, n_init=1, random_state=3).fit(X)
kmeans20 = KMeans(n_clusters=3, n_init=20, random_state=3).fit(X)

kmeans1.inertia_, kmeans20.inertia_

plt.scatter(X[:,0], X[:,1], c=kmeans1.labels_)
plt.title("n_init = 1")
plt.show()

plt.scatter(X[:,0], X[:,1], c=kmeans20.labels_)
plt.title("n_init = 20")
plt.show()