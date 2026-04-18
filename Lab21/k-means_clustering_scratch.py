import numpy as np
import matplotlib.pyplot as plt

def initialize_centroids(X,k):
    indices = np.random.choice(len(X),k,replace=False)
    return X[indices]

def assign_clusters(X,centroids):
    clusters=[]

    for point in X:
        distances = []
        for c in centroids:
            distance = np.sqrt(np.sum((point - c)**2))
            distances.append(distance)

        cluster = np.argmin(distances)
        clusters.append(cluster)

    return np.array(clusters)

def recompute_centroids(X,clentroids,k):
    centroids=[]
    for i in range(k):
        points = X[clentroids==i]
        if len(points)==0:
            centroids.append(X[np.random.randint(len(X))])

        else:
            centroid = np.mean(points,axis=0)
            centroids.append(centroid)

    return np.array(centroids)


def k_means(X,k,max_iter=100):
    centroids=initialize_centroids(X,k)
    for iteration in range(max_iter):
        clusters=assign_clusters(X,centroids)
        new_centroids=recompute_centroids(X,clusters,k)

        if np.allclose(centroids,new_centroids):
            break

        centroids=new_centroids
    return centroids,clusters

def main():
    X = np.array([
        [1, 2], [2, 1], [3, 2],
        [8, 8], [9, 9], [10, 8]
    ])

    k = 2

    centroids,clusters=k_means(X,k)
    print("Final Centroids:")
    print(centroids)
    print("\nCluster Assignments:")
    print(clusters)


    plt.scatter(X[:, 0], X[:, 1], c=clusters)
    plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='X', s=200)
    plt.title("K-Means Clustering")
    plt.show()

if __name__ == "__main__":
    main()