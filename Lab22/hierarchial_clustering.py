"""Work on NCI data - build classification model after reducing the gene expression
features using hierarchical clustering. Compare this with the PCA approach"""

from ISLP import load_data
from lightning_fabric.accelerators import Accelerator
from pandas.core.config_init import pc_large_repr_doc
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def data():
    Ic = load_data("NCI60")  #this data is high dimensional i.e. p>>n thus we will carry out dimensionality reduction
    X = pd.DataFrame(Ic['data'])
    y = Ic['labels']
    return X,y

def hierarchial_clustering(X,clusters):
    Z = linkage(X.T,method="complete")  #linkage type- complete  #why did we transpose? bcoz we want to cluster genes and not samples
    f__cluster = fcluster(Z,t=50,criterion="maxclust")  #cuts the dendogram into 5 clusters , at this stage all the genes gets a cluster label (1 to 50)
    return f__cluster

def reduce_features_by_clusters(X,clusters,k):
    X_reduced = np.zeros((X.shape[0],k))  #we are creating an empty matrix to store your new features same number of rows different number of columns

    for i in range(1,k+1):  # for each cluster since k= no. of clusters
        cols = np.where(clusters==i)[0]  #find genes in that cluster
        if len(cols) > 0:
            X_reduced[:,i-1] = X.iloc[:,cols].mean(axis=1)  #computing mean expression
            #taking all genes in the cluster and computing average
    return X_reduced

def pca_reduction(X,n_components):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    return X_pca

def train_lda(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)
    lda = LinearDiscriminantAnalysis() #separate classes, maximize btwn-class variance , minimize within-class variance
    lda.fit(X_train,y_train)

    accuracy = lda.score(X_test,y_test)
    return accuracy


def plot_dendrogram(X, num_genes=100):
    """
    Plotting dendrogram using a subset of genes (for readability)
    """
    # Use subset (full 6830 genes will be too messy)
    X_subset = X.iloc[:, :num_genes]

    Z = linkage(X_subset.T, method='complete')

    plt.figure(figsize=(12, 6))
    dendrogram(Z, no_labels=True)
    plt.title(f"Dendrogram (First {num_genes} Genes)")
    plt.xlabel("Genes")
    plt.ylabel("Distance")
    plt.show()


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def plot_pca(X, y):
    """
    Plot first 2 principal components
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))

    for label in set(y):
        idx = (y == label)
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=label)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Visualization (2D)")
    plt.legend(fontsize=8)
    plt.show()

def main():
    X,y = data()
    f__cluster = hierarchial_clustering(X,clusters=50)
    x_clustered = reduce_features_by_clusters(X,f__cluster,50)
    acc_cluster =train_lda(x_clustered,y)

    X_pca = pca_reduction(X,n_components=10)
    acc_pca = train_lda(X_pca,y)

    print("Clustering Accuracy",acc_cluster)
    print("PCA Accuracy",acc_pca)

if __name__ == "__main__":
    main()
