import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.datasets import get_rdataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ISLP import load_data

USArrests = get_rdataset('USArrests').data
print(USArrests.head())

scaler = StandardScaler(with_std=True,with_mean=True)
USArrests_scaled = scaler.fit_transform(USArrests)

#PCA
pcaUS = PCA()  #centers the variables to have mean zero
scores=pcaUS.fit_transform(USArrests_scaled)
print("PCA mean:",pcaUS.mean_)
print("Components are:",pcaUS.components_)

#choose components PCA biplot Not Scaled
i, j = 0, 1  #Pc1 and PC2

#plot
fig , ax = plt.subplots (1, 1, figsize =(8, 8))

#scatter plot (scores)
ax.scatter(scores [:,0], scores [: ,1])

#axis labels
ax.set_xlabel(f'PC{i+1}')
ax.set_ylabel(f'PC{j+1}')

for k in range(pcaUS.components_.shape[1]):
    ax.arrow(0, 0,
             pcaUS.components_[i,k],
             pcaUS.components_[j,k],
             color='red')

    ax.text(pcaUS.components_[i,k],
            pcaUS.components_[j,k],
            USArrests.columns[k])

plt.title("PCA biplot (Unscaled)")
plt.grid()
plt.show()


#PCA Bipilot Scaled
scale_arrow = 2

scores [:,1] *= -1
pcaUS.components_ [1] *= -1 # flip the y-axis

fig , ax = plt.subplots (1, 1, figsize =(8, 8))
ax.scatter(scores [:,0], scores [: ,1])

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')

for k in range(pcaUS.components_.shape[1]):
    ax.arrow(0, 0,
             scale_arrow*pcaUS.components_[i,k],
             scale_arrow*pcaUS.components_[j,k],
             color='red')
    ax.text(scale_arrow*pcaUS.components_[i,k],
            scale_arrow*pcaUS.components_[j,k],
            USArrests.columns[k])

plt.title("PCA biplot (Scaled)")
plt.grid()
plt.show()

#Std Dev of scores
scores.std(axis=0,ddof=1)

#Variance
pcaUS.explained_variance_
pcaUS.explained_variance_ratio_


#Scree Plot
fig, axes = plt.subplots(1, 2, figsize=(15,5))

ticks = np.arange(pcaUS.n_components_) + 1

# Scree Plot
axes[0].plot(ticks, pcaUS.explained_variance_ratio_, marker='o')
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Proportion of Variance Explained")
axes[0].set_xticks(ticks)

# Cumulative PVE
axes[1].plot(ticks, pcaUS.explained_variance_ratio_.cumsum(), marker='o')
axes[1].set_xlabel("Principal Component")
axes[1].set_ylabel("Cumulative PVE")
axes[1].set_xticks(ticks)

plt.show()



