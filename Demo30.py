import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans

X = np.r_[np.random.randn(500, 2) + [2, 2],
          np.random.randn(500, 2) + [0, -2],
          np.random.randn(500, 2) + [-2, 2]]
kmean = KMeans(n_init=1, n_clusters=4)
kmean.fit(X)
print(kmean.cluster_centers_)
print(kmean.inertia_)
colors = ['c', 'm', 'y', 'k']
markers = ['o', 's', '*', '^']
for i in range(4):
    dataX = X[kmean.labels_ == i]
    plt.scatter(dataX[:, 0], dataX[:, 1], c=colors[i], marker=markers[i])
    print(dataX.size)
plt.scatter(kmean.cluster_centers_[:, 0], kmean.cluster_centers_[:, 1],
            marker='*', s=200, c='#C0FFEE')
plt.show()