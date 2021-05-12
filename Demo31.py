import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

X = np.r_[np.random.randn(5000, 2) + [2, 2],
          np.random.randn(5000, 2) + [0, -2],
          np.random.randn(5000, 2) + [-2, 2]]
inertias = []
for k in range(1, 10):
    kmean = KMeans(n_clusters=k)
    kmean.fit(X)
    inertias.append(kmean.inertia_)

plt.plot(range(1, 10), inertias)
plt.show()