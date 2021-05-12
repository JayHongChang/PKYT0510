from sklearn import datasets, svm
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
pca = PCA(n_components=2)
data = pca.fit(iris.data).transform(iris.data)
print(iris.data.shape, data.shape)
print(iris.data[0:5, ])
print(data[0:5, ])
datamax = data.max(axis=0) + 1
datamin = data.min(axis=0) - 1
n = 1000
X, Y = np.meshgrid(np.linspace(datamin[0], datamax[0], n),
                   np.linspace(datamin[1], datamax[1], n))
print(len(X), len(Y))
# svc = svm.SVC(kernel='linear')
svc = svm.SVC(kernel='rbf',C=40)
# svc = svm.SVC(kernel='rbf')
# svc = svm.SVC(kernel='sigmoid')
svc.fit(data, iris.target)
Z = svc.predict(np.c_[X.ravel(), Y.ravel()])
print(Z.shape)
plt.contour(X, Y, Z.reshape(X.shape))

for c, s in zip([0, 1, 2], ['.', '^', '*']):
    d = data[iris.target == c]
    plt.scatter(d[:, 0], d[:, 1], c='k', marker=s)

print(f"score = {svc.score(data, iris.target)}")
# default 0.96
# linear 0.967
# poly 0.946
# rbf 0.96
# sigmoid (0.86)
plt.show()