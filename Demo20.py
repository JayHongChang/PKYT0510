from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
print(list(iris.keys()))
print(iris['feature_names'])
# print(iris['target'])
X = iris["data"][:, 3:]  # petal width
y = (iris['target'] == 2).astype(np.int)
print(X)
print(y)

regression1 = LogisticRegression()
regression1.fit(X, y)
print(regression1.coef_, regression1.intercept_)

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_prob = regression1.predict_proba(X_new)
y_calculate = 1 / (1 + np.exp(-(regression1.coef_ * X_new + regression1.intercept_)))
plt.plot(X_new, y_prob[:, 1], 'b--', label="virginica", linewidth=5)
plt.plot(X_new, y_prob[:, 0], 'g--', label="not virginica")
plt.plot(X_new, y_calculate, 'r', label="calculated")
plt.plot(X, y, 'g.')
plt.legend(loc=2)
plt.show()