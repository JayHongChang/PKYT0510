import numpy as np
from sklearn.naive_bayes import GaussianNB

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
nb1 = GaussianNB()
nb1.fit(X, Y)
print(nb1.predict([[0, 0], [-2, 0], [0, 2], [0, -2]]))

nb2 = GaussianNB()
nb2.partial_fit(X, Y, np.unique(Y))
print(nb2.predict([[0, 0], [-2, 0], [0, 2], [0, -2]]))
nb2.partial_fit([[-1, 0]], [2])
print("predict again,", nb2.predict([[0, 0], [-2, 0], [0, 2], [0, -2]]))