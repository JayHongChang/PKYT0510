import matplotlib.pyplot as plt
from sklearn import linear_model

features = [[0, 1], [1, 3], [2, 8], [3, 9]]
values = [1, 4, 5.5, 8]
regression1 = linear_model.LinearRegression()
regression1.fit(features, values)

print("coefficient", regression1.coef_)
print("intercept", regression1.intercept_)

print(regression1.predict([[0, 0], [2, 2], [4, 4]]))
print(regression1.predict([[0, 1], [1, 3], [2, 8]]))
print(regression1.score([[0, 1], [1, 3], [2, 8]], [1, 4, 5.5]))