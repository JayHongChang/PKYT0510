from sklearn import linear_model
import matplotlib.pyplot as plt

regression1 = linear_model.LinearRegression()
print(type(regression1))
features = [[1], [2], [3], [5]]
values = [1, 4, 15, 20]
plt.scatter(features, values, c='green')
plt.show()
regression1.fit(features, values)
print(type(regression1))

print("coefficient", regression1.coef_)
print("intercept", regression1.intercept_)
print("score", regression1.score(features, values))
range1 = [1, 5]

plt.plot(range1, regression1.coef_*range1+regression1.intercept_, c='gray')
plt.scatter(features, values, c='green')
plt.show()