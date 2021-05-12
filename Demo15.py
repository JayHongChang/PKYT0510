from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot
from sklearn.feature_selection import SelectKBest, f_regression

X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=511)
print(X.shape)
print(y.shape)
model = LinearRegression()
model.fit(X, y)
importance = model.coef_
for index, value in enumerate(importance):
    print(f'Feature #{index} score:{value:.4f}')
pyplot.bar([x for x in range(len(importance))], importance)

kBest = SelectKBest(f_regression, k=4).fit(X, y)
print(type(kBest))
print("get support vectors:")
print(kBest.get_support())
newX = kBest.fit_transform(X, y)
print(X.shape, newX.shape)
print(X[:1])
print(newX[:1])
pyplot.show()