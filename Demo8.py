import matplotlib.pyplot as plt

from sklearn import datasets

regressionData = datasets.make_regression(10, 6, noise=5)
for i in range(0,6):
    x1 = regressionData[0][:, i]
    y = regressionData[1]

    plt.scatter(x1, y)
    plt.show()