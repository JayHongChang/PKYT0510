from pprint import pprint

from sklearn import datasets


def sortBySecond(record):
    return record[1]


regressionData = datasets.make_regression(10, 6, noise=5)
regressionX = regressionData[0]
print(type(regressionX), regressionX.shape)
print(regressionX)
r1 = sorted(regressionX, key=lambda tup: tup[0])
print("sort by first column")
pprint(r1)
r2 = sorted(regressionX, key=sortBySecond)
print("sort by second column")
pprint(r2)