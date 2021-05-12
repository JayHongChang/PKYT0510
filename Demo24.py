from sklearn import tree
from matplotlib import pyplot as plt

X = [[0, 0], [1, 1]]
Y = [0, 1]

classifier = tree.DecisionTreeClassifier()
classifier.fit(X, Y)
print(classifier.predict([[2, 2], [-1, -1], [2, 1], [1, 2], [-1, 2]]))

tree.plot_tree(classifier)
plt.show()