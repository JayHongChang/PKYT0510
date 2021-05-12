from subprocess import check_call

import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import export_graphviz

X = [[0, 0], [1, 1], [0, 1], [1, 0]]
Y = [0, 0, 1, 1]
color = ['red', 'green']
marker = ['o', 'd']
index = 0
while index < len(X):
    type = Y[index]
    plt.scatter(X[index][0], X[index][1], c=color[type], marker=marker[type])
    index += 1
plt.show()

classifier = tree.DecisionTreeClassifier()
classifier.fit(X, Y)
print(classifier)
# make a directory graph
dot_file = "graph/demo25.dot"
# png, svg, pdf
image_format = 'pdf'
output_file = 'graph/demo25.%s'
export_graphviz(classifier, out_file=dot_file,
                filled=True, rounded=True, special_characters=True,
                class_names=["red_circle", "green_diamond"])
check_call(['dot', '-T%s' % image_format, dot_file, '-o',
            output_file % image_format])