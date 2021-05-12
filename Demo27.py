import sklearn.datasets as datasets
from sklearn import model_selection, svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

iris = datasets.load_iris()
data = iris.data
target = iris.target

logisticRegression1 = LogisticRegression()
svc1 = svm.SVC(kernel='linear')
svc2 = svm.SVC(kernel='rbf')
svc3 = svm.SVC(kernel='poly')
tree = DecisionTreeClassifier()
classifiers = [logisticRegression1, svc1, svc2, svc3, tree]

for classifier in classifiers:
    score = model_selection.cross_val_score(classifier, data, target, cv=3)
    print(f"classifier:{classifier} score={score}")
