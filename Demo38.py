import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(iris.target)
print(iris.target_names)
df['species'] = np.array([iris.target_names[i] for i in iris.target])
seaborn.pairplot(df, hue='species')
plt.show()
X_train, X_test, y_train, y_test = train_test_split(df[iris.feature_names], iris.target,
                                                    test_size=0.5, stratify=iris.target)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf = RandomForestClassifier(n_estimators=100, oob_score=True)
rf.fit(X_train, y_train)
predicted = rf.predict(X_test)
accuracy = accuracy_score(y_test, predicted)
print(f"out of bag score:{rf.oob_score_:.3f}")
print(f"accuracy={accuracy}")

from sklearn.metrics import confusion_matrix

cm = pd.DataFrame(confusion_matrix(y_test, predicted), columns=iris.target_names, index=iris.target_names)
print(cm)
seaborn.heatmap(cm, annot=True)
plt.show()