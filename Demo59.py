import numpy
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

dataset1 = numpy.loadtxt("data/diabetes.csv", skiprows=1, delimiter=",")

inputList = dataset1[:, 0:8]
resultList = dataset1[:, 8]

fiveFold = StratifiedKFold(n_splits=5, shuffle=True)    #利用k-fold分成5份資料
totalScores = []

for train, test in fiveFold.split(inputList, resultList):
    print("start a new fold")
    model = Sequential()
    model.add(Dense(14, input_dim=8, activation=tf.nn.relu))
    model.add(Dense(8, activation=tf.nn.relu))
    output_layer = Dense(1, activation=tf.nn.sigmoid)
    model.add(output_layer)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.summary()

    model.fit(inputList[train], resultList[train], epochs=200, batch_size=20, verbose=0)
    scores = model.evaluate(inputList[test], resultList[test], verbose=0)
    print(model.metrics_names)
    print(model.metrics_names[1], scores[1])
    print(model.metrics_names[0], scores[0])
    totalScores.append(scores[1] * 100)

print("5 fold result:", totalScores)
print(f"mean={numpy.mean(totalScores)}, std={numpy.std(totalScores)}")