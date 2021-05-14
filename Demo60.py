import numpy
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

dataset1 = numpy.loadtxt("data/diabetes.csv", skiprows=1, delimiter=",")
print(dataset1.shape)

inputList = dataset1[:, 0:8]
resultList = dataset1[:, 8]
print(inputList.shape)
print(resultList.shape)


def create_default_model():
    m = Sequential()
    m.add(Dense(14, input_dim=8, activation=tf.nn.relu))
    m.add(Dense(8, activation=tf.nn.relu))
    output_layer = Dense(1, activation=tf.nn.sigmoid)
    m.add(output_layer)
    m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    m.summary()
    return m


model = KerasClassifier(build_fn=create_default_model,
                        epochs=200, batch_size=20, verbose=0)
fiveFold = StratifiedKFold(n_splits=5, shuffle=True)
result = cross_val_score(model, inputList, resultList, cv=fiveFold)
print(f"result mean={result.mean()}, std={result.std()}")