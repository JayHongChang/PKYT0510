import numpy
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV

dataset1 = numpy.loadtxt("data/diabetes.csv", skiprows=1, delimiter=",")
print(dataset1.shape)

inputList = dataset1[:, 0:8]
resultList = dataset1[:, 8]
print(inputList.shape)
print(resultList.shape)


def create_default_model(optimizer='adam', init='uniform'):
    m = Sequential()
    m.add(Dense(14, input_dim=8, kernel_initializer=init, activation=tf.nn.relu))
    m.add(Dense(8, activation=tf.nn.relu))
    output_layer = Dense(1, activation=tf.nn.sigmoid)
    m.add(output_layer)
    m.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    m.summary()
    return m


model = KerasClassifier(build_fn=create_default_model, verbose=0)
optimizers = ['sgd', 'rmsprop', 'adam']
inits = ['normal', 'uniform']
epochs = [50, 100, 150]
batches = [5, 10, 15]
parameterGrid = dict(optimizer=optimizers, epochs=epochs, init=inits, batch_size=batches)
grid = GridSearchCV(estimator=model, param_grid=parameterGrid, cv=3)
grid_result = grid.fit(inputList, resultList)