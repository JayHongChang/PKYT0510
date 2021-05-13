import numpy
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import matplotlib.pyplot as plt

dataset1 = numpy.loadtxt("data/diabetes.csv", skiprows=1, delimiter=",")
print(dataset1.shape)

inputList = dataset1[:, 0:8]
resultList = dataset1[:, 8]
print(inputList.shape)
print(resultList.shape)

model = Sequential()
model.add(Dense(14, input_dim=8, activation=tf.nn.relu))
model.add(Dense(8, activation=tf.nn.relu))
output_layer = Dense(1, activation=tf.nn.sigmoid)
model.add(output_layer)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(inputList, resultList, validation_split=0.1, epochs=200, batch_size=20)
scores = model.evaluate(inputList, resultList)
print(type(model.metrics_names))
print(model.metrics_names)
print(model.metrics_names[1], scores[1])
print(model.metrics_names[0], scores[0])

plt.plot(history.history['accuracy'])             #用history屬性來叫出accuracy,loss
plt.plot(history.history['val_accuracy'])
plt.legend(['accuracy', 'validation accuracy'], loc='upper right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'validation loss'], loc='upper right')
plt.show()