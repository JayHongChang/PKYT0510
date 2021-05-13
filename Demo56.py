import numpy
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense
import tensorflow as tf

dataset1 = numpy.loadtxt("data/diabetes.csv", skiprows=1, delimiter=",")
print(dataset1.shape)

inputList = dataset1[:, 0:8]
resultList = dataset1[:, 8]
print(inputList.shape)
print(resultList.shape)


def createModel():
    model = Sequential()
    model.add(Dense(14, input_dim=8, activation=tf.nn.relu))
    model.add(Dense(8, activation=tf.nn.relu))
    output_layer = Dense(1, activation=tf.nn.sigmoid)
    model.add(output_layer)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

model1 = createModel()

model1.fit(inputList, resultList, epochs=200, batch_size=20, verbose=0)
MODEL_FILENAME = "data/demo56.model"
save_model(model1, MODEL_FILENAME)
scores = model1.evaluate(inputList, resultList)
# print(type(model1.metrics_names))
# print(model1.metrics_names)
print("model1:", model1.metrics_names[1], scores[1])
print("model1:", model1.metrics_names[0], scores[0])

model2 = createModel()
scores = model2.evaluate(inputList, resultList)
print("model2:", model2.metrics_names[1], scores[1])
print("model2:", model2.metrics_names[0], scores[0])

model3 = load_model(MODEL_FILENAME)
scores = model3.evaluate(inputList, resultList)
print("model3:", model3.metrics_names[1], scores[1])
print("model3:", model3.metrics_names[0], scores[0])