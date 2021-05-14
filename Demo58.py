import numpy
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from sklearn.model_selection import train_test_split

dataset1 = numpy.loadtxt("data/diabetes.csv", skiprows=1, delimiter=",")
print(dataset1.shape)

inputList = dataset1[:, 0:8]
resultList = dataset1[:, 8]
print(inputList.shape)
print(resultList.shape)

feature_train, feature_test, label_train, label_test = train_test_split(inputList,
                                                                        resultList,
                                                                        test_size=0.2,
                                                                        stratify=resultList)

for data in [resultList, label_train, label_test]:
    classes, counts = numpy.unique(data, return_counts=True)
    for cl, co in zip(classes, counts):
        print(f"{int(cl)}==>{co / sum(counts)}")
    print("---")

model = Sequential()
model.add(Dense(14, input_dim=8, activation=tf.nn.relu))
model.add(Dense(8, activation=tf.nn.relu))
output_layer = Dense(1, activation=tf.nn.sigmoid)
model.add(output_layer)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
print("before training, model coef:", output_layer.get_weights()[0])
print("before training, model intercept:", output_layer.get_weights()[1])

model.fit(feature_train, label_train, validation_data=(feature_test, label_test),
          epochs=200, batch_size=20, verbose=1)
# scores = model.evaluate(inputList, resultList)
# print(type(model.metrics_names))
# print(model.metrics_names)
# print(model.metrics_names[1], scores[1])
# print(model.metrics_names[0], scores[0])
# print("after training, model coef:", output_layer.get_weights()[0])
# print("after training, model intercept:", output_layer.get_weights()[1])