from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.utils import to_categorical
import tensorflow as tf
from keras.callbacks import TensorBoard

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape)
FLATTEN_DIM = 28 * 28
TRAINING_SIZE = len(train_images)
TEST_SIZE = len(test_images)

trainImages = np.reshape(train_images, (TRAINING_SIZE, FLATTEN_DIM))
testImages = np.reshape(test_images, (TEST_SIZE, FLATTEN_DIM))
print(trainImages[0])

trainImages = trainImages.astype(np.float32)
testImages = testImages.astype(np.float32)
trainImages /= 255
testImages /= 255
print(trainImages[0])

NUM_DIGITS = 10
trainLabels = to_categorical(train_labels, NUM_DIGITS)
testLabels = to_categorical(test_labels, NUM_DIGITS)
print(trainLabels[0])

model = Sequential()
model.add(Dense(units=200, activation=tf.nn.relu, input_shape=(FLATTEN_DIM,)))
model.add(Dense(units=200, activation=tf.nn.relu))
model.add(Dense(units=10, activation=tf.nn.softmax))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
tbCallback = TensorBoard(log_dir="./logs/demo70", histogram_freq=0,
                         write_graph=True, write_images=True)
model.fit(trainImages, trainLabels, epochs=20, callbacks=[tbCallback])
predictLabels = model.predict_classes(testImages)
print("predicted resut=",predictLabels[:10])

loss, accuracy = model.evaluate(testImages, testLabels)
print(f"test accuracy:{accuracy}")