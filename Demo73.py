import pandas as pd
import keras
from sklearn.preprocessing import LabelBinarizer
from keras import callbacks
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard

csv = pd.read_csv("data/bmi.csv")
print(csv.shape, csv.columns)
csv['height'] = csv['height'] / 200
csv['weight'] = csv['weight'] / 100
encoder = LabelBinarizer()
transformedLabel = encoder.fit_transform(csv['label'])
print(csv['label'][:10])
print(transformedLabel[:10])

test_csv = csv[25000:]
test_pat = test_csv[["weight", 'height']]
test_ans = transformedLabel[25000:]
train_csv = csv[:25000]
train_pat = train_csv[["weight", 'height']]
train_ans = transformedLabel[:25000]

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(2,)))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.summary()
tb = TensorBoard(log_dir="logs/demo73", histogram_freq=1)
history = model.fit(train_pat, train_ans, batch_size=50, epochs=100,
                    verbose=1, validation_data=(test_pat, test_ans),
                    callbacks=[tb])
score = model.evaluate(test_pat, test_ans, verbose=0)
print("test loss:", score[0])
print("test accuracy:", score[1])
