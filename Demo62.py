from pandas import read_csv
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.layers import Dense
from keras.models import Sequential
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier

PATH = "data/iris.data"
df1 = read_csv(PATH, header=None)
print(type(df1), df1.shape)
print(df1.head())
dataset = df1.values
print(type(dataset))
features = dataset[:, :4].astype(float)
labels = dataset[:, 4]
print(features)
print(labels)
encoder = LabelEncoder()
encoder.fit(labels)
encoded_Y = encoder.transform(labels)
print(type(encoded_Y), encoded_Y)
dummy_y = np_utils.to_categorical(encoded_Y)
print(type(dummy_y), dummy_y)


def baseline_model():
    m = Sequential()
    m.add(Dense(10, input_dim=4, activation=tf.nn.relu))
    m.add(Dense(3, activation=tf.nn.softmax))
    m.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
    m.summary()
    return m


estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=10, verbose=1)

kfold = KFold(n_splits=3, shuffle=True)
results = cross_val_score(estimator, features, dummy_y, cv=kfold)
print(f"Acc mean:{results.mean() * 100}%, std={results.std()}")