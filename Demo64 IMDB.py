import numpy as np
from keras import layers, models
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print(train_data[0])
print(max([max(sequence) for sequence in train_data]))
word_index = imdb.get_word_index()
reverse_word_index = dict((v, k) for k, v in word_index.items())
decoded_review = ' '.join(reverse_word_index.get(i - 3, '?') for i in train_data[3])
print(decoded_review)


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, seq in enumerate(sequences):
        results[i, seq] = 1.
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
print(x_train[0])

model = models.Sequential()
model.add(Dense(20, activation='relu', input_shape=(10000,)))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    validation_data=(x_val, y_val),
                    epochs=30,
                    batch_size=64)