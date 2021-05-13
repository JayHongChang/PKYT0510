import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
learning_rate = 0.01
training_epochs = 100

x_train = np.linspace(-1, 1, 101)
# print(x_train)
y_train = 5 * x_train + np.random.randn(*x_train.shape) * 0.33


# print(y_train)


def model(X, w):
    return tf.multiply(X, w)


X = tf.compat.v1.placeholder(tf.float32)
Y = tf.compat.v1.placeholder(tf.float32)

w = tf.Variable(0.0, name="weights")
y_model = model(X, w)
cost = tf.square(Y - y_model)
print("cost=", cost)

train_op = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session() as session1:
    session1.run(init)
    for epoch in range(training_epochs):
        print(f"#{epoch}#")
        for x, y in zip(x_train, y_train):
            session1.run(train_op, feed_dict={X: x, Y: y})
    w_val = session1.run(w)
y_learned = x_train * w_val
plt.scatter(x_train, y_train)
plt.plot(x_train, y_learned, 'r')
plt.show()