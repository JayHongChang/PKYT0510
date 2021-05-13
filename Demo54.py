import tensorflow as tf

#vector = [3.0, -1.0, 2.4, 5.9, 0.0001, 8.5, -0.000000001]
vector = tf.constant([3.0, -1.0, 2.4, 5.9, 0.0001, 8.5, -0.000000001])
result1 = tf.nn.relu(vector)
result2 = tf.nn.sigmoid(vector)

print(result1)
print(result2)