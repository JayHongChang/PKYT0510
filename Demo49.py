import tensorflow as tf


@tf.function
def add(p, q):
    return tf.math.add(p, q)


print(add([1, 2, 3], [4, 5, 6]))
print(add([1, 2, 3, 4], [5, 6, 7, 8]))
print(add([500], [3000]))