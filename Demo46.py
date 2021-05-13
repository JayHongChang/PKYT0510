import tensorflow as tf
import numpy as np

print(tf.__version__)
a1 = np.array([5, 3, 8])
a2 = np.array([3, -1, 2])
a3 = np.add(a1, a2)
print(a3)

t1 = tf.constant([5, 3, 8])
t2 = tf.constant([3, -1, 2])
t3 = tf.add(t1, t2)
print("tensor use tensor operator", t3.numpy())
t4 = np.add(t1, t2)
print("tensor use numpy operator", t4)
t5 = tf.add(a1, a2)
print("numpy use tensor operator", t5.numpy())