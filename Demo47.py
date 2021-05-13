import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()
print(tf.__version__)
a1 = np.array([5, 3, 8])
a2 = np.array([3, -1, 2])
a3 = np.add(a1, a2)
print(a3)

t1 = tf.constant([5, 3, 8])
t2 = tf.constant([3, -1, 2])
t3 = tf.add(t1, t2)
print("tensor use tensor operator", t3)
session1 = tf.compat.v1.Session()
result1 = session1.run(t3)
print("result=", result1)
session1.close()