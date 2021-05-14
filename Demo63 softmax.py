import tensorflow as tf
from numpy import array

v1 = [3.0, 4.0, 5.0]

s1 = tf.nn.softmax(v1)
print("original result=", array(v1) / array(v1).sum())
print("softmax result=", s1.numpy())