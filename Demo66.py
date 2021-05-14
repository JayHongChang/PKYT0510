import tensorflow as tf
from numpy import array
import numpy as np

v1 = [3.0, 4.0, 5.0]

s1 = tf.nn.softmax(v1)


def softmax(x):
    x = np.array(x)
    return np.exp(x) / np.sum(np.exp(x), axis=0)


print("original result=", array(v1) / array(v1).sum())
print("softmax result=", s1.numpy())
print("hand writing softmax result=", softmax(v1))    #softmax用於數值大於1的時候