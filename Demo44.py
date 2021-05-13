import tensorflow as tf

t1 = tf.constant("Hello tensorflow")
t2 = tf.constant(500)
t3 = tf.constant(3.14)
print(type(t1), t1, t1.numpy())
print(type(t2), t2, t2.numpy())
print(type(t3), t3, t3.numpy())