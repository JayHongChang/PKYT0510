import tensorflow as tf

tf.compat.v1.disable_eager_execution()

a = tf.compat.v1.placeholder(dtype=tf.int32, shape=(None,))
b = tf.compat.v1.placeholder(dtype=tf.int32, shape=(None,))
c = tf.add(a, b)
with tf.compat.v1.Session() as session1:
    result = session1.run(c, feed_dict={a: [3, 4, 5], b: [-1, 2, 3]})
    print(result)
    result2 = session1.run(c, feed_dict={a: [3, 4, 5, 6], b: [-1, 2, 3, 4]})
    print(result2)
    result3 = session1.run(c, feed_dict={a: [500], b: [3000]})
    print(result3)