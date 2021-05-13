import tensorflow as tf


def computeArea(sides):
    a = sides[:, 0]
    b = sides[:, 1]
    c = sides[:, 2]

    s = (a + b + c) / 2
    areaSquare = s * (s - a) * (s - b) * (s - c)
    return areaSquare ** 0.5


with tf.compat.v1.Session() as session1:
    sides = tf.compat.v1.placeholder(tf.float32, shape=(None, 3))
    area = computeArea(sides)
    result = session1.run(area, feed_dict={
        sides: [[6.0, 6.0, 6.0],
                [3.0, 4.0, 5.0],
                [6.0, 8.0, 10.0]]
    })
    print(result)