
import tensorflow as tf
from datetime import datetime


@tf.function
def computeArea(sides):
    a = sides[:, 0]
    b = sides[:, 1]
    c = sides[:, 2]
    s = (a + b + c) / 2
    areaSquare = s * (s - a) * (s - b) * (s - c)
    return areaSquare ** 0.5


current_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
# 手動創建logs目錄
print(current_timestamp)
logdir = 'logs/demo51/%s' % current_timestamp
writer = tf.summary.create_file_writer(logdir)
tf.summary.trace_on(graph=True, profiler=True)
print(computeArea(tf.constant([[3.0, 4.0, 5.0],
                               [6.0, 8.0, 10.0],
                               [6.0, 6.0, 6.0]])))
with writer.as_default():
    tf.summary.trace_export(name="heron formula", step=0, profiler_outdir=logdir)