import tensorflow as tf
import numpy as np

sess = tf.Session()

def run_print(exp, data_dict={}):
    global sess
    print(sess.run(exp, feed_dict=data_dict))

x = tf.constant(2, name="input_x")
y = tf.constant(3, name="input_y")
z = tf.add(x, y, name='sum')

print(x, y, z)
run_print(z)


mat = tf.placeholder(dtype=tf.int32, shape=(3,2))


x_data = np.linspace(-1, 1, 10)

xs = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="xs")
avg = tf.reduce_sum(xs)
sess.run(avg)

W = tf.Variable(
    tf.random_uniform(shape=(13, 1),
                      minval=-1.0, maxval=1.0,
                      dtype=tf.float32),
    name = 'W')

tf.squared_difference()
