import numpy as np
import tensorflow as tf


def add_layer(inputs, in_size, out_size, activation_function=None):
    """
    Y = X * W + b, X: n_batch x in_size, W: in_size x out_size, b: 1 x out_size
    :param inputs:
    :param in_size: input dimension
    :param out_size: output dimension
    :param activation_function:
    :return: X * W + b
    """
    Weights = tf.Variable(tf.random_normal(shape=[in_size, out_size]))
    biases = tf.Variable(tf.zeros(shape=[1, out_size]))
    wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function == None:
        return wx_plus_b
    else:
        return activation_function(wx_plus_b)


x_data = np.linspace(-1, 1, 100)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# print(x_data)
# print(y_data)

xs = tf.placeholder(tf.float32, shape=[None, 1])
ys = tf.placeholder(tf.float32, shape=[None, 1])
L1 = add_layer(xs, 1, 10, tf.nn.relu)
Prediction = add_layer(L1, 10, 1, None)

loss = tf.reduce_mean(
    tf.reduce_sum(tf.square(y_data-Prediction)),
    )

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(100):
    sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
    if i%50 == 0:
        print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))


tf.gradients()