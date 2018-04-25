# -*- coding: utf-8 -*-
import tensorflow as tf
from src.input_data import *
from src.input import train

img_width = 28
data, num = read_data_sets(path='../data/faces_dir', img_width=img_width)

x = tf.placeholder("float", shape=[None, img_width * img_width])
y_ = tf.placeholder("float", [None, num])

W = tf.Variable(tf.zeros([img_width * img_width, num]) + 0.1)
b = tf.Variable(tf.zeros([num]) + 0.1)

y = tf.nn.softmax(tf.matmul(x, W) + b)


tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
regularizer = tf.contrib.layers.l2_regularizer(scale=5.0/50000)
reg_term = tf.contrib.layers.apply_regularization(regularizer)
cross_entropy = -tf.reduce_sum(y_ * tf.log(y) + 1e-8) + reg_term

train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

merged_summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('/tmp/mnist_logs', sess.graph)

writer = tf.summary.FileWriter("tmp/mnist_demo/1")
writer.add_graph(sess.graph)

validate_data = get_validate_data()
for i in range(500):
    for batch in train.get_batches(10):
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})
    if i % 5 == 0:
        # summary_str = sess.run(merged_summary_op)
        # summary_writer.add_summary(summary_str, i)

        with sess.as_default():
            print(accuracy.eval(feed_dict={x: validate_data[0], y_: validate_data[1]}))
test_data = get_test_data()
with sess.as_default():
    print(accuracy.eval(feed_dict={x: test_data[0], y_: test_data[1]}))
