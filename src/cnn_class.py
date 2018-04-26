# -*- coding: utf-8 -*-
import time

import tensorflow as tf
import numpy as np
import os

from src import logger
from src.input import train
from src.input_data import get_validate_data, get_test_data
from src.logger import Logger


class CNN:

    def __init__(self, learning_rate=1e-3, img_width=32, input_size=1024, output_size=10,
                 model_path='models/net_32_4.ckpt'):
        self.conv_layer = 0
        self.fc_layer = 0
        self.x = tf.placeholder(tf.float32, shape=[None, input_size], name='x')
        self.y_ = tf.placeholder(tf.float32, shape=[None, output_size], name='y_')
        self.phase = tf.placeholder(tf.bool, name='phase')
        self.learning_rate = learning_rate
        self.img_width = img_width
        self.model_path = model_path
        self.logger = Logger()
        self.cross_entropy = None
        self.update_ops = None
        self.train_step = None
        self.accuracy = None
        self.sess = tf.InteractiveSession()

    @staticmethod
    def __weight_variable(shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    @staticmethod
    def __bias_variable(shape, name):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    @staticmethod
    def __conv2d(x, W):
        # [1, x, y, 1]
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def __max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    def format_input(self, channel=1):
        x_image = tf.reshape(self.x, [-1, self.img_width, self.img_width, channel])
        tf.summary.image('input', x_image, 3)
        return x_image

    def conv(self, input_data, input_axis, kernel_size, feature_map_num, name='conv'):
        self.conv_layer += 1
        with tf.name_scope(name):
            W_conv = self.__weight_variable([kernel_size, kernel_size, input_axis, feature_map_num], 'W')
            b_conv = self.__bias_variable([feature_map_num], 'b')
            # x_image = tf.reshape(x, [-1, img_width, img_width, 1])
            h_conv = self.__conv2d(input_data, W_conv) + b_conv
            h_conv_bn = tf.contrib.layers.batch_norm(h_conv)
            h_conv_relu = tf.nn.relu(h_conv_bn)
            h_pool = self.__max_pool_2x2(h_conv_relu)

            h_pool_bn = tf.contrib.layers.batch_norm(h_pool)

            tf.summary.histogram("w_conv", W_conv)
            tf.summary.histogram("b_conv", b_conv)
            tf.summary.histogram("a_conv", h_conv_relu)
            return h_pool_bn

    def fc(self, input_data, input_size, fully_connect_size, name='fc'):
        self.fc_layer += 1
        with tf.name_scope(name):
            W_fc = self.__weight_variable([input_size, fully_connect_size], 'W')
            b_fc = self.__bias_variable([fully_connect_size], 'b')
            # h_pool_bn_flat = tf.reshape(h_pool3_bn, [-1, int(flat_size)])
            h_fc = tf.matmul(input_data, W_fc) + b_fc
            h_fc_bn = tf.contrib.layers.batch_norm(h_fc)
            h_fc_relu = tf.nn.relu(h_fc_bn)
            h_fc_relu_bn = tf.contrib.layers.batch_norm(h_fc_relu)
            self.keep_prob = tf.placeholder("float")
            h_fc_drop = tf.nn.dropout(h_fc_relu_bn, self.keep_prob)
            h_fc1_drop_bn = tf.contrib.layers.batch_norm(h_fc_drop)

            tf.summary.histogram("w_fc", W_fc)
            tf.summary.histogram("b_fc", b_fc)
            tf.summary.histogram("a_fc", h_fc_relu)
            return h_fc1_drop_bn

    def output_layer(self, input_data, input_size, output_size, name='output'):
        with tf.name_scope(name):
            W_fc = self.__weight_variable([input_size, output_size], 'W_fc')
            b_fc = self.__bias_variable([output_size], 'b_fc')
            # h_pool_bn_flat = tf.reshape(h_pool3_bn, [-1, int(flat_size)])
            h_fc = tf.matmul(input_data, W_fc) + b_fc
            self.y_conv = tf.nn.softmax(h_fc)
            self.output_num = output_size
            return self.y_conv

    def flat(self, input, input_size):
        divide = 2 ** self.conv_layer
        flat_size = int(self.img_width / divide) ** 2 * input_size if int(
            self.img_width / divide) == self.img_width / divide else (int(
            self.img_width / divide) + 1) ** 2 * input_size
        input_flat = tf.reshape(input, [-1, int(flat_size)])
        return input_flat, flat_size

    def train(self, is_test=False, is_load=False, summary_file='tmp/1'):
        self.cross_entropy = -tf.reduce_sum(self.y_ * tf.log(self.y_conv + 1e-8))
        tf.summary.scalar('cross_entropy', self.cross_entropy)

        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy)
        # train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))

        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        tf.summary.scalar('accuracy', self.accuracy)

        merged_summary, writer = self.summary(file=summary_file)

        saver = tf.train.Saver()
        if is_load:
            saver.restore(self.sess, self.model_path)
        else:
            self.sess.run(tf.global_variables_initializer())
        # merged_summary_op = tf.summary.merge_all()
        # summary_writer = tf.summary.FileWriter('/tmp/mnist_logs', self.sess.graph)

        print(self.output_num)
        j = 0
        for i in range(0, 5):
            start = time.clock()
            train_accuracy = []
            train_loss = []
            for batch in train.get_batches(64):
                j += 1
                if is_test:
                    print(np.shape(batch[0]), np.shape(batch[1]))
                    print(type(batch[0]), type(batch[1]))
                    print(batch[0])
                    print(batch[1])
                    abc = input('abc')
                self.train_batch(i, batch, train_accuracy, train_loss, 0.8)
                s = self.sess.run(merged_summary,
                                  feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.9, self.phase: 1})
                writer.add_summary(s, j)
            self.evaluate(i)
            end = time.clock()
            print("step %d, training accuracy %g, training loss %g" % (
                i, sum(train_accuracy) / len(train_accuracy), sum(train_loss) / len(train_loss)))
            print("running time: %s" % str((end - start) / 3))
            saver.save(self.sess, self.model_path)
        self.test()

    def train_batch(self, step, batch, train_accuracy, train_loss, keep_prob):
        train_accuracy.append(self.accuracy.eval(feed_dict={
            self.x: batch[0], self.y_: batch[1], self.keep_prob: keep_prob, self.phase: 1}))
        train_loss.append(self.cross_entropy.eval(feed_dict={
            self.x: batch[0], self.y_: batch[1], self.keep_prob: keep_prob, self.phase: 1}))
        self.logger.info("step %d, training accuracy %g, training loss %g"
                         % (step, sum(train_accuracy) / len(train_accuracy), sum(train_loss) / len(train_loss)))
        self.train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.9, self.phase: 1})

    def evaluate(self, step, merged_summary=None, writer=None):
        validate_data = get_validate_data()
        test_accuracy = self.accuracy.eval(feed_dict={
            self.x: validate_data[0], self.y_: validate_data[1], self.keep_prob: 1.0, self.phase: 0})
        test_loss = self.cross_entropy.eval(feed_dict={
            self.x: validate_data[0], self.y_: validate_data[1], self.keep_prob: 1.0, self.phase: 0})

        """if merged_summary is not None:
            s = self.sess.run(merged_summary,
                              feed_dict={self.x: validate_data[0], self.y_: validate_data[1], self.keep_prob: 1.0,
                                      self.phase: 1})
            writer.add_summary(s, step)
        """
        self.logger.info("step %d, validate set: accuracy %g, loss %g" % (step, test_accuracy, test_loss))
        print("step %d, validate set: accuracy %g, loss %g" % (step, test_accuracy, test_loss))

    def test(self):
        test_data = get_test_data()
        test_accuracy = self.accuracy.eval(feed_dict={
            self.x: test_data[0], self.y_: test_data[1], self.keep_prob: 1.0, self.phase: 0})
        test_loss = self.cross_entropy.eval(feed_dict={
            self.x: test_data[0], self.y_: test_data[1], self.keep_prob: 1.0, self.phase: 0})
        self.logger.info("test set: accuracy %g, loss %g" % (test_accuracy, test_loss))
        self.logger.info('end')
        print("test set: accuracy %g, loss %g" % (test_accuracy, test_loss))

    def summary(self, file='tmp/mnist_demo/3'):
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(file)
        writer.add_graph(self.sess.graph)
        return merged_summary, writer
