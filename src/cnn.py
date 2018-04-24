# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from input_data import *
from logger import *
from input import train
from input import test
from numpy import shape
from model import *
import os
import time

from src1.input_data import read_data_sets
from src1.logger import Logger

img_width = 64
kernel_size = 5
# kernel_size_small = 3
feature_map1_num = 32
feature_map2_num = 64
# feature_map3_num = 128
fully_connect_size = 512
model_path = 'models/net_32_4.ckpt'

data, num = read_data_sets(img_width=img_width)
logger = Logger()
logger.info('start\nimg_width=%d,kernel_size=%d,feature_map1_num=%d,feature_map2_num=%d,fully_connect_size=%d' 
  % (img_width, kernel_size, feature_map1_num, feature_map2_num, fully_connect_size))

def CNN(is_test=False, is_load=False):

  x = tf.placeholder(tf.float32, shape=[None, img_width ** 2], name='x')
  y_ = tf.placeholder(tf.float32, shape=[None, num], name='y_')

  def weight_variable(shape, name):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial, name=name)

  def bias_variable(shape, name):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial, name=name)

  def conv2d(x, W):
      # [1, x, y, 1]
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

  def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')


  W_conv1 = weight_variable([kernel_size, kernel_size, 1, feature_map1_num], 'W_conv1')
  b_conv1 = bias_variable([feature_map1_num], 'b_conv1')
  x_image = tf.reshape(x, [-1, img_width, img_width, 1])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)

  W_conv2 = weight_variable([kernel_size, kernel_size, feature_map1_num, feature_map2_num], name='W_conv2')
  b_conv2 = bias_variable([feature_map2_num], name='b_conv2')
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)

  # W_conv3 = weight_variable([kernel_size_small, kernel_size_small, feature_map2_num, feature_map3_num], name='W_conv3')
  # b_conv3 = bias_variable([feature_map3_num], name='b_conv3')
  # h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
  # h_pool3 = max_pool_2x2(h_conv3)

  W_fc1 = weight_variable([int(img_width * img_width  * feature_map2_num / 16), fully_connect_size], 'W_fc1')
  b_fc1 = bias_variable([fully_connect_size], 'b_fc1')
  h_pool2_flat = tf.reshape(h_pool2, [-1, int(img_width * img_width  * feature_map2_num / 16)])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  keep_prob = tf.placeholder("float")
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  W_fc2 = weight_variable([fully_connect_size, num], name='W_fc2')
  b_fc2 = bias_variable([num], name='b_fc2')

  y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


  """
  tf.add_to_collection(tf.GraphKeys.WEIGHTS, W_conv1)
  tf.add_to_collection(tf.GraphKeys.WEIGHTS, W_conv2)
  tf.add_to_collection(tf.GraphKeys.WEIGHTS, W_fc1)
  tf.add_to_collection(tf.GraphKeys.WEIGHTS, W_fc2)
  regularizer = tf.contrib.layers.l2_regularizer(scale=5.0/50000)
  reg_term = tf.contrib.layers.apply_regularization(regularizer)

  # loss = (tf.reduce_mean(
  #     tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=z_3)) +
  #     reg_term)

  cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv + 1e-8)) + reg_term
  """
  cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv + 1e-8))
  train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
  # sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
  sess = tf.InteractiveSession()
  # config = tf.ConfigProto()
  # config.gpu_options.allow_growth = True
  # sess = tf.Session(config=config)

  # config = tf.ConfigProto()
  # config.gpu_options.per_process_gpu_memory_fraction = 0.4
  # sess = tf.Session(config=config)

  saver = tf.train.Saver()
  if is_load:
    saver.restore(sess, model_path)
  else:
    sess.run(tf.global_variables_initializer())
  
  print(num)
  j = 0
  for i in range(0, 5000):
    start = time.clock()
    train_accuracy = []
    train_loss = []
    for batch in train.get_batches(10):
      j += 1
      if is_test:
        print(np.shape(batch[0]), np.shape(batch[1]))
        print(type(batch[0]), type(batch[1]))
        print(batch[0])
        print(batch[1])
        abc = input('abc')
      # if j % 10 == 0:
      train_accuracy.append(accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0}))
      train_loss.append(cross_entropy.eval(feed_dict={
          x: batch[0], y_: batch[1], keep_prob: 1.0}))
        # logger.info("step %d, training accuracy %g, training loss %g" 
        #   % (i, train_accuracy, train_loss))
        # print "step %d, training accuracy %g, training loss %g" % (i, train_accuracy, train_loss)
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.3})

    validate_data = get_validate_data()
    test_accuracy = accuracy.eval(feed_dict={
      x: validate_data[0], y_: validate_data[1], keep_prob: 1.0})
    test_loss = cross_entropy.eval(feed_dict={
      x: validate_data[0], y_: validate_data[1], keep_prob: 1.0})
    end = time.clock()
    logger.info("step %d, validate set: accuracy %g, loss %g" % (i, test_accuracy, test_loss))
    print("running time: %s" % (end-start))
    print("step %d, training accuracy %g, training loss %g" % (i, sum(train_accuracy)/len(train_accuracy), sum(train_loss)/len(train_loss)))
    print("step %d, validate set: accuracy %g, loss %g" % (i, test_accuracy, test_loss))

    saver.save(sess, model_path)


  test_data = get_test_data()
  test_accuracy = accuracy.eval(feed_dict={
      x: test_data[0], y_: test_data[1], keep_prob: 1.0})
  test_loss = cross_entropy.eval(feed_dict={
      x: test_data[0], y_: test_data[1], keep_prob: 1.0})
  logger.info("test set: accuracy %g, loss %g" % (test_accuracy, test_loss))
  logger.info('end')
  print("test set: accuracy %g, loss %g" % (test_accuracy, test_loss))



if __name__ == '__main__':
  CNN(is_load=False)