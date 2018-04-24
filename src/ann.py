# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from src.input_data import *
from src.logger import *
from src.input import train
from src.input import test
from numpy import shape
# 加载数据
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
img_width = 28
data, num = read_data_sets(img_width=img_width)

# x 不是一个特定的值，而是一个占位符placeholder
# 我们在TensorFlow运行计算时输入这个值，也就是说，x 是输入值
# [None, 784] 表示所有的输入图像， None 表示输入图像的个数是可以变的
x = tf.placeholder("float", shape=[None, img_width * img_width])
# 为了计算交叉熵，我们首先需要添加一个新的占位符用于输入正确值：
y_ = tf.placeholder("float", [None,num])

# Variable 代表可修改的张量，存在在TensorFlow的用于描述交互性操作的图中
# 它们可以用于计算输入值，也可以在计算中被修改
# 对于各种机器学习应用，一般都会有模型参数，可以用Variable表示
W = tf.Variable(tf.zeros([img_width*img_width,num]))
b = tf.Variable(tf.zeros([num]))

# 模型定义
# tf.matmul(X, W)表示矩阵乘法
# tf.nn.softmax 表示 softmax 函数
y = tf.nn.softmax(tf.matmul(x,W) + b)


# 其实，在机器学习，我们通常定义指标来表示一个模型是坏的，这个指标称为成本（cost）或损失（loss），然后尽量最小化这个指标。但是，这两种方式是相同的。
# 一个非常常见的，非常漂亮的成本函数是“交叉熵”（cross-entropy）。交叉熵产生于信息论里面的信息压缩编码技术，但是它后来演变成为从博弈论到机器学习等其他领域里的重要技术手段。


# 计算交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# 在这里，我们要求TensorFlow用梯度下降算法（gradient descent algorithm）以0.01的学习速率最小化交叉熵。
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 现在，我们已经设置好了我们的模型。在运行计算之前，我们需要添加一个操作来初始化我们创建的变量：
init = tf.initialize_all_variables()

# 现在我们可以在一个Session里面启动我们的模型，并且初始化变量：
sess = tf.Session()
sess.run(init)

# tf.argmax 是一个非常有用的函数，它能给出某个tensor对象在某一维上的其数据最大值所在的索引值。
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# 这行代码会给我们一组布尔值。为了确定正确预测项的比例，我们可以把布尔值转换成浮点数，然后取平均值。
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 然后开始训练模型，这里我们让模型循环训练1000次！
test = False
validate_data = get_validate_data()
for i in range(1000):
	for batch in train.get_batches(10):
		# batch_xs, batch_ys = mnist.train.next_batch(10)
		if test:
			print(type(batch[0]), type(batch[1]))
			print(shape(batch[0]), shape(batch[1]))
			print(batch[0])
			print(batch[1])
			abc = eval(input('shit'))
		sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})
	if i % 5 == 0:
		# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
		# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		with sess.as_default():
			print(accuracy.eval(feed_dict={x: validate_data[0], y_: validate_data[1]}))
# 该循环的每个步骤中，我们都会随机抓取训练数据中的100个批处理数据点，然后我们用这些数据点作为参数替换之前的占位符来运行train_step。
# 使用一小部分的随机数据来进行训练被称为随机训练（stochastic training）- 在这里更确切的说是随机梯度下降训练。


# print correct_prediction.eval(session=sess)
# print correct_prediction
test_data = get_test_data()
with sess.as_default():
	print(accuracy.eval(feed_dict={x: test_data[0], y_: test_data[1]}))
# print type(sess.run(tf.constant([1,2,3])))



