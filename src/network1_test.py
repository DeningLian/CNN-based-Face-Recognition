# -*- coding: utf-8 -*-
# network1_test.py
# import mnist_loader
# training_data, validation_data, test_data = mnist_loader.load_data_wrapper()


import input_data
import network
import numpy as np






if __name__ == '__main__':
	# face_num, face_name_list, face_train_data_array = extract_face_train(
	# 	xml_path, img_path_train, img_width)
	data, num = input_data.read_data_sets(img_width=28)
	# print num
	# print np.shape(data[0])
	# print np.shape(data[1])
	# print np.shape(data[2])
	# print data[1][1]
	net = network.Network([28*28, 100, num])
	net.SGD(data[0][0], data[0][1], 300, 10, 0.3, data[2][0], data[2][1])
