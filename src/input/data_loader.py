# -*- coding: utf-8 -*-
import os
import sys
import random
import cv2
import numpy
import logging
import numpy as np
# from clean_DS_Store import *
from src.clean_DS_Store import clean

person_num = 0

def load_img_from_dir(dir_path, label, img_width):
	"""从目录中加载人脸图片和标签
	   对图片进行左右翻转，生成新的训练数据
	   返回[图片，标签]
	   其中图片为二维数组，标签为one-hot向量
		
	Arguments:
		dir_path {string} -- 人脸文件夹路径

	Returns:
		list([图片, 标签])，其中图片为二维数组，标签为one-hot向量
	"""
	ret = []
	face_names = os.listdir(dir_path)
	for face_name in face_names:
		img = cv2.imread(dir_path+'/'+face_name)
		try:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			img = cv2.resize(img, (img_width, img_width))
			ret.append([img, label])
			# img = cv2.flip(img, 1)
			# ret.append([img, label])
		except:
			os.remove(dir_path+'/'+face_name)
	return ret


		


def encode_data(data):
	"""将图像数据和标签进行编码
	
	将图像数据压缩到一维，标签附加到图像数据后面
	
	Arguments:
		data {list(图像，标签)} -- [图像为二维数组，标签为一维数组]
	
	Returns:
		编码后的数据，为一个大的列表，列表中每一项均为图像和其标签数据
	"""
	code = []
	for img, label in data:
		code_per_img = []
		for row in img.tolist():
			code_per_img.extend(row)
		code_per_img = [[(pixel * 2)/float(255) - 1 for pixel in code_per_img]]
		# code_per_img = [[(pixel)/float(255) for pixel in code_per_img]]
		code_per_img.append(label)
		code.append(code_per_img)
	# print code
	return code

def load(dir_path='../data/faces', img_width=64, 
	training_ratio=0.6, validation_ratio=0.1, testing_ratio=0.3):

	global person_num
	person_names = os.listdir(dir_path)
	person_num = person_names.__len__()

	label = [0.0 for i in range(0, person_num)]
	label[0] = 1.0

	training_data, validation_data, testing_data = [], [], []
	training_label, validation_label, testing_label = [], [], []

	left, right = training_ratio, training_ratio + validation_ratio
	for person_name in person_names:
		person_data = load_img_from_dir(dir_path+'/'+person_name, label, img_width)
		person_data = encode_data(person_data)
		# for d in person_data:
		# 	print d
			# print '\n'
		# print person_data
		random.shuffle(person_data)
		data_num = person_data.__len__()
		# print int(data_num*left)
		training_data.extend([data[0] for data in 
			person_data[0:int(data_num*left)]])
		# print training_data[0].__len__()
		validation_data.extend([data[0] for data in 
			person_data[int(data_num*left):int(data_num*right)]])
		testing_data.extend([data[0] for data in 
			person_data[int(data_num*right):data_num]])
		training_label.extend([data[1] for data in 
			person_data[0:int(data_num*left)]])
		validation_label.extend([data[1] for data in 
			person_data[int(data_num*left):int(data_num*right)]])
		testing_label.extend([data[1] for data in 
			person_data[int(data_num*right):data_num]])
		# print training_data

		label = [label[i-1] for i in range(0, person_num)]
	# print '验证数据'
	# print validation_data
	# print validation_label
		# print training_data[0]
		# print validation_data[0]
		# print testing_data[0]
	
	return [[np.array(training_data), np.array(training_label)], 
	[np.array(validation_data), np.array(validation_label)], 
	[np.array(testing_data), np.array(testing_label)]], person_num

if __name__ == '__main__':
	clean()
	data, person_num = load('../../data/faces', 2)
	[train, validate, test] = data
	print(train[1])
	print(validate[1])
	print(test[1])
	# print data[0]
	print(person_num)