# -*- coding: utf-8 -*-
# from data_loader import *

import numpy as np

from src.clean_DS_Store import clean
from src.input import data_loader, train, validate, test
from src.logger import Logger


def read_data_sets(path='../data/faces', img_width=64, training_ratio=0.7,
	validation_ratio=0.2, testing_ratio=0.1):
	clean()
	data, num = data_loader.load(path, img_width, training_ratio,
		validation_ratio, testing_ratio)
	logger = Logger()
	logger.info('data set read completed')
	[train.data, train.label] = data[0]
	[validate.data, validate.label] = data[1]
	[test.data, test.label] = data[2]
	logger.info('train set, validate set, test set allocated completed')
	return data, num

def get_train_data():
	return [np.array(train.data), np.array(train.label)]

def get_validate_data():
	return [np.array(validate.data), np.array(validate.label)]

def get_test_data():
	return [np.array(test.data), np.array(test.label)]

if __name__ == '__main__':
	read_data_sets(img_width=2)
	# print train.data.__len__()
	# batch = train.get_batches(4)[0]
	
	validate_data = get_validate_data()
	datas = validate_data[0]
	labels = validate_data[1]
	print((train.data.shape))
	print((train.data[0]))
	
	# print train.data[1]
	# print train.label
	print((labels.shape))
	print((test.data.shape))
	# for label in labels:
	# 	print label.shape
	# for batch in train.get_batches(4):
	# 	print batch[1]
