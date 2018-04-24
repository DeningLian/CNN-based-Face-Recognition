# -*- coding: utf-8 -*-
import random
import numpy as np

data = []
label = []
batch = []
# batch_size = 0

def shuffle():
	global data, label, batch
	feature=[(_data, _label) for _data, _label in zip(data,label)]
	random.shuffle(feature)
	data = [_data for _data, _label in feature]
	label = [_label for _data, _label in feature]

def get_batches(batch_size=10):
	global data, label, batch
	shuffle()
	left, right = 0, 0
	right += batch_size
	while left < data.__len__():
		right = right if right < data.__len__() else data.__len__()
		batch = [np.array(data[left:right]), np.array(label[left:right])]
		yield batch
		left = right
		right += batch_size

def get_current_batch():
	return batch

if __name__ == '__main__':
	batch = get_batches()
	batch = get_batches()
	batch = get_batches()
	batch = get_batches()