# -*- coding: utf-8 -*-
# from data_loader import *

import numpy as np

from input import data_loader, train, validate, test
from logger import Logger


def wightening_train(data):
    X = data
    mean = np.mean(X, axis=0)
    X -= mean  # 减去均值，使得以0为中心

    cov = np.dot(X.T, X) / X.shape[0]  # 计算协方差矩阵
    U, S, V = np.linalg.svd(cov)  # 矩阵的奇异值分解

    # 提取前k个奇异值
    s = np.sum(S)
    _s = 0
    for i in range(0, X.shape[1]):
        _s += S[i]
        if _s >= s * 0.99:
            k = i
            sqrt = np.sqrt(k)
            # log = np.log2(k)
            while int(sqrt) != sqrt:
                k += 1
                sqrt = np.sqrt(k)
            break

    S, U = S[0:k], U[0:k]
    Xrot = np.dot(X, U.T)
    Xwhite = Xrot / np.sqrt(S + 1e-5)
    # print(k)
    return Xwhite, S, U


def wightening_validate(data, S, U):
    X = data
    Xrot = np.dot(X, U.T)
    Xwhite = Xrot / np.sqrt(S + 1e-5)
    return Xwhite


def read_data_sets(path='../data/faces_dir', is_whitening=False, img_width=64, training_ratio=0.7,
                   validation_ratio=0.2, testing_ratio=0.1):
    data, num = data_loader.load(path, img_width, training_ratio,
                                 validation_ratio, testing_ratio)
    if is_whitening:
        data[0][0], S, U = wightening_train(data[0][0])
        data[1][0] = wightening_validate(data[1][0], S, U)
        data[2][0] = wightening_validate(data[2][0], S, U)
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
    # print(np.sqrt(9))
    # a = 3.0
    # print(int(a) == a)
    # print train.data.__len__()
    # batch = train.get_batches(4)[0]

    """validate_data = get_validate_data()
    datas = validate_data[0]
    labels = validate_data[1]
    print((train.data.shape))
    print((train.data[0]))

    # print train.data[1]
    # print train.label
    print((labels.shape))
    print((test.data.shape))"""
# for label in labels:
# 	print label.shape
# for batch in train.get_batches(4):
# 	print batch[1]
