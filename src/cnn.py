# -*- coding: utf-8 -*-

from cnn_class import CNN
from input_data import *
from logger import *
import os

os.system('find ./../. -name ".DS_Store" | xargs rm -rf')
print('.DS_Store 文件清除成功')

E = [0.001]
dataset = ['door']
for e in E:
    img_width = 16
    kernel_size1 = 5
    kernel_size2 = 3
    kernel_size3 = 3
    feature_map1_num = 32
    feature_map2_num = 64
    feature_map3_num = 128
    fully_connect_size = 512
    model_path = 'models/net_32_4.ckpt'
    data, num = read_data_sets(path='../data/' + dataset[0], is_whitening=False, img_width=img_width)
    img_width = int(np.sqrt(data[0][0].shape[1]))

    logger = Logger()
    logger.info('start\nimg_width=%d,kernel_size=%d,feature_map1_num=%d,feature_map2_num=%d,fully_connect_size=%d'
                % (img_width, kernel_size1, feature_map1_num, feature_map2_num, fully_connect_size))

    cnn = CNN(learning_rate=e, img_width=img_width, input_size=img_width ** 2, output_size=num)

    x_image = cnn.format_input()

    conv_output1 = cnn.conv(x_image, 1, kernel_size1, feature_map1_num, 'conv1')
    conv_output2 = cnn.conv(conv_output1, feature_map1_num, kernel_size2, feature_map2_num, 'conv2')
    conv_output3 = cnn.conv(conv_output2, feature_map2_num, kernel_size3, feature_map3_num, 'conv3')

    conv_output3_flat, flat_size = cnn.flat(conv_output3, feature_map3_num)

    fc_output1 = cnn.fc(conv_output3_flat, flat_size, fully_connect_size, 'fc1')
    y_conv = cnn.output_layer(fc_output1, fully_connect_size, num, 'fc2')

    cnn.train(is_summary=False, summary_file='../tmp/test/' + dataset[0], epoch=5, is_load=False)
