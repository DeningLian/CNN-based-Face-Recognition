# -*- coding: utf-8 -*-
"""将该目录下所有的DS_Store文件清除"""
import os
import logger

def clean():
	os.system('find ./../. -name ".DS_Store" | xargs rm -rf')
	print('.DS_Store 文件清除成功')

if __name__ == '__main__':
	# handler = logger.__doc__
	handler1 = logging.FileHandler('../logs/test1.log')
	root_logger = logging.getLogger()
	root_logger.addHandler(handler)
	root_logger.addHandler(handler1)
	print(root_logger.level)
	root_logger.level = logging.NOTSET
	root_logger.info('shit')
	print(root_logger.handlers)