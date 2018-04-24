# -*- coding: utf-8 -*-
import logging

class Logger:
	def __init__(self, file='journal'):
		self.dir_path = '../logs'
		logging.basicConfig(
                   filename = self.dir_path + '/' + file + '.log',
                   filemode = 'a',
                   # stream = sys.stdout,
                   format = '%(asctime)s [%(levelname)s] %(message)s',
                   datefmt = '%m/%d/%Y %I:%M:%S',
                   level = logging.DEBUG
                   )
		self.root_logger = logging.getLogger(file)
		self.root_logger.info('\n\n\n')
		# self.info('文件关联成功')

	def info(self, str):
		self.root_logger.info(str)

	def debug(self, str):
		self.root_logger.debug(str)

	def error(self, str):
		self.root_logger.error(str)


if __name__ == '__main__':
	logger = Logger()
	logger.info('info')
	logger.debug('debug')
	logger.error('shit')