import tensorflow as tf

model_path = '../models'

def save(sess, path, cnt=0):
	saver = tf.train.Saver()
	path = path +'/' + str(cnt) + '.ckpt'
	save_path = saver.save(sess, path)

def load(sess, path, cnt=0):
	saver = tf.train.Saver()
	path = path +'/' + str(cnt) + '.ckpt'
	saver.restore(sess, path)