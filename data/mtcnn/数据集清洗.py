# coding: utf-8
import os
import cv2
import shutil
import random
from main import *

file_dir = '../full'
person_names = os.listdir(file_dir)
random.shuffle(person_names)
os.system('find ./../. -name ".DS_Store" | xargs rm -rf')
for person_name in person_names:
	face_names = os.listdir(file_dir + '/' + person_name)
	random.shuffle(face_names)
	for face_name in face_names:
		try:
			img = cv2.imread(file_dir + '/' + person_name + '/' + face_name)
			cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			if not is_have_face(file_dir + '/' + person_name + '/' + face_name):
				os.remove(file_dir + '/' + person_name + '/' + face_name)
				print('remove ' + face_name + ' ' + person_name)
		except:
			os.remove(file_dir + '/' + person_name + '/' + face_name)
	shutil.copytree(file_dir + '/' + person_name, '../complete/' + person_name)
	shutil.rmtree(file_dir + '/' + person_name)

# os.system('python ./mtcnn/main.py')