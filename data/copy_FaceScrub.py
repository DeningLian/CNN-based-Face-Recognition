# -*- coding: utf-8 -*-
import os
import shutil

def copy():
	src = '/Users/seapatrol/ProgramData/full'
	dst = '/Users/seapatrol/ProgramData/人脸识别/data/full'

	person_names = os.listdir(src)
	for person_name in person_names:
		try:
			shutil.copytree(src + '/' + person_name + '/face', dst + '/' + person_name)
		except :
			pass

def minus():
	src = '/Users/seapatrol/ProgramData/人脸识别/data/faces'
	person_names = os.listdir(src)
	
	for person_name in person_names:
		i = 0
		face_names = os.listdir(src + '/' + person_name)
		for face_name in face_names:
			if i >= 10:
				os.remove(src + '/' + person_name + '/' + face_name)
			else:
				i += 1

if __name__ == '__main__':
	os.system('find ./../. -name ".DS_Store" | xargs rm -rf')
	copy()
	# minus()