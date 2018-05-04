# -*- coding: utf-8 -*-
import cv2

img = cv2.imread('sigmoid.png')
print(img[0][0][0])
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img.shape)
for i in range(img.shape[0]):
	for j in range(img.shape[1]):
		# print(img[i][j])
		if img[i][j] <= 15:
			img[i][j] = 255
		# abc = input('shit')
# ret, img = cv2.threshold(img,200,255,cv2.THRESH_BINARY_INV)
cv2.imwrite("cc.png", img)
