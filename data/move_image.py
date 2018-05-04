# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def show(X):
	x = [i for i in X.T[0]]
	y = [j for j in X.T[1]]
	plt.figure(1) # 新建绘画窗口，独立显示绘画的图片
	# plt.subplot(1, 1, 1) # 行，列，活跃区。活跃区的意思就是绘画区
	# x = np.linspace(0, np.pi * 2, 50)  # 在 0 到 2pi 之间，均匀产生 50 点的数组

	plt.plot(x, y, 'g.', label='y-x')
	# plt.plot(x, np.cos(x), 'm', label='cos(x)')
	plt.legend() # 展示图例
	plt.xlabel('x') # 给 x 轴添加标签
	# plt.ylabel('y') # 给 y 轴添加标签
	plt.title('figure') # 添加图形标题
	plt.show()

	# 直方图
	# x = np.random.randn(1000)
	# plt.hist(x, 50)
	# plt.show()

def whiten(X,fudge=1E-18):

   # the matrix X should be observations-by-components
   X = X - np.mean(X)
   # get the covariance matrix
   Xcov = np.dot(X.T,X)

   # eigenvalue decomposition of the covariance matrix
   d, V = np.linalg.eigh(Xcov)

   print(d, V)
   d = d[0:1]
   V = V[0:1]
   print(d, V)
   # a fudge factor can be used so that eigenvectors associated with
   # small eigenvalues do not get overamplified.
   D = np.diag(1. / np.sqrt(d+fudge))

   # whitening matrix
   W = np.dot(np.dot(V.T, D), V)

   # multiply by the whitening matrix
   X_white = np.dot(X, W)

   return X_white, W

# X= np.random.random(size=(80, 2))

x = np.linspace(0, 10, 10)
bias = []
for i in range(0, x.shape[0]):
	bias.append(3 * (np.random.random_sample()-0.5))
y = x + np.array(bias)
X = np.array([[i, j] for i, j in zip(x, y)])
whiten(X)
# show(X)
"""
X -= np.mean(X, axis = 0) # 减去均值，使得以0为中心
show(X)
print(X.shape)
cov = np.dot(X.T, X) / X.shape[0] #计算协方差矩阵
print(cov.shape)
U,S,V = np.linalg.svd(cov) #矩阵的奇异值分解
print(U.shape)
print(S.shape)
print(V.shape)

# print(X)
s = np.sum(S)
_s = 0
for i in range(0, X.shape[1]):
	_s += S[i]
	if _s >= s * 0.99:
		# S, U = S[0:i], U[0:i]
		break
print('abs')
print(U.shape)
print(S.shape)
Xrot = np.dot(X, U.T)
print(Xrot.shape)
# print(Xrot)
Xwhite = Xrot / np.sqrt(S + 1e-5)
print(Xwhite.shape)
show(Xwhite)
"""