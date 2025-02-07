## AVMNIST 데이터셋 중, MNIST 부분을 처리해 준다!!

import torch.utils.data
from torch.utils.data import DataLoader		# 보통 데이터 불러오는 건 얘 역할!!
import torchvision.transforms as transforms		# 데이터 변환
import torch.nn.functional as F


import numpy as np
import pandas as pd
import os
import cv2 
import random
# import scipy.io as scio
from PIL import Image
import math

class MNIST(torch.utils.data.Dataset):
	""" UTD-MHAD dataset """
	def __init__(self, root, per_class_num=105,train=True):

		self.root = root
		self.train = train
		self.per_class_num = per_class_num

		if self.train:
			self.train_list = self.get_train_list(self.root,self.per_class_num)
		else:
			self.test_list = self.get_test_list(self.root)
	
	# train 데이터 로드하기
	def get_train_list(self, root, per_class_num):
		"""
		Args:
			root (string) - The root path image dataset
		"""
		tr_root = os.path.join(root+'train/')
		tr_number_list = sorted(os.listdir(tr_root))
		train_list = list() # image for training 
		for i in tr_number_list:
			images = sorted(os.listdir(os.path.join(tr_root+i)))
			for j in range(per_class_num):
				path = os.path.join(tr_root+i+'/'+images[j])
				train_list.append(path)

		return train_list

	# test 데이터 로드하기
	def get_test_list(self, root):
		te_root = os.path.join(root+'test/')
		te_number_list = sorted(os.listdir(te_root))
		test_list = list() # image for training 
		for i in te_number_list:
			images = sorted(os.listdir(os.path.join(te_root+i)))
			# for j in range(45):
			for j in range(len(images)):
				path = os.path.join(te_root+i+'/'+images[j])
				test_list.append(path)

		return test_list

	# 데이터셋 크기 변환(train, test 여부에 따라)
	def get_length(self):

		if self.train:
			length = len(self.train_list)
		else:
			length = len(self.test_list)

		return length 

	def __len__(self):
		""" Returns size of the dataset
		returns:
			int - number of samples in the dataset
		"""
		return self.get_length()

	def __getitem__(self, index):
		""" get image and label  """
		if self.train:
			image_path = self.train_list[index]
			image_label = int(image_path.split('/')[-2])		# 전체 경로 중, 클래스 라벨을 선별 ㄱㄱ
		else:
			image_path = self.test_list[index]
			image_label = int(image_path.split('/')[-2])
		transformations = transforms.Compose([
											  transforms.ToTensor(),
											  transforms.Normalize([0.5], [0.5])])
		img = Image.open(image_path)
		# print(image_label)
		im = transformations(img)	# 이미지를 정규화된 텐서로 변환!!
		label = torch.tensor(image_label).long()
		# label = torch.zeros(10).long()
		# label[image_label] = 1
		return im, label

if __name__ == '__main__':
	from PIL import Image
	import torch
	# dir = '/home/halima/AVmnist/data/'
	dir = '/home/etri01/jy/harmonicnas/HNAS_AVMNIST/soundmnist'
	root = dir+'mnist/'
	dataset = MNIST(root,per_class_num=105, train=False)		# train=False 이므로, test 데이터 폴더에서 가져옴!
	# im, label =dataset[3]
	# # im.show()
	print(len(dataset))
