## 이번엔, MNIST 이미지와 오디오 데이터를 함께 처리..!!
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F

from ..data_transform import get_data_transform

import matplotlib.pyplot as plt
import librosa.display

import numpy as np
import pandas as pd
import os
import cv2 
import random
# import scipy.io as scio
from PIL import Image
import math
import sys
from .wav2mfcc import wav2mfcc



class SoundMNIST(torch.utils.data.Dataset):
	"""  soundmnist dataset """

	def __init__(self, img_root, sound_root, per_class_num=105,train=True, aug="default"):

		self.img_root = img_root
		self.sound_root = sound_root

		self.train = train
		self.per_class_num = per_class_num

		self.augment_type = aug

		# 이미지 데이터 증강 관련 인사이트 추가
		self.image_transform = get_data_transform(is_training=self.train, augment=self.augment_type)

		# train 일때랑 test 일때랑 다르게 호출!!
		if self.train:
			self.train_img_list = self.get_image_train_list(self.img_root, self.per_class_num)
			self.train_sound_list = self.get_sound_train_list(self.sound_root, self.per_class_num)
			# print(self.train_list)
		else:
			self.test_img_list = self.get_image_test_list(self.img_root, self.per_class_num)
			self.test_sound_list = self.get_sound_test_list(self.sound_root, self.per_class_num)

	def get_image_train_list(self, root, per_class_num):
		tr_root = os.path.join(root+'train/')
		tr_number_list = sorted(os.listdir(tr_root))
		train_img_list = list() 		# image for training 
		for i in tr_number_list:		# 각 클래스별 폴더에 대해
			images = sorted(os.listdir(os.path.join(tr_root+i)))
			for j in range(per_class_num):		# 해당 pr_class_num 만큼 가져와~
				path = os.path.join(tr_root+i+'/'+images[j])
				train_img_list.append(path)

		return train_img_list

	def get_image_test_list(self, root, per_class_num):
		te_root = os.path.join(root+'test/')
		te_number_list = sorted(os.listdir(te_root))
		test_img_list = list() # image for training 
		for i in te_number_list:
			images = sorted(os.listdir(os.path.join(te_root+i)))
			for j in range(len(images)-45, len(images)):		# train 데이터와 달리, 모든 클래스 균등하게 샘플링 X, 정해진 45개만 사용!!
			# for j in range(45):
				path = os.path.join(te_root+i+'/'+images[j])
				test_img_list.append(path)

		return test_img_list

	def get_sound_train_list(self, root, per_class_num):
		tr_root = os.path.join(root+'train/')
		tr_number_list = sorted(os.listdir(tr_root))
		train_sound_list = list() # image for training 
		for i in tr_number_list:
			jack_list = list()
			nico_list = list()
			theo_list = list()
			sounds = sorted(os.listdir(tr_root+ i))
			for j in sounds:
				if j.split('_')[1] == 'jackson':
					if len(jack_list) < int(per_class_num/3):
						jack_list.append(os.path.join(tr_root+i+'/'+j))
				elif j.split('_')[1] == 'nicolas':
					if len(nico_list) < int(per_class_num/3):
						nico_list.append(os.path.join(tr_root+i+'/'+j))
				else:
					if len(theo_list) < int(per_class_num/3):
						theo_list.append(os.path.join(tr_root+i+'/'+j))
			temp_list = jack_list + nico_list + theo_list
			train_sound_list += temp_list

		return train_sound_list

	def get_sound_test_list(self, root, per_class_num):
		te_root = os.path.join(root+'test/')
		te_number_list = sorted(os.listdir(te_root))
		test_sound_list = list() # image for training 
		for i in te_number_list:
			sounds = sorted(os.listdir(os.path.join(te_root+i)))
			for j in sounds:
				path = os.path.join(te_root+i+'/' + j)
				test_sound_list.append(path)

		return test_sound_list



	def get_length(self):
		
		if self.train:
			length = len(self.train_img_list)
		else:
			length = len(self.test_img_list)

		return length 


	def __len__(self):
		""" Returns size of the dataset
		returns:
			int - number of samples in the dataset
		"""
		return self.get_length()

	def __getitem__(self, index):
		""" get image and label  """
		"""이미지 & 오디오 데이터 로드하기"""
		if self.train:
			image_path = self.train_img_list[index]
			image_label = int(image_path.split('/')[-2])
			sound_path = self.train_sound_list[index]
			sound_label = int(sound_path.split('/')[-2])
		else:
			image_path = self.test_img_list[index]
			image_label = int(image_path.split('/')[-2])
			sound_path = self.test_sound_list[index]
			sound_label = int(sound_path.split('/')[-2])

		assert image_label == sound_label

		# transformations_img = transforms.Compose([transforms.ToTensor(),
		# 									  transforms.Normalize([0.5], [0.5])])

		apply_aug = self.train and (random.random() < 0.3)		# 증강 확률은 30%로!
		
		## 이미지 데이터 변환
		img = Image.open(image_path).convert("L")		# 원본 이미지 불러오기
		trans_img = self.image_transform(img) if apply_aug else transforms.ToTensor()(img)

		# ## 원본 오디오
		# origin_sound = np.asarray(wav2mfcc(sound_path, apply_aug=False))
		# origin_sound = torch.tensor(origin_sound, dtype=torch.float32).unsqueeze(0)
		# origin_sound = (origin_sound - origin_sound.mean()) / (origin_sound.std() + 1e-6)

		## 오디오 데이터 변환
		trans_sound = np.asarray(wav2mfcc(sound_path, apply_aug=apply_aug))	# 오디오 to MFCC 형태 변환!!
		trans_sound = torch.tensor(trans_sound, dtype=torch.float32).unsqueeze(0)
		trans_sound = (trans_sound - trans_sound.mean()) / (trans_sound.std() + 1e-6)

		# # 그 결과로, 이미지 + 오디오 MFCC 텐서 + 레이블 같이 반환 ㄱㄱ
		# im = transformations(img)
		# sound = transformations(sound)
		
		label = torch.tensor(image_label).long()
		# label = torch.zeros(10).long()
		# label[image_label] = 1
		return trans_img, trans_sound, label



if __name__ == '__main__':
	from PIL import Image
	import torch
	# dir = '/home/hbouzidi/hbouzidi/datasets/AVMNIST'
	dir = '/home/etri01/jy/harmonicnas/HNAS_AVMNIST/soundmnist/'
	img_root = dir+'mnist/'
	sound_root = dir+'sound_450/'


	## SoundMNIST 데이터셋 로드하기
	dataset = SoundMNIST(img_root, sound_root, per_class_num=30, train=True, aug='default')
	# im, sd, label = dataset[0]
	print(len(dataset))					# 450
	# print(sd.size)
	# print(len(dataset))
	loader = DataLoader(dataset, batch_size=8, shuffle= False)

	## 배치에서 하나의 샘플 가져오기
	batch = next(iter(loader))
	print(batch[0].shape)		# [batch_size, channel, height, width] => [8, 1, 28, 28]
	print(batch[1].shape)		# [batch_size, channel, n_mfcc, time_steps] => [8, 1, 20, 20]
	print(batch[0].view(-1, 784).shape) # => [8, 784]
	print(batch[1].view(-1, 400).shape) # => [8, 400]
	print(batch[2].shape)				# [8]

	## 랜덤으로 아무 이미지나 시각화 ㄱㄱ(증강 잘 되었는지?)
	idx = np.random.randint(len(dataset))
	img, sound, label = dataset[idx]

	# 이미지 시각화
	plt.imshow(img.squeeze(0), cmap="gray")  # 1채널 흑백 이미지
	plt.title(f"Label: {label.item()}")
	plt.axis("off")
	plt.show()

	## MFCC 스펙트로그램 시각화
	plt.subplot(1, 2, 2)
	sound_np = sound.squeeze(0).numpy()
	librosa.display.specshow(sound_np, x_axis="time", cmap="coolwarm")
	plt.colorbar(label="MFCC Coefficients")
	plt.title("MFCC Features")
	plt.xlabel("Time")
	plt.ylabel("MFCC Coefficients")
	plt.tight_layout()
	plt.show()

	# ##랜덤 샘플 선택
	# idx = np.random.randint(len(dataset))
	# original_img, transformed_img, original_mfcc, transformed_mfcc, label = dataset[idx]

	# # 원본 vs 변형 이미지
	# fig, ax = plt.subplots(1, 2, figsize=(10, 4))
	# ax[0].imshow(original_img, cmap='gray')
	# ax[0].set_title("Original Image")

	# ax[1].imshow(transformed_img.squeeze(0), cmap="gray")
	# ax[1].set_title("Transformed Img")
	# plt.show()

	# # 원본 vs 변형 MFCC
	# fig, ax = plt.subplots(1, 2, figsize=(12, 4))
	# librosa.display.specshow(original_mfcc.squeeze(0).numpy(), x_axis="time", cmap="coolwarm", ax=ax[0])	
	# ax[0].set_title("Original MFCC")
	
	# librosa.display.specshow(transformed_mfcc.squeeze(0).numpy(), x_axis="time", cmap="coolwarm", ax=ax[1])
	# ax[1].set_title("Transformed MFCC")
	# plt.show()
