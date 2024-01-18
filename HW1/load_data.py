import numpy as np
import os
import random
import torch
from torch.utils.data import IterableDataset
import time
import imageio

class DataGenerator(IterableDataset):
	"""
	Data Generator generating batches of Omniglot data.
	A "class" is considered as a class of omniglot digit
	"""

	def __init__(self,
				 num_classes,
				 num_samples_per_class,
				 batch_type,
				 config={},
				 device=torch.device('cpu'),
				 cache=True):
		"""
		:param num_classes:
			N: N-way
			number of classes for classification
		:param num_samples_per_class:
			K+1: K-shot
			number of samples per class
		:param batch_type:
			"train"/"val"/"test"
		:param config:
		:param device:
		:param cache:
		"""
		self.num_samples_per_class = num_samples_per_class
		self.num_classes = num_classes

		data_folder = config.get('data_folder', "./omniglot_resized")
		self.img_size = config.get("img_size", (28, 28))
		self.dim_input = np.prod(self.img_size)
		self.dim_output = self.num_classes

		char_folders = [
			os.path.join(data_folder, family, character)
			for family in os.listdir(data_folder)   # each family is a task
				if os.path.isdir(os.path.join(data_folder, family))
			for character in os.listdir(os.path.join(data_folder, family))
				if os.path.isdir(os.path.join(data_folder, family, character))
		]
		# one task is to classify a certain combinations of chars


		random.seed(1)
		random.shuffle(char_folders)
		num_val = 100
		num_train = 1100
		self.metatrain_char_folders = char_folders[:num_train]
		self.metaval_char_folders = char_folders[num_train:num_train + num_val]
		self.metatest_char_folders = char_folders[num_train + num_val:]
		self.device = 'device'
		self.image_caching = cache
		self.stored_images = {}

		if batch_type == 'train':
			self.folders = self.metatrain_char_folders
		elif batch_type == 'val':
			self.folders = self.metaval_char_folders
		else:
			self.folders = self.metatest_char_folders

	def image_file_to_array(self, filename, dim_input):
		"""
		takes an image path and returns numpy array
		:param filename:
		:param dim_input:
		:return:
		"""
		if self.image_caching and (filename in self.stored_images):
			return self.stored_images[filename]
		image = imageio.v2.imread(filename)  # misc.imread(filename)
		image = image.reshape([dim_input])
		image = image.astype(np.float32)  # / 255.0
		image = 1.0 - image
		if self.image_caching:
			self.stored_images[filename] = image
		return image

	def _sample(self):
		"""
		Samples a batch for training, validation or testing set
		:return:
			imgs: [K+1, N, 784]
				img[:-1] is the support set, img[-1] is the query set
			labels: [K+1, N, N]
				labels[:-1] is the support set, labels[-1] is the query set
		"""
		imgs = []
		chars = np.random.choice(self.folders, self.num_classes, replace=False)


		for idx, char in enumerate(chars): # num_classes iterations

			# randomly reading K+1 samples from this char
			paths = os.listdir(char)
			paths = [os.path.join(char, str(p)[2:-1]) for p in paths]
			paths = np.random.choice(paths, self.num_samples_per_class, replace=False) # K+1 samples

			img_class = np.stack([
				self.image_file_to_array(p, dim_input=self.dim_input) for p in paths
			])
			imgs.append(img_class)
		imgs = np.stack(imgs, axis=1) # [K+1, N, 784]
		labels = np.repeat(
			np.eye(self.num_classes)[None, ...],
			repeats=[self.num_samples_per_class],
			axis=0
		)

		# shuffle the examples in query set
		query_shuffle_idx = np.random.permutation(self.num_classes)
		# print(query_shuffle_idx)
		imgs[-1] = imgs[-1][query_shuffle_idx, :]
		labels[-1] = labels[-1][query_shuffle_idx, :]

		return imgs, labels

	def __iter__(self):
		while True:
			yield self._sample()


if __name__ == '__main__':
	d = DataGenerator(4, 3,  'val')
	imgs, labels = d._sample()
	print(labels)