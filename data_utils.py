import numpy as np
import pandas as pd
import scipy.sparse as sp

import torch.utils.data as data

def load_all_custom(fold_num,test_num=100, dataset=None):
	""" We load all the three file here to save time in each epoch. """

	total_data = pd.read_csv(f'./folds_data/processed_csv_folds/all_data.csv')
	total_data = total_data[['uid', 'sid']]
	total_data['uid'] = total_data['uid'].apply(lambda x : int(x))
	total_data['sid'] = total_data['sid'].apply(lambda x : int(x))
	user_num = total_data['uid'].max() + 1
	item_num = total_data['sid'].max() + 1

	train_data = pd.read_csv(f'./folds_data/processed_csv_folds/train_fold_'+str(fold_num)+'.csv')
	train_data = train_data[['uid', 'sid']]
	train_data_len = train_data.shape[0]

	return user_num, item_num, train_data_len

class BPRData(data.Dataset):
	def __init__(self, train_data_length):
		super(BPRData, self).__init__()
		self.train_data_length = train_data_length
		self.features_fill = None

	def get_data(self,fold_num, dataset, current_epoch):
		import pickle
		with open('./folds_data/train_samples_'+str(fold_num)+f'/train_samples_{current_epoch}', 'rb') as fp:
			b = pickle.load(fp)
			self.features_fill = b

	def __len__(self):
		return self.train_data_length

	def __getitem__(self, idx):
		features = self.features_fill
		if True:
			user = features[idx][0]
			pos = features[idx][1]
			neg = features[idx][2]
			return user, pos, neg