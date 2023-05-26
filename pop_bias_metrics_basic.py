import time
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn

device = "cuda" if torch.cuda.is_available() else "cpu"

def predictions_func(model_used,data, filter):

	data['uid'] = data['uid'].apply(lambda x : int(x))
	data['sid'] = data['sid'].apply(lambda x : int(x))

	filter_users = data.uid.value_counts()[data.uid.value_counts() > filter].index
	data = data[data.uid.isin(filter_users)] ; data = data.reset_index()[['uid', 'sid']]

	data_len = data.shape[0] ; frac = 50 ; frac_user_num = int(data_len/frac)
	model_used.eval() ; model_used.to(device)

	predictions_list = []
	for itr in range(frac+1):
		if itr == frac: tmp = data.iloc[frac_user_num*(itr):].values
		else: tmp = data.iloc[frac_user_num*itr:frac_user_num*(itr+1)].values
		user = tmp[:, 0] ; user = user.astype(np.int32) ; user = torch.from_numpy(user).to(device)
		item = tmp[:, 1] ; item = item.astype(np.int32) ; item = torch.from_numpy(item).to(device)
		predictions = model_used.forward_one_item(user, item)
		predictions_list += predictions.detach().to(device).tolist()
	data['pred'] = predictions_list

	return data

def pred_item_rank(model_here, test_data, sid_pop_total):

	data = predictions_func(model_here,test_data,1)

	user_item_rank = data.groupby('uid')['pred'].rank('average', ascending = False)
	data['user_item_rank'] = user_item_rank.values - 1

	user_count = data.groupby('uid')['pred'].count().reset_index()
	user_count.columns = ['uid', 'user_count']
	user_count['user_count'] = user_count['user_count'] - 1
	user_count_dict = dict(user_count.values)

	data['user_count'] = data['uid'].map(user_count_dict)
	data['user_item_rank2'] = data['user_item_rank'] / data['user_count']

	item_rank = data[['sid','user_item_rank2']].groupby('sid').mean().reset_index()
	item_rank.columns = ['sid', 'rank']

	sid_pop_dict = dict(sid_pop_total.values)
	item_rank['sid_pop_count'] = item_rank['sid'].map(sid_pop_dict)

	return item_rank

def pred_item_score(model_here, test_data, sid_pop_total):

	data = predictions_func(model_here,test_data,1)

	item_score = data[['sid','pred']].groupby('sid').mean().reset_index()

	sid_pop_dict = dict(sid_pop_total.values)
	item_score['sid_pop_count'] = item_score['sid'].map(sid_pop_dict)

	return item_score

def pred_item_stdscore(model_here, test_data, sid_pop_total):

	data = predictions_func(model_here,test_data,1)

	user_mean_dict = dict(data.groupby('uid')['pred'].mean().reset_index().values)
	user_std_dict = dict(data.groupby('sid')['pred'].std().reset_index().values)

	data['mean'] = data['uid'].map(user_mean_dict)
	data['std'] = data['uid'].map(user_std_dict)

	data['z'] = (data['pred'] - data['mean']) / data['std']
	item_z_score = data[['sid','z']].groupby('sid').mean().reset_index()

	sid_pop_dict = dict(sid_pop_total.values)
	item_z_score['sid_pop_count'] = item_z_score['sid'].map(sid_pop_dict)

	return item_z_score

def pred_item_rankdist(model_here, test_data, sid_pop_total):

	data = predictions_func(model_here,test_data,4)

	sid_pop_dict = dict(sid_pop_total.values)
	data['sid_pop_count'] = data['sid'].map(sid_pop_dict)

	user_item_pop_rank = data.groupby('uid')['sid_pop_count'].rank('average', ascending = False)
	user_item_pop_rank = user_item_pop_rank - 1
	data['user_item_pop_rank'] = user_item_pop_rank

	user_item_score_rank = data.groupby('uid')['pred'].rank('average', ascending = False)
	user_item_score_rank = user_item_score_rank # 코드 오류
	data['user_item_score_rank'] = user_item_score_rank

	user_count = data.groupby('uid')['pred'].count().reset_index()
	user_count.columns = ['uid', 'user_count']
	user_count['user_count'] = user_count['user_count'] - 1
	user_count_dict = dict(user_count.values)

	data['user_count'] = data['uid'].map(user_count_dict)
	data['user_item_pop_rank2'] = data['user_item_pop_rank'] / data['user_count']
	data['user_item_score_rank2'] = data['user_item_score_rank'] / data['user_count']
	data = data.sort_values(['uid', 'user_item_score_rank2'], ascending = (True, True))
	item_rankdist = data.groupby('uid')['user_item_pop_rank2'].head(1)
	return item_rankdist

def pred_item_rankdist_modified(model_here, test_data, sid_pop_total):

	data = predictions_func(model_here,test_data,4)

	sid_pop_dict = dict(sid_pop_total.values)
	data['sid_pop_count'] = data['sid'].map(sid_pop_dict)

	user_item_pop_rank = data.groupby('uid')['sid_pop_count'].rank('average', ascending = False)
	user_item_pop_rank = user_item_pop_rank - 1
	data['user_item_pop_rank'] = user_item_pop_rank

	user_item_score_rank = data.groupby('uid')['pred'].rank('average', ascending = False)
	user_item_score_rank = user_item_score_rank - 1 # 코드 오류
	data['user_item_score_rank'] = user_item_score_rank

	user_count = data.groupby('uid')['pred'].count().reset_index()
	user_count.columns = ['uid', 'user_count']
	user_count['user_count'] = user_count['user_count'] - 1
	user_count_dict = dict(user_count.values)

	data['user_count'] = data['uid'].map(user_count_dict)
	data['user_item_pop_rank2'] = data['user_item_pop_rank'] / data['user_count']
	data['user_item_score_rank2'] = data['user_item_score_rank'] / data['user_count']

	data = data.sort_values(['uid', 'user_item_score_rank2'], ascending = (True, True))
	item_rankdist = data.groupby('uid')['user_item_pop_rank2'].head(1)
	return item_rankdist

def pred_item_rankdist2(model_here, test_data, sid_pop_total):

	data = predictions_func(model_here,test_data,4)

	sid_pop_dict = dict(sid_pop_total.values)
	data['sid_pop_count'] = data['sid'].map(sid_pop_dict)

	user_item_pop_rank = data.groupby('uid')['sid_pop_count'].rank('average', ascending = False)
	user_item_pop_rank = user_item_pop_rank - 1
	data['user_item_pop_rank'] = user_item_pop_rank

	user_item_score_rank = data.groupby('uid')['pred'].rank('average', ascending = False)
	user_item_score_rank = user_item_score_rank -1
	data['user_item_score_rank'] = user_item_score_rank

	user_count = data.groupby('uid')['pred'].count().reset_index()
	user_count.columns = ['uid', 'user_count']
	user_count['user_count'] = user_count['user_count'] - 1
	user_count_dict = dict(user_count.values)

	data['user_count'] = data['uid'].map(user_count_dict)
	data['user_item_pop_rank2'] = data['user_item_pop_rank'] / data['user_count']
	data['user_item_score_rank2'] = data['user_item_score_rank'] / data['user_count']

	data = data.sort_values(['uid', 'user_item_score_rank2'], ascending = (True, True))

	res = data[['user_item_pop_rank2', 'user_item_score_rank2']]
	res.columns = ['pop_rank', 'score_rank']

	bins = np.linspace(0, 1, 20)

	res['bins'] = pd.cut(res['pop_rank'], bins=bins, include_lowest=True)

	return res

def raw_pred_score(model_here, test_data):
	data = predictions_func(model_here,test_data,0)
	return data

def uPO(model_here, without_neg_data, sid_pop_total):
	# https://www.statology.org/pandas-groupby-correlation/
	# https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.corr.html

	data = predictions_func(model_here,without_neg_data,3)

	data = data.sort_values(['uid', 'sid'], ascending = [True, False])
	sid_pop_dict = dict(sid_pop_total.values)
	data['sid_pop_count'] = data['sid'].map(sid_pop_dict)
	result = data.groupby('uid')[['sid_pop_count', 'pred']].corr(method = 'spearman')
	result2 = result.unstack().iloc[:, 1].fillna(0).values.mean()

	return result2

# pearson correlation coefficient
def pcc_train(model_here, train_data, sid_pop):

	data = predictions_func(model_here,train_data,1)

	sid_pop_dict = dict(sid_pop.values)
	data['sid_pop_count'] = data['sid'].map(sid_pop_dict)

	X = np.array(data.pred.values)
	Y = np.array(data.sid_pop_count.values) # item pop

	pcc = ((X - X.mean())*(Y - Y.mean())).sum() / np.sqrt(((X - X.mean())*(X- X.mean())).sum()) / np.sqrt(((Y - Y.mean())*(Y- Y.mean())).sum())

	return pcc

# test pearson correlation coefficient
def pcc_test(model_here, test_data, sid_pop):

	data = predictions_func(model_here,test_data,1)

	sid_pop_dict = dict(sid_pop.values)
	data['sid_pop_count'] = data['sid'].map(sid_pop_dict)

	X = np.array(data.pred.values)
	Y = np.array(data.sid_pop_count.values) # item pop

	pcc = ((X - X.mean())*(Y - Y.mean())).sum() / np.sqrt(((X - X.mean())*(X- X.mean())).sum()) / np.sqrt(((Y - Y.mean())*(Y- Y.mean())).sum())

	return pcc

def pcc_test_check(model_here, without_neg_data, sid_pop_total):
	# https://www.statology.org/pandas-groupby-correlation/
	# https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.corr.html

	data = predictions_func(model_here,without_neg_data,1)
 
	sid_pop_dict = dict(sid_pop_total.values)
	data['sid_pop_count'] = data['sid'].map(sid_pop_dict)
	result = data[['sid_pop_count', 'pred']].corr(method = 'pearson')

	return result