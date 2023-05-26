import numpy as np
import torch
import pandas as pd

def arp_custom_new(pred_items, sid_pop_total):
	d = sid_pop_total
	l = pred_items

	r = [ [ d[v] for v in lv] for lv in l]
	ARP = np.mean(np.sum(r, axis = 1))
	return ARP

def metrics_custom_new_bpr(model, test_data, top_k, sid_pop_total, user_num):

	HR, NDCG, ARP = [], [], []

	device = "cuda" if torch.cuda.is_available() else "cpu"

	test_data = test_data[['uid', 'sid', 'type']]
	test_data['uid'] = test_data['uid'].apply(lambda x : int(x))
	test_data['sid'] = test_data['sid'].apply(lambda x : int(x))
	test_users_num = len(test_data['uid'].unique())

	data_len = test_data.shape[0]
	frac = 50
	frac_user_num = int(data_len/frac)

	model.eval()
	model.to(device)

	predictions_list = []

	for itr in range(frac):
		tmp = test_data.iloc[ (frac_user_num * itr) : (frac_user_num* (itr+1) ) ].values
		user = tmp[:, 0]
		user = user.astype(np.int32)
		user = torch.from_numpy(user).to(device)
		item = tmp[:, 1]
		item = item.astype(np.int32)
		item = torch.from_numpy(item).to(device)
		predictions_tmp = model.forward_one_item(user, item)
		predictions_list += predictions_tmp.detach().to(device).tolist()
	if itr+1 == frac:
		tmp = test_data.iloc[ (frac_user_num * (itr+1)):].values
		user = tmp[:, 0]
		user = user.astype(np.int32)
		user = torch.from_numpy(user).to(device)
		item = tmp[:, 1]
		item = item.astype(np.int32)
		item = torch.from_numpy(item).to(device)

		pred_temp = tmp[:,2]

		predictions_tmp = model.forward_one_item(user, item)
		predictions_list += predictions_tmp.detach().to(device).tolist()

	test_data['pred'] = predictions_list

	### compute ARP
	pred_result_for_arp = test_data.sort_values(['pred'], ascending = False).groupby('uid').head(top_k)
	pred_result_for_arp = pred_result_for_arp.sid.astype(int).values
	pred_result_for_arp = pred_result_for_arp.reshape(test_users_num, top_k)
	pop_count_dict = dict(zip(sid_pop_total.sid, sid_pop_total.total_counts))
	ARP.append(arp_custom_new(pred_result_for_arp.tolist(), pop_count_dict))

	### compute accuracy, ndcg @ top k
	test_data_pos = test_data[test_data.type == 'pos']
	test_data_pos = test_data_pos.reset_index()[['uid', 'sid', 'pred']]
	test_data_neg = test_data[test_data.type == 'neg']
	test_data_neg = test_data_neg.reset_index()[['uid', 'sid', 'pred']]

	test_data_neg = test_data_neg.sort_values(by = ['uid', 'pred'], ascending = [True, False])
	test_data_neg['order'] = list(range(100)) * user_num
	neg_scores_table_alluser = test_data_neg.pivot_table(index = 'uid', columns = 'order', values = 'pred')   # user_num x 100
	neg_scores_table_testuser = neg_scores_table_alluser.loc[test_data_pos.uid.values, :]       # test_pos_item_num x 100

	pos_score_final = pd.DataFrame(test_data_pos.pred.values) # test pos item num x 1
	neg_score_final = pd.DataFrame(neg_scores_table_testuser.values) # test pos item num x 100
	final_df = pd.concat([pos_score_final, neg_score_final], axis = 1) # test_pos_item_num x 101
	final_df.columns = list(range(101))
	rank_list = final_df.iloc[:, 0:(top_k+1)]
	rank_score = rank_list.rank(1, ascending=False, method='max').iloc[:,0].values
	hits = (rank_score < (top_k + 1))*1
	ndcgs = hits*np.reciprocal(np.log2(rank_score+1))

	HR += hits.tolist()
	NDCG += ndcgs.tolist()

	return np.mean(HR), np.mean(NDCG), np.mean(ARP)

def metrics_graph_bpr(model, test_data, top_k, sid_pop_total, user_num):

	HR, NDCG, ARP = [], [], []

	device = "cuda" if torch.cuda.is_available() else "cpu"

	test_data = test_data[['uid', 'sid', 'type']]
	test_data['uid'] = test_data['uid'].apply(lambda x : int(x))
	test_data['sid'] = test_data['sid'].apply(lambda x : int(x))
	test_users_num = len(test_data['uid'].unique())

	user = test_data.values[:, 0]
	user = np.array(user).astype(np.int32)
	user = user.tolist()
	user = torch.LongTensor(user).cuda()

	item = test_data.values[:, 1] #.tolist()
	item = np.array(item).astype(np.int32)
	item = item.tolist()
	item = torch.LongTensor(item).cuda()

	u_emb, pos_i_emb, neg_i_emb = model(user, item, item, drop_flag = False)
	predictions = torch.sum(torch.mul(u_emb, pos_i_emb), axis=1)
	test_data['pred'] = predictions.detach().to(device)

	### compute ARP
	pred_result_for_arp = test_data.sort_values(['pred'], ascending = False).groupby('uid').head(top_k)
	pred_result_for_arp = pred_result_for_arp.sid.astype(int).values
	pred_result_for_arp = pred_result_for_arp.reshape(test_users_num, top_k)
	pop_count_dict = dict(zip(sid_pop_total.sid, sid_pop_total.total_counts))
	ARP.append(arp_custom_new(pred_result_for_arp.tolist(), pop_count_dict))

	### compute accuracy, ndcg @ top k
	test_data_pos = test_data[test_data.type == 'pos']
	test_data_pos = test_data_pos.reset_index()[['uid', 'sid', 'pred']]

	test_data_neg = test_data[test_data.type == 'neg']
	test_data_neg = test_data_neg.reset_index()[['uid', 'sid', 'pred']]
	test_data_neg = test_data_neg.sort_values(by = ['uid', 'pred'], ascending = [True, False])
	test_data_neg['order'] = list(range(100)) * user_num
	neg_scores_table_alluser = test_data_neg.pivot_table(index = 'uid', columns = 'order', values = 'pred')   # user_num x 100
	neg_scores_table_testuser = neg_scores_table_alluser.loc[test_data_pos.uid.values, :]       # test_pos_item_num x 100

	pos_score_final = pd.DataFrame(test_data_pos.pred.values) # test pos item num x 1
	neg_score_final = pd.DataFrame(neg_scores_table_testuser.values) # test pos item num x 100

	final_df = pd.concat([pos_score_final, neg_score_final], axis = 1) # test_pos_item_num x 101
	final_df.columns = list(range(101))
	rank_list = final_df.iloc[:, 0:(top_k+1)]
	rank_score = rank_list.rank(1, ascending=False, method='max').iloc[:,0].values
	hits = (rank_score < (top_k + 1))*1
	ndcgs = hits*np.reciprocal(np.log2(rank_score+1))

	HR += hits.tolist()
	NDCG += ndcgs.tolist()

	return np.mean(HR), np.mean(NDCG), np.mean(ARP)