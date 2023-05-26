# Train Function and Eval-Test Function
import model_basic
import evaluate

import time
import os
import numpy as np
import scipy.stats as stats
from scipy.stats import skew

import torch
import torch.optim as optim

import copy

from pop_bias_metrics_basic import\
pred_item_rank, pred_item_score, pred_item_rankdist, pcc_test, uPO

def val_test_model( model,
					top_k,
					user_num,
					val_data_with_neg,
					pop_sid_all_data,
					val_data_without_neg,
					val = False,
					fold = None):

	model.eval()

	if val == True: print('entered val evaluated')
	else: print('entered test evaluated')

	HR, NDCG, ARP = evaluate.metrics_custom_new_bpr(model, val_data_with_neg, top_k, pop_sid_all_data, user_num)

	PCC_TEST = pcc_test(model, val_data_without_neg, pop_sid_all_data)

	score = pred_item_score(model, val_data_without_neg, pop_sid_all_data)
	SCC_score_test = stats.spearmanr(score.dropna()['sid_pop_count'].values, score.dropna()['pred'].values)
	rank = pred_item_rank(model, val_data_without_neg, pop_sid_all_data)
	SCC_rank_test = stats.spearmanr(rank.dropna()['sid_pop_count'].values, rank.dropna()['rank'].values)

	upo = uPO(model, val_data_without_neg, pop_sid_all_data)

	rankdist = pred_item_rankdist(model, val_data_without_neg, pop_sid_all_data)
	mean_test = np.mean(rankdist[rankdist.notna()].values)
	skew_test = skew(rankdist[rankdist.notna()].values)

	print("HR: {:.3f}\tNDCG: {:.3f}\tmean test (POPQ@K): {:.3f}".format(np.mean(HR), np.mean(NDCG), np.round(mean_test, 3)))
	if val != True:
		print("ARP: {:.3f}".format(np.mean(ARP)))
		print('PCC_TEST (Pearson Correlation Coefficient): ', np.round(PCC_TEST, 3))
		print('SCC_score_test : ', np.round(SCC_score_test[0], 3))
		print('SCC_rank_test (IPO = PRI): ', np.round(SCC_rank_test[0], 3))
		print('upo is :', np.round(upo, 3))
		print('skew_test : ', np.round(skew_test, 3))
		print(' ')
		return HR , NDCG , mean_test , SCC_rank_test

	if val == True: return HR , NDCG , SCC_rank_test , mean_test

def train_model(model_name,
				train_loader,
				dataset,
				fold_num,
				sample,
				epochs,
				batch_size,
				weight,
				lr,
				factor_num,
				top_k,
				user_num,
				item_num,
				val_results,
				val_data_with_neg,
				pop_sid_all_data,
				val_data_without_neg,
				model_path,
				out,
				device,
				burnin,
				reg,
				time_start):

	model = model_basic.MF_BPR(user_num, item_num, factor_num, model_name) ; model.to(device)
	optimizer = optim.Adam(model.parameters(), lr=lr)

	print('\nentered training\n')

	best_hr = 0 ; acc_w = weight ; pop_w = 1-weight ; doit = True

	for epoch in range(epochs):
		print('epoch is : ',epoch)
		model.train()

		train_loader.dataset.get_data(fold_num,dataset, epoch)

		if burnin == 'yes':
			if epoch < epochs/4 and doit == True:	sam = sample ;	sample = 'none' ;	doit = False
			elif epoch >= epochs/4 and doit == False:	sample = sam ; doit = True

		for user, pos, neg in train_loader:
			user = user.to(device) ; pos = pos.to(device) ; neg = neg.to(device)
			model.zero_grad()

			# Forward
			pos_scores, neg_scores = model(user, pos, neg)

			# Loss
			if sample == 'none':
				loss = - (pos_scores - neg_scores).sigmoid().log().mean()
			elif sample == 'posneg':
				acc_loss = - (pos_scores - neg_scores).sigmoid().log().mean()/2
				pop_loss =  -(1 -(pos_scores + neg_scores).abs().tanh() ).log().mean()/2
				loss = acc_loss*acc_w + pop_loss*pop_w

			elif sample == 'new_posneg':

				pos_new = torch.where((neg_scores < 0)&(pos_scores > 0),pos_scores,float('nan')) ; pos_pos = pos_new[pos_new > 0]
				neg_new = torch.where((neg_scores < 0)&(pos_scores > 0),neg_scores,float('nan')) ; neg_neg = neg_new[neg_new < 0]

				num_nan = torch.tensor(pos_new[torch.isnan(pos_new) == True].size())/batch_size ; num_nan = num_nan.to(device)

				acc_loss = - (pos_scores - neg_scores).sigmoid().log().mean()/2 ; acc_loss.to(device)
				pop_loss =  -(1 -(pos_pos + neg_neg).abs().tanh() ).log().mean()/2; pop_loss.to(device)

				loss = acc_loss + pop_loss*pop_w + num_nan

			elif sample == 'new_posneg_bigger':

				bigger = torch.where((pos_scores > neg_scores),1,float('nan'))
				bigger = torch.tensor((bigger[torch.isnan(bigger) == True]).size())/batch_size ; bigger = bigger.to(device)

				pos_new = torch.where((neg_scores < 0)&(pos_scores > 0),pos_scores,float('nan')) ; pos_pos = pos_new[pos_new > 0]
				neg_new = torch.where((neg_scores < 0)&(pos_scores > 0),neg_scores,float('nan')) ; neg_neg = neg_new[neg_new < 0]

				num_nan = torch.tensor(pos_new[torch.isnan(pos_new) == True].size())/batch_size ; num_nan = num_nan.to(device)

				acc_loss = - (pos_scores - neg_scores).sigmoid().log().mean()/2 ; acc_loss.to(device)

				pop_loss =  -(1 -(pos_pos + neg_neg).abs().tanh() ).log().mean()/2 ; pop_loss.to(device)

				loss = acc_loss + pop_loss*pop_w + num_nan + bigger

			if reg == 'yes':
				user_emb_w = model.embed_user_MLP.weight[user]
				pos_emb_w = model.embed_item_MLP.weight[pos]
				neg_emb_w = model.embed_item_MLP.weight[neg]
				reg = (torch.norm(user_emb_w) ** 2 + torch.norm(pos_emb_w)** 2 + torch.norm(neg_emb_w) ** 2)/3 / batch_size
				loss += 1e-5*reg

			loss.backward()
			optimizer.step()

		HR , NDCG , SCC_rank_test , mean_test = val_test_model( model,
																													top_k,
																													user_num,
																													val_data_with_neg,
																													pop_sid_all_data,
																													val_data_without_neg,
																													val = True)

		epoch_val_result = [epoch, round(HR,4), round(NDCG,4), round(mean_test,4), round(SCC_rank_test[0],4),round((time.time() - time_start),4)]
		val_results.append(epoch_val_result)

		if HR > best_hr:
			best_model = copy.deepcopy(model)
			best_hr, best_ndcg, best_popq, best_epoch = HR, NDCG, mean_test, epoch

	if out:
		if not os.path.exists(model_path):	os.mkdir(model_path)
		torch.save(best_model,'{}{}_{}_{}_{}.pth'.format(model_path,f'final_{dataset}_', f'{model_name}_{sample}', weight, epochs))
	print('')
	print("End. Best epoch{:3d}: HR = {:.3f}, NDCG = {:.3f}, Mean Test (POPQ@K) = {:.3f}".format(best_epoch, best_hr, best_ndcg, best_popq))

	return best_model, val_results