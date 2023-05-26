# %%
############################## MATRIX FACTORIZATION WITH BPR LOSS AND REG TERM ################################
############################## LIBRARIES ###################################################

import os
import csv
import sys
import time
import numpy as np
import pandas as pd
import random as random

import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn

import data_utils
import Train_Val_Test_Functions

from pop_bias_metrics_basic import\
pred_item_rank, pred_item_rankdist, raw_pred_score

import matplotlib.pyplot as plt

print('Libraries Imported')

############################## PARAMETERS OF THE TRAIN ######################################

# Define seed or comment this section to have no seed
# random_seed = 0
# torch.manual_seed(random_seed)
# random.seed(random_seed)
# print('Seed =',random_seed)

model_name = 'MF'	# model used, only MF ('Matrix Factorization') available
load_model = False # Load or train a model? True or False
dataset = 'movielens'	# dataset used (only movielens available)
sample = 'new_posneg_bigger'	# sample used - [new_posneg_bigger,new_posneg, posneg , none]
epochs = 1	# training epochs, max 20, or redefine in data generation code
batch_size = 2048	# batch size for training
weight = 0.9	# weight between acuracy and regularization, does not influence 'none' sample
lr = 0.001	# learning rate
factor_num = 64	# predictive factor numbers in the model
top_k = 10	# compute metrics@top_k
num_ng = 3	# sample negative items for training
gpu = '0'	# gpu used
out = True	# save model or no
burnin = 'no'	# yes or no
reg = 'no'	# yes or no

if sample not in ['none', 'posneg','new_posneg','new_posneg_bigger'] and model_name != 'MF':	
	print('sample must be "none" or "posneg" or "new_posneg" or "new_posneg_bigger"') ;	sys.exit()
print('Parameters Defined')

############################## INITIAL CONFIGURATIONS ###################################

model_path = './models/'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
cudnn.benchmark = True

os.environ["CUDA_VISIBLE_DEVICES"] = gpu
device = "cuda" if torch.cuda.is_available() else "cpu"
print('Configurations Set: device = ',device)

############################## GET NUMBER OF FOLDS ######################################

folders_path = "./folds_data"
num_folds = 0 ; i = 1
while i == 1:
	num_folds += 1
	for root, dirs, files in os.walk(folders_path):
		i = dirs.count('train_samples_'+str(num_folds))
		break
num_folds -= 1
print('There are',num_folds,'folds.') # Can be redefined in the data generation jupyter notebook
print('')
print('# PARAMETERS')
print('model_name =',model_name,".'.",'load_model =',load_model,".'.",'dataset =',dataset,".'.",'sample =',sample)
print('epochs =',epochs,".'.",'batch_size =',batch_size,".'.",'burnin =',burnin,".'.",'reg =',reg)
print('weight =',weight,".'.",'lr =',lr,".'.",'factor_num =',factor_num,".'.",'top_k =',top_k,".'.",'num_ng =',num_ng)
print('gpu =',gpu,".'.",'out =',out,".'.")
print('')

############################## CREATE RESULTS FILES AND BEGINNING FOLD LOOP #################################

results_path = "./experiments/"+sample+'_'+str(epochs)+'_'+str(batch_size)+'_'+str(weight)+'_'+str(factor_num)+'_'+str(lr)+'_'+str(top_k)
if not os.path.isdir(results_path):
	os.mkdir(results_path)
test_results_list = [] ; test_hr_avg = test_ndcg_avg = test_popq_avg = test_pri_avg = time_avg = 0

for fold_num in range(num_folds):
	val_results = []
	fold_num += 1

	############################## PREPARE DATASET ########################################################

	user_num, item_num, train_data_len = data_utils.load_all_custom(fold_num, dataset=dataset )

	raw_train_data = pd.read_csv(f'./folds_data/processed_csv_folds/train_fold_'+str(fold_num)+'.csv')
	val_data_without_neg = pd.read_csv(f'./folds_data/processed_csv_folds/val_fold_'+str(fold_num)+'.csv')
	val_data_with_neg = pd.read_csv(f'./folds_data/processed_csv_folds/val_fold_'+str(fold_num)+'_with_neg'+'.csv')
	test_data_without_neg = pd.read_csv(f'./folds_data/processed_csv_folds/test_fold_'+str(fold_num)+'.csv')
	test_data_with_neg = pd.read_csv(f'./folds_data/processed_csv_folds/test_fold_'+str(fold_num)+'_with_neg'+'.csv')

	pop_sid_all_data = pd.read_csv(f'./folds_data/processed_csv_folds/pop_sid_all_data.csv')

	train_dataset = data_utils.BPRData(train_data_len*num_ng)

	train_loader = data.DataLoader(train_dataset,batch_size=batch_size, shuffle=True, pin_memory=True)
	print('Dataset Ready - fold ' + str(fold_num) + ' #########################################################################################')

	########################### LOAD MODEL #########################################################

	start_time = time.time()

	if load_model:
		model = torch.load(f'./models/final_{dataset}__{model_name}_{sample}_{weight}_{epochs}.pth')
		model.to(device) ; print('') , print('Model Loaded')

	########################### CREATE MODEL AND TRAIN ##############################################

	else:
		best_model, val_results = Train_Val_Test_Functions.train_model( model_name,
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
																					start_time)

		print('\nFinished Training')
		print("--- %s seconds ---" % (time.time() - start_time))

		# Save validation results in csv
		with open(results_path+'/Val_Results_Fold_'+str(fold_num)+'.csv', 'w', newline='') as file:
			vals_results_csv = csv.writer(file)
			vals_results_csv.writerow(["Epoch","HR", "NDCG", "POPQ","PRI","Time(seconds)"])
			for ep in range(epochs):	vals_results_csv.writerow(val_results[ep])

		val_results_df = pd.read_csv(results_path+'/Val_Results_Fold_'+str(fold_num)+'.csv')

		# Plot validation results
		plt.figure(figsize=(8,5)) ; plt.title('Val_Results_Fold_'+str(fold_num)+'_HR', fontdict={'fontweight':'bold', 'fontsize': 18})
		plt.plot(val_results_df.Epoch, val_results_df.HR, 'b.-')
		plt.xticks(val_results_df.Epoch.tolist())
		plt.xlabel('Epoch') ; plt.ylabel('HR')
		plt.savefig(results_path + '/Val_Results_Fold_'+str(fold_num)+'_HR.png', dpi=300)
		plt.figure().clear() ; plt.close() ; plt.cla() ; plt.clf()

		plt.figure(figsize=(8,5)) ; plt.title('Val_Results_Fold_'+str(fold_num)+'_NDCG', fontdict={'fontweight':'bold', 'fontsize': 18})
		plt.plot(val_results_df.Epoch, val_results_df.NDCG, 'b.-')
		plt.xticks(val_results_df.Epoch.tolist())
		plt.xlabel('Epoch') ; plt.ylabel('NDCG')
		plt.savefig(results_path + '/Val_Results_Fold_'+str(fold_num)+'_NDCG.png', dpi=300)
		plt.figure().clear() ; plt.close() ; plt.cla() ; plt.clf()

		plt.figure(figsize=(8,5)) ; plt.title('Val_Results_Fold_'+str(fold_num)+'_POPQ', fontdict={'fontweight':'bold', 'fontsize': 18})
		plt.plot(val_results_df.Epoch, val_results_df.POPQ, 'b.-')
		plt.xticks(val_results_df.Epoch.tolist())
		plt.xlabel('Epoch') ; plt.ylabel('POPQ')
		plt.savefig(results_path + '/Val_Results_Fold_'+str(fold_num)+'_POPQ.png', dpi=300)
		plt.figure().clear() ; plt.close() ; plt.cla() ; plt.clf()

		plt.figure(figsize=(8,5)) ; plt.title('Val_Results_Fold_'+str(fold_num)+'_PRI', fontdict={'fontweight':'bold', 'fontsize': 18})
		plt.plot(val_results_df.Epoch, val_results_df.PRI, 'b.-')
		plt.xticks(val_results_df.Epoch.tolist())
		plt.xlabel('Epoch') ; plt.ylabel('PRI')
		plt.savefig(results_path + '/Val_Results_Fold_'+str(fold_num)+'_PRI.png', dpi=300)
		plt.figure().clear() ; plt.close() ; plt.cla() ; plt.clf()

	############################## TEST ########################################################
	print('')

	test_hr, test_ndcg, test_popq, test_pri = Train_Val_Test_Functions.val_test_model(best_model,
																									top_k,
																									user_num,
																									test_data_with_neg,
																									pop_sid_all_data,
																									test_data_without_neg,
																									fold = fold_num)

	test_results_list.append([fold_num,round(test_hr,4), round(test_ndcg,4), round(test_popq,4),\
							  round(test_pri[0],4),round((time.time() - start_time),4)])

	test_hr_avg += test_hr
	test_ndcg_avg += test_ndcg
	test_popq_avg += test_popq
	test_pri_avg += test_pri[0]
	time_avg += round((time.time() - start_time),4)

test_hr_avg = round(test_hr_avg/num_folds,4)
test_ndcg_avg = round(test_ndcg_avg/num_folds,4)
test_popq_avg = round(test_popq_avg/num_folds,4)
test_pri_avg = round(test_pri_avg/num_folds,4)
time_avg = round(time_avg/num_folds,4)

# Save test results in csv
with open(results_path+'/Tests_Results.csv', 'w', newline='') as file:
	tests_results_csv = csv.writer(file)
	tests_results_csv.writerow(["Fold","HR", "NDCG", "POPQ","PRI","Time(seconds)"])
	for tr in range(len(test_results_list)):	tests_results_csv.writerow(test_results_list[tr])
	tests_results_csv.writerow(["Average",test_hr_avg, test_ndcg_avg, test_popq_avg,test_pri_avg,time_avg])

tests_results_df = pd.read_csv(results_path+'/Tests_Results.csv')

# Plot test results
plt.figure(figsize=(8,5)) ; plt.title('Tests_Results_HR.png', fontdict={'fontweight':'bold', 'fontsize': 18})
plt.plot(tests_results_df[tests_results_df['Fold'] != 'Average'].Fold, tests_results_df[tests_results_df['Fold'] != 'Average'].HR, 'b.-')
plt.xticks(tests_results_df[tests_results_df['Fold'] != 'Average'].Fold.tolist())
plt.xlabel('Fold') ; plt.ylabel('HR')
plt.savefig(results_path + '/Tests_Results_HR.png', dpi=300)
plt.figure().clear() ; plt.close() ; plt.cla() ; plt.clf()

plt.figure(figsize=(8,5)) ; plt.title('Tests_Results_NDCG.png', fontdict={'fontweight':'bold', 'fontsize': 18})
plt.plot(tests_results_df[tests_results_df['Fold'] != 'Average'].Fold, tests_results_df[tests_results_df['Fold'] != 'Average'].NDCG, 'b.-')
plt.xticks(tests_results_df[tests_results_df['Fold'] != 'Average'].Fold.tolist())
plt.xlabel('Fold') ; plt.ylabel('NDCG')
plt.savefig(results_path + '/Tests_Results_NDCG.png', dpi=300)
plt.figure().clear() ; plt.close() ; plt.cla() ; plt.clf()

plt.figure(figsize=(8,5)) ; plt.title('Tests_Results_POPQ.png', fontdict={'fontweight':'bold', 'fontsize': 18})
plt.plot(tests_results_df[tests_results_df['Fold'] != 'Average'].Fold, tests_results_df[tests_results_df['Fold'] != 'Average'].POPQ, 'b.-')
plt.xticks(tests_results_df[tests_results_df['Fold'] != 'Average'].Fold.tolist())
plt.xlabel('Fold') ; plt.ylabel('POPQ')
plt.savefig(results_path + '/Tests_Results_POPQ.png', dpi=300)
plt.figure().clear() ; plt.close() ; plt.cla() ; plt.clf()

plt.figure(figsize=(8,5)) ; plt.title('Tests_Results_PRI.png', fontdict={'fontweight':'bold', 'fontsize': 18})
plt.plot(tests_results_df[tests_results_df['Fold'] != 'Average'].Fold, tests_results_df[tests_results_df['Fold'] != 'Average'].PRI, 'b.-')
plt.xticks(tests_results_df[tests_results_df['Fold'] != 'Average'].Fold.tolist())
plt.xlabel('Fold') ; plt.ylabel('PRI')
plt.savefig(results_path + '/Tests_Results_PRI.png', dpi=300)
plt.figure().clear() ; plt.close() ; plt.cla() ; plt.clf()