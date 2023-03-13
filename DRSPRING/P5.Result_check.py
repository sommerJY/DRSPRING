
import rdkit
import os
import os.path as osp
from math import ceil
import pandas as pd
import numpy as np
import json
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader
from torch.utils.data import Dataset

from torch_geometric.nn import DenseGCNConv as GCNConv, dense_diff_pool #DiffPool
from torch_geometric.nn import SAGEConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import pickle
import joblib
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
import sklearn
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = sklearn.preprocessing.OneHotEncoder
import datetime
from datetime import *
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error
												
# Graph convolution model
import torch_geometric.nn as pyg_nn
# Graph utility function
import torch_geometric.utils as pyg_utils
import torch.optim as optim
import torch_geometric.nn.conv
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs


import networkx as nx
import copy
from scipy.sparse import coo_matrix
from scipy import sparse
from scipy import stats
import sklearn.model_selection

import sys
import random
import shutil
import math

import ray
from ray import tune
from functools import partial
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune import ExperimentAnalysis

import numpy as np
from rdkit import Chem
from rdkit.Chem.QED import qed
import sys
import os
import pandas as pd

#######

import pandas as pd 
import matplotlib.pyplot as plt
from ray.tune import ExperimentAnalysis
import pickle
import math
import torch






RAY_dir = '22.10.01.PRJ01.TRIAL.5_2_3'
G_NAME = '5_2_3.MJ_1_MJB' 
K_W_Dir = 'TRIAL.5_2_3' #  TRIAL_5.1 , TRIAL.5_2_3 , TRIAL.5_3_3


RAY_dir = '22.10.02.PRJ01.TRIAL.5_3_3'
G_NAME = '5_3_3.MJ_1_MJB' 
K_W_Dir = 'TRIAL.5_3_3' #  TRIAL_5.1 , TRIAL.5_2_3 , TRIAL.5_3_3


RAY_dir = '22.10.02.PRJ01.TRIAL.5_2_3.MJ_2_MJB'
G_NAME = '5_2_3.MJ_2_MJB' 
K_W_Dir = 'TRIAL.5_2_3' #  TRIAL_5.1 , TRIAL.5_2_3 , TRIAL.5_3_3


RAY_dir = '22.10.03.PRJ01.TRIAL.5_3_3.MJ_2_MJB'
G_NAME = '5_3_3.MJ_2_MJB' 
K_W_Dir = 'TRIAL.5_3_3' #  TRIAL_5.1 , TRIAL.5_2_3 , TRIAL.5_3_3



RAY_dir = '22.10.03.PRJ01.TRIAL.5_2_3.MJ_2_NF'
G_NAME = '5_2_3.MJ_2_NF' 
K_W_Dir = 'TRIAL.5_2_3' #  TRIAL_5.1 , TRIAL.5_2_3 , TRIAL.5_3_3


RAY_dir = '22.10.03.PRJ01.TRIAL.5_3_3.MJ_2_NF'
G_NAME = '5_3_3.MJ_2_NF' 
K_W_Dir = 'TRIAL.5_3_3' #  TRIAL_5.1 , TRIAL.5_2_3 , TRIAL.5_3_3


RAY_dir = '22.10.06.PRJ01.TRIAL.5_2_3.MJ_1_NF'
G_NAME = '5_2_3.MJ_1_NF' 
K_W_Dir = 'TRIAL.5_2_3' #  TRIAL_5.1 , TRIAL.5_2_3 , TRIAL.5_3_3



RAY_dir = '22.10.06.PRJ01.TRIAL.5_3_3.MJ_1_NF'
G_NAME = '5_3_3.MJ_1_NF' 
K_W_Dir = 'TRIAL.5_3_3' #  TRIAL_5.1 , TRIAL.5_2_3 , TRIAL.5_3_3
# 아직 안끝났지만 한번 확인만 하려고 





# 다시 시도한거 
anal_df = ExperimentAnalysis("~/ray_results/{}".format(RAY_dir))

ANA_DF = anal_df.dataframe()
ANA_ALL_DF = anal_df.trial_dataframes


ANA_DF.to_csv('/home01/k006a01/PRJ.01/{}/RAY_ANA_DF.{}.csv'.format(K_W_Dir, G_NAME))
import pickle
with open("/home01/k006a01/PRJ.01/{}/RAY_ANA_DF.{}.pickle".format(K_W_Dir, G_NAME), "wb") as fp:
	pickle.dump(ANA_ALL_DF,fp) 


DF_KEY = list(ANA_DF.sort_values('ValLoss')['logdir'])[0] # 126
DF_KEY
# get /home01/k006a01/ray_results/22.10.01.PRJ01.TRIAL.5_2_3/RAY_MY_train_9561642c_91_G_hiddim=32,G_layer=2,batch_size=16,dropout_1=0.2000,dropout_2=0.5000,epoch=1000,feat_size_0=128,feat_siz_2022-10-03_07-32-43/model.pth M1_model.pth
# get /home01/k006a01/ray_results/22.10.02.PRJ01.TRIAL.5_3_3/RAY_MY_train_a198b33e_60_G_hiddim=32,G_layer=3,batch_size=64,dropout_1=0.5000,dropout_2=0.0100,epoch=1000,feat_size_0=512,feat_siz_2022-10-04_06-50-05/model.pth M1_model.pth
# get /home01/k006a01/ray_results/22.10.02.PRJ01.TRIAL.5_2_3.MJ_2_MJB/RAY_MY_train_f2fce574_85_G_hiddim=512,G_layer=3,batch_size=128,dropout_1=0.0100,dropout_2=0.2000,epoch=1000,feat_size_0=64,feat_si_2022-10-05_11-38-03/model.pth M1_model.pth
# get /home01/k006a01/ray_results/22.10.03.PRJ01.TRIAL.5_3_3.MJ_2_MJB/RAY_MY_train_b5cde67a_91_G_hiddim=64,G_layer=3,batch_size=64,dropout_1=0.2000,dropout_2=0.2000,epoch=1000,feat_size_0=256,feat_siz_2022-10-05_15-49-04/model.pth M1_model.pth
/home01/k006a01/ray_results/22.10.03.PRJ01.TRIAL.5_2_3.MJ_2_NF/RAY_MY_train_ab48b9a6_39_G_hiddim=512,G_layer=2,batch_size=64,dropout_1=0.5000,dropout_2=0.0100,epoch=1000,feat_size_0=4096,feat_s_2022-10-05_13-25-17
/home01/k006a01/ray_results/22.10.03.PRJ01.TRIAL.5_3_3.MJ_2_NF/RAY_MY_train_a5e96910_39_G_hiddim=512,G_layer=4,batch_size=64,dropout_1=0.0100,dropout_2=0.5000,epoch=1000,feat_size_0=256,feat_si_2022-10-04_11-34-30
/home01/k006a01/ray_results/22.10.06.PRJ01.TRIAL.5_2_3.MJ_1_NF/RAY_MY_train_2e4dfddc_63_G_hiddim=64,G_layer=2,batch_size=128,dropout_1=0.0100,dropout_2=0.2000,epoch=1000,feat_size_0=128,feat_si_2022-10-08_05-46-05
/home01/k006a01/ray_results/22.10.06.PRJ01.TRIAL.5_3_3.MJ_1_NF/RAY_MY_train_ad10752c_29_G_hiddim=512,G_layer=3,batch_size=64,dropout_1=0.5000,dropout_2=0.2000,epoch=1000,feat_size_0=256,feat_si_2022-10-09_06-11-29



TOPVAL_PATH = DF_KEY

mini_df = ANA_ALL_DF[DF_KEY]

cck_num =mini_df[mini_df.ValLoss==min(mini_df.ValLoss)].index.item() # 121.34
checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
TOPVAL_PATH = DF_KEY + checkpoint
TOPVAL_PATH

# get /home01/k006a01/ray_results/22.10.01.PRJ01.TRIAL.5_2_3/RAY_MY_train_9561642c_91_G_hiddim=32,G_layer=2,batch_size=16,dropout_1=0.2000,dropout_2=0.5000,epoch=1000,feat_size_0=128,feat_siz_2022-10-03_07-32-43/checkpoint_000787/checkpoint M2_checkpoint
# get /home01/k006a01/ray_results/22.10.02.PRJ01.TRIAL.5_3_3/RAY_MY_train_a198b33e_60_G_hiddim=32,G_layer=3,batch_size=64,dropout_1=0.5000,dropout_2=0.0100,epoch=1000,feat_size_0=512,feat_siz_2022-10-04_06-50-05/checkpoint_000958/checkpoint M2_checkpoint
# get /home01/k006a01/ray_results/22.10.02.PRJ01.TRIAL.5_2_3.MJ_2_MJB/RAY_MY_train_f2fce574_85_G_hiddim=512,G_layer=3,batch_size=128,dropout_1=0.0100,dropout_2=0.2000,epoch=1000,feat_size_0=64,feat_si_2022-10-05_11-38-03/checkpoint_000807/checkpoint M2_checkpoint
# get /home01/k006a01/ray_results/22.10.03.PRJ01.TRIAL.5_3_3.MJ_2_MJB/RAY_MY_train_b5cde67a_91_G_hiddim=64,G_layer=3,batch_size=64,dropout_1=0.2000,dropout_2=0.2000,epoch=1000,feat_size_0=256,feat_siz_2022-10-05_15-49-04/checkpoint_000878/checkpoint M2_checkpoint
/home01/k006a01/ray_results/22.10.03.PRJ01.TRIAL.5_2_3.MJ_2_NF/RAY_MY_train_ab48b9a6_39_G_hiddim=512,G_layer=2,batch_size=64,dropout_1=0.5000,dropout_2=0.0100,epoch=1000,feat_size_0=4096,feat_s_2022-10-05_13-25-17/checkpoint_000926
/home01/k006a01/ray_results/22.10.03.PRJ01.TRIAL.5_3_3.MJ_2_NF/RAY_MY_train_a5e96910_39_G_hiddim=512,G_layer=4,batch_size=64,dropout_1=0.0100,dropout_2=0.5000,epoch=1000,feat_size_0=256,feat_si_2022-10-04_11-34-30/checkpoint_000586
/home01/k006a01/ray_results/22.10.06.PRJ01.TRIAL.5_2_3.MJ_1_NF/RAY_MY_train_2e4dfddc_63_G_hiddim=64,G_layer=2,batch_size=128,dropout_1=0.0100,dropout_2=0.2000,epoch=1000,feat_size_0=128,feat_si_2022-10-08_05-46-05/checkpoint_000972
/home01/k006a01/ray_results/22.10.06.PRJ01.TRIAL.5_3_3.MJ_1_NF/RAY_MY_train_ad10752c_29_G_hiddim=512,G_layer=3,batch_size=64,dropout_1=0.5000,dropout_2=0.2000,epoch=1000,feat_size_0=256,feat_si_2022-10-09_06-11-29/checkpoint_000851



import numpy as np
TOT_min = np.Inf
TOT_key = ""
for key in ANA_ALL_DF.keys():
	trial_min = min(ANA_ALL_DF[key]['ValLoss'])
	if trial_min < TOT_min :
		TOT_min = trial_min
		TOT_key = key

TOT_min
TOT_key

mini_df = ANA_ALL_DF[TOT_key]
TOPVAL_PATH = TOT_key

cck_num =mini_df[mini_df.ValLoss==min(mini_df.ValLoss)].index.item()
checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
TOPVAL_PATH = TOT_key + checkpoint
TOPVAL_PATH

# 130
# get /home01/k006a01/ray_results/22.10.01.PRJ01.TRIAL.5_2_3/RAY_MY_train_9561642c_91_G_hiddim=32,G_layer=2,batch_size=16,dropout_1=0.2000,dropout_2=0.5000,epoch=1000,feat_size_0=128,feat_siz_2022-10-03_07-32-43/checkpoint_000787/checkpoint M4_checkpoint
# get /home01/k006a01/ray_results/22.10.02.PRJ01.TRIAL.5_3_3/RAY_MY_train_a198b33e_60_G_hiddim=32,G_layer=3,batch_size=64,dropout_1=0.5000,dropout_2=0.0100,epoch=1000,feat_size_0=512,feat_siz_2022-10-04_06-50-05/checkpoint_000958/checkpoint M4_checkpoint
# get /home01/k006a01/ray_results/22.10.02.PRJ01.TRIAL.5_2_3.MJ_2_MJB/RAY_MY_train_c1826b56_60_G_hiddim=64,G_layer=3,batch_size=64,dropout_1=0.0100,dropout_2=0.0100,epoch=1000,feat_size_0=256,feat_siz_2022-10-05_00-04-00/checkpoint_000638/checkpoint M4_checkpoint
# get /home01/k006a01/ray_results/22.10.03.PRJ01.TRIAL.5_3_3.MJ_2_MJB/RAY_MY_train_073d3d5e_89_G_hiddim=64,G_layer=3,batch_size=64,dropout_1=0.2000,dropout_2=0.2000,epoch=1000,feat_size_0=256,feat_siz_2022-10-05_15-28-03/checkpoint_000627/checkpoint M4_checkpoint
/home01/k006a01/ray_results/22.10.03.PRJ01.TRIAL.5_2_3.MJ_2_NF/RAY_MY_train_ab48b9a6_39_G_hiddim=512,G_layer=2,batch_size=64,dropout_1=0.5000,dropout_2=0.0100,epoch=1000,feat_size_0=4096,feat_s_2022-10-05_13-25-17/checkpoint_000926
/home01/k006a01/ray_results/22.10.03.PRJ01.TRIAL.5_3_3.MJ_2_NF/RAY_MY_train_b3f0e564_90_G_hiddim=128,G_layer=3,batch_size=64,dropout_1=0.2000,dropout_2=0.5000,epoch=1000,feat_size_0=512,feat_si_2022-10-05_16-13-53/checkpoint_000929
/home01/k006a01/ray_results/22.10.06.PRJ01.TRIAL.5_2_3.MJ_1_NF/RAY_MY_train_ea6357b8_62_G_hiddim=256,G_layer=2,batch_size=128,dropout_1=0.0100,dropout_2=0.2000,epoch=1000,feat_size_0=128,feat_s_2022-10-08_05-25-27/checkpoint_000611
/home01/k006a01/ray_results/22.10.06.PRJ01.TRIAL.5_3_3.MJ_1_NF/RAY_MY_train_ad10752c_29_G_hiddim=512,G_layer=3,batch_size=64,dropout_1=0.5000,dropout_2=0.2000,epoch=1000,feat_size_0=256,feat_si_2022-10-09_06-11-29/checkpoint_000851



##########################################
##########################################
##########################################


print('input ok')

def normalize(X, means1=None, std1=None, means2=None, std2=None,
	feat_filt=None, norm='tanh_norm'):
	if std1 is None:
		std1 = np.nanstd(X, axis=0) # nan 무시하고 표준편차 구하기 
	if feat_filt is None:
		feat_filt = std1!=0
	X = X[:,feat_filt]
	X = np.ascontiguousarray(X)
	if means1 is None:
		means1 = np.mean(X, axis=0)
	X = (X-means1)/std1[feat_filt]
	if norm == 'norm':
		return(X, means1, std1, feat_filt)
	elif norm == 'tanh':
		return(np.tanh(X), means1, std1, feat_filt)
	elif norm == 'tanh_norm':
		X = np.tanh(X)
		if means2 is None:
			means2 = np.mean(X, axis=0)
		if std2 is None:
			std2 = np.std(X, axis=0)
		X = (X-means2)/std2
		X[:,std2==0]=0
		return(X, means1, std1, means2, std2, feat_filt)


# normalizing 과정에 대한 고민 필요 ... GCN 이니까 상관 없지 않을까
def prepare_data_GCN(MY_chem_A_feat, MY_chem_B_feat, MY_chem_A_adj, MY_chem_B_adj, MY_g_feat_A, MY_g_feat_B, MY_Cell, MY_syn, norm ) :
	#chem_feat_A_train, chem_feat_A_tv, chem_feat_B_train, chem_feat_B_tv, chem_adj_A_train, chem_adj_A_tv, chem_adj_B_train, chem_adj_B_tv, exp_A_train, exp_A_tv, exp_B_train, exp_B_tv,  cell_train, cell_tv, syn_train, syn_tv  = sklearn.model_selection.train_test_split(
	#		MY_chem_A_feat, MY_chem_B_feat, MY_chem_A_adj, MY_chem_B_adj, MY_g_feat_A, MY_g_feat_B, MY_Cell, MY_syn,
	#		test_size= 0.2 , random_state=42 )
	#chem_feat_A_val, chem_feat_A_test, chem_feat_B_val, chem_feat_B_test, chem_adj_A_val, chem_adj_A_test, chem_adj_B_val, chem_adj_B_test, exp_A_val, exp_A_test, exp_B_val, exp_B_test, cell_val, cell_test, syn_val, syn_test   = sklearn.model_selection.train_test_split(
	#		chem_feat_A_tv, chem_feat_B_tv, chem_adj_A_tv, chem_adj_B_tv, exp_A_tv, exp_B_tv, cell_tv, syn_tv,
	#		test_size=0.5, random_state=42 )
	#
	train_data = {}
	val_data = {}
	test_data = {}
	#
	train_data['drug1_feat'] = torch.concat([chem_feat_A_train, chem_feat_B_train], axis = 0)
	val_data['drug1_feat'] = chem_feat_A_val
	test_data['drug1_feat'] = chem_feat_A_test
	#
	train_data['drug2_feat'] = torch.concat([chem_feat_B_train, chem_feat_A_train], axis = 0)
	val_data['drug2_feat'] = chem_feat_B_val
	test_data['drug2_feat'] = chem_feat_B_test
	#
	train_data['drug1_adj'] = torch.concat([chem_adj_A_train, chem_adj_B_train], axis = 0)
	val_data['drug1_adj'] = chem_adj_A_val
	test_data['drug1_adj'] = chem_adj_A_test
	#
	train_data['drug2_adj'] = torch.concat([chem_adj_B_train, chem_adj_A_train], axis = 0)
	val_data['drug2_adj'] = chem_adj_B_val
	test_data['drug2_adj'] = chem_adj_B_test
	#
	train_data['EXP_A'] = torch.concat([exp_A_train, exp_B_train], axis = 0)
	val_data['EXP_A'] = exp_A_val
	test_data['EXP_A'] = exp_A_test
	#
	train_data['EXP_B'] = torch.concat([exp_B_train, exp_A_train], axis = 0)
	val_data['EXP_B'] = exp_B_val
	test_data['EXP_B'] = exp_B_test
	#
	train_data['cell'] = np.concatenate((cell_train, cell_train), axis=0)
	val_data['cell'] = cell_val
	test_data['cell'] = cell_test
	#               
	train_data['y'] = np.concatenate((syn_train, syn_train), axis=0)
	val_data['y'] = syn_val
	test_data['y'] = syn_test
	#
	print(train_data['drug1_feat'].shape, flush=True)
	print(val_data['drug1_feat'].shape, flush=True)
	print(test_data['drug1_feat'].shape, flush=True)
	return train_data, val_data, test_data





class DATASET_GCN_W_FT(Dataset):
	def __init__(self, gcn_drug1_F, gcn_drug2_F, gcn_drug1_ADJ, gcn_drug2_ADJ, gcn_exp_A, gcn_exp_B, gcn_adj, gcn_adj_weight, cell_info, syn_ans):
		self.gcn_drug1_F = gcn_drug1_F
		self.gcn_drug2_F = gcn_drug2_F
		self.gcn_drug1_ADJ = gcn_drug1_ADJ
		self.gcn_drug2_ADJ = gcn_drug2_ADJ
		self.gcn_exp_A = gcn_exp_A
		self.gcn_exp_B = gcn_exp_B
		self.gcn_adj = gcn_adj
		self.gcn_adj_weight = gcn_adj_weight
		self.syn_ans = syn_ans
		self.cell_info = cell_info
		#
	#
	def __len__(self):
		return len(self.gcn_drug1_F)
			#
	def __getitem__(self, index):
		self.adj_re_A = self.gcn_drug1_ADJ[index].long().to_sparse().indices()
		self.adj_re_B = self.gcn_drug2_ADJ[index].long().to_sparse().indices()
		return self.gcn_drug1_F[index], self.gcn_drug2_F[index], self.adj_re_A, self.adj_re_B, self.gcn_exp_A[index], self.gcn_exp_B[index], self.gcn_adj, self.gcn_adj_weight, self.cell_info[index], self.syn_ans[index] 


def graph_collate_fn(batch):
	drug1_f_list = []
	drug2_f_list = []
	drug1_adj_list = []
	drug2_adj_list = []
	expA_list = []
	expB_list = []
	exp_adj_list = []
	exp_adj_w_list = []
	y_list = []
	cell_list = []
	EXP_num_nodes_seen = 0
	DRUG_1_num_nodes_seen = 0
	DRUG_2_num_nodes_seen = 0
	for drug1_f, drug2_f, drug1_adj, drug2_adj, expA, expB, exp_adj, exp_adj_w, y, cell in batch :
		drug1_f_list.append(drug1_f)
		drug2_f_list.append(drug2_f)
		drug1_adj_list.append(drug1_adj+DRUG_1_num_nodes_seen)
		drug2_adj_list.append(drug2_adj+DRUG_2_num_nodes_seen)
		expA_list.append(expA)
		expB_list.append(expB)
		exp_adj_list.append(exp_adj+EXP_num_nodes_seen)
		exp_adj_w_list.append(exp_adj_w)
		y_list.append(y)
		cell_list.append(cell)
		EXP_num_nodes_seen += expA.shape[0]
		DRUG_1_num_nodes_seen += drug1_f.shape[0]
		DRUG_2_num_nodes_seen += drug2_f.shape[0]
	drug1_f_new = torch.cat(drug1_f_list, 0)
	drug2_f_new = torch.cat(drug2_f_list, 0)
	drug1_adj_new = torch.cat(drug1_adj_list, 1)
	drug2_adj_new = torch.cat(drug2_adj_list, 1)
	expA_new = torch.cat(expA_list, 0)
	expB_new = torch.cat(expB_list, 0)
	exp_adj_new = torch.cat(exp_adj_list, 1)
	exp_adj_w_new = torch.cat(exp_adj_w_list, 1)
	y_new = torch.stack(y_list, 0)
	cell_new = torch.stack(cell_list, 0)
	return drug1_f_new, drug2_f_new, drug1_adj_new, drug2_adj_new, expA_new, expB_new, exp_adj_new, exp_adj_w_new, y_new, cell_new



def weighted_mse_loss(input, target, weight):
	return (weight * (input - target) ** 2).mean()


def result_pearson(y, pred):
	pear = stats.pearsonr(y, pred)
	pear_value = pear[0]
	pear_p_val = pear[1]
	print("Pearson correlation is {} and related p_value is {}".format(pear_value, pear_p_val), flush=True)


def result_spearman(y, pred):
	spear = stats.spearmanr(y, pred)
	spear_value = spear[0]
	spear_p_val = spear[1]
	print("Spearman correlation is {} and related p_value is {}".format(spear_value, spear_p_val), flush=True)


def plot_loss(train_loss, valid_loss, path, plotname):
	fig = plt.figure(figsize=(10,8))
	plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
	plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')
	plt.xlabel('epochs')
	plt.ylabel('loss')
	plt.ylim(0, math.ceil(max(train_loss+valid_loss))) # 일정한 scale
	plt.xlim(0, len(train_loss)+1) # 일정한 scale
	plt.grid(True)
	plt.legend()
	plt.tight_layout()
	fig.savefig('{}/{}.loss_plot.png'.format(path, plotname), bbox_inches = 'tight')


############################ MAIN
print('MAIN')



class MY_expGCN_parallel_model(torch.nn.Module):
	def __init__(self, G_layer_chem, G_indim_chem, G_hiddim_chem, G_layer_exp, G_indim_exp, G_hiddim_exp, layers_1, layers_2, layers_3, cell_dim ,out_dim, inDrop, drop):
		super(MY_expGCN_parallel_model, self).__init__()
		self.G_layer_chem = G_layer_chem
		self.G_indim_chem = G_indim_chem
		self.G_hiddim_chem = G_hiddim_chem
		self.G_layer_exp = G_layer_exp
		self.G_indim_exp = G_indim_exp
		self.G_hiddim_exp = G_hiddim_exp
		self.layers_1 = [int(a) for a in layers_1]
		self.layers_2 = [int(a) for a in layers_2]
		self.layers_3 = [int(a) for a in layers_3]
		self.cell_dim = cell_dim
		self.out_dim = out_dim
		self.inDrop = inDrop
		self.drop = drop
		self.tanh = torch.nn.Tanh()
		self.pool = pyg_nn.global_mean_pool
		#
		self.G_convs_1_chem = torch.nn.ModuleList([pyg_nn.GCNConv(self.G_indim_chem, self.G_hiddim_chem)])
		self.G_convs_1_chem.extend([pyg_nn.GCNConv(self.G_hiddim_chem, self.G_hiddim_chem) for i in range(self.G_layer_chem-2)])
		self.G_convs_1_chem.extend([pyg_nn.GCNConv(self.G_hiddim_chem, self.G_hiddim_chem)])
		self.G_bns_1_chem = torch.nn.ModuleList([torch.nn.BatchNorm1d(self.G_hiddim_chem) for i in range(self.G_layer_chem-1)])
		#
		self.G_convs_2_chem = torch.nn.ModuleList([pyg_nn.GCNConv(self.G_indim_chem, self.G_hiddim_chem)])
		self.G_convs_2_chem.extend([pyg_nn.GCNConv(self.G_hiddim_chem, self.G_hiddim_chem) for i in range(self.G_layer_chem-2)])
		self.G_convs_2_chem.extend([pyg_nn.GCNConv(self.G_hiddim_chem, self.G_hiddim_chem)])
		self.G_bns_2_chem = torch.nn.ModuleList([torch.nn.BatchNorm1d(self.G_hiddim_chem) for i in range(self.G_layer_chem-1)])
		#
		self.G_convs_1_exp = torch.nn.ModuleList([pyg_nn.GCNConv(self.G_indim_exp, self.G_hiddim_exp)])
		self.G_convs_1_exp.extend([pyg_nn.GCNConv(self.G_hiddim_exp, self.G_hiddim_exp) for i in range(self.G_layer_exp-2)])
		self.G_convs_1_exp.extend([pyg_nn.GCNConv(self.G_hiddim_exp, self.G_hiddim_exp)])
		self.G_bns_1_exp = torch.nn.ModuleList([torch.nn.BatchNorm1d(self.G_hiddim_exp) for i in range(self.G_layer_exp-1)])
		#
		self.G_convs_2_exp = torch.nn.ModuleList([pyg_nn.GCNConv(self.G_indim_exp, self.G_hiddim_exp)])
		self.G_convs_2_exp.extend([pyg_nn.GCNConv(self.G_hiddim_exp, self.G_hiddim_exp) for i in range(self.G_layer_exp-2)])
		self.G_convs_2_exp.extend([pyg_nn.GCNConv(self.G_hiddim_exp, self.G_hiddim_exp)])
		self.G_bns_2_exp = torch.nn.ModuleList([torch.nn.BatchNorm1d(self.G_hiddim_exp) for i in range(self.G_layer_exp-1)])
		#
		self.Convs_1 = torch.nn.ModuleList([torch.nn.Linear(self.G_hiddim_chem+self.G_hiddim_exp, self.layers_1[0] )])
		self.Convs_1.extend([torch.nn.Linear(self.layers_1[a], self.layers_1[a+1]) for a in range(len(self.layers_1)-1)])
		#
		self.Convs_2 = torch.nn.ModuleList([torch.nn.Linear(self.G_hiddim_chem+self.G_hiddim_exp, self.layers_2[0] )])
		self.Convs_2.extend([torch.nn.Linear(self.layers_2[a], self.layers_2[a+1]) for a in range(len(self.layers_2)-1)])
		#
		self.SNPs = torch.nn.ModuleList([torch.nn.Linear(self.layers_1[-1]+self.layers_2[-1]+self.cell_dim , self.layers_3[0] )])
		self.SNPs.extend([torch.nn.Linear(self.layers_3[a], self.layers_3[a+1]) for a in range(len(self.layers_3)-1)])
		self.SNPs.extend([torch.nn.Linear(self.layers_3[-1], self.out_dim)])
		#
		self.reset_parameters()
	#
	def reset_parameters(self):
		for conv in self.G_convs_1_chem :
			conv.reset_parameters()
		for bns in self.G_bns_1_chem :
			bns.reset_parameters()
		for conv in self.G_convs_2_chem :
			conv.reset_parameters()
		for bns in self.G_bns_2_chem :
			bns.reset_parameters()
		for conv in self.G_convs_1_exp :
			conv.reset_parameters()
		for bns in self.G_bns_1_exp :
			bns.reset_parameters()
		for conv in self.G_convs_2_exp :
			conv.reset_parameters()
		for bns in self.G_bns_2_exp :
			bns.reset_parameters()
		for conv in self.Convs_1 :
			conv.reset_parameters()
		for conv in self.Convs_2:
			conv.reset_parameters()
		for conv in self.SNPs:
			conv.reset_parameters()
	#
	def calc_batch_label (self, syn, feat) :
		batchnum = syn.shape[0]
		nodenum = feat.shape[0]/batchnum
		Num = [a for a in range(batchnum)]
		Rep = np.repeat(Num, nodenum)
		batch_labels = torch.Tensor(Rep).long()
		if torch.cuda.is_available():
			batch_labels = batch_labels.cuda()
		return batch_labels
	#
	def forward(self, Drug1_F, Drug2_F, Drug1_ADJ, Drug2_ADJ, EXP1, EXP2, EXP_ADJ, EXP_ADJ_WGT, syn, cell ):
		Drug_batch_label = self.calc_batch_label(syn, Drug1_F)
		Exp_batch_label = self.calc_batch_label(syn, EXP1)
		#
		for G_1_C in range(len(self.G_convs_1_chem)):
			if G_1_C == len(self.G_convs_1_chem)-1 :
				Drug1_F = self.G_convs_1_chem[G_1_C](x=Drug1_F, edge_index=Drug1_ADJ)
				Drug1_F = F.dropout(Drug1_F, p=self.inDrop, training=self.training)
				Drug1_F = self.pool(Drug1_F, Drug_batch_label )
				Drug1_F = self.tanh(Drug1_F)
				G_1_C_out = Drug1_F
			else :
				Drug1_F = self.G_convs_1_chem[G_1_C](x=Drug1_F, edge_index=Drug1_ADJ)
				Drug1_F = self.G_bns_1_chem[G_1_C](Drug1_F)
				Drug1_F = F.elu(Drug1_F)
		#
		for G_2_C in range(len(self.G_convs_2_chem)):
			if G_2_C == len(self.G_convs_2_chem)-1 :
				Drug2_F = self.G_convs_2_chem[G_2_C](x=Drug2_F, edge_index=Drug2_ADJ)
				Drug2_F = F.dropout(Drug2_F, p=self.inDrop, training=self.training)
				Drug2_F = self.pool(Drug2_F, Drug_batch_label )
				Drug2_F = self.tanh(Drug2_F)
				G_2_C_out = Drug2_F
			else :
				Drug2_F = self.G_convs_2_chem[G_2_C](x=Drug2_F, edge_index=Drug2_ADJ)
				Drug2_F = self.G_bns_2_chem[G_2_C](Drug2_F)
				Drug2_F = F.elu(Drug2_F)
		#
		for G_1_E in range(len(self.G_convs_1_exp)):
			if G_1_E == len(self.G_convs_1_exp)-1 :
				EXP1 = self.G_convs_1_exp[G_1_E](x=EXP1, edge_index=EXP_ADJ, edge_weight= EXP_ADJ_WGT)
				EXP1 = F.dropout(EXP1, p=self.inDrop, training=self.training)
				EXP1 = self.pool(EXP1, Exp_batch_label )
				EXP1 = self.tanh(EXP1)
				G_1_E_out = EXP1
			else :
				EXP1 = self.G_convs_1_exp[G_1_E](x=EXP1, edge_index=EXP_ADJ, edge_weight= EXP_ADJ_WGT)
				EXP1 = self.G_bns_1_exp[G_1_E](EXP1)
				EXP1 = F.elu(EXP1)
		#
		for G_2_E in range(len(self.G_convs_2_exp)):
			if G_2_E == len(self.G_convs_2_exp)-1 :
				EXP2 = self.G_convs_2_exp[G_2_E](x=EXP2, edge_index=EXP_ADJ, edge_weight= EXP_ADJ_WGT)
				EXP2 = F.dropout(EXP2, p=self.inDrop, training=self.training)
				EXP2 = self.pool(EXP2, Exp_batch_label )
				EXP2 = self.tanh(EXP2)
				G_2_E_out = EXP2
			else :
				EXP2 = self.G_convs_2_exp[G_2_E](x=EXP2, edge_index=EXP_ADJ, edge_weight= EXP_ADJ_WGT)
				EXP2 = self.G_bns_2_exp[G_2_E](EXP2)
				EXP2 = F.elu(EXP2)
		#
		input_drug1 = torch.concat( (G_1_C_out, G_1_E_out), 1 ) # normalization 추가 해야할것 같은데 
		input_drug2 = torch.concat( (G_2_C_out, G_2_E_out), 1 )
		#
		for L1 in range(len(self.Convs_1)):
			if L1 != len(self.Convs_1)-1 :
				input_drug1 = self.Convs_1[L1](input_drug1)
				input_drug1 = F.dropout(input_drug1, p=self.inDrop, training = self.training)
				input_drug1 = F.relu(input_drug1)
			else :
				input_drug1 = self.Convs_1[L1](input_drug1)
		#
		for L2 in range(len(self.Convs_2)):
			if L2 != len(self.Convs_2)-1 :
				input_drug2 = self.Convs_2[L2](input_drug2)
				input_drug2 = F.dropout(input_drug2, p=self.inDrop, training = self.training)
				input_drug2 = F.relu(input_drug2)
			else :
				input_drug2 = self.Convs_2[L2](input_drug2)
		#
		X = torch.cat(( input_drug1, input_drug2, cell ), 1)
		for L3 in range(len(self.SNPs)):
			if L3 != len(self.SNPs)-1 :
				X = self.SNPs[L3](X)
				X = F.dropout(X, p=self.drop, training = self.training)
				X = F.relu(X)
			else :
				X = self.SNPs[L3](X)
		return X






WORK_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_GPU/PJ01.TRIAL.{}/'.format(G_NAME)

MY_chem_A_feat = torch.load(WORK_PATH+'1006.{}.MY_chem_A_feat.pt'.format(G_NAME))
MY_chem_B_feat = torch.load(WORK_PATH+'1006.{}.MY_chem_B_feat.pt'.format(G_NAME))
MY_chem_A_adj = torch.load(WORK_PATH+'1006.{}.MY_chem_A_adj.pt'.format(G_NAME))
MY_chem_B_adj = torch.load(WORK_PATH+'1006.{}.MY_chem_B_adj.pt'.format(G_NAME))
MY_g_feat_A = torch.load(WORK_PATH+'1006.{}.MY_g_feat_A.pt'.format(G_NAME))
MY_g_feat_B = torch.load(WORK_PATH+'1006.{}.MY_g_feat_B.pt'.format(G_NAME))
MY_Cell = torch.load(WORK_PATH+'1006.{}.MY_Cell.pt'.format(G_NAME))
MY_syn = torch.load(WORK_PATH+'1006.{}.MY_syn.pt'.format(G_NAME))

A_B_C_S_SET_RE = pd.read_csv(WORK_PATH+'5_3_3.MJ_1_NF.A_B_C_S.csv')




# for_CAN_smiles = pd.read_csv(PC_PATH+'CID_SMILES.csv',sep = '\t', low_memory = False)
# for_CAN_smiles = copy.deepcopy(PC_FILTER)
#for_CAN_smiles = for_CAN_smiles[['CID','CAN_SMILES']]
for_CAN_smiles.columns = ['drug_row_cid','ROW_CAN_SMILES']
A_B_C_S_SM1 = pd.merge(A_B_C_S_SET_RE, for_CAN_smiles, on='drug_row_cid', how ='left' )
for_CAN_smiles.columns = ['drug_col_cid','COL_CAN_SMILES']
A_B_C_S_SM2 = pd.merge(A_B_C_S_SM1, for_CAN_smiles, on='drug_col_cid', how ='left' )
for_CAN_smiles.columns = ['CID','CAN_SMILES']

aa = list(A_B_C_S_SM2['ROW_CAN_SMILES'])
bb = list(A_B_C_S_SM2['COL_CAN_SMILES'])
cc = list(A_B_C_S_SM2['DrugCombCello'])
A_B_C_S_SM2['SM_SM'] = [aa[i] + '___' + bb[i]+ '___' + cc[i] for i in range(A_B_C_S_SM2.shape[0])]

A_B_C_S_SM3 = [aa[i] + '___' + bb[i]+ '___' + cc[i] for i in range(A_B_C_S_SM2.shape[0])]







seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# gcn_drug1_F, gcn_drug2_F, gcn_drug1_ADJ, gcn_drug2_ADJ, gcn_exp_A, gcn_exp_B, gcn_tgt_A, gcn_tgt_B, gcn_adj, gcn_adj_weight, syn_ans, cell_info
norm = 'tanh_norm'
chem_feat_A_train, chem_feat_A_tv, chem_feat_B_train, chem_feat_B_tv, chem_adj_A_train, chem_adj_A_tv, chem_adj_B_train, chem_adj_B_tv, exp_A_train, exp_A_tv, exp_B_train, exp_B_tv,  cell_train, cell_tv, syn_train, syn_tv, SM_t, SM_tv  = sklearn.model_selection.train_test_split(
		MY_chem_A_feat, MY_chem_B_feat, MY_chem_A_adj, MY_chem_B_adj, MY_g_feat_A, MY_g_feat_B, MY_Cell, MY_syn, A_B_C_S_SM3,
		test_size= 0.2 , random_state=42 )
chem_feat_A_val, chem_feat_A_test, chem_feat_B_val, chem_feat_B_test, chem_adj_A_val, chem_adj_A_test, chem_adj_B_val, chem_adj_B_test, exp_A_val, exp_A_test, exp_B_val, exp_B_test, cell_val, cell_test, syn_val, syn_test, SM_v, SM_test  = sklearn.model_selection.train_test_split(
		chem_feat_A_tv, chem_feat_B_tv, chem_adj_A_tv, chem_adj_B_tv, exp_A_tv, exp_B_tv, cell_tv, syn_tv, SM_tv,
		test_size=0.5, random_state=42 )

train_data, val_data, test_data = prepare_data_GCN(MY_chem_A_feat, MY_chem_B_feat, MY_chem_A_adj, MY_chem_B_adj, MY_g_feat_A, MY_g_feat_B, MY_Cell, MY_syn, norm)



rmv_ind = [a for a in range(len(SM_test)) if SM_test[a] in SM_t+SM_v]
len(rmv_ind)
selec_ind = [a for a in range(len(SM_test)) if a not in rmv_ind]



# WEIGHT 
ys = train_data['y'].squeeze().tolist()
min_s = np.amin(ys)
loss_weight = np.log(train_data['y'] - min_s + np.e)
JY_IDX_WEIGHT_T = torch.Tensor(JY_IDX_WEIGHT).view(1,-1)





# DATA check 
T_train = DATASET_GCN_W_FT(
	torch.Tensor(train_data['drug1_feat']), torch.Tensor(train_data['drug2_feat']), 
	torch.Tensor(train_data['drug1_adj']), torch.Tensor(train_data['drug2_adj']),
	torch.Tensor(train_data['EXP_A']), torch.Tensor(train_data['EXP_B']), 
	JY_ADJ_IDX, JY_IDX_WEIGHT_T,
	torch.Tensor(train_data['cell']), 
	torch.Tensor(train_data['y']))

T_val = DATASET_GCN_W_FT(
	torch.Tensor(val_data['drug1_feat']), torch.Tensor(val_data['drug2_feat']), 
	torch.Tensor(val_data['drug1_adj']), torch.Tensor(val_data['drug2_adj']),
	val_data['EXP_A'], val_data['EXP_B'], 
	JY_ADJ_IDX, JY_IDX_WEIGHT_T,
	torch.Tensor(val_data['cell']), 
	torch.Tensor(val_data['y']))
	
T_test = DATASET_GCN_W_FT(
	torch.Tensor(test_data['drug1_feat']), torch.Tensor(test_data['drug2_feat']), 
	torch.Tensor(test_data['drug1_adj']), torch.Tensor(test_data['drug2_adj']),
	test_data['EXP_A'], test_data['EXP_B'], 
	JY_ADJ_IDX, JY_IDX_WEIGHT_T,
	torch.Tensor(test_data['cell']), 
	torch.Tensor(test_data['y']))

T_test = DATASET_GCN_W_FT(
	torch.Tensor(test_data['drug1_feat'][selec_ind]), torch.Tensor(test_data['drug2_feat'][selec_ind]), 
	torch.Tensor(test_data['drug1_adj'][selec_ind]), torch.Tensor(test_data['drug2_adj'][selec_ind]),
	test_data['EXP_A'][selec_ind], test_data['EXP_B'][selec_ind], 
	JY_ADJ_IDX, JY_IDX_WEIGHT_T,
	torch.Tensor(test_data['cell'][selec_ind]), 
	torch.Tensor(test_data['y'][selec_ind]))

RAY_train = ray.put(T_train)
RAY_val = ray.put(T_val)
RAY_test = ray.put(T_test)
RAY_loss_weight = ray.put(loss_weight)




import pandas as pd 
import matplotlib.pyplot as plt
from ray.tune import ExperimentAnalysis
import pickle
import math
import torch




def RAY_TEST_MODEL(my_config, model_path, model_name, PRJ_PATH, G_NAME, number): 
    use_cuda =  False
    T_test = ray.get(RAY_test)
    Test_loader = torch.utils.data.DataLoader(T_test, batch_size = my_config['config/batch_size'].item(), collate_fn = graph_collate_fn, shuffle =False, num_workers=my_config['config/n_workers'].item())
    #
    G_layer = my_config['config/G_layer'].item()
    G_hiddim = my_config['config/G_hiddim'].item()
    dsn1_layers = [my_config['config/feat_size_0'].item(), my_config['config/feat_size_1'].item(), my_config['config/feat_size_2'].item()]
    dsn2_layers = [my_config['config/feat_size_0'].item(), my_config['config/feat_size_1'].item(), my_config['config/feat_size_2'].item()] 
    snp_layers = [my_config['config/feat_size_3'].item() , my_config['config/feat_size_4'].item()]
    inDrop = my_config['config/dropout_1'].item()
    Drop = my_config['config/dropout_2'].item()
    #    
    best_model = MY_expGCN_parallel_model(
                G_layer, T_test.gcn_drug1_F.shape[-1] , G_hiddim,
                G_layer, 2, G_hiddim,
                dsn1_layers, dsn2_layers, snp_layers, MY_Cell.shape[1], 1,
                inDrop, Drop
                )
    #
    if torch.cuda.is_available():
        best_model = best_model.cuda()
        print('model to cuda', flush = True)
        if torch.cuda.device_count() > 1 :
            best_model = torch.nn.DataParallel(best_model)
            print('model to multi cuda', flush = True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        state_dict = torch.load(os.path.join(model_path, model_name))
    else:
        state_dict = torch.load(os.path.join(model_path, model_name), map_location=torch.device('cpu'))
    #
    if type(state_dict) == tuple:
        best_model.load_state_dict(state_dict[0])
    else : 
        best_model.load_state_dict(state_dict)	#
    #
    #
    best_model.eval()
    test_loss = 0.0
    PRED_list = []
    Y_list = []
    with torch.no_grad():
        best_model.eval()
        for batch_idx_t, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y) in enumerate(Test_loader):
            expA = expA.view(-1,2)
            expB = expB.view(-1,2)
            adj_w = adj_w.squeeze()
            if use_cuda:
                drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell = drug1_f.cuda(), drug2_f.cuda(), drug1_a.cuda(), drug2_a.cuda(), expA.cuda(), expB.cuda(), adj.cuda(), adj_w.cuda(), y.cuda(), cell.cuda()
            output = best_model(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w,  y, cell) 
            MSE = torch.nn.MSELoss()
            loss = MSE(output, y)
            test_loss = test_loss + loss.item()
            Y_list = Y_list + y.view(-1).tolist()
            outputs = output.view(-1).tolist()
            PRED_list = PRED_list+outputs
        TEST_LOSS = test_loss/(batch_idx_t+1)
        R__T = TEST_LOSS
        R__1 , R__2 = jy_corrplot(PRED_list, Y_list, PRJ_PATH,'P.{}.{}_model'.format( G_NAME, number) )
        return  R__T, R__1, R__2


def jy_corrplot(PRED_list, Y_list, path, plotname ):
	jplot = sns.jointplot(x=PRED_list, y=Y_list, ci=68, kind='reg')
	pr,pp = stats.pearsonr(PRED_list, Y_list)
	print("Pearson correlation is {} and related p_value is {}".format(pr, pp))
	sr,sp = stats.spearmanr(PRED_list, Y_list)
	print("Spearman correlation is {} and related p_value is {}".format(sr, sp))
	jplot.ax_joint.annotate(f'$pearson = {pr:.3f}, spearman = {sr:.3f}$',xy=(min(PRED_list)+ 0.01, max(Y_list)- 0.01 ), ha='left', va='center',)
	jplot.ax_joint.scatter(PRED_list, Y_list)
	jplot.set_axis_labels(xlabel='Predicted', ylabel='Answer', size=15)
	jplot.figure.savefig('{}/{}.corrplot.png'.format(path, plotname), bbox_inches = 'tight')
	return pr, sr



def plot_loss(train_loss, valid_loss, path, plotname):
	fig = plt.figure(figsize=(10,8))
	plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
	plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')
	plt.xlabel('epochs')
	plt.ylabel('loss')
	plt.ylim(0, math.ceil(max(train_loss+valid_loss))) # 일정한 scale
	plt.xlim(0, len(train_loss)+1) # 일정한 scale
	plt.grid(True)
	plt.legend()
	plt.tight_layout()
	fig.savefig('{}/{}.loss_plot.png'.format(path, plotname), bbox_inches = 'tight')





PRJ_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_GPU/PJ01.TRIAL.{}/'.format(G_NAME)
ANA_DF = pd.read_csv(PRJ_PATH+'RAY_ANA_DF.{}.csv'.format(G_NAME))
with open(PRJ_PATH+'RAY_ANA_DF.{}.pickle'.format(G_NAME), 'rb') as f:
	ANA_ALL_DF = pickle.load(f)



# 5_3_3.MJ_1_NF

PRJ_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_GPU/PJ01.TRIAL.5_3_3.MJ_1_NF/'
ANA_DF = pd.read_csv(PRJ_PATH+'RAY_ANA_DF.5_3_3.MJ_1_NF.csv.1')
with open(PRJ_PATH+'RAY_ANA_DF.5_3_3.MJ_1_NF.pickle.1', 'rb') as f:
	ANA_ALL_DF = pickle.load(f)


# G_NAME = '5_3_3.MJ_2_MJB' 
TOPVAL_PATH = PRJ_PATH



# 1) best final
#
#
DF_KEY = list(ANA_DF.sort_values('ValLoss')['logdir'])[0]
print('best final', flush=True)
print(DF_KEY, flush=True)
#TOPVAL_PATH = DF_KEY
mini_df = ANA_ALL_DF[DF_KEY]
my_config = ANA_DF[ANA_DF.logdir==DF_KEY]
R_1_V = list(ANA_DF.sort_values('ValLoss')['ValLoss'])[0]
R_1_T, R_1_1, R_1_2 = RAY_TEST_MODEL(my_config, TOPVAL_PATH, 'M1_model.pth', PRJ_PATH, G_NAME, 'M1')


plot_loss(list(mini_df.TrainLoss), list(mini_df.ValLoss), PRJ_PATH, "M1_M2")


cck_num =mini_df[mini_df.ValLoss==min(mini_df.ValLoss)].index.item()
checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
#TOPVAL_PATH = DF_KEY + checkpoint
print('best final check', flush=True)
print(TOPVAL_PATH, flush=True)
R_2_V = min(mini_df.ValLoss)
R_2_T, R_2_1, R_2_2 = RAY_TEST_MODEL(my_config, TOPVAL_PATH, 'M2_checkpoint', PRJ_PATH, G_NAME, 'M2')

import numpy as np
TOT_min = np.Inf
TOT_key = ""
for key in ANA_ALL_DF.keys():
	trial_min = min(ANA_ALL_DF[key]['ValLoss'])
	if trial_min < TOT_min :
		TOT_min = trial_min
		TOT_key = key

print('best val', flush=True)
print(TOT_key, flush=True)
mini_df = ANA_ALL_DF[TOT_key]
#TOPVAL_PATH = TOT_key
my_config = ANA_DF[ANA_DF.logdir==TOT_key]
cck_num =mini_df[mini_df.ValLoss==min(mini_df.ValLoss)].index.item()
checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
#TOPVAL_PATH = TOT_key + checkpoint
print('best val check', flush=True)
print(TOPVAL_PATH, flush=True)
R_3_V = min(mini_df.ValLoss)
R_3_T, R_3_1, R_3_2 = RAY_TEST_MODEL(my_config, TOPVAL_PATH, 'M4_checkpoint', PRJ_PATH, G_NAME, 'M4')






def final_result(R_1_V, R_1_T, R_1_1, R_1_2, R_2_V, R_2_T, R_2_1, R_2_2, R_3_V, R_3_T, R_3_1, R_3_2) :
	print('---1---', flush=True)
	print('- Val MSE : {:.2f}'.format(R_1_V), flush=True)
	print('- Test MSE : {:.2f}'.format(R_1_T), flush=True)
	print('- Test Pearson : {:.2f}'.format(R_1_1), flush=True)
	print('- Test Spearman : {:.2f}'.format(R_1_2), flush=True)
	print('---2---', flush=True)
	print('- Val MSE : {:.2f}'.format(R_2_V), flush=True)
	print('- Test MSE : {:.2f}'.format(R_2_T), flush=True)
	print('- Test Pearson : {:.2f}'.format(R_2_1), flush=True)
	print('- Test Spearman : {:.2f}'.format(R_2_2), flush=True)
	print('---3---', flush=True)
	print('- Val MSE : {:.2f}'.format(R_3_V), flush=True)
	print('- Test MSE : {:.2f}'.format(R_3_T), flush=True)
	print('- Test Pearson : {:.2f}'.format(R_3_1), flush=True)
	print('- Test Spearman : {:.2f}'.format(R_3_2), flush=True)


final_result(R_1_V, R_1_T, R_1_1, R_1_2, R_2_V, R_2_T, R_2_1, R_2_2, R_3_V, R_3_T, R_3_1, R_3_2)








































use_cuda =  False
T_test = ray.get(RAY_test)
Test_loader = torch.utils.data.DataLoader(T_test, batch_size = my_config['config/batch_size'].item(), collate_fn = graph_collate_fn, shuffle =False, num_workers=my_config['config/n_workers'].item())
#
G_layer = my_config['config/G_layer'].item()
G_hiddim = my_config['config/G_hiddim'].item()
dsn1_layers = [my_config['config/feat_size_0'].item(), my_config['config/feat_size_1'].item(), my_config['config/feat_size_2'].item()]
dsn2_layers = [my_config['config/feat_size_0'].item(), my_config['config/feat_size_1'].item(), my_config['config/feat_size_2'].item()] 
snp_layers = [my_config['config/feat_size_3'].item() , my_config['config/feat_size_4'].item()]
inDrop = my_config['config/dropout_1'].item()
Drop = my_config['config/dropout_2'].item()
#    
best_model = MY_expGCN_parallel_model(
			G_layer, T_test.gcn_drug1_F.shape[-1] , G_hiddim,
			G_layer, 2, G_hiddim,
			dsn1_layers, dsn2_layers, snp_layers, MY_Cell.shape[1], 1,
			inDrop, Drop
			)
#
device = "cuda:0" if torch.cuda.is_available() else "cpu"


state_dict = torch.load(os.path.join(PRJ_PATH, 'M1_model.pth'), map_location=torch.device('cpu'))
#
if type(state_dict) == tuple:
	best_model.load_state_dict(state_dict[0])
else : 
	best_model.load_state_dict(state_dict)	#
#
#
best_model.eval()
test_loss = 0.0
PRED_list = []
Y_list = []
with torch.no_grad():
	best_model.eval()
	for batch_idx_t, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y) in enumerate(Test_loader):
		expA = expA.view(-1,2)
		expB = expB.view(-1,2)
		adj_w = adj_w.squeeze()
		if use_cuda:
			drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell = drug1_f.cuda(), drug2_f.cuda(), drug1_a.cuda(), drug2_a.cuda(), expA.cuda(), expB.cuda(), adj.cuda(), adj_w.cuda(), y.cuda(), cell.cuda()
		output = best_model(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w,  y, cell) 
		MSE = torch.nn.MSELoss()
		loss = MSE(output, y)
		test_loss = test_loss + loss.item()
		Y_list = Y_list + y.view(-1).tolist()
		outputs = output.view(-1).tolist()
		PRED_list = PRED_list+outputs
TEST_LOSS = test_loss/(batch_idx_t+1)
R__T = TEST_LOSS
R__1 , R__2 = jy_corrplot(PRED_list, Y_list, PRJ_PATH,'P.{}.{}_model_retest'.format( G_NAME, 'M1') )



