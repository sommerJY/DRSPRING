# 349 version





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

#import ray
#from ray import tune
#from functools import partial
#from ray.tune.schedulers import ASHAScheduler
#from ray.tune import CLIReporter
#from ray.tune.suggest.optuna import OptunaSearch
#from ray.tune import ExperimentAnalysis

import numpy as np
from rdkit import Chem
from rdkit.Chem.QED import qed
import sys
import os
import pandas as pd

import torch.multiprocessing as mp
import torch.distributed as dist

from torch.utils.data.distributed import DistributedSampler


import torch.distributed as dist
from argparse import ArgumentParser
import torch.multiprocessing as mp

import layers




NETWORK_PATH = '/home01/k020a01/01.Data/HumanNet/'
LINCS_PATH = '/home01/k020a01/01.Data/LINCS/'
DC_PATH = '/home01/k020a01/01.Data/DrugComb/'
PRJ_PATH ='/home01/k020a01/TEST'




class DATASET_GCN_W_FT(Dataset):
	def __init__(self, gcn_drug1_F, gcn_drug2_F, gcn_drug1_ADJ, gcn_drug2_ADJ, 
	gcn_gene_A, gcn_gene_B, target_A, target_B, cell_basal, gcn_adj, gcn_adj_weight, 
	cell_info, syn_ans ):
		self.gcn_drug1_F = gcn_drug1_F
		self.gcn_drug2_F = gcn_drug2_F
		self.gcn_drug1_ADJ = gcn_drug1_ADJ
		self.gcn_drug2_ADJ = gcn_drug2_ADJ
		self.gcn_gene_A = gcn_gene_A
		self.gcn_gene_B = gcn_gene_B
		self.target_A = target_A
		self.target_B = target_B
		self.cell_basal = cell_basal
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
		adj_re_A = self.gcn_drug1_ADJ[index].long().to_sparse().indices()
		adj_re_B = self.gcn_drug2_ADJ[index].long().to_sparse().indices()
		#
		FEAT_A = torch.Tensor(np.array([ self.gcn_gene_A[index].squeeze().tolist() , self.target_A[index].tolist(), self.cell_basal[index].tolist()]).T)
		FEAT_B = torch.Tensor(np.array([ self.gcn_gene_B[index].squeeze().tolist() , self.target_B[index].tolist(), self.cell_basal[index].tolist()]).T)
		#
		return self.gcn_drug1_F[index], self.gcn_drug2_F[index],adj_re_A, adj_re_B, FEAT_A, FEAT_B, self.gcn_adj, self.gcn_adj_weight , self.cell_info[index], self.syn_ans[index]



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
	#
	for drug1_f, drug2_f, drug1_adj, drug2_adj, expA, expB, exp_adj, exp_adj_w, cell, y in batch :
		drug1_f_list.append(drug1_f)
		drug2_f_list.append(drug2_f)
		drug1_adj_list.append(drug1_adj+DRUG_1_num_nodes_seen)
		drug2_adj_list.append(drug2_adj+DRUG_2_num_nodes_seen)
		expA_list.append(expA)
		expB_list.append(expB)
		exp_adj_list.append(exp_adj+EXP_num_nodes_seen)
		exp_adj_w_list.append(exp_adj_w)
		y_list.append(torch.Tensor(y))
		cell_list.append(torch.Tensor(cell))
		EXP_num_nodes_seen += expA.shape[0]
		DRUG_1_num_nodes_seen += drug1_f.shape[0]
		DRUG_2_num_nodes_seen += drug2_f.shape[0]
	#
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
	return drug1_f_new, drug2_f_new, drug1_adj_new, drug2_adj_new, expA_new, expB_new, exp_adj_new, exp_adj_w_new, cell_new, y_new






def prepare_data_GCN(CV_ND_INDS, data_nodup_df3, data_nodup_df2, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, 
MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, 
cell_one_hot, MY_syn_RE2, norm ) : 
	#        
	CV_num = 0
	train_key = 'CV{}_train'.format(CV_num)
	val_key = 'CV{}_val'.format(CV_num)
	test_key = 'CV{}_test'.format(CV_num)
	#
	train_no_dup = data_nodup_df3.loc[CV_ND_INDS[train_key]]
	val_no_dup = data_nodup_df3.loc[CV_ND_INDS[val_key]]
	test_no_dup = data_nodup_df2.loc[CV_ND_INDS[test_key]]
	#
	ABCS_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(train_no_dup.setset)]
	ABCS_val = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(val_no_dup.setset)]
	ABCS_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(test_no_dup.setset)]
	#
	train_ind = list(ABCS_train.index)
	val_ind = list(ABCS_val.index)
	test_ind = list(ABCS_test.index)
	tv_ind = list(ABCS_train.index) + list(ABCS_val.index)
	# 
	chem_feat_A_train = MY_chem_A_feat_RE2[train_ind]; chem_feat_A_val = MY_chem_A_feat_RE2[val_ind]; chem_feat_A_test = MY_chem_A_feat_RE2[test_ind]; chem_feat_A_tv = MY_chem_A_feat_RE2[tv_ind]
	chem_feat_B_train = MY_chem_B_feat_RE2[train_ind]; chem_feat_B_val = MY_chem_B_feat_RE2[val_ind]; chem_feat_B_test = MY_chem_B_feat_RE2[test_ind]; chem_feat_B_tv = MY_chem_B_feat_RE2[tv_ind]
	chem_adj_A_train = MY_chem_A_adj_RE2[train_ind]; chem_adj_A_val = MY_chem_A_adj_RE2[val_ind]; chem_adj_A_test = MY_chem_A_adj_RE2[test_ind]; chem_adj_A_tv = MY_chem_A_adj_RE2[tv_ind]
	chem_adj_B_train = MY_chem_B_adj_RE2[train_ind]; chem_adj_B_val = MY_chem_B_adj_RE2[val_ind]; chem_adj_B_test = MY_chem_B_adj_RE2[test_ind]; chem_adj_B_tv = MY_chem_B_adj_RE2[tv_ind]
	gene_A_train = MY_g_EXP_A_RE2[train_ind]; gene_A_val = MY_g_EXP_A_RE2[val_ind]; gene_A_test = MY_g_EXP_A_RE2[test_ind]; gene_A_tv = MY_g_EXP_A_RE2[tv_ind]
	gene_B_train = MY_g_EXP_B_RE2[train_ind]; gene_B_val = MY_g_EXP_B_RE2[val_ind]; gene_B_test = MY_g_EXP_B_RE2[test_ind]; gene_B_tv = MY_g_EXP_B_RE2[tv_ind]
	target_A_train = MY_Target_A2[train_ind]; target_A_val = MY_Target_A2[val_ind]; target_A_test = MY_Target_A2[test_ind]; target_A_tv = MY_Target_A2[tv_ind]
	target_B_train = MY_Target_B2[train_ind]; target_B_val = MY_Target_B2[val_ind]; target_B_test = MY_Target_B2[test_ind]; target_B_tv = MY_Target_B2[tv_ind]
	cell_basal_train = MY_CellBase_RE2[train_ind]; cell_basal_val = MY_CellBase_RE2[val_ind]; cell_basal_test = MY_CellBase_RE2[test_ind]; cell_basal_tv = MY_CellBase_RE2[tv_ind]
	cell_train = cell_one_hot[train_ind]; cell_val = cell_one_hot[val_ind]; cell_test = cell_one_hot[test_ind]; cell_tv = cell_one_hot[tv_ind]
	syn_train = MY_syn_RE2[train_ind]; syn_val = MY_syn_RE2[val_ind]; syn_test = MY_syn_RE2[test_ind]; syn_tv = MY_syn_RE2[tv_ind]
	#
	train_data = {}
	val_data = {}
	test_data = {}
	tv_data = {}
	#
	train_data['drug1_feat'] = torch.concat([chem_feat_A_train, chem_feat_B_train], axis = 0)
	val_data['drug1_feat'] = chem_feat_A_val
	test_data['drug1_feat'] = chem_feat_A_test
	tv_data['drug1_feat'] = torch.concat([chem_feat_A_tv, chem_feat_B_tv], axis = 0)
	#
	train_data['drug2_feat'] = torch.concat([chem_feat_B_train, chem_feat_A_train], axis = 0)
	val_data['drug2_feat'] = chem_feat_B_val
	test_data['drug2_feat'] = chem_feat_B_test
	tv_data['drug2_feat'] = torch.concat([chem_feat_B_tv, chem_feat_A_tv], axis = 0)
	#
	train_data['drug1_adj'] = torch.concat([chem_adj_A_train, chem_adj_B_train], axis = 0)
	val_data['drug1_adj'] = chem_adj_A_val
	test_data['drug1_adj'] = chem_adj_A_test
	tv_data['drug1_adj'] = torch.concat([chem_adj_A_tv, chem_adj_B_tv], axis = 0)
	#
	train_data['drug2_adj'] = torch.concat([chem_adj_B_train, chem_adj_A_train], axis = 0)
	val_data['drug2_adj'] = chem_adj_B_val
	test_data['drug2_adj'] = chem_adj_B_test
	tv_data['drug2_adj'] = torch.concat([chem_adj_A_tv, chem_adj_B_tv], axis = 0)
	#
	train_data['GENE_A'] = torch.concat([gene_A_train, gene_B_train], axis = 0)
	val_data['GENE_A'] = gene_A_val
	test_data['GENE_A'] = gene_A_test
	tv_data['GENE_A'] = torch.concat([gene_A_tv, gene_B_tv], axis = 0)
	#
	train_data['GENE_B'] = torch.concat([gene_B_train, gene_A_train], axis = 0)
	val_data['GENE_B'] = gene_B_val
	test_data['GENE_B'] = gene_B_test
	tv_data['GENE_B'] = torch.concat([gene_B_tv, gene_A_tv], axis = 0)
	#
	train_data['TARGET_A'] = torch.concat([target_A_train, target_B_train], axis = 0)
	val_data['TARGET_A'] = target_A_val
	test_data['TARGET_A'] = target_A_test
	tv_data['TARGET_A'] = torch.concat([target_A_tv, target_B_tv], axis = 0)
	#
	train_data['TARGET_B'] = torch.concat([target_B_train, target_A_train], axis = 0)
	val_data['TARGET_B'] = target_B_val
	test_data['TARGET_B'] = target_B_test
	tv_data['TARGET_B'] = torch.concat([target_B_tv, target_A_tv], axis = 0)
	#   #
	train_data['cell_BASAL'] = torch.concat((cell_basal_train, cell_basal_train), axis=0)
	val_data['cell_BASAL'] = cell_basal_val
	test_data['cell_BASAL'] = cell_basal_test
	tv_data['cell_BASAL'] = torch.concat((cell_basal_tv, cell_basal_tv), axis=0)
	##
	train_data['cell'] = torch.concat((cell_train, cell_train), axis=0)
	val_data['cell'] = cell_val
	test_data['cell'] = cell_test
	tv_data['cell'] = torch.concat((cell_tv, cell_tv), axis=0)
	#            
	train_data['y'] = torch.concat((syn_train, syn_train), axis=0)
	val_data['y'] = syn_val
	test_data['y'] = syn_test
	tv_data['y'] = torch.concat((syn_tv, syn_tv), axis=0)
	#
	print(train_data['drug1_feat'].shape, flush=True)
	print(val_data['drug1_feat'].shape, flush=True)
	print(test_data['drug1_feat'].shape, flush=True)
	print(tv_data['drug1_feat'].shape, flush=True)
	return train_data, val_data, test_data, tv_data



# WEIGHT 
def get_loss_weight(train_data) :
	#train_data = globals()['train_data_'+str(CV)]
	ys = train_data['y'].squeeze().tolist()
	min_s = np.amin(ys)
	loss_weight = np.log(train_data['y'] - min_s + np.e)
	return loss_weight




# 
def make_merged_data(train_data, val_data, test_data, tv_data, JY_ADJ_IDX, JY_IDX_WEIGHT_T) :
	#train_data = globals()['train_data_'+str(CV)]
	#val_data = globals()['val_data_'+str(CV)]
	#test_data = globals()['test_data_'+str(CV)]
	#
	T_train = DATASET_GCN_W_FT(
		torch.Tensor(train_data['drug1_feat']), torch.Tensor(train_data['drug2_feat']), 
		torch.Tensor(train_data['drug1_adj']), torch.Tensor(train_data['drug2_adj']),
		torch.Tensor(train_data['GENE_A']), torch.Tensor(train_data['GENE_B']), 
		torch.Tensor(train_data['TARGET_A']), torch.Tensor(train_data['TARGET_B']), torch.Tensor(train_data['cell_BASAL']), 
		JY_ADJ_IDX, JY_IDX_WEIGHT_T, 
		train_data['cell'].float(),
		torch.Tensor(train_data['y'])
		)
	#
	T_val = DATASET_GCN_W_FT(
		torch.Tensor(val_data['drug1_feat']), torch.Tensor(val_data['drug2_feat']), 
		torch.Tensor(val_data['drug1_adj']), torch.Tensor(val_data['drug2_adj']),
		torch.Tensor(val_data['GENE_A']), torch.Tensor(val_data['GENE_B']), 
		torch.Tensor(val_data['TARGET_A']), torch.Tensor(val_data['TARGET_B']), torch.Tensor(val_data['cell_BASAL']), 
		JY_ADJ_IDX, JY_IDX_WEIGHT_T, 
		val_data['cell'].float(),
		torch.Tensor(val_data['y'])
		)
	#	
	T_test = DATASET_GCN_W_FT(
		torch.Tensor(test_data['drug1_feat']), torch.Tensor(test_data['drug2_feat']), 
		torch.Tensor(test_data['drug1_adj']), torch.Tensor(test_data['drug2_adj']),
		torch.Tensor(test_data['GENE_A']), torch.Tensor(test_data['GENE_B']), 
		torch.Tensor(test_data['TARGET_A']), torch.Tensor(test_data['TARGET_B']), torch.Tensor(test_data['cell_BASAL']), 
		JY_ADJ_IDX, JY_IDX_WEIGHT_T, 
		test_data['cell'].float(),
		torch.Tensor(test_data['y'])
		)
	#
	T_tv = DATASET_GCN_W_FT(
		torch.Tensor(tv_data['drug1_feat']), torch.Tensor(tv_data['drug2_feat']), 
		torch.Tensor(tv_data['drug1_adj']), torch.Tensor(tv_data['drug2_adj']),
		torch.Tensor(tv_data['GENE_A']), torch.Tensor(tv_data['GENE_B']), 
		torch.Tensor(tv_data['TARGET_A']), torch.Tensor(tv_data['TARGET_B']), torch.Tensor(tv_data['cell_BASAL']), 
		JY_ADJ_IDX, JY_IDX_WEIGHT_T, 
		tv_data['cell'].float(),
		torch.Tensor(tv_data['y'])
		)
	#
	return T_train, T_val, T_test, T_tv









def prepare_input(MJ_NAME, WORK_DATE, MISS_NAME, file_name, WORK_NAME, CELL_CUT):
	#
	print('NETWORK')
	#
	#
	hunet_gsp = pd.read_csv(NETWORK_PATH+'HS-DB.tsv', sep = '\t', header = None)
	hunet_gsp.columns = ['G_A','G_B','SC']
	#
	LINCS_gene_file = pd.read_csv(LINCS_PATH+'geneinfo_beta.txt', sep = '\t')
	LINCS_978 = LINCS_gene_file[LINCS_gene_file.feature_space == 'landmark']
	LINCS_978 = LINCS_978[['gene_id','gene_symbol']]
	LINCS_978['new_node'] = [str(list(LINCS_978.gene_id)[i]) + "__" + list(LINCS_978.gene_symbol)[i] for i in range(978)]
	LINCS_978 = LINCS_978.reset_index(drop=True)
	lm_entrezs = list(LINCS_978.gene_id)
	#
	hnet_L1 = hunet_gsp[hunet_gsp['G_A'].isin(lm_entrezs)]
	hnet_L2 = hnet_L1[hnet_L1['G_B'].isin(lm_entrezs)] # 
	len(set(list(hnet_L2['G_A']) + list(hnet_L2['G_B']))) # 
	hnet_L3 = hnet_L2[hnet_L2.SC >= 3.5]
	ID_G = nx.from_pandas_edgelist(hnet_L3, 'G_A', 'G_B')
	#
	#
	# edge 3871
	ID_GENE_ORDER_mini = list(ID_G.nodes()) # 978
	ID_ADJ = nx.adjacency_matrix(ID_G)
	ID_ADJ_tmp = torch.LongTensor(ID_ADJ.toarray())
	ID_ADJ_IDX = ID_ADJ_tmp.to_sparse().indices()  # [2, 7742]
	ID_WEIGHT = [] # len : 3871 -> 7742
	ID_WEIGHT_SCORE = [1 for a in range(ID_ADJ_IDX.shape[1])]
	#
	#
	#
	# 유전자 이름으로 붙이기 
	new_node_names = []
	for a in ID_G.nodes():
		tmp_name = LINCS_978[LINCS_978.gene_id == a ]['gene_symbol'].item() # 6118
		new_node_name = str(a) + '__' + tmp_name
		new_node_names = new_node_names + [new_node_name]
	#
	mapping = {list(ID_G.nodes())[a]:new_node_names[a] for a in range(len(new_node_names))}
	#
	ID_G_RE = nx.relabel_nodes(ID_G, mapping)
	MY_G = ID_G_RE 
	MY_WEIGHT_SCORE = ID_WEIGHT_SCORE # SCORE
	#
	# Graph 확인 
	#
	JY_GRAPH = MY_G
	JY_GRAPH_ORDER = MY_G.nodes()
	JY_ADJ = nx.adjacency_matrix(JY_GRAPH)
	#
	JY_ADJ_tmp = torch.LongTensor(JY_ADJ.toarray())
	JY_ADJ_IDX = JY_ADJ_tmp.to_sparse().indices()
	JY_IDX_WEIGHT = MY_WEIGHT_SCORE
	#
	#
	# LINCS exp order 따지기 
	BETA_ORDER_pre = [list(LINCS_978.new_node).index(a) for a in JY_GRAPH_ORDER]
	BETA_ORDER_DF = LINCS_978.iloc[BETA_ORDER_pre] # 어차피 ref 다르고 같은 애들이라 괜춘 
	BETA_ENTREZ_ORDER = list(BETA_ORDER_DF.gene_id)
	BETA_SYMBOL_ORDER = list(BETA_ORDER_DF.gene_symbol)
	BETA_NEWNOD_ORDER = list(BETA_ORDER_DF.new_node)
	#
	#
	#MJ_NAME = 'M3V4'
	#WORK_DATE = '23.03.16'
	#MISS_NAME = 'MIS2'
	# SAVE_PATH = '/home01/k020a01/02.VER3/{}_FULL_DATA/'.format(MJ_NAME)
	##### SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V4_CCLE_FULL/'
	SAVE_PATH = '/home01/k020a01/02.M3V5/{}_FULL_DATA/'.format(MJ_NAME)
	#
	#
	# file_name = 'M3V4ccle_MISS2_FULL'
	A_B_C_S_SET_ADD = pd.read_csv(SAVE_PATH+'{}.A_B_C_S_SET_ADD.csv'.format(file_name), low_memory=False)
	MY_chem_A_feat = torch.load(SAVE_PATH+'{}.MY_chem_A_feat.pt'.format(file_name))
	MY_chem_B_feat = torch.load(SAVE_PATH+'{}.MY_chem_B_feat.pt'.format(file_name))
	MY_chem_A_adj = torch.load(SAVE_PATH+'{}.MY_chem_A_adj.pt'.format(file_name))
	MY_chem_B_adj = torch.load(SAVE_PATH+'{}.MY_chem_B_adj.pt'.format(file_name))
	MY_g_EXP_A = torch.load(SAVE_PATH+'{}.MY_g_EXP_A.pt'.format(file_name))
	MY_g_EXP_B = torch.load(SAVE_PATH+'{}.MY_g_EXP_B.pt'.format(file_name))
	MY_Target_1_A = torch.load(SAVE_PATH+'{}.MY_Target_1_A.pt'.format(file_name))
	MY_Target_1_B = torch.load(SAVE_PATH+'{}.MY_Target_1_B.pt'.format(file_name))
	MY_Target_2_A = torch.load(SAVE_PATH+'{}.MY_Target_2_A.pt'.format(file_name))
	MY_Target_2_B = torch.load(SAVE_PATH+'{}.MY_Target_2_B.pt'.format(file_name))
	MY_CellBase = torch.load(SAVE_PATH+'{}.MY_CellBase.pt'.format(file_name))
	MY_syn = torch.load(SAVE_PATH+'{}.MY_syn.pt'.format(file_name))
	#
	# WORK_NAME = 'WORK_20' # full / new target / cut1 / 
	#MISS_filter = ['AOBO']
	#MISS_filter = ['AOBO','AXBO','AOBX']
	MISS_filter = ['AOBO','AXBO','AOBX','AXBX']
	#
	A_B_C_S_SET = A_B_C_S_SET_ADD[A_B_C_S_SET_ADD.Basal_Exp == 'O']
	#
	A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.SYN_OX == 'O']
	#
	## A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.T2OX == 'O'] ###################### old targets 
	A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.T1OX == 'O'] ####################### new targets 
	#
	A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.type.isin(MISS_filter)]
	#
	# basal node exclude -> CCLE match 만 사용
	CCLE_PATH = '/home01/k020a01/01.Data/CCLE/'
	# CCLE_PATH = '/st06/jiyeonH/13.DD_SESS/CCLE.22Q1/'
	ccle_exp = pd.read_csv(CCLE_PATH+'CCLE_expression.csv', low_memory=False)
	ccle_info= pd.read_csv(CCLE_PATH+'sample_info.csv', low_memory=False)
	#
	ori_col = list( ccle_exp.columns ) # entrez!
	for_gene = ori_col[1:]
	for_gene2 = [int(a.split('(')[1].split(')')[0]) for a in for_gene]
	new_col = ['DepMap_ID']+for_gene2 
	ccle_exp.columns = new_col
	#
	ccle_cell_info = ccle_info[['DepMap_ID','CCLE_Name']]
	ccle_cell_info.columns = ['DepMap_ID','DrugCombCCLE']
	ccle_exp2 = pd.merge(ccle_exp, ccle_cell_info, on = 'DepMap_ID' , how='left')
	ccle_exp3 = ccle_exp2[['DepMap_ID','DrugCombCCLE']+BETA_ENTREZ_ORDER]
	ccle_names = [a for a in ccle_exp3.DrugCombCCLE if type(a) == str]
	#
	A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.DrugCombCCLE.isin(ccle_names)]
	#
	#
	data_ind = list(A_B_C_S_SET.index)
	#
	MY_chem_A_feat_RE = MY_chem_A_feat[data_ind]
	MY_chem_B_feat_RE = MY_chem_B_feat[data_ind]
	MY_chem_A_adj_RE = MY_chem_A_adj[data_ind]
	MY_chem_B_adj_RE = MY_chem_B_adj[data_ind]
	MY_g_EXP_A_RE = MY_g_EXP_A[data_ind]
	MY_g_EXP_B_RE = MY_g_EXP_B[data_ind]
	#
	# MY_Target_A = copy.deepcopy(MY_Target_2_A)[data_ind] ############## OLD TARGET !!!!!! #####
	# MY_Target_B = copy.deepcopy(MY_Target_2_B)[data_ind] ############## OLD TARGET !!!!!! #####
	#
	MY_Target_A = copy.deepcopy(MY_Target_1_A)[data_ind] ############## NEW TARGET !!!!!! #####
	MY_Target_B = copy.deepcopy(MY_Target_1_B)[data_ind] ############## NEW TARGET !!!!!! #####
	#
	MY_CellBase_RE = MY_CellBase[data_ind]
	MY_syn_RE = MY_syn[data_ind]
	#
	A_B_C_S_SET = A_B_C_S_SET.reset_index(drop = True)
	#
	#
	#
	# cell line vector 
	DC_CELL_DF2 = pd.read_csv(DC_PATH+'DC_CELL_INFO.csv', sep = '\t')
	DC_CELL_info_filt = DC_CELL_DF2[DC_CELL_DF2.DrugCombCCLE.isin(A_B_C_S_SET.DrugCombCCLE)] # 38
	A_B_C_S_SET_COH = pd.merge(A_B_C_S_SET, DC_CELL_info_filt[['DrugCombCCLE','DC_cellname']], on = 'DrugCombCCLE', how = 'left'  )
	#
	# sample number filter 
	# 빈도 확인 
	C_names = list(set(A_B_C_S_SET_COH.DC_cellname))
	C_freq = [list(A_B_C_S_SET_COH.DC_cellname).count(a) for a in C_names]
	C_df = pd.DataFrame({'cell' : C_names, 'freq' : C_freq })
	C_df = C_df.sort_values('freq')
	CELL_CUT = 200 ############ WORK 20 ##############
	#
	C_freq_filter = C_df[C_df.freq > CELL_CUT ] 
	#
	A_B_C_S_SET_COH = A_B_C_S_SET_COH[A_B_C_S_SET_COH.DC_cellname.isin(C_freq_filter.cell)]
	#
	DC_CELL_info_filt_re = DC_CELL_info_filt[DC_CELL_info_filt.DC_cellname.isin(C_freq_filter.cell)]
	DC_CELL_info_filt_re['cell_onehot'] = [a for a in range(len(set(DC_CELL_info_filt_re.DrugCombCCLE)))]
	#
	DC_CELL_info_filt_re = DC_CELL_info_filt_re.reset_index(drop = True)
	#
	#
	data_ind = list(A_B_C_S_SET_COH.index)
	#
	MY_chem_A_feat_RE2 = MY_chem_A_feat_RE[data_ind]
	MY_chem_B_feat_RE2 = MY_chem_B_feat_RE[data_ind]
	MY_chem_A_adj_RE2 = MY_chem_A_adj_RE[data_ind]
	MY_chem_B_adj_RE2 = MY_chem_B_adj_RE[data_ind]
	MY_g_EXP_A_RE2 = MY_g_EXP_A_RE[data_ind]
	MY_g_EXP_B_RE2 = MY_g_EXP_B_RE[data_ind]
	MY_Target_A2 = copy.deepcopy(MY_Target_A)[data_ind]
	MY_Target_B2 = copy.deepcopy(MY_Target_B)[data_ind]
	MY_CellBase_RE2 = MY_CellBase_RE[data_ind]
	MY_syn_RE2 = MY_syn_RE[data_ind]
	#
	# merge 전 후로 index 달라지므로 뒤에 넣어줬음 
	A_B_C_S_SET_COH2 = pd.merge(A_B_C_S_SET_COH, DC_CELL_info_filt_re[['DC_cellname','cell_onehot']], on = 'DC_cellname', how='left')
	cell_one_hot = torch.nn.functional.one_hot(torch.Tensor(A_B_C_S_SET_COH2['cell_onehot']).long())
	#
	#
	# SMILES 기준 & cell line 잘 섞이게 데이터 나누기 
	#
	#
	A_B_C_S_SET_SM = copy.deepcopy(A_B_C_S_SET_COH2)
	#
	A_B_C_S_SET_SM['ori_index'] = list(A_B_C_S_SET_SM.index)
	aaa = list(A_B_C_S_SET_SM['drug_row_CID'])
	bbb = list(A_B_C_S_SET_SM['drug_col_CID'])
	aa = list(A_B_C_S_SET_SM['ROW_CAN_SMILES'])
	bb = list(A_B_C_S_SET_SM['COL_CAN_SMILES'])
	cc = list(A_B_C_S_SET_SM['DrugCombCCLE'])
	#
	# 앞뒤 고려하는거 들어감 
	A_B_C_S_SET_SM['CID_CID_CCLE'] = [str(int(aaa[i])) + '___' + str(int(bbb[i]))+ '___' + cc[i] if aaa[i] < bbb[i] else str(int(bbb[i])) + '___' + str(int(aaa[i]))+ '___' + cc[i] for i in range(A_B_C_S_SET_SM.shape[0])]
	A_B_C_S_SET_SM['SM_C_CHECK'] = [aa[i] + '___' + bb[i]+ '___' + cc[i] if aa[i] < bb[i] else bb[i] + '___' + aa[i]+ '___' + cc[i] for i in range(A_B_C_S_SET_SM.shape[0])]
	#
	A_B_C_S_SET_SM[['drug_row_CID','drug_col_CID','DrugCombCCLE']].drop_duplicates() # 52152
	A_B_C_S_SET_SM[['ROW_CAN_SMILES','COL_CAN_SMILES','DrugCombCCLE']].drop_duplicates() # 52120
	len(set(A_B_C_S_SET_SM['CID_CID_CCLE'])) # 51212
	len(set(A_B_C_S_SET_SM['SM_C_CHECK'])) # 51160
	#
	#
	#
	# get unique values, remove duplicates, but keep original counts
	data_no_dup, counts = np.unique(list(A_B_C_S_SET_SM['SM_C_CHECK']), return_counts=True)
	data_no_dup_cells = [setset.split('___')[2] for setset in data_no_dup]
	data_nodup_df = pd.DataFrame({'setset' : data_no_dup.tolist(), 'cell' : data_no_dup_cells })
	data_nodup_df2 = data_nodup_df.sort_values('cell')
	data_nodup_df2 = data_nodup_df2.reset_index(drop =True)
	#
	grouped_df = data_nodup_df2.groupby('cell')
	TrainVal_list = []; Test_list =[]
	#
	for i, g in grouped_df:
		if len(g) > CELL_CUT :
			nums = int(.20 * len(g)) 
			bins = []
			g2 = sklearn.utils.shuffle(g, random_state=42)
			for ii in list(range(0, len(g2), nums)):
				if len(bins)< 5 :
					bins.append(ii)
			#
			bins = bins[1:]
			res = np.split(g2, bins)
			TrainVal_list = TrainVal_list + res[0].index.tolist() + res[1].index.tolist() + res[2].index.tolist() + res[3].index.tolist()
			Test_list = Test_list + res[4].index.tolist()
		else :
			print(i)
	#
	#
	data_nodup_df3 = data_nodup_df2.loc[TrainVal_list]
	data_nodup_df3 = data_nodup_df3.reset_index(drop=True)
	grouped_df2 = data_nodup_df3.groupby('cell')
	#
	CV_1_list = []; CV_2_list = []; CV_3_list = []; CV_4_list = []; CV_5_list = []
	for i, g in grouped_df2:
		nums = int(.2 * len(g)) 
		bins = []
		g2 = sklearn.utils.shuffle(g, random_state=42)
		for ii in list(range(0, len(g2), nums)):
			if len(bins)< 5 :
				bins.append(ii)
		#
		bins = bins[1:]
		res = np.split(g2, bins)
		CV_1_list = CV_1_list + res[0].index.tolist()
		CV_2_list = CV_2_list + res[1].index.tolist()
		CV_3_list = CV_3_list + res[2].index.tolist()
		CV_4_list = CV_4_list + res[3].index.tolist()
		CV_5_list = CV_5_list + res[4].index.tolist()
	#
	CV_ND_INDS = {
		'CV0_train' : CV_1_list + CV_2_list + CV_3_list + CV_4_list, 
		'CV0_val' : CV_5_list, 'CV0_test' : Test_list,
		'CV1_train' : CV_2_list + CV_3_list + CV_4_list + CV_5_list , 
		'CV1_val' : CV_1_list, 'CV1_test' : Test_list,
		'CV2_train' : CV_3_list + CV_4_list + CV_5_list + CV_1_list, 
		'CV2_val' : CV_2_list, 'CV2_test' : Test_list,
		'CV3_train' : CV_4_list + CV_5_list + CV_1_list + CV_2_list,
		'CV3_val' : CV_3_list, 'CV3_test' : Test_list,
		'CV4_train' : CV_5_list + CV_1_list + CV_2_list + CV_3_list,
		'CV4_val' : CV_4_list, 'CV4_test' : Test_list 
	}
	#
	norm = 'tanh_norm'
	train_data, val_data, test_data, tv_data = prepare_data_GCN (CV_ND_INDS, data_nodup_df3, data_nodup_df2, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, 
	MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, 
	cell_one_hot, MY_syn_RE2, norm)
	#
	seed = 42
	random.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)
	#
	#
	#LOSS_WEIGHT_0 = get_loss_weight(train_data) ######## change it !!!! 
	LOSS_WEIGHT_0 = get_loss_weight(tv_data)
	JY_IDX_WEIGHT_T = torch.Tensor(JY_IDX_WEIGHT).view(1,-1)
	#
	#
	T_train_0, T_val_0, T_test_0, T_tv_0 = make_merged_data(train_data, val_data, test_data, tv_data, JY_ADJ_IDX, JY_IDX_WEIGHT_T)
	batch_cut_weight = LOSS_WEIGHT_0
	#
	return A_B_C_S_SET_SM, T_train_0, T_val_0, T_test_0, T_tv_0, batch_cut_weight
