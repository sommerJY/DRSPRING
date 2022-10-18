

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


G_NAME = '5_2_3.MJ_1_NF'

WORK_PATH = '/home01/k006a01/PRJ.01/TRIAL.5_2_3/'
DC_PATH = '/home01/k006a01/01.DATA/DrugComb/'
IDK_PATH = '/home01/k006a01/01.DATA/IDK/'
LINCS_PATH = '/home01/k006a01/01.DATA/LINCS/' #
TARGET_PATH = '/home01/k006a01/01.DATA/TARGET/'
cid_PATH = '/home01/k006a01/01.DATA/PC_ATC_LIST/'

# WORK_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_6/DATA.5_2_3/'

MY_chem_A_feat = torch.load(WORK_PATH+'1001.{}.MY_chem_A_feat.pt'.format(G_NAME))
MY_chem_B_feat = torch.load(WORK_PATH+'1001.{}.MY_chem_B_feat.pt'.format(G_NAME))
MY_chem_A_adj = torch.load(WORK_PATH+'1001.{}.MY_chem_A_adj.pt'.format(G_NAME))
MY_chem_B_adj = torch.load(WORK_PATH+'1001.{}.MY_chem_B_adj.pt'.format(G_NAME))
MY_g_feat_A = torch.load(WORK_PATH+'1001.{}.MY_g_feat_A.pt'.format(G_NAME))
MY_g_feat_B = torch.load(WORK_PATH+'1001.{}.MY_g_feat_B.pt'.format(G_NAME))
MY_Cell = torch.load(WORK_PATH+'1001.{}.MY_Cell.pt'.format(G_NAME))
MY_syn = torch.load(WORK_PATH+'1001.{}.MY_syn.pt'.format(G_NAME))



print('NETWORK')

hunet_dir = '/home01/k006a01/01.DATA/HumanNet/'
# hunet_dir = '/st06/jiyeonH/13.DD_SESS/HumanNetV3/'


hunet_gsp = pd.read_csv(hunet_dir+'HumanNet-GSP.tsv', sep = '\t', header = None)
hunet_gsp.columns = ['G_A','G_B']

L_matching_list = pd.read_csv(IDK_PATH+'L_12_string.csv', sep = '\t')

hnet_IAS_L1 = hunet_gsp[hunet_gsp['G_A'].isin(L_matching_list.entrez)]
hnet_IAS_L2 = hnet_IAS_L1[hnet_IAS_L1['G_B'].isin(L_matching_list.entrez)] # 20232

len(set(list(hnet_IAS_L2['G_A']) + list(hnet_IAS_L2['G_B']))) # 611
ID_G = nx.from_pandas_edgelist(hnet_IAS_L2, 'G_A', 'G_B')

ESSEN_NAMES = list(set(L_matching_list['entrez']))

MSSNG = [a for  a in ESSEN_NAMES if a not in list(ID_G.nodes)]
# MSSNG = L_matching_list[L_matching_list['PPI_name'].isin(list(ID_G.nodes))==False]['PPI_name']
for nn in list(MSSNG):
	ID_G.add_node(nn)

# edge 3871
ID_GENE_ORDER_mini = list(ID_G.nodes()) # 978
ID_ADJ = nx.adjacency_matrix(ID_G)
ID_ADJ_tmp = torch.LongTensor(ID_ADJ.toarray())
ID_ADJ_IDX = ID_ADJ_tmp.to_sparse().indices()  # [2, 7742]
ID_WEIGHT = [] # len : 3871 -> 7742

ID_WEIGHT_SCORE = [1 for a in range(ID_ADJ_IDX.shape[1])]


new_node_name = []
for a in ID_G.nodes():
	tmp_name = list(set(L_matching_list[L_matching_list.entrez == a ]['L_gene_symbol'])) # 6118
	new_node_name = new_node_name + tmp_name

mapping = {list(ID_G.nodes())[a]:new_node_name[a] for a in range(len(new_node_name))}

ID_G_RE = nx.relabel_nodes(ID_G, mapping)

MY_G = ID_G_RE 
MY_WEIGHT_SCORE = ID_WEIGHT_SCORE # SCORE



# TARGET # 민지가 다시 올려준다고 함 

TARGET_DB = pd.read_csv(TARGET_PATH+'combined_target.csv', low_memory=False)



# Graph 확인 

JY_GRAPH = MY_G
JY_GRAPH_ORDER = MY_G.nodes()
JY_ADJ = nx.adjacency_matrix(JY_GRAPH)

JY_ADJ_tmp = torch.LongTensor(JY_ADJ.toarray())
JY_ADJ_IDX = JY_ADJ_tmp.to_sparse().indices()
JY_IDX_WEIGHT = MY_WEIGHT_SCORE







print('input ok', flush=True)

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
	chem_feat_A_train, chem_feat_A_tv, chem_feat_B_train, chem_feat_B_tv, chem_adj_A_train, chem_adj_A_tv, chem_adj_B_train, chem_adj_B_tv, exp_A_train, exp_A_tv, exp_B_train, exp_B_tv,  cell_train, cell_tv, syn_train, syn_tv  = sklearn.model_selection.train_test_split(
			MY_chem_A_feat, MY_chem_B_feat, MY_chem_A_adj, MY_chem_B_adj, MY_g_feat_A, MY_g_feat_B, MY_Cell, MY_syn,
			test_size= 0.2 , random_state=42 )
	chem_feat_A_val, chem_feat_A_test, chem_feat_B_val, chem_feat_B_test, chem_adj_A_val, chem_adj_A_test, chem_adj_B_val, chem_adj_B_test, exp_A_val, exp_A_test, exp_B_val, exp_B_test, cell_val, cell_test, syn_val, syn_test   = sklearn.model_selection.train_test_split(
			chem_feat_A_tv, chem_feat_B_tv, chem_adj_A_tv, chem_adj_B_tv, exp_A_tv, exp_B_tv, cell_tv, syn_tv,
			test_size=0.5, random_state=42 )
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
	for drug1_f, drug2_f, drug1_adj, drug2_adj, expA, expB, exp_adj, exp_adj_w, cell, y in batch :
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
	return drug1_f_new, drug2_f_new, drug1_adj_new, drug2_adj_new, expA_new, expB_new, exp_adj_new, exp_adj_w_new, cell_new, y_new



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




seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# gcn_drug1_F, gcn_drug2_F, gcn_drug1_ADJ, gcn_drug2_ADJ, gcn_exp_A, gcn_exp_B, gcn_tgt_A, gcn_tgt_B, gcn_adj, gcn_adj_weight, syn_ans, cell_info
norm = 'tanh_norm'
train_data, val_data, test_data = prepare_data_GCN(MY_chem_A_feat, MY_chem_B_feat, MY_chem_A_adj, MY_chem_B_adj, MY_g_feat_A, MY_g_feat_B, MY_Cell, MY_syn, norm)


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

RAY_train = ray.put(T_train)
RAY_val = ray.put(T_val)
RAY_test = ray.put(T_test)
RAY_loss_weight = ray.put(loss_weight)


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





def RAY_MY_train(config, checkpoint_dir=None):
	n_epochs = config["epoch"]
	criterion = weighted_mse_loss
	use_cuda =  False# True #  False
	#
	T_train = ray.get(RAY_train)
	T_val = ray.get(RAY_val)
	T_test = ray.get(RAY_test)
	T_loss_weight = ray.get(RAY_loss_weight)
	batch_cut_weight = [T_loss_weight[i:i+config["batch_size"]] for i in range(0,len(T_loss_weight), config["batch_size"])]
	#
	loaders = {
			'train' : torch.utils.data.DataLoader(T_train, batch_size = config["batch_size"], collate_fn = graph_collate_fn, shuffle =False, num_workers=config['n_workers']),
			'eval' : torch.utils.data.DataLoader(T_val, batch_size = config["batch_size"], collate_fn = graph_collate_fn, shuffle =False, num_workers=config['n_workers']),
			'test' : torch.utils.data.DataLoader(T_test, batch_size = config["batch_size"], collate_fn = graph_collate_fn, shuffle =False, num_workers=config['n_workers']),
	}
	#
	dsn1_layers = [config["feat_size_0"], config["feat_size_1"] , config["feat_size_2"] ]
	dsn2_layers = [config["feat_size_0"], config["feat_size_1"] , config["feat_size_2"] ]
	snp_layers = [config["feat_size_3"] ,config["feat_size_4"] ,]
	inDrop = config["dropout_1"]
	Drop = config["dropout_2"]
	#
	#
	MM_MODEL = MY_expGCN_parallel_model(
			config["G_layer"], T_train.gcn_drug1_F.shape[-1] , config["G_hiddim"],
			config["G_layer"], 2 , config["G_hiddim"],
			dsn1_layers, dsn2_layers, snp_layers, MY_Cell.shape[1], 1,
			inDrop, Drop
			)
	#
	if torch.cuda.is_available():
		MM_MODEL = MM_MODEL.cuda()
		if torch.cuda.device_count() > 1 :
			MM_MODEL = torch.nn.DataParallel(MM_MODEL)
	#       
	optimizer = torch.optim.Adam(MM_MODEL.parameters(), lr = config["lr"] )
	if checkpoint_dir :
		checkpoint = os.path.join(checkpoint_dir, "checkpoint")
		model_state, optimizer_state = torch.load(checkpoint)
		model.load_state_dict(model_state)
		optimizer.load_state_dict(optimizer_state)
	#
	train_loss_all = []
	valid_loss_all = []
	#
	for epoch in range(n_epochs):
		now = datetime.now()
		train_loss = 0.0
		valid_loss = 0.0
		#
		###################
		# train the model #
		###################
		MM_MODEL.train()
		for batch_idx_t, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y) in enumerate(loaders['train']):
			expA = expA.view(-1,2)#### 다른점 
			expB = expB.view(-1,2)#### 다른점 
			adj_w = adj_w.squeeze()
			# move to GPU
			if use_cuda:
				drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell = drug1_f.cuda(), drug2_f.cuda(), drug1_a.cuda(), drug2_a.cuda(), expA.cuda(), expB.cuda(), adj.cuda(), adj_w.cuda(), y.cuda(), cell.cuda() 
			## find the loss and update the model parameters accordingly
			# clear the gradients of all optimized variables
			optimizer.zero_grad()
			output = MM_MODEL(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell)
			wc = torch.Tensor(batch_cut_weight[batch_idx_t]).view(-1,1)
			if torch.cuda.is_available():
				wc = wc.cuda()
			loss = criterion(output, y, wc ) # weight 더해주기 
			loss.backward()
			optimizer.step()
			## record the average training loss, using something like
			## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
			train_loss = train_loss + loss.item()
		#
		######################    
		# validate the model #
		######################
		MM_MODEL.eval()
		for batch_idx_v, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y) in enumerate(loaders['eval']):
			expA = expA.view(-1,2)#### 다른점 
			expB = expB.view(-1,2)#### 다른점 
			adj_w = adj_w.squeeze()
			# move to GPU
			if use_cuda:
				drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell = drug1_f.cuda(), drug2_f.cuda(), drug1_a.cuda(), drug2_a.cuda(), expA.cuda(), expB.cuda(), adj.cuda(), adj_w.cuda(), y.cuda(), cell.cuda()
			## update the average validation loss
			output = MM_MODEL(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell)
			MSE = torch.nn.MSELoss()
			loss = MSE(output, y) # train 이 아니라서 weight 안넣어줌. 그냥 nn.MSE 넣어주기 
			# update average validation loss 
			valid_loss = valid_loss + loss.item()
		#
		# calculate average losses
		TRAIN_LOSS = train_loss/(batch_idx_t+1)
		train_loss_all.append(TRAIN_LOSS)
		VAL_LOSS = valid_loss/(batch_idx_v+1)
		valid_loss_all.append(VAL_LOSS)
		#
		# print training/validation statistics 
		#
		done = datetime.now()
		time_spent = done-now
		#
		with tune.checkpoint_dir(step = epoch) as checkpoint_dir:
			trial_name = '_'.join(checkpoint_dir.split('/')[-2].split('_')[0:4])
			print('trial : {}, epoch : {}, TrainLoss : {}, ValLoss : {}'.format(trial_name, epoch,TRAIN_LOSS,VAL_LOSS), flush=True)
			path = os.path.join(checkpoint_dir, "checkpoint")
			torch.save((MM_MODEL.state_dict(), optimizer.state_dict()), path)
			torch.save(MM_MODEL.state_dict(), './model.pth')
		tune.report(TrainLoss= TRAIN_LOSS,  ValLoss=VAL_LOSS )
	#
	print("Finished Training")
 

def RAY_TEST_MODEL(my_config, model_path, model_name, PRJ_PATH, Trial_name, G_NAME, number): 
	use_cuda =  False# True # False 
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
	print("state_dict_done", flush = True)
	print(type(state_dict))
	print(len(state_dict))
	if type(state_dict) == tuple:
		best_model.load_state_dict(state_dict[0])
	else : 
		best_model.load_state_dict(state_dict)	#
	print("state_load_done", flush = True)
	#
	#
	best_model.eval()
	test_loss = 0.0
	PRED_list = []
	Y_list = test_data['y'].squeeze().tolist()
	with torch.no_grad():
		best_model.eval()
		for batch_idx_t, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y) in enumerate(Test_loader):
			expA = expA.view(-1,2)
			expB = expB.view(-1,2)
			adj_w = adj_w.squeeze()
			if use_cuda:
				drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell = drug1_f.cuda(), drug2_f.cuda(), drug1_a.cuda(), drug2_a.cuda(), expA.cuda(), expB.cuda(), adj.cuda(), adj_w.cuda(), y.cuda(), cell.cuda()
			output= best_model(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell) 
			MSE = torch.nn.MSELoss()
			loss = MSE(output, y)
			test_loss = test_loss + loss.item()
			outputs = output.squeeze().tolist()
			PRED_list = PRED_list+outputs
	TEST_LOSS = test_loss/(batch_idx_t+1)
	R__T = TEST_LOSS
	R__1 , R__2 = jy_corrplot(PRED_list, Y_list, PRJ_PATH,'P.{}.{}.{}_model'.format(Trial_name, G_NAME, number) )
	return  R__T, R__1, R__2






def jy_corrplot(PRED_list, Y_list, path, plotname ):
	jplot = sns.jointplot(x=PRED_list, y=Y_list, ci=68, kind='reg')
	pr,pp = stats.pearsonr(PRED_list, Y_list)
	print("Pearson correlation is {} and related p_value is {}".format(pr, pp), flush=True)
	sr,sp = stats.spearmanr(PRED_list, Y_list)
	print("Spearman correlation is {} and related p_value is {}".format(sr, sp), flush=True)
	jplot.ax_joint.annotate(f'$pearson = {pr:.3f}, spearman = {sr:.3f}$',xy=(min(PRED_list)+ 0.01, max(Y_list)- 0.01 ), ha='left', va='center',)
	jplot.ax_joint.scatter(PRED_list, Y_list)
	jplot.set_axis_labels(xlabel='Predicted', ylabel='Answer', size=15)
	jplot.figure.savefig('{}/{}.corrplot.png'.format(path, plotname), bbox_inches = 'tight')
	return pr, sr






def final_result(R_1_V, R_1_T, R_1_1, R_1_2, R_2_V, R_2_T, R_2_1, R_2_2, R_3_V, R_3_T, R_3_1, R_3_2) :
	print('---1---')
	print('- Val MSE : {:.2f}'.format(R_1_V))
	print('- Test MSE : {:.2f}'.format(R_1_T))
	print('- Test Pearson : {:.2f}'.format(R_1_1))
	print('- Test Spearman : {:.2f}'.format(R_1_2))
	print('---2---')
	print('- Val MSE : {:.2f}'.format(R_2_V))
	print('- Test MSE : {:.2f}'.format(R_2_T))
	print('- Test Pearson : {:.2f}'.format(R_2_1))
	print('- Test Spearman : {:.2f}'.format(R_2_2))
	print('---3---')
	print('- Val MSE : {:.2f}'.format(R_3_V))
	print('- Test MSE : {:.2f}'.format(R_3_T))
	print('- Test Pearson : {:.2f}'.format(R_3_1))
	print('- Test Spearman : {:.2f}'.format(R_3_2))




from ray.tune import ExperimentAnalysis



def MAIN(ANAL_name, WORK_PATH, PRJ_PATH, Trial_name, G_NAME, num_samples= 10, max_num_epochs=1000, grace_period = 150, cpus_per_trial = 16, gpus_per_trial = 1):
	# OPTUNA CONFIGURE 
	CONFIG={
		'n_workers' : tune.choice([cpus_per_trial]),
		"epoch" : tune.choice([max_num_epochs]),
		"G_layer" : tune.choice([2, 3, 4]), # 
		"G_hiddim" : tune.choice([512, 256, 128, 64, 32]), # 
		"batch_size" : tune.choice([ 128, 64, 32, 16]), # CPU 니까 
		"feat_size_0" : tune.choice([4096, 2048, 1024, 512, 256, 128, 64, 32]), # 4096, 2048, 1024, 512, 256, 128, 64, 32
		"feat_size_1" : tune.choice([4096, 2048, 1024, 512, 256, 128, 64, 32]),# 4096, 2048, 1024, 512, 256, 128, 64, 32
		"feat_size_2" : tune.choice([4096, 2048, 1024, 512, 256, 128, 64, 32]),# 4096, 2048, 1024, 512, 256, 128, 64, 32
		"feat_size_3" : tune.choice([4096, 2048, 1024, 512, 256, 128, 64, 32]),# 4096, 2048, 1024, 512, 256, 128, 64, 32
		"feat_size_4" : tune.choice([4096, 2048, 1024, 512, 256, 128, 64, 32]),# 4096, 2048, 1024, 512, 256, 128, 64, 32
		"dropout_1" : tune.choice([0.01, 0.2, 0.5, 0.8]), # 0.01, 0.2, 0.5, 0.8
		"dropout_2" : tune.choice([0.01, 0.2, 0.5, 0.8]), # 0.01, 0.2, 0.5, 0.8
		"lr" : tune.choice([0.00001, 0.0001, 0.001]),# 0.00001, 0.0001, 0.001
	}
	#
	#
	reporter = CLIReporter(
		metric_columns=["TrainLoss", "ValLoss",  "training_iteration"])
	#
	optuna_search = OptunaSearch(metric="ValLoss", mode="min")
	#
	ASHA_scheduler = tune.schedulers.ASHAScheduler(
		time_attr='training_iteration', metric="ValLoss", mode="min", max_t= max_num_epochs, grace_period = grace_period )
	#
	#
	ANALYSIS = tune.run( # 끝내지 않음 
		tune.with_parameters(RAY_MY_train),
		name = ANAL_name,
		num_samples=num_samples,
		config=CONFIG,
		resources_per_trial={'cpu': cpus_per_trial }, # , 'gpu' : gpus_per_trial
		progress_reporter = reporter,
		search_alg = optuna_search,
		scheduler = ASHA_scheduler
	)
	best_trial = ANALYSIS.get_best_trial("ValLoss", "min", "last")
	print("Best trial config: {}".format(best_trial.config), flush=True)
	print("Best trial final validation loss: {}".format(
	best_trial.last_result["ValLoss"]), flush=True)
	#
	#
	anal_df = ExperimentAnalysis("~/ray_results/{}".format(ANAL_name))
	#
	# 1) best final
	#
	ANA_DF = anal_df.dataframe()
	ANA_ALL_DF = anal_df.trial_dataframes
	#
	DF_KEY = list(ANA_DF.sort_values('ValLoss')['logdir'])[0]
	print('best final', flush=True)
	print(DF_KEY, flush=True)
	TOPVAL_PATH = DF_KEY
	mini_df = ANA_ALL_DF[DF_KEY]
	my_config = ANA_DF[ANA_DF.logdir==DF_KEY]
	R_1_V = list(ANA_DF.sort_values('ValLoss')['ValLoss'])[0]
	R_1_T, R_1_1, R_1_2 = RAY_TEST_MODEL(my_config, TOPVAL_PATH, 'model.pth', PRJ_PATH, Trial_name, G_NAME, 'M1')
	#
	# 2) best final's checkpoint
	# 
	cck_num =mini_df[mini_df.ValLoss==min(mini_df.ValLoss)].index.item()
	checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
	TOPVAL_PATH = DF_KEY + checkpoint
	print('best final check', flush=True)
	print(TOPVAL_PATH, flush=True)
	R_2_V = min(mini_df.ValLoss)
	R_2_T, R_2_1, R_2_2 = RAY_TEST_MODEL(my_config, TOPVAL_PATH, 'checkpoint', PRJ_PATH, Trial_name, G_NAME, 'M2')
	#
	# 3) total checkpoint best 
	#	
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
	TOPVAL_PATH = TOT_key
	my_config = ANA_DF[ANA_DF.logdir==TOT_key]
	cck_num =mini_df[mini_df.ValLoss==min(mini_df.ValLoss)].index.item()
	checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
	TOPVAL_PATH = TOT_key + checkpoint
	print('best val check', flush=True)
	print(TOPVAL_PATH, flush=True)
	R_3_V = min(mini_df.ValLoss)
	R_3_T, R_3_1, R_3_2 = RAY_TEST_MODEL(my_config, TOPVAL_PATH, 'checkpoint', PRJ_PATH, Trial_name, G_NAME, 'M4')
	#
	final_result(R_1_V, R_1_T, R_1_1, R_1_2, R_2_V, R_2_T, R_2_1, R_2_2, R_3_V, R_3_T, R_3_1, R_3_2)
	return ANALYSIS






# local
MAIN('22.10.01.PRJ01.TRIAL.5_2_3', WORK_PATH, WORK_PATH, '5_2_3', 'MJ_1_MJB', 3, 3, 1, 32, 1)

# 4gpu
MAIN('22.10.01.PRJ01.TRIAL.5_2_3', WORK_PATH, WORK_PATH, '5_2_3', 'MJ_1_MJB', 100, 1000, 150, 32, 1)

# 8gpu
MAIN('22.10.01.PRJ01.TRIAL.5_2_3', WORK_PATH, WORK_PATH, '5_2_3', 'MJ_1_MJB', 100, 1000, 150, 16, 1)
MAIN('22.10.01.PRJ01.TRIAL.5_2_3', WORK_PATH, WORK_PATH, '5_2_3', 'MJ_1_MJB', 100, 1000, 150, 8, 0.5)




