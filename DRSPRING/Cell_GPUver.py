
import rdkit
import os
import sys
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


#NETWORK_PATH = '/st06/jiyeonH/13.DD_SESS/HumanNetV3/'
#LINCS_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/' 
#DC_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/'
#TARGET_PATH = '/home01/k020a01/01.Data/TARGET/'



NETWORK_PATH = '/home01/k020a01/01.Data/HumanNet/'
LINCS_PATH = '/home01/k020a01/01.Data/LINCS/'
DC_PATH = '/home01/k020a01/01.Data/DrugComb/'
TARGET_PATH = '/home01/k020a01/01.Data/TARGET/'



print('NETWORK', flush = True)

# NETWORK_PATH = '/st06/jiyeonH/13.DD_SESS/HumanNetV3/'

# 349 
hunet_gsp = pd.read_csv(NETWORK_PATH+'HS-DB.tsv', sep = '\t', header = None)
hunet_gsp.columns = ['G_A','G_B','SC']

LINCS_gene_file = pd.read_csv(LINCS_PATH+'geneinfo_beta.txt', sep = '\t')
LINCS_978 = LINCS_gene_file[LINCS_gene_file.feature_space == 'landmark']
LINCS_978 = LINCS_978[['gene_id','gene_symbol']]
LINCS_978['new_node'] = [str(list(LINCS_978.gene_id)[i]) + "__" + list(LINCS_978.gene_symbol)[i] for i in range(978)]
LINCS_978 = LINCS_978.reset_index(drop=True)
lm_entrezs = list(LINCS_978.gene_id)


hnet_L1 = hunet_gsp[hunet_gsp['G_A'].isin(lm_entrezs)]
hnet_L2 = hnet_L1[hnet_L1['G_B'].isin(lm_entrezs)] # 3885
hnet_L3 = hnet_L2[hnet_L2.SC >= 3.5]

len(set(list(hnet_L3['G_A']) + list(hnet_L3['G_B']))) # 611

ID_G = nx.from_pandas_edgelist(hnet_L3, 'G_A', 'G_B')

#MSSNG = [a for a in lm_entrezs if a not in list(ID_G.nodes)]

#for nn in list(MSSNG):
#	ID_G.add_node(nn)


# edge 3871
ID_GENE_ORDER_mini = list(ID_G.nodes()) # 978
ID_ADJ = nx.adjacency_matrix(ID_G)
ID_ADJ_tmp = torch.LongTensor(ID_ADJ.toarray())
ID_ADJ_IDX = ID_ADJ_tmp.to_sparse().indices()  # [2, 7742]
ID_WEIGHT = [] # len : 3871 -> 7742

ID_WEIGHT_SCORE = [1 for a in range(ID_ADJ_IDX.shape[1])]



# 유전자 이름으로 붙이기 

new_node_names = []
for a in ID_G.nodes():
	tmp_name = LINCS_978[LINCS_978.gene_id == a ]['gene_symbol'].item() # 6118
	new_node_name = str(a) + '__' + tmp_name
	new_node_names = new_node_names + [new_node_name]

mapping = {list(ID_G.nodes())[a]:new_node_names[a] for a in range(len(new_node_names))}

ID_G_RE = nx.relabel_nodes(ID_G, mapping)

MY_G = ID_G_RE 
MY_WEIGHT_SCORE = ID_WEIGHT_SCORE # SCORE



# Graph 확인 

JY_GRAPH = MY_G
JY_GRAPH_ORDER = MY_G.nodes()
JY_ADJ = nx.adjacency_matrix(JY_GRAPH)

JY_ADJ_tmp = torch.LongTensor(JY_ADJ.toarray())
JY_ADJ_IDX = JY_ADJ_tmp.to_sparse().indices()
JY_IDX_WEIGHT = MY_WEIGHT_SCORE


# LINCS exp order 따지기 
BETA_ORDER_pre = [list(LINCS_978.new_node).index(a) for a in JY_GRAPH_ORDER]
BETA_ORDER_DF = LINCS_978.iloc[BETA_ORDER_pre] # 어차피 ref 다르고 같은 애들이라 괜춘 
BETA_ENTREZ_ORDER = list(BETA_ORDER_DF.gene_id)
BETA_SYMBOL_ORDER = list(BETA_ORDER_DF.gene_symbol)
BETA_NEWNOD_ORDER = list(BETA_ORDER_DF.new_node)




# ABCS check 

MJ_NAME = 'M3V4'
MISS_NAME = 'MIS2'

SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V4_349_FULL/'

file_name = 'M3V4_349_MISS2_FULL'
A_B_C_S_SET_ADD = pd.read_csv(SAVE_PATH+'{}.A_B_C_S_SET_ADD.csv'.format(file_name), low_memory=False)




# A_B_C_S SET filter check
WORK_NAME = 'WORK_20v1'


#MISS_filter = ['AOBO']
#MISS_filter = ['AOBO','AXBO','AOBX']
MISS_filter = ['AOBO','AXBO','AOBX','AXBX']


A_B_C_S_SET = A_B_C_S_SET_ADD[A_B_C_S_SET_ADD.SYN_OX == 'O']
###A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.T2OX == 'O'] # old targets 
A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.T1OX == 'O'] # new targets 
A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.type.isin(MISS_filter)]


CCLE_PATH = '/home01/k020a01/01.Data/CCLE/'
# CCLE_PATH = '/st06/jiyeonH/13.DD_SESS/CCLE.22Q1/'
ccle_exp = pd.read_csv(CCLE_PATH+'CCLE_expression.csv', low_memory=False)
ccle_info= pd.read_csv(CCLE_PATH+'sample_info.csv', low_memory=False)

ori_col = list( ccle_exp.columns ) # entrez!
for_gene = ori_col[1:]
for_gene2 = [int(a.split('(')[1].split(')')[0]) for a in for_gene]
new_col = ['DepMap_ID']+for_gene2 
ccle_exp.columns = new_col

ccle_cell_info = ccle_info[['DepMap_ID','CCLE_Name']]
ccle_cell_info.columns = ['DepMap_ID','DrugCombCCLE']
ccle_exp2 = pd.merge(ccle_exp, ccle_cell_info, on = 'DepMap_ID' , how='left')
ccle_exp3 = ccle_exp2[['DepMap_ID','DrugCombCCLE']+BETA_ENTREZ_ORDER]
ccle_cello_names = [a for a in ccle_exp3.DrugCombCCLE if type(a) == str]

A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.DrugCombCCLE.isin(ccle_cello_names)]



# cell line vector 
# DC_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/'
DC_CELL_DF2 = pd.read_csv(DC_PATH+'DC_CELL_INFO.csv', sep = '\t')

DC_CELL_info_filt = DC_CELL_DF2[DC_CELL_DF2.DrugCombCCLE.isin(A_B_C_S_SET.DrugCombCCLE)] # 38
DC_CELL_info_filt['cell_onehot'] = [a for a in range(len(set(A_B_C_S_SET.DrugCombCCLE)))]
DC_CELL_info_filt = DC_CELL_info_filt.reset_index(drop = True)

A_B_C_S_SET_COH = pd.merge(A_B_C_S_SET, DC_CELL_info_filt[['DrugCombCCLE','DC_cellname']], on = 'DrugCombCCLE', how = 'left'  )


# sample number filter 

# 빈도 확인 

C_names = list(set(A_B_C_S_SET_COH.DC_cellname))
C_freq = [list(A_B_C_S_SET_COH.DC_cellname).count(a) for a in C_names]
C_df = pd.DataFrame({'cell' : C_names, 'freq' : C_freq })
C_df = C_df.sort_values('freq')

# 그래서 결론적으로 cell one hot 은? 

C_freq_filter = C_df[C_df.freq > 200] 



A_B_C_S_SET_COH = A_B_C_S_SET_COH[A_B_C_S_SET_COH.DC_cellname.isin(C_freq_filter.cell)]

DC_CELL_info_filt_re = DC_CELL_info_filt[DC_CELL_info_filt.DC_cellname.isin(C_freq_filter.cell)]
DC_CELL_info_filt_re['cell_onehot'] = [a for a in range(len(set(DC_CELL_info_filt_re.DrugCombCCLE)))]

DC_CELL_info_filt_re = DC_CELL_info_filt_re.reset_index(drop = True)


# merge 전 후로 index 달라지므로 뒤에 넣어줬음 
A_B_C_S_SET_COH2 = pd.merge(A_B_C_S_SET_COH, DC_CELL_info_filt_re[['DC_cellname','cell_onehot']], on = 'DC_cellname', how='left')
cell_one_hot = torch.nn.functional.one_hot(torch.Tensor(A_B_C_S_SET_COH2['cell_onehot']).long())





# 모델 가져오기 

import pandas as pd
import matplotlib.pyplot as plt
from ray.tune import ExperimentAnalysis
##from ray.tune import Analysis
import pickle
import math
import torch
import os




print('MAIN')

class MY_expGCN_parallel_model(torch.nn.Module):
	def __init__(self, G_layer_chem, G_indim_chem, G_hiddim_chem,
	G_layer_exp, G_indim_exp, G_hiddim_exp,
	layers_1, layers_2, layers_3, cell_dim ,
	out_dim, inDrop, drop):
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
		for conv in self.G_convs_1_exp :
			conv.reset_parameters()
		for bns in self.G_bns_1_exp :
			bns.reset_parameters()
		for conv in self.G_convs_2_chem :
			conv.reset_parameters()
		for bns in self.G_bns_2_chem :
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
	def forward(self, Drug1_F, Drug2_F, Drug1_ADJ, Drug2_ADJ, EXP1, EXP2, EXP_ADJ, EXP_ADJ_WGT, cell, syn ):
		Drug_batch_label = self.calc_batch_label(syn, Drug1_F)
		Exp_batch_label = self.calc_batch_label(syn, EXP1)
		#
		#print('chem')
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
		#print('exp')
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
		#print('snp')
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


WORK_DATE = '23.03.19'
PRJ_NAME = 'M3V4'
MISS_NAME = 'MIS2'
WORK_NAME = 'WORK_20v1'
W_NAME = 'W20v1' # 349


PRJ_PATH = '/home01/k020a01/02.VER3/{}_{}_{}/'.format(PRJ_NAME, MISS_NAME, W_NAME)
# PRJ_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V4_MIS2_W20v1/'

ANA_DF = pd.read_csv(PRJ_PATH+'RAY_ANA_DF.{}_{}_{}.csv'.format(MJ_NAME, MISS_NAME, W_NAME))
with open(PRJ_PATH+'RAY_ANA_DF.{}_{}_{}.pickle'.format(MJ_NAME, MISS_NAME, W_NAME), 'rb') as f:
		ANA_ALL_DF = pickle.load(f)









# GPU 니까 놉 
# TOPVAL_PATH = PRJ_PATH



####################################################
# 일단 추가 필요 파일들 
#
import os 

CELVAL_PATH = PRJ_PATH + 'VAL/'
os.makedirs(CELVAL_PATH, exist_ok = True)
# CELVAL_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V4_MIS2_W20v1.OLD/VAL/'

all_chem_DF = pd.read_csv(CELVAL_PATH+'DC_ALL_7555_ORDER.csv')
all_chem_feat_TS = torch.load(CELVAL_PATH+'DC_ALL.MY_chem_feat.pt')
all_chem_feat_adj = torch.load(CELVAL_PATH+'DC_ALL.MY_chem_adj.pt')


def check_drug_f_ts(CID) :
	INDEX = all_chem_DF[all_chem_DF.CID == CID].index.item()
	adj_pre = all_chem_feat_adj[INDEX]
	adj_proc = adj_pre.long().to_sparse().indices()
	return all_chem_feat_TS[INDEX], adj_proc



avail_LINCS_DF = pd.read_csv(CELVAL_PATH+'AVAIL_LINCS_EXP_cell.csv')
avail_LINCS_TS = torch.load(CELVAL_PATH+'AVAIL_LINCS_EXP_cell.pt')
avail_LINCS_DF['tuple'] = [(avail_LINCS_DF['CID'][a], avail_LINCS_DF['ccle_name'][a]) for a in range(avail_LINCS_DF.shape[0]) ]
avail_LINCS_TPs = list(avail_LINCS_DF['tuple'])

mj_exp_DF = pd.read_csv(CELVAL_PATH+'AVAIL_EXP_TOT.csv')
mj_exp_TS = torch.load(CELVAL_PATH+'AVAIL_EXP_TOT.pt')
mj_exp_DF['tuple'] = [( int(a.split('__')[0]) , a.split('__')[1]) for a in mj_exp_DF['sample']]
mj_exp_TPs = list(mj_exp_DF['tuple'])

targets_DF = pd.read_csv(CELVAL_PATH+'DC_ALL_TARGET.csv')
targets_TS = torch.load(CELVAL_PATH+'DC_ALL_TARGET.pt')

all_CellBase_DF = pd.read_csv(CELVAL_PATH+'AVAIL_CELL_DF.csv')
all_CellBase_TS = torch.load(CELVAL_PATH+'AVAIL_CLL_MY_CellBase.pt')

TPs_all = avail_LINCS_TPs + mj_exp_TPs
TPs_all_2 = [str(a[0])+"__"+a[1] for a in TPs_all]





# NEW targets 

# TARGET_PATH = '/st06/jiyeonH/13.DD_SESS/01.PRJ2/'

TARGET_DB = pd.read_csv(TARGET_PATH+'TARGET_CID_ENTREZ.csv', sep ='\t', index_col = 0)




                # LINCS 값을 우선시 하는 버전 (마치 MISS 2)
                def check_exp_f_ts(CID, CELLO) :
                    TUPLE = (int(CID), CELLO)
                    # Gene EXP
                    if TUPLE in avail_LINCS_TPs:
                        L_index = avail_LINCS_DF[avail_LINCS_DF['tuple'] == TUPLE].index[0].item() # 이건 나중에 고쳐야해 
                        EXP_vector = avail_LINCS_TS[L_index]
                    elif TUPLE in mj_exp_TPs :
                        M_index = mj_exp_DF[mj_exp_DF['tuple'] == TUPLE].index.item()
                        EXP_vector = mj_exp_TS[M_index]
                    else :
                        print('error')
                    #
                    # TARGET 
                    T_index = targets_DF[targets_DF['CID'] == CID].index.item()
                    TG_vector = targets_TS[T_index]
                    #
                    # BASAL EXP 
                    B_index = all_CellBase_DF[all_CellBase_DF.DrugCombCCLE == CELLO].index.item()
                    B_vector = all_CellBase_TS[B_index]
                    #
                    #
                    FEAT = torch.Tensor(np.array([ EXP_vector.squeeze().tolist() , TG_vector.squeeze().tolist(), B_vector.squeeze().tolist()]).T)
                    return FEAT.view(-1,3)







                    # MJ 값을 우선시 하는 버전 (마치 MISS 4)
                    def check_exp_f_ts(CID, CELLO) :
                        TUPLE = (int(CID), CELLO)
                        # Gene EXP
                        if TUPLE in avail_LINCS_TPs:
                            L_index = avail_LINCS_DF[avail_LINCS_DF['tuple'] == TUPLE].index[0].item() # 이건 나중에 고쳐야해 
                            EXP_vector = avail_LINCS_TS[L_index]
                        elif TUPLE in mj_exp_TPs :
                            M_index = mj_exp_DF[mj_exp_DF['tuple'] == TUPLE].index.item()
                            EXP_vector = mj_exp_TS[M_index]
                        else :
                            print('error')
                        #
                        # TARGET 
                        T_index = targets_DF[targets_DF['CID'] == CID].index.item()
                        TG_vector = targets_TS[T_index]
                        #
                        # BASAL EXP 
                        B_index = all_CellBase_DF[all_CellBase_DF.DrugCombCCLE == CELLO].index.item()
                        B_vector = all_CellBase_TS[B_index]
                        #
                        #
                        FEAT = torch.Tensor(np.array([ EXP_vector.squeeze().tolist() , TG_vector.squeeze().tolist(), B_vector.squeeze().tolist()]).T)
                        return FEAT.view(-1,3)



def check_cell_oh(CELLO) :
	oh_index = DC_CELL_info_filt[DC_CELL_info_filt.DrugCombCCLE == CELLO].index.item()
	cell_vec = cell_one_hot[oh_index]
	return cell_vec






class CellTest_Dataset(Dataset): 
	def __init__(self, tuple_list):
		self.tuple_list = tuple_list
    #
	def __len__(self): 
		return len(self.tuple_list)
	def __getitem__(self, idx): 
		ROW_CID, COL_CID, CELLO = self.tuple_list[idx]
		TUP_1 = (int(ROW_CID), CELLO)
		TUP_2 = (int(COL_CID), CELLO)
		#
		if (TUP_1 in TPs_all) & (TUP_2 in TPs_all) :
			drug1_f , drug1_a = check_drug_f_ts(ROW_CID)
			drug2_f , drug2_a = check_drug_f_ts(COL_CID)
			#
			expA = check_exp_f_ts(ROW_CID, CELLO)
			expB = check_exp_f_ts(COL_CID, CELLO)
			#
			adj = copy.deepcopy(JY_ADJ_IDX).long()
			adj_w = torch.Tensor(JY_IDX_WEIGHT).squeeze()
			#
			cell = check_cell_oh(CELLO)
			cell = cell.unsqueeze(0)
			#
			y = torch.Tensor([1]).float().unsqueeze(1)
		#
		else :
			drug1_f , drug1_a = torch.zeros(size = (50, 64)), torch.zeros(size = (2, 1))
			drug2_f , drug2_a = torch.zeros(size = (50, 64)), torch.zeros(size = (2, 1))
			adj = copy.deepcopy(JY_ADJ_IDX).long()
			adj_w = torch.Tensor(JY_IDX_WEIGHT).squeeze()
			#expA, expB = torch.zeros(size = (978, 3)), torch.zeros(size = (978, 3))
			expA, expB = torch.zeros(size = (349, 3)), torch.zeros(size = (349, 3))
			cell = torch.zeros(size = (1, 25))
			y = torch.Tensor([0]).float().unsqueeze(1)
		#
		return ROW_CID, COL_CID, CELLO, drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y




def graph_collate_fn(batch):
	tup_list = []
	#
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
	for ROW_CID, COL_CID, CELLO, drug1_f, drug2_f, drug1_adj, drug2_adj, expA, expB, exp_adj, exp_adj_w, cell, y in batch :
		tup_list.append( (ROW_CID, COL_CID, CELLO) )
		#
		drug1_f_list.append(drug1_f)
		drug2_f_list.append(drug2_f)
		drug1_adj_list.append(drug1_adj+DRUG_1_num_nodes_seen)
		drug2_adj_list.append(drug2_adj+DRUG_2_num_nodes_seen)
		expA_list.append(expA)
		expB_list.append(expB)
		exp_adj_list.append(exp_adj+EXP_num_nodes_seen)
		exp_adj_w_list.append(exp_adj_w.unsqueeze(0))
		y_list.append(torch.Tensor(y))
		cell_list.append(cell)
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
	return tup_list, drug1_f_new, drug2_f_new, drug1_adj_new, drug2_adj_new, expA_new, expB_new, exp_adj_new, exp_adj_w_new, cell_new, y_new






def Cell_Test(CELLO, MODEL_NAME, use_cuda = True) :
    print(CELLO)
    print(datetime.now(), flush = True)
    #
    with open(CELVAL_PATH+'{}.testtest.json'.format(CELLO)) as f: ######################### 바꿔야해 
        lst_check = [tuple(x) for x in json.load(f)]
    #
    #
    tt_df = pd.DataFrame()
    tt_df['tuple'] = lst_check
    tt_df['cid1'] = [str(int(a[0])) for a in lst_check]
    tt_df['cid2'] = [str(int(a[1])) for a in lst_check]
    tt_df['cello'] = [a[2] for a in lst_check]
    #
    tt_df['cid1_celo'] = tt_df.cid1 +'__' +tt_df.cello
    tt_df['cid2_celo'] = tt_df.cid2 +'__' +tt_df.cello
    #
    tt_df_re1 = tt_df[tt_df.cid1_celo.isin(TPs_all_2)]
    tt_df_re2 = tt_df_re1[tt_df_re1.cid2_celo.isin(TPs_all_2)]
    #
    tg_lists = [str(int(a)) for a in list(set(TARGET_DB.CID))]
    tt_df_re3 = tt_df_re2[tt_df_re2.cid1.isin(tg_lists)] # 5289702
    tt_df_re4 = tt_df_re3[tt_df_re3.cid2.isin(tg_lists)] # 2359767
    #
    tt_df_re4 = tt_df_re4.reset_index(drop=True)
    #
    tuple_list = tt_df_re4['tuple']
    #
    #
    #
    #
    CELL_PRED_DF = pd.DataFrame(columns = ['ROW_CID','COL_CID','CCLE','PRED_RES'])
    PRED_list = []
    #
    G_chem_layer = my_config['config/G_chem_layer'].item()
    G_chem_hdim = my_config['config/G_chem_hdim'].item()
    G_exp_layer = my_config['config/G_exp_layer'].item()
    G_exp_hdim = my_config['config/G_exp_hdim'].item()
    dsn1_layers = [my_config['config/feat_size_0'].item(), my_config['config/feat_size_1'].item(), my_config['config/feat_size_2'].item()]
    dsn2_layers = [my_config['config/feat_size_0'].item(), my_config['config/feat_size_1'].item(), my_config['config/feat_size_2'].item()] 
    snp_layers = [my_config['config/feat_size_3'].item() , my_config['config/feat_size_4'].item()]
    inDrop = my_config['config/dropout_1'].item()
    Drop = my_config['config/dropout_2'].item()
    #      
    # 만약에 이렇게 안하고 마지막에 한번 더 다시 돌리기로 하면 이렇게 할 필요 음슴 
    # 이건 차후에 다시 final 모델 생기면 수정해도 될듯! 
    #
    best_model_0 = MY_expGCN_parallel_model(
                G_chem_layer, 64 , G_chem_hdim,
                G_exp_layer, 3, G_exp_hdim,
                dsn1_layers, dsn2_layers, snp_layers, 
                cell_one_hot.shape[-1], 1,
                inDrop, Drop
                )
    best_model_1 = MY_expGCN_parallel_model(
                G_chem_layer, 64 , G_chem_hdim,
                G_exp_layer, 3, G_exp_hdim,
                dsn1_layers, dsn2_layers, snp_layers, 
                cell_one_hot.shape[-1], 1,
                inDrop, Drop
                )
    best_model_2 = MY_expGCN_parallel_model(
                G_chem_layer, 64 , G_chem_hdim,
                G_exp_layer, 3, G_exp_hdim,
                dsn1_layers, dsn2_layers, snp_layers, 
                cell_one_hot.shape[-1], 1,
                inDrop, Drop
                )
    best_model_3 = MY_expGCN_parallel_model(
                G_chem_layer, 64 , G_chem_hdim,
                G_exp_layer, 3, G_exp_hdim,
                dsn1_layers, dsn2_layers, snp_layers, 
                cell_one_hot.shape[-1], 1,
                inDrop, Drop
                )
    best_model_4 = MY_expGCN_parallel_model(
                G_chem_layer, 64 , G_chem_hdim,
                G_exp_layer, 3, G_exp_hdim,
                dsn1_layers, dsn2_layers, snp_layers, 
                cell_one_hot.shape[-1], 1,
                inDrop, Drop
                )
    # 
    #
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        state_dict_0 = torch.load(os.path.join(PRJ_PATH, "{}_CV_0_model.pth".format(MODEL_NAME))) #### change ! 
        state_dict_1 = torch.load(os.path.join(PRJ_PATH, "{}_CV_1_model.pth".format(MODEL_NAME))) #### change ! 
        state_dict_2 = torch.load(os.path.join(PRJ_PATH, "{}_CV_2_model.pth".format(MODEL_NAME))) #### change ! 
        state_dict_3 = torch.load(os.path.join(PRJ_PATH, "{}_CV_3_model.pth".format(MODEL_NAME))) #### change ! 
        state_dict_4 = torch.load(os.path.join(PRJ_PATH, "{}_CV_4_model.pth".format(MODEL_NAME))) #### change ! 
    else:
        state_dict_0 = torch.load(os.path.join(PRJ_PATH, "{}_CV_0_model.pth".format(MODEL_NAME)), map_location=torch.device('cpu'))
        state_dict_1 = torch.load(os.path.join(PRJ_PATH, "{}_CV_1_model.pth".format(MODEL_NAME)), map_location=torch.device('cpu'))
        state_dict_2 = torch.load(os.path.join(PRJ_PATH, "{}_CV_2_model.pth".format(MODEL_NAME)), map_location=torch.device('cpu'))
        state_dict_3 = torch.load(os.path.join(PRJ_PATH, "{}_CV_3_model.pth".format(MODEL_NAME)), map_location=torch.device('cpu'))
        state_dict_4 = torch.load(os.path.join(PRJ_PATH, "{}_CV_4_model.pth".format(MODEL_NAME)), map_location=torch.device('cpu'))
    #
    #
    print("state_dict_done", flush = True)
    if type(state_dict_0) == tuple:
        best_model_0.load_state_dict(state_dict_0[0])
        best_model_1.load_state_dict(state_dict_1[0])
        best_model_2.load_state_dict(state_dict_2[0])
        best_model_3.load_state_dict(state_dict_3[0])
        best_model_4.load_state_dict(state_dict_4[0])
    else : 
        best_model_0.load_state_dict(state_dict_0)
        best_model_1.load_state_dict(state_dict_1)
        best_model_2.load_state_dict(state_dict_2)
        best_model_3.load_state_dict(state_dict_3)
        best_model_4.load_state_dict(state_dict_4)
    #
    print("state_load_done", flush = True)
    #
    best_model_0.eval()
    best_model_1.eval()
    best_model_2.eval()
    best_model_3.eval()
    best_model_4.eval()
    #
    #
    dataset = CellTest_Dataset(tuple_list)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = my_config['config/batch_size'].item(), collate_fn = graph_collate_fn, shuffle = False )# , num_workers=my_config['config/n_workers'].item()
    #
    CELL_PRED_DF = pd.DataFrame(columns = ['CV0','CV1','CV2','CV3','CV4','ROW_CID','COL_CID','CCLE','Y'])
    CELL_PRED_DF.to_csv(CELVAL_PATH+'PRED_{}.FINAL_ing.csv'.format(CELLO), index=False)
    #
    with torch.no_grad():
        for batch_idx_t, (tup_list, drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y) in enumerate(dataloader):
            print("{} / {}".format(batch_idx_t, len(dataloader)) , flush = True)
            print(datetime.now(), flush = True)
            list_ROW_CID = [a[0] for a in tup_list]
            list_COL_CID = [a[1] for a in tup_list]
            list_CELLO = [a[2] for a in tup_list]
            #
            if use_cuda:
                drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell = drug1_f.cuda(), drug2_f.cuda(), drug1_a.cuda(), drug2_a.cuda(), expA.cuda(), expB.cuda(), adj.cuda(), adj_w.cuda(), y.cuda(), cell.cuda()
            #
            output_0= best_model_0(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w.squeeze(), cell.squeeze(), y)
            output_1= best_model_1(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w.squeeze(), cell.squeeze(), y)
            output_2= best_model_2(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w.squeeze(), cell.squeeze(), y)
            output_3= best_model_3(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w.squeeze(), cell.squeeze(), y)
            output_4= best_model_4(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w.squeeze(), cell.squeeze(), y)
            outputs_0 = output_0.squeeze().tolist() # [output.squeeze().item()]
            outputs_1 = output_1.squeeze().tolist()
            outputs_2 = output_2.squeeze().tolist()
            outputs_3 = output_3.squeeze().tolist()
            outputs_4 = output_4.squeeze().tolist()
            #print(outputs)
            #
            tmp_df = pd.DataFrame({
            'CV0': outputs_0,
            'CV1': outputs_1,
            'CV2': outputs_2,
            'CV3': outputs_3,
            'CV4': outputs_4,
            'ROW_CID' : list_ROW_CID,
            'COL_CID' : list_COL_CID,
            'CCLE' : list_CELLO,
            'Y' : y.squeeze().tolist()
            })
            CELL_PRED_DF = pd.concat([CELL_PRED_DF, tmp_df])
            tmp_df.to_csv(CELVAL_PATH+'PRED_{}.FINAL_ing.csv'.format(CELLO), mode='a', index=False, header = False)
    return CELL_PRED_DF






PRJ_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V4_MIS2_W20v1/'
os.makedirs( os.path.join(PRJ_PATH,'VAL'), exist_ok = True)
ANA_DF_CSV = pd.read_csv(os.path.join(PRJ_PATH,'RAY_ANA_DF.{}_{}_{}.csv'.format(MJ_NAME, MISS_NAME, W_NAME)))

my_config = ANA_DF_CSV[ANA_DF_CSV.trial_id=='98d5812c']

avail_cell_list = []

for CELLO in avail_cell_list : 
    # CELLO = 'A549_LUNG'
    MODEL_NAME = 'C4'
    CELL_PRED_DF = Cell_Test(CELLO, MODEL_NAME, use_cuda = False)
    CELL_PRED_DF.to_csv(CELVAL_PATH+'PRED_{}.FINAL.csv'.format(CELLO), index=False)



실제로 돌릴때 고칠거 

1) workers
2) json 파일이름 
3) cell list
4) 모델 미리 폴더에 가져와야함 
5) 아직 GPU 실험 못해봄 








# Cell_Test(CELLO, MODEL_NAME, use_cuda = True)






                    예시로 하나만 실험해보자면 
                    CELLO = 'A549_LUNG'
                    tmp_path = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V4_MIS2_W20v1.OLD/VAL/'
                    with open(tmp_path+'{}.json'.format(CELLO)) as f:
                        lst_check = [tuple(x) for x in json.load(f)]

                    with open(tmp_path+'{}.testtest.json'.format(CELLO), 'w') as f:
                        json.dump(lst_check[0:100], f)

                    with open(tmp_path+'{}.testtest.json'.format(CELLO)) as f:
                        lst_check2 = [tuple(x) for x in json.load(f)]










