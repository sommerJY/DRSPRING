

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



NETWORK_PATH = '/home01/k040a01/01.Data/HumanNet/'
LINCS_PATH = '/home01/k040a01/01.Data/LINCS/'
DC_PATH = '/home01/k040a01/01.Data/DrugComb/'
TARGET_PATH = '/home01/k040a01/01.Data/TARGET/'



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


# Cell info

DC_CELL_DF2 = pd.read_csv(DC_PATH+'DC_CELL_INFO.csv', sep = '\t')










import pandas as pd
import matplotlib.pyplot as plt
from ray.tune import ExperimentAnalysis
##from ray.tune import Analysis
import pickle
import math
import torch
import os






class MY_expGCN_parallel_model(torch.nn.Module):
	def __init__(self, G_layer_chem, G_indim_chem, G_hiddim_chem, 
	G_layer_exp, G_indim_exp, G_hiddim_exp, 
	layers_1, layers_2, layers_3, 
	out_dim, inDrop, drop):
		super(MY_expGCN_parallel_model, self).__init__()
		self.G_layer_chem = G_layer_chem
		self.G_indim_chem = G_indim_chem
		self.G_hiddim_chem = G_hiddim_chem
		self.G_layer_exp = G_layer_exp
		self.G_indim_exp = G_indim_exp
		self.G_hiddim_exp = G_hiddim_exp
		self.G_Common_dim = min([G_hiddim_chem,G_hiddim_exp])
		self.layers_1 = [int(a) for a in layers_1]
		self.layers_2 = [int(a) for a in layers_2]
		self.layers_3 = [int(a) for a in layers_3]
		self.out_dim = out_dim
		self.inDrop = inDrop
		self.drop = drop
		self.tanh = torch.nn.Tanh()
		self.pool = pyg_nn.global_mean_pool
		#
		self.G_convs_1_chem = torch.nn.ModuleList([pyg_nn.GCNConv(self.G_indim_chem, self.G_hiddim_chem)])
		self.G_convs_1_chem.extend([pyg_nn.GCNConv(self.G_hiddim_chem, self.G_hiddim_chem) for i in range(self.G_layer_chem-2)])
		self.G_convs_1_chem.extend([pyg_nn.GCNConv(self.G_hiddim_chem, self.G_Common_dim)])
		self.G_bns_1_chem = torch.nn.ModuleList([torch.nn.BatchNorm1d(self.G_hiddim_chem) for i in range(self.G_layer_chem-1)])
		##
		self.G_convs_1_exp = torch.nn.ModuleList([pyg_nn.GCNConv(self.G_indim_exp, self.G_hiddim_exp)])
		self.G_convs_1_exp.extend([pyg_nn.GCNConv(self.G_hiddim_exp, self.G_hiddim_exp) for i in range(self.G_layer_exp-2)])
		self.G_convs_1_exp.extend([pyg_nn.GCNConv(self.G_hiddim_exp, self.G_Common_dim)])
		self.G_bns_1_exp = torch.nn.ModuleList([torch.nn.BatchNorm1d(self.G_hiddim_exp) for i in range(self.G_layer_exp-1)])
		##
		self.Convs_1 = torch.nn.ModuleList([torch.nn.Linear(self.G_Common_dim+self.G_Common_dim, self.layers_1[0] )])
		self.Convs_1.extend([torch.nn.Linear(self.layers_1[a], self.layers_1[a+1]) for a in range(len(self.layers_1)-1)])
		##
		self.SNPs = torch.nn.ModuleList([torch.nn.Linear(self.layers_1[-1]+self.layers_2[-1], self.layers_3[0] )])
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
		for conv in self.Convs_1 :
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
		for G_2_C in range(len(self.G_convs_1_chem)):
			if G_2_C == len(self.G_convs_1_chem)-1 :
				Drug2_F = self.G_convs_1_chem[G_2_C](x=Drug2_F, edge_index=Drug2_ADJ)
				Drug2_F = F.dropout(Drug2_F, p=self.inDrop, training=self.training)
				Drug2_F = self.pool(Drug2_F, Drug_batch_label )
				Drug2_F = self.tanh(Drug2_F)
				G_2_C_out = Drug2_F
			else :
				Drug2_F = self.G_convs_1_chem[G_2_C](x=Drug2_F, edge_index=Drug2_ADJ)
				Drug2_F = self.G_bns_1_chem[G_2_C](Drug2_F)
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
		for G_2_E in range(len(self.G_convs_1_exp)):
			if G_2_E == len(self.G_convs_1_exp)-1 :
				EXP2 = self.G_convs_1_exp[G_2_E](x=EXP2, edge_index=EXP_ADJ, edge_weight= EXP_ADJ_WGT)
				EXP2 = F.dropout(EXP2, p=self.inDrop, training=self.training)
				EXP2 = self.pool(EXP2, Exp_batch_label )
				EXP2 = self.tanh(EXP2)
				G_2_E_out = EXP2
			else :
				EXP2 = self.G_convs_1_exp[G_2_E](x=EXP2, edge_index=EXP_ADJ, edge_weight= EXP_ADJ_WGT)
				EXP2 = self.G_bns_1_exp[G_2_E](EXP2)
				EXP2 = F.elu(EXP2)
		#
		input_drug1 = torch.concat( (G_1_C_out, G_1_E_out), 1 ) # normalization 추가 해야할것 같은데 
		input_drug2 = torch.concat( (G_2_C_out, G_2_E_out), 1 )
		#
		for L1 in range(len(self.Convs_1)):
			if L1 != len(self.Convs_1)-1 :
				input_drug1 = self.Convs_1[L1](input_drug1)
				input_drug1 = F.dropout(input_drug1, p=self.inDrop, training = self.training)
				input_drug1 = F.elu(input_drug1)
			else :
				input_drug1 = self.Convs_1[L1](input_drug1)
		#
		for L2 in range(len(self.Convs_1)):
			if L2 != len(self.Convs_1)-1 :
				input_drug2 = self.Convs_1[L2](input_drug2)
				input_drug2 = F.dropout(input_drug2, p=self.inDrop, training = self.training)
				input_drug2 = F.elu(input_drug2)
			else :
				input_drug2 = self.Convs_1[L2](input_drug2)
		#
		X = torch.cat(( input_drug1, input_drug2 ), 1)
		for L3 in range(len(self.SNPs)):
			if L3 != len(self.SNPs)-1 :
				X = self.SNPs[L3](X)
				X = F.dropout(X, p=self.drop, training = self.training)
				X = F.elu(X)
			else :
				X = self.SNPs[L3](X)
		return X




PRJ_PATH = '/home01/k040a01/02.M3V8/M3V8_W414_349_MIS2/'


CELVAL_PATH = PRJ_PATH

os.makedirs(CELVAL_PATH, exist_ok = True)

all_chem_DF = pd.read_csv(CELVAL_PATH+'DC_ALL_7555_ORDER.csv')
all_chem_feat_TS = torch.load(CELVAL_PATH+'DC_ALL.MY_chem_feat.pt')
all_chem_feat_adj = torch.load(CELVAL_PATH+'DC_ALL.MY_chem_adj.pt')

avail_LINCS_DF = pd.read_csv(CELVAL_PATH+'AVAIL_LINCS_EXP_cell.csv')
avail_LINCS_TS = torch.load(CELVAL_PATH+'AVAIL_LINCS_EXP_cell.pt')
avail_LINCS_DF['tuple'] = [(int(avail_LINCS_DF['CID'][a]), avail_LINCS_DF['CCLE_Name'][a]) for a in range(avail_LINCS_DF.shape[0]) ]
avail_LINCS_TPs = list(avail_LINCS_DF['tuple'])



targets_DF = pd.read_csv(CELVAL_PATH+'DC_ALL_TARGET.csv')
targets_TS = torch.load(CELVAL_PATH+'DC_ALL_TARGET.pt')

all_CellBase_DF = pd.read_csv(CELVAL_PATH+'AVAIL_CELL_DF.csv')
all_CellBase_TS = torch.load(CELVAL_PATH+'AVAIL_CLL_MY_CellBase.pt')





TARGET_DB = pd.read_csv(TARGET_PATH+'TARGET_CID_ENTREZ.csv', sep ='\t', index_col = 0)



# LINCS 값을 우선시 하는 버전 (마치 MISS 2)
def check_exp_f_ts(CID, CELLO, mj_exp_DF, mj_exp_TS, mj_exp_TPs) :
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





def check_drug_f_ts(CID) :
	INDEX = all_chem_DF[all_chem_DF.CID == CID].index.item()
	adj_pre = all_chem_feat_adj[INDEX]
	adj_proc = adj_pre.long().to_sparse().indices()
	return all_chem_feat_TS[INDEX], adj_proc



class CellTest_Dataset(Dataset): 
	def __init__(self, tuple_list, TPs_all, mj_exp_DF, mj_exp_TS, mj_exp_TPs):
		self.tuple_list = tuple_list
		self.TPs_all = TPs_all
		self.mj_exp_DF = mj_exp_DF
		self.mj_exp_TS = mj_exp_TS
		self.mj_exp_TPs = mj_exp_TPs
	#
	def __len__(self): 
		return len(self.tuple_list)
	def __getitem__(self, idx): 
		ROW_CID, COL_CID, CELLO = self.tuple_list[idx]
		TUP_1 = (int(ROW_CID), CELLO)
		TUP_2 = (int(COL_CID), CELLO)
		#
		if (TUP_1 in self.TPs_all) & (TUP_2 in self.TPs_all) :
			drug1_f , drug1_a = check_drug_f_ts(ROW_CID)
			drug2_f , drug2_a = check_drug_f_ts(COL_CID)
			#
			expA = check_exp_f_ts(ROW_CID, CELLO, self.mj_exp_DF, self.mj_exp_TS, self.mj_exp_TPs)
			expB = check_exp_f_ts(COL_CID, CELLO, self.mj_exp_DF, self.mj_exp_TS, self.mj_exp_TPs)
			#
			adj = copy.deepcopy(JY_ADJ_IDX).long()
			adj_w = torch.Tensor(JY_IDX_WEIGHT).squeeze()
			#
			cell = torch.zeros(size = (1, 25)) # no need 
			#
			y = torch.Tensor([1]).float().unsqueeze(1)
		#
		else :
			drug1_f , drug1_a = torch.zeros(size = (50, 64)), torch.zeros(size = (2, 1))
			drug2_f , drug2_a = torch.zeros(size = (50, 64)), torch.zeros(size = (2, 1))
			adj = copy.deepcopy(JY_ADJ_IDX).long()
			adj_w = torch.Tensor(JY_IDX_WEIGHT).squeeze()
			#expA, expB = torch.zeros(size = (978, 3)), torch.zeros(size = (978, 3))
			expA, expB = torch.zeros(size = (349, 3)), torch.zeros(size = (349, 3)) ###################################
			cell = torch.zeros(size = (1, 25)) # no need
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



def Cell_Json(CELL, TISSUE) :
	CELLO = CELL+ '_' +TISSUE
	print(datetime.now(), flush = True)
	print(CELLO)
	#
	with open(CELVAL_PATH+'cell_json/'+'{}.json'.format(CELLO)) as f: 
		lst_check = [tuple(x) for x in json.load(f)]
	print(datetime.now(), flush = True)
	#
	mj_exp_DF = pd.read_csv(CELVAL_PATH+'tissue/'+'{}.AVAIL_EXP_TOT.csv'.format(TISSUE))
	mj_exp_TS = torch.load(CELVAL_PATH+'tissue/'+'{}.AVAIL_EXP_TOT.pt'.format(TISSUE))
	mj_exp_DF['tuple'] = [( int(a.split('__')[0]) , a.split('__')[1]) for a in mj_exp_DF['sample']]
	mj_exp_TPs = list(mj_exp_DF['tuple'])
	TPs_all = avail_LINCS_TPs + mj_exp_TPs
	TPs_all_2 = [str(a[0])+"__"+a[1] for a in TPs_all]
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
	tuple_list = tt_df_re2['tuple']
	dataset = CellTest_Dataset(tuple_list, TPs_all, mj_exp_DF, mj_exp_TS, mj_exp_TPs)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size = 512 , collate_fn = graph_collate_fn, shuffle = False, num_workers = 16)
	return dataloader



avail_cell_dict = {
	
	(1)414_1 - 'BONE': ['A673']
	(1)414_2 - 'KIDNEY': ['CAKI1', 'UO31', 'ACHN', 'A498', '786O'], 
	(1)414_10 'LUNG': ['A549', 'A427', 'HOP92', 'NCIH23', 'NCIH1650', 'NCIH520', 'HOP62', 'NCIH460', 
	'NCIH2122', 'NCIH226', 'EKVX', 'SKMES1', 'NCIH522'],
	(1)414_12 - 'OVARY': ['NIHOVCAR3', 'OVCAR4', 'OVCAR5', 'IGROV1', 'OVCAR8','PA1']
	(8)414_L1 LARGE_INTESTINE 'SW620', 'LOVO', 'RKO', 'KM12', 'HCT15', 'HCT116', 'HT29', 'DLD1', 
	(1)414_5 - 'LARGE_INTESTINE': ['SW837'],  
	(1)414_13 - 'OVARY': ['UWB1289', 'SKOV3', 'CAOV3', 'ES2', 'A2780', 'OV90'], 
	(1)414_6 - 'CENTRAL_NERVOUS_SYSTEM': ['T98G', 'SF539', 'U251MG', 'SNB75', 'SF268', 'SF295'], 
	
	(1)414_0 - 'PROSTATE': ['VCAP', 'PC3']
		'PLEURA': ['MSTO211H']

	414_BL: : 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE': ['UHO1', 'K562', 'L1236', 'SR786', 'RPMI8226'], 
	(1)414_3 - 'BREAST1': ['BT549', 'BT474', 'MDAMB361', 'MDAMB231', 'UACC812', 'KPL1', 'MDAMB468', 'MDAMB175VII', ]
	(1)414_4 - BREAST2 'HCC1500', 'HCC1419', 'MDAMB436', 'ZR751', 'MCF7', 'HS578T', 'T47D', 'CAMA1'], 
	
	(1)414_7 - SKIN - 'SKMEL5', 'HT144', 'RVH421', 'SKMEL28', 'A375', 'A101D', 'COLO792', 'UACC257', 
	(1)414_9 - 'SKIN': ['UACC62', 'LOXIMVI', 'WM115', 'G361', 'SKMEL30', 'COLO829'], 
	(8)414S - SKIN - 'COLO800', 'MEWO', 'MELHO', 'A2058', 'RPMI7951', 'IPC298', 'MALME3M', 'SKMEL2', 
	

		 


TB01, worker 16, 8192 -> lav 20, 100/2000 30분 : PROSTATE - 2023-09-17 17:31:13 ~ 20:40:57 : 643/2000
TB02, worker 8, 4096 -> lav 10, 200/4000 30분 : BLOOD - 2023-09-17 17:23:37 ~ 20:41:02 : 879 / 4000


CELL_1 = Cell_Json('VCAP', 'PROSTATE')
CELL_2 = Cell_Json('PC3', 'PROSTATE')
CELL_3 = Cell_Json('A673' , 'BONE') 
CELL_4 = Cell_Json('CAKI1', 'KIDNEY')
CELL_5 = Cell_Json('UO31', 'KIDNEY')
CELL_6 = Cell_Json('ACHN', 'KIDNEY')
CELL_7 = Cell_Json('A498', 'KIDNEY')
CELL_8 = Cell_Json('786O', 'KIDNEY')



RAY_CELL_1 = ray.put(CELL_1)
RAY_CELL_2 = ray.put(CELL_2)
RAY_CELL_3 = ray.put(CELL_3)
RAY_CELL_4 = ray.put(CELL_4)
RAY_CELL_5 = ray.put(CELL_5)
RAY_CELL_6 = ray.put(CELL_6)
RAY_CELL_7 = ray.put(CELL_7)
RAY_CELL_8 = ray.put(CELL_8)



def ray_learn (config, checkpoint_dir=None) :
	use_cuda = True
	#
	input_list = [RAY_CELL_1, RAY_CELL_2, RAY_CELL_3, RAY_CELL_4, RAY_CELL_5, RAY_CELL_6, RAY_CELL_7, RAY_CELL_8]
	cell_list = ['VCAP', 'PC3', 'A673', 'CAKI1', 'UO31', 'ACHN', 'A498', '786O']
	#
	CELL_NUM = config['CV']
	CV_CELL = ray.get(input_list[CELL_NUM])
	CELLO = cell_list[CELL_NUM]
	#
	OLD_PATH = '/home01/k040a01/02.M3V8/M3V8_W402_349_MIS2'
	ANA_DF_CSV = pd.read_csv(os.path.join(OLD_PATH,'RAY_ANA_DF.{}.csv'.format('M3V8_W402_349_MIS2')))
	my_config = ANA_DF_CSV[ANA_DF_CSV.trial_id=='2b5ce4c0'] 
	CKP_PATH = '/home01/k040a01/ray_results/PRJ02.23.08.27.M3V8.WORK_404.349.MIS2/RAY_MY_train_874a8_00000_0_G_chem_hdim=32,G_chem_layer=3,G_exp_hdim=32,G_exp_layer=3,batch_size=512,dropout_1=0.1000,dropout_2=0.1_2023-08-27_19-43-16/checkpoint_000999/checkpoint'
	#
	G_chem_layer = my_config['config/G_chem_layer'].item()
	G_chem_hdim = my_config['config/G_chem_hdim'].item()
	G_exp_layer = my_config['config/G_exp_layer'].item()
	G_exp_hdim = my_config['config/G_exp_hdim'].item()
	dsn_layers = [int(a) for a in my_config["config/dsn_layer"].item().split('-') ]
	snp_layers = [int(a) for a in my_config["config/snp_layer"].item().split('-') ]
	inDrop = my_config['config/dropout_1'].item()
	Drop = my_config['config/dropout_2'].item()
	#      
	best_model = MY_expGCN_parallel_model(
				G_chem_layer, 64 , G_chem_hdim,
				G_exp_layer, 3, G_exp_hdim,
				dsn_layers, dsn_layers, snp_layers, 
				1,
				inDrop, Drop
				) 
	#
	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	if torch.cuda.is_available():
		state_dict = torch.load(CKP_PATH) #### change ! 
	else:
		state_dict = torch.load(CKP_PATH, map_location=torch.device('cpu'))
	# 
	# 
	print("state_dict_done", flush = True)
	if type(state_dict) == tuple:
		best_model.load_state_dict(state_dict[0])
	else : 
		best_model.load_state_dict(state_dict)
	#
	print("state_load_done", flush = True)
	#
	#
	best_model.to(device)
	best_model.eval()
	#
	#
	CELL_PRED_DF = pd.DataFrame(columns = ['PRED','ROW_CID','COL_CID','CCLE','Y'])
	CELL_PRED_DF.to_csv(CELVAL_PATH+'/CELL_VAL/'+'PRED_{}.FINAL_ing.csv'.format(CELLO), index=False)
	#
	with torch.no_grad():
		for batch_idx_t, (tup_list, drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y) in enumerate(CV_CELL):
			print("{} / {}".format(batch_idx_t, len(CV_CELL)) , flush = True)
			print(datetime.now(), flush = True)
			list_ROW_CID = [a[0] for a in tup_list]
			list_COL_CID = [a[1] for a in tup_list]
			list_CELLO = [a[2] for a in tup_list]
			#
			if use_cuda:
				drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell = drug1_f.cuda(), drug2_f.cuda(), drug1_a.cuda(), drug2_a.cuda(), expA.cuda(), expB.cuda(), adj.cuda(), adj_w.cuda(), y.cuda(), cell.cuda()
			#
			output = best_model(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w.squeeze(), cell.squeeze(), y) 
			outputs = output.squeeze().tolist() # [output.squeeze().item()]
			#print(outputs)
			#
			tmp_df = pd.DataFrame({
			'PRED': outputs,
			'ROW_CID' : list_ROW_CID,
			'COL_CID' : list_COL_CID,
			'CCLE' : list_CELLO,
			'Y' : y.squeeze().tolist()
			})
			CELL_PRED_DF = pd.concat([CELL_PRED_DF, tmp_df])
			tmp_df.to_csv(CELVAL_PATH+'/CELL_VAL/'+'PRED_{}.FINAL_ing.csv'.format(CELLO), mode='a', index=False, header = False)
	


CONFIG = {"CV" : tune.grid_search([0,1,2,3,4,5,6,7])}

reporter = CLIReporter()

ANALYSIS = tune.run( # 끝내지 않음 
		tune.with_parameters(ray_learn),
		name = 'cell',
		num_samples=1,
		config=CONFIG,
		resources_per_trial={'cpu': 16,'gpu' : 1 }, # , 
		progress_reporter = reporter
	)



이렇게 진행하는건 어렵다.. 
'PROSTATE': ['VCAP', 'PC3']

'BONE': ['A673']

'KIDNEY': ['CAKI1', 'UO31', 'ACHN', 'A498', '786O']

'BREAST': [
	'BT549', 'BT474', 'MDAMB361', 'MDAMB231', 'UACC812', 'KPL1', 'MDAMB468', 'MDAMB175VII', 
	'HCC1500', 'HCC1419', 'MDAMB436', 'ZR751', 'MCF7', 'HS578T', 'T47D', 'CAMA1'], 

'LARGE_INTESTINE': [
		'SW620', 'LOVO', 'RKO', 'KM12', 'HCT15', 'HCT116', 'HT29', 'DLD1', 'SW837']

'CENTRAL_NERVOUS_SYSTEM': ['T98G', 'SF539', 'U251MG', 'SNB75', 'SF268', 'SF295']

'SKIN': [
	'SKMEL5', 'HT144', 'RVH421', 'SKMEL28', 'A375', 'A101D', 'COLO792', 'UACC257', 
	'COLO800', 'MEWO', 'MELHO', 'A2058', 'RPMI7951', 'IPC298', 'MALME3M', 'SKMEL2', 
	'UACC62', 'LOXIMVI', 'WM115', 'G361', 'SKMEL30', 'COLO829']

'LUNG': [
	'A549', 'A427', 'HOP92', 'NCIH23', 'NCIH1650', 'NCIH520', 'HOP62',
	 'NCIH460', 'NCIH2122', 'NCIH226', 'EKVX', 'SKMES1', 'NCIH522']
	
'OVARY': ['NIHOVCAR3', 'OVCAR4', 'OVCAR5', 'IGROV1', 'OVCAR8',
 'PA1', 'UWB1289', 'SKOV3', 'CAOV3', 'ES2', 'A2780', 'OV90']

'PLEURA': ['MSTO211H']

'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE': ['UHO1', 'K562', 'L1236', 'SR786', 'RPMI8226']

gpu4 쓰게 결국은 바꿔야할것 같긴 함.. 




#######################################
################################################

뭐가 맞는지 봐가면서 해야함... 

# 그놈의 색깔 
PRJ_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V6_W213_349_MIS2/CELL_VAL/'

cell_path = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V6_W213_349_MIS2/CELL_VAL/'
cell_path2 = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V6_W213_349_MIS2/'

cell_path = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V8_W414_349_MIS2/CELL_VAL/DONE'
cell_path2 = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V8_W414_349_MIS2/'

# kisti 
cell_path = '/home01/k040a01/02.M3V6/M3V6_W213_349_MIS2/CELL_VAL' 



# 다시 진행해주기 
import glob
from matplotlib import colors as mcolors

CVCL_files_raw = glob.glob(cell_path+'/PRED_*.FINAL.csv')
tissues = ['_'.join(a.split('/')[-1].split('.')[0].split('PRED_CELL_')[1].split('_')[1:]) for a in CVCL_files_raw]
strips = [a.split('/')[-1].split('.')[0].split('PRED_CELL_')[1].split('_')[0] for a in CVCL_files_raw]

pred_df = pd.DataFrame(columns = ['PRED', 'ROW_CID','COL_CID','CCLE', 'Y','tissue','strip'])
#pred_df_pos = pd.DataFrame(columns = ['PRED', 'ROW_CID','COL_CID','CCLE', 'Y','tissue','strip'])
#pred_df_neg = pd.DataFrame(columns = ['PRED', 'ROW_CID','COL_CID','CCLE', 'Y','tissue','strip'])

for indd in range(len(CVCL_files_raw)) :
	fifi = CVCL_files_raw[indd]
	tiss = tissues[indd]
	stripname = strips[indd]
	print(stripname)
	tmp_df = pd.read_csv(fifi,  low_memory=False, index_col = False)
	tmp_df2 = tmp_df[['ROW_CID','COL_CID','CCLE', 'PRED', 'Y']]
	tmp_df2['tissue'] = tiss
	tmp_df2['strip'] = stripname
	#tmp_df_pos = tmp_df2[tmp_df2.PRED>0]
	#tmp_df_neg = tmp_df2[tmp_df2.PRED<0]
	#pred_df_pos = pd.concat([pred_df_pos, tmp_df_pos])
	#pred_df_neg = pd.concat([pred_df_neg, tmp_df_neg])
	pred_df = pd.concat([pred_df, tmp_df2])
	
	#tmp_df3.to_csv(cell_path + 'TOP.{}.csv'.format(fifi.split('/')[8].split('.')[0]))



tissue_set = ['CENTRAL_NERVOUS_SYSTEM', 'LUNG', 'BREAST', 'BONE', 'OVARY', 'PROSTATE', 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'LARGE_INTESTINE', 'SKIN', 'PLEURA', 'KIDNEY' ] # list(set(test_cell_df['tissue']))
color_set = ['#CF3476','#FF7514','#025669','#308446','#84C3BE','#D53032','#4ddcfd','#ffcd36','#ac8cff',"#0000ffff","#7bff68ff"] # "#1E1E1E"
color_dict = {a : color_set[tissue_set.index(a)] for a in tissue_set}


my_order = pred_df.groupby(by=["CCLE"])["PRED"].median().iloc[::-1].index
my_order2 = my_order[::-1].index

order_tissue = ['_'.join(a.split('_')[1:]) for a in my_order]
order_tissue_col = [color_dict[a] for a in order_tissue]

fig, ax = plt.subplots(figsize=(40, 8))
sns.violinplot(ax = ax, data  = pred_df, x = 'CCLE', y = 'PRED', linewidth=1,  edgecolor="dimgrey", order=my_order)
violins = [c for i,c in enumerate(ax.collections) if i%2 == 0] # pollycollection 가져오기 위함 
for tiss_num in range(len(my_order)) : 
	violins[tiss_num].set_facecolor(mcolors.to_rgba(order_tissue_col[tiss_num]))

ax.set_xlabel('cell names', fontsize=20)
ax.set_ylabel('pred_synergy', fontsize=20)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
#ax.set_xticklabels('', rotation = 90)
ax.tick_params(axis='both', which='major', labelsize=20 )
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.grid(False)

PRJ_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V8_W414_349_MIS2/'

plt.savefig(os.path.join(PRJ_PATH,'cell_pred2.png'), dpi = 300)
plt.savefig(os.path.join(PRJ_PATH,'cell_pred0925.pdf'), format="pdf", bbox_inches = 'tight')






# BOX plot -> 너무 퍼져서 violin plot 이 의미가 없었음 
PRJ_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V6_W213_349_MIS2/for_png/'
PRJ_PATH = '/home01/k040a01/02.M3V6/M3V6_W213_349_MIS2/CELL_VAL'


#cell_list = list(set(merged_CVCL_RE.DrugCombCCLE)) # 
my_order = pred_df.groupby(by=["CCLE"])["PRED"].median().iloc[::-1].index
my_order2 = my_order.iloc[::-1].index

# 이게 무슨 순서더라 
# 안돼 그래도 median 순서로 가고싶음 
#order = ['A-673', 'MDA-MB-361', 'CAMA-1', 'MDA-MB-175-VII', 'L-1236', 'UACC-812', 'HCC1500', 'HCC1419', 'U-HO1', 'T98G', 'BT-474', 'LOX IMVI', 'ZR751', 'NCIH2122', 'MDAMB436', 'COLO 829', 'KPL1', 'EKVX', 'A498', 'MDA-MB-231', 'HT144', 'T-47D', 'A375', 'RPMI7951', 'A2058', 'G-361', 'KM12', 'MCF7', 'CAOV3', 'COLO 800', 'Mel Ho', 'NCI-H226', 'UACC-257', 'OVCAR3', 'SF-539', 'PC-3', 'CAKI-1', 'UACC62', 'SR', 'UO-31', 'NCI-H522', 'SK-MEL-5', 'HOP-92', 'ACHN', '786O', 'HS 578T', 'RVH-421', 'K-562', 'HOP-62', 'OVCAR-5', 'MSTO', 'U251', 'COLO 792', 'NCIH23', 'OV90', 'LOVO', 'RPMI-8226', 'A101D', 'NCIH520', 'SK-MEL-2', 'MALME-3M', 'IPC-298', 'HT29', 'SF-295', 'UWB1289', 'SW-620', 'OVCAR-4', 'SKMES1', 'SW837', 'MDA-MB-468', 'HCT-15', 'OVCAR-8', 'SNB-75', 'SF-268', 'HCT116', 'BT-549', 'A549', 'IGROV1', 'A427', 'SK-MEL-28', 'SK-OV-3', 'NCIH1650', 'WM115', 'NCI-H460', 'MeWo', 'ES2', 'SKMEL30', 'A2780', 'RKO', 'VCAP', 'PA1', 'DLD1']
#order2= ['A673_BONE', 'MDAMB361_BREAST', 'CAMA1_BREAST', 'MDAMB175VII_BREAST', 'L1236_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'UACC812_BREAST', 'HCC1500_BREAST', 'HCC1419_BREAST', 'UHO1_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'T98G_CENTRAL_NERVOUS_SYSTEM', 'BT474_BREAST', 'LOXIMVI_SKIN', 'ZR751_BREAST', 'NCIH2122_LUNG', 'MDAMB436_BREAST', 'COLO829_SKIN', 'KPL1_BREAST', 'EKVX_LUNG', 'A498_KIDNEY', 'MDAMB231_BREAST', 'HT144_SKIN', 'T47D_BREAST', 'A375_SKIN', 'RPMI7951_SKIN', 'A2058_SKIN', 'G361_SKIN', 'KM12_LARGE_INTESTINE', 'MCF7_BREAST', 'CAOV3_OVARY', 'COLO800_SKIN', 'MELHO_SKIN', 'NCIH226_LUNG', 'UACC257_SKIN', 'NIHOVCAR3_OVARY', 'SF539_CENTRAL_NERVOUS_SYSTEM', 'PC3_PROSTATE', 'CAKI1_KIDNEY', 'UACC62_SKIN', 'SR786_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'UO31_KIDNEY', 'NCIH522_LUNG', 'SKMEL5_SKIN', 'HOP92_LUNG', 'ACHN_KIDNEY', '786O_KIDNEY', 'HS578T_BREAST', 'RVH421_SKIN', 'K562_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'HOP62_LUNG', 'OVCAR5_OVARY', 'MSTO211H_PLEURA', 'U251MG_CENTRAL_NERVOUS_SYSTEM', 'COLO792_SKIN', 'NCIH23_LUNG', 'OV90_OVARY', 'LOVO_LARGE_INTESTINE', 'RPMI8226_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'A101D_SKIN', 'NCIH520_LUNG', 'SKMEL2_SKIN', 'MALME3M_SKIN', 'IPC298_SKIN', 'HT29_LARGE_INTESTINE', 'SF295_CENTRAL_NERVOUS_SYSTEM', 'UWB1289_OVARY', 'SW620_LARGE_INTESTINE', 'OVCAR4_OVARY', 'SKMES1_LUNG', 'SW837_LARGE_INTESTINE', 'MDAMB468_BREAST', 'HCT15_LARGE_INTESTINE', 'OVCAR8_OVARY', 'SNB75_CENTRAL_NERVOUS_SYSTEM', 'SF268_CENTRAL_NERVOUS_SYSTEM', 'HCT116_LARGE_INTESTINE', 'BT549_BREAST', 'A549_LUNG', 'IGROV1_OVARY', 'A427_LUNG', 'SKMEL28_SKIN', 'SKOV3_OVARY', 'NCIH1650_LUNG', 'WM115_SKIN', 'NCIH460_LUNG', 'MEWO_SKIN', 'ES2_OVARY', 'SKMEL30_SKIN', 'A2780_OVARY', 'RKO_LARGE_INTESTINE', 'VCAP_PROSTATE', 'PA1_OVARY', 'DLD1_LARGE_INTESTINE']



order_tissue = ['_'.join(a.split('_')[1:]) for a in my_order]
order_tissue_col = [color_dict[a] for a in order_tissue]


fig, ax = plt.subplots(figsize=(40, 8))
sns.violinplot(ax = ax, data  = pred_df, x = 'CCLE', y = 'PRED', linewidth=1,  
edgecolor="dimgrey", order=my_order)
violins = [c for i,c in enumerate(ax.collections) if i%2 == 0] # pollycollection 가져오기 위함 
for tiss_num in range(len(my_order)) : 
	violins[tiss_num].set_facecolor(mcolors.to_rgba(order_tissue_col[tiss_num]))

ax.set_xlabel('cell names', fontsize=20)
ax.set_ylabel('pred_synergy', fontsize=20)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20 )
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.grid(False)
# plt.savefig(os.path.join(PRJ_PATH,'cell_pred.png'), dpi = 300)
fig.savefig('{}/{}.pdf'.format(PRJ_PATH, 'cell_pred'), format="pdf", bbox_inches = 'tight')
fig.savefig('{}/{}.pdf'.format(PRJ_PATH, 'cell_pred1'), format="pdf", bbox_inches = 'tight')

plt.close()

# row num : 84,153,008





#######################################
각기 살펴보는 경우 


DC_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/'

DC_DRUG_DF_FULL = pd.read_csv('/st06/jiyeonH/11.TOX/MY_TRIAL_6/DC_DRUG_DF_PC.csv', sep ='\t')

def top_bot_30 (CELL) : 
	#CELL = 'A427_LUNG'
	DONE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V8_W414_349_MIS2/CELL_VAL/DONE/'
	mini_csv = pd.read_csv(DONE_PATH+'PRED_CELL_{}.FINAL.csv'.format(CELL))
	#
	drug_name = DC_DRUG_DF_FULL[['dname','CID']]
	#
	drug_name.columns = ['ROW_NAME','ROW_CID']
	mini_csv_1 = pd.merge(mini_csv,drug_name, on = 'ROW_CID', how='left' )
	drug_name.columns = ['COL_NAME','COL_CID']
	mini_csv_2 = pd.merge(mini_csv_1, drug_name, on = 'COL_CID', how='left' )
	#
	mini_csv_3 = mini_csv_2.sort_values('PRED', ascending = False)
	#
	top_30 = mini_csv_3.iloc[:30,]
	bot_30 = mini_csv_3.iloc[-30:,]
	#
	top_30.to_csv(DONE_PATH+'top30_{}.csv'.format(CELL), sep = '\t', index= False)
	bot_30.to_csv(DONE_PATH+'bot30_{}.csv'.format(CELL), sep = '\t', index= False)
	return (mini_csv_3)




DONE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V8_W414_349_MIS2/CELL_VAL/DONE/'

CVCL_files_raw = glob.glob(DONE_PATH+'/PRED_*SKIN*.FINAL.csv')
tissues = ['_'.join(a.split('/')[-1].split('.')[0].split('PRED_')[1].split('_')[2:]) for a in CVCL_files_raw]
strips = [a.split('/')[-1].split('.')[0].split('PRED_')[1].split('_')[1] for a in CVCL_files_raw]
ccle = [a.split('/')[-1].split('.')[0].split('PRED_CELL_')[1] for a in CVCL_files_raw]

for cell in ccle :
	top_bot_30(cell)


NCIH520 = top_bot_30('NCIH520_LUNG')

NCIH520[(NCIH520.ROW_NAME!='dioxybenzone') & (NCIH520.COL_NAME!='dioxybenzone')]

A549_LUNG = top_bot_30('A549_LUNG')

DLD1_LARGE_INTESTINE = top_bot_30('DLD1_LARGE_INTESTINE')
row = list(DLD1_LARGE_INTESTINE.ROW_CID)
col = list(DLD1_LARGE_INTESTINE.COL_CID)

row2 = [5282049 if a == 443593 else a for a in row ]
col2 = [5282049 if a == 443593 else a for a in col ]

DLD1_LARGE_INTESTINE['row2'] = row2
DLD1_LARGE_INTESTINE['col2'] = col2

DLD1_LARGE_INTESTINE['CID_CID'] = [str(int(row2[i])) + '___' + str(int(col2[i])) if row2[i] < col2[i] else str(int(col2[i])) + '___' + str(int(row2[i])) for i in range(DLD1_LARGE_INTESTINE.shape[0])]
DLD1_LARGE_INTESTINE_tmp = DLD1_LARGE_INTESTINE[['CID_CID']].drop_duplicates()
DLD1_LARGE_INTESTINE_tmp.iloc[:30]

11178236

LOVO = top_bot_30('LOVO_LARGE_INTESTINE')
row = list(LOVO.ROW_CID)
col = list(LOVO.COL_CID)

row2 = [10126189 if a == 11178236 else a for a in row ]
col2 = [10126189 if a == 11178236 else a for a in col ]

LOVO['row2'] = row2
LOVO['col2'] = col2

LOVO['CID_CID'] = [str(int(row2[i])) + '___' + str(int(col2[i])) if row2[i] < col2[i] else str(int(col2[i])) + '___' + str(int(row2[i])) for i in range(LOVO.shape[0])]
LOVO_tmp = LOVO[['CID_CID']].drop_duplicates()
LOVO_tmp.iloc[:30]


#########################




DONE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V8_W414_349_MIS2/CELL_VAL/DONE/'
CVCL_files_raw = glob.glob(DONE_PATH+'/PRED_*LUNG*.FINAL.csv')
tissues = ['_'.join(a.split('/')[-1].split('.')[0].split('PRED_')[1].split('_')[2:]) for a in CVCL_files_raw]
strips = [a.split('/')[-1].split('.')[0].split('PRED_')[1].split('_')[1] for a in CVCL_files_raw]
ccle = [a.split('/')[-1].split('.')[0].split('PRED_CELL_')[1] for a in CVCL_files_raw]

LUNG_ALL = pd.DataFrame(columns = ['PRED', 'ROW_CID','COL_CID','CCLE', 'Y','tissue','strip'])

for indd in range(len(CVCL_files_raw)) :
	fifi = CVCL_files_raw[indd]
	tiss = tissues[indd]
	stripname = strips[indd]
	print(stripname)
	tmp_df = pd.read_csv(fifi,  low_memory=False, index_col = False)
	tmp_df2 = tmp_df[['ROW_CID','COL_CID','CCLE', 'PRED', 'Y']]
	tmp_df2['tissue'] = tiss
	tmp_df2['strip'] = stripname
	LUNG_ALL = pd.concat([LUNG_ALL, tmp_df2])


def get_pair(cid1, cid2) :
	tmp = LUNG_ALL[(LUNG_ALL.ROW_CID==cid1 ) & (LUNG_ALL.COL_CID==cid2)]
	tmp1 = tmp.sort_values('PRED')
	#
	tmp = LUNG_ALL[(LUNG_ALL.ROW_CID==cid2 ) & (LUNG_ALL.COL_CID==cid1)]
	tmp2 = tmp.sort_values('PRED')
	return tmp1, tmp2


# lung 
res1, res2 = get_pair(2907, 31703)
res01, res02 = get_pair(2907, 5978)
res03, res04 = get_pair(31703, 5978)
res05, res06 = get_pair(5702198, 135410875)
res07, res08 = get_pair(5702198, 60750)
res09, res10 = get_pair(5702198, 148124)
res11, res12 = get_pair(426756, 135410875)
res13, res14 = get_pair(426756, 60750)
res15, res16 = get_pair(426756, 148124)

	
	
	


DONE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V8_W414_349_MIS2/CELL_VAL/DONE/'
CVCL_files_raw = glob.glob(DONE_PATH+'/PRED_*LARGE_INTESTINE*.FINAL.csv')
tissues = ['_'.join(a.split('/')[-1].split('.')[0].split('PRED_')[1].split('_')[2:]) for a in CVCL_files_raw]
strips = [a.split('/')[-1].split('.')[0].split('PRED_')[1].split('_')[1] for a in CVCL_files_raw]
ccle = [a.split('/')[-1].split('.')[0].split('PRED_CELL_')[1] for a in CVCL_files_raw]

COLON_ALL = pd.DataFrame(columns = ['PRED', 'ROW_CID','COL_CID','CCLE', 'Y','tissue','strip'])

for indd in range(len(CVCL_files_raw)) :
	fifi = CVCL_files_raw[indd]
	tiss = tissues[indd]
	stripname = strips[indd]
	print(stripname)
	tmp_df = pd.read_csv(fifi,  low_memory=False, index_col = False)
	tmp_df2 = tmp_df[['ROW_CID','COL_CID','CCLE', 'PRED', 'Y']]
	tmp_df2['tissue'] = tiss
	tmp_df2['strip'] = stripname
	COLON_ALL = pd.concat([COLON_ALL, tmp_df2])


def get_pair(cid1, cid2) :
	tmp = COLON_ALL[(COLON_ALL.ROW_CID==cid1 ) & (COLON_ALL.COL_CID==cid2)]
	tmp1 = tmp.sort_values('PRED')
	#
	tmp = COLON_ALL[(COLON_ALL.ROW_CID==cid2 ) & (COLON_ALL.COL_CID==cid1)]
	tmp2 = tmp.sort_values('PRED')
	return tmp1, tmp2


# colon
res01, res02 = get_pair(60953, 9887053)
res03, res04 = get_pair(3385 ,135403648 )
res05, res06 = get_pair(3385 , 60838)
res07, res08 = get_pair( 135403648,60838 )
res09, res10 = get_pair(3385 ,135403648 )
res11, res12 = get_pair( 3385, 9887053)
res13, res14 = get_pair( 135403648,9887053 )
res15, res16 = get_pair( 41867, 5702198)
res17, res18 = get_pair( 41867,3385 )
res19, res20 = get_pair( 41867,60953 )
res21, res22 = get_pair( 5702198 ,3385 )
res23, res24 = get_pair( 5702198 ,60953  )
res25, res26 = get_pair(  3385, 60953)









def get_pair(cid1, cid2) :
	tmp = pred_df[(pred_df.ROW_CID==cid1 ) & (pred_df.COL_CID==cid2)]
	tmp1 = tmp.sort_values('PRED')
	#
	tmp = pred_df[(pred_df.ROW_CID==cid2 ) & (pred_df.COL_CID==cid1)]
	tmp2 = tmp.sort_values('PRED')
	#
	if (tmp1.shape[0] >0) & (tmp2.shape[0] >0) :
		res = tmp1, tmp2
	elif tmp1.shape[0] >0 :
		res = tmp1
	else :
		res = tmp2 
	return res


# hogkin
res0 = get_pair(31703, 5360373) X
res1 = get_pair(31703, 13342) X
res2 = get_pair(31703, 135398738) 
res3 = get_pair(5360373, 13342) x
res4 = get_pair(5360373, 135398738) x
res5 = get_pair(13342,	135398738) x
res6 = get_pair(5360373,36462) x
res7 = get_pair(5360373,31703) x 
res8 = get_pair(5360373,2907) x 
res9 = get_pair(5360373,5978) x 
res10 = get_pair(5360373,4915) x 
res11 = get_pair(5360373,5865) x 
res12 = get_pair(36462,31703)
res13 = get_pair(36462,2907)
res14 = get_pair(36462,5978) x 
res15 = get_pair(36462,4915)
res16 = get_pair(36462,	5865)
res17 = get_pair(31703,	2907)
res18 = get_pair(31703,	5978) x 
res19 = get_pair(31703,	4915)
res20 = get_pair(31703,	5865)
res21 = get_pair(2907	,5978) x 
res22 = get_pair(2907	,4915)
res23 = get_pair(2907,	5865)
res24 = get_pair(5978,	4915) x 
res25 = get_pair(5978,	5865) x 
res26 = get_pair(4915,	5865)




# non hodgkin 
res30 = get_pair(2907,	5978) x 
res31 = get_pair(2907,	5865)
res32 = get_pair(31703,	5978) x 
res33 = get_pair(31703,	5865)
res34 = get_pair(5978,	5865) x 
res35 = get_pair(36462,	5865)
res36 = get_pair(36462,	5978) x 
res37 = get_pair(36462,	2907)
res38 = get_pair(36462,	31703)
res39 = get_pair(5865,	5978) x 

res40 = get_pair(5865,	2907)
res41 = get_pair(5865,	31703)
res42 = get_pair(5978,	2907) x 
res43 = get_pair(5978,	31703) x 
res44 = get_pair(29072907,	31703) x 


res0 = get_pair()
res1 = get_pair()
res2 = get_pair()
res3 = get_pair()
res4 = get_pair()
res5 = get_pair()
res6 = get_pair()
res7 = get_pair()
res8 = get_pair()
res9 = get_pair()







# leukemia

res50 = get_pair(5743,	2907)
res51 = get_pair(5743,	5978) x 
res52 = get_pair(5743,	31703)
res53 = get_pair(6253,	30323)






# myeloma 

res60 = get_pair(216326,	5743)
res61 = get_pair(387447,	5426)
res62 = get_pair(387447,	5743) # 적음 
res63 = get_pair(5426,	5743)



# breast 

res70 = get_pair(2907,	126941) # 적음 
res71 = get_pair(2907,	3385) # 적음 
res72 = get_pair(126941,	3385) # 적음 



tmp1 = pred_df[ (pred_df.ROW_CID== 5702198) ]
tmp11 = tmp1.sort_values('PRED')
tmp2 = pred_df[ (pred_df.COL_CID== 5702198) ]
tmp22 = tmp2.sort_values('PRED')

# gemcitabine, MK2206
res80 = get_pair(60750, 24964624) 

# cisplatin, MK2206 
res81 = get_pair(5702198  , 24964624) 

# gefitinib, antimycin A 
res82 = get_pair(123631, 14957)
...             ...      ...       ...                            ...  ...                     ...          ...
3721098    8.242529  14957.0  123631.0                MDAMB468_BREAST  1.0                  BREAST     MDAMB468
562385     8.940926  14957.0  123631.0                    T47D_BREAST  1.0                  BREAST         T47D
1874894    9.713179  14957.0  123631.0                   OVCAR8_OVARY  1.0                   OVARY       OVCAR8
7910001   10.106432  14957.0  123631.0                   ZR751_BREAST  1.0                  BREAST        ZR751
13569963  11.383194  14957.0  123631.0           DLD1_LARGE_INTESTINE  1.0         LARGE_INTESTINE         DLD1


# cardamonin, cisplatin 
res83 = get_pair(641785, 5702198)
517254     8.581692  5702198.0  641785.0           NCIH520_LUNG  1.0             LUNG    NCIH520
8424841    9.345329  5702198.0  641785.0  SW620_LARGE_INTESTINE  1.0  LARGE_INTESTINE      SW620
12760039  10.109173  5702198.0  641785.0           BT549_BREAST  1.0           BREAST      BT549
12236433  10.421917  5702198.0  641785.0           NCIH460_LUNG  1.0             LUNG    NCIH460
7143583   19.277130  5702198.0  641785.0         HCC1419_BREAST  1.0           BREAST    HCC1419


31703 doxorubicin 
4477 Niclosamide

res84 = get_pair(31703, 4477)

+ ciaplatin 
res85 = get_pair(31703, 5702198)

6256 L triflu 
mk 2206 
res86 = get_pair(6256, 24964624)

# ciprofloxacin / 5-FU
res86 = get_pair(2764, 3385)

# ciprofloxacin / olaparib 
res86 = get_pair(2764, 23725625)

# Antimycin A 


tmp1 = pred_df[ (pred_df.ROW_CID==  14957 ) ]
tmp11 = tmp1.sort_values('PRED')
tmp2 = pred_df[ (pred_df.COL_CID== 5702198) ]
tmp22 = tmp2.sort_values('PRED')






import json

with open("/st06/jiyeonH/11.TOX/DR_SPRING/PC_cancerdrug_NCI.json", "r") as json_file:
	nci_cancer = json.load(json_file)


tmptmp = nci_cancer['Annotations']['Annotation']

NCI_ver = pd.DataFrame(columns = ['NAME','CID'])

for i in tmptmp :
	NAME = i['Name']
	if 'LinkedRecords' in i.keys():
		if 'CID' in i['LinkedRecords'].keys() :
			CID = i['LinkedRecords']['CID']
			tmp_df = pd.DataFrame({
			'CID' : CID, 'NAME' : NAME
				})
			NCI_ver = pd.concat([NCI_ver, tmp_df])


NCI_ver = NCI_ver.drop_duplicates() # 404 

NCI_ver_cids = list(set(NCI_ver.CID))



pred_df.shape # 1,617,857,317


pred_df_nci = pred_df[pred_df.COL_CID.isin(NCI_ver_cids)]
# 47,270,683

pred_df_nci2 = pred_df_nci[pred_df_nci.ROW_CID.isin(NCI_ver_cids)]
# 1,175,649



def top_bot_30 (CELL) : 
	mini_csv  = pred_df_nci2[pred_df_nci2.CCLE == CELL]
	drug_name = DC_DRUG_DF_FULL[['dname','CID']]
	#
	drug_name.columns = ['ROW_NAME','ROW_CID']
	mini_csv_1 = pd.merge(mini_csv,drug_name, on = 'ROW_CID', how='left' )
	drug_name.columns = ['COL_NAME','COL_CID']
	mini_csv_2 = pd.merge(mini_csv_1, drug_name, on = 'COL_CID', how='left' )
	#
	mini_csv_3 = mini_csv_2.sort_values('PRED', ascending = False)
	#
	top_30 = mini_csv_3.iloc[:30,]
	print(top_30[top_30.PRED>=10])
	#
	top_30.to_csv(DONE_PATH+'NCI_top30_{}.csv'.format(CELL), sep = '\t', index= False)
	



CVCL_files_raw = glob.glob(DONE_PATH+'/PRED_*.FINAL.csv')
tissues = ['_'.join(a.split('/')[-1].split('.')[0].split('PRED_')[1].split('_')[2:]) for a in CVCL_files_raw]
strips = [a.split('/')[-1].split('.')[0].split('PRED_')[1].split('_')[1] for a in CVCL_files_raw]
ccle = [a.split('/')[-1].split('.')[0].split('PRED_CELL_')[1] for a in CVCL_files_raw]

for cell in ccle :
	top_bot_30(cell)
















import json

tmp_all = []
for num in range(8) :
	nn = num+1 
	with open("/st06/jiyeonH/11.TOX/DR_SPRING/FDA.{}.json".format(nn), "r") as json_file:
		fda = json.load(json_file)
	#
	tmptmp = fda1['Annotations']['Annotation']
	tmp_all = tmp_all + tmptmp





FDA_ver = pd.DataFrame(columns = ['NAME','CID'])

for i in tmp_all :
	NAME = i['Name']
	if 'LinkedRecords' in i.keys():
		if 'CID' in i['LinkedRecords'].keys() :
			CID = i['LinkedRecords']['CID']
			tmp_df = pd.DataFrame({
			'CID' : CID, 'NAME' : NAME
				})
			FDA_ver = pd.concat([FDA_ver, tmp_df])


FDA_ver = FDA_ver.drop_duplicates() # 808

FDA_ver_cids = list(set(FDA_ver.CID))



pred_df.shape # 1,617,857,317


pred_df_fda = pred_df[pred_df.COL_CID.isin(FDA_ver_cids)]
# 68,747,916


pred_df_fda2 = pred_df_fda[pred_df_fda.ROW_CID.isin(FDA_ver_cids)]
# 3,654,044




def top_bot_30 (CELL) : 
	mini_csv  = pred_df_fda2[pred_df_fda2.CCLE == CELL]
	drug_name = DC_DRUG_DF_FULL[['dname','CID']]
	#
	drug_name.columns = ['ROW_NAME','ROW_CID']
	mini_csv_1 = pd.merge(mini_csv,drug_name, on = 'ROW_CID', how='left' )
	drug_name.columns = ['COL_NAME','COL_CID']
	mini_csv_2 = pd.merge(mini_csv_1, drug_name, on = 'COL_CID', how='left' )
	#
	mini_csv_3 = mini_csv_2.sort_values('PRED', ascending = False)
	#
	top_30 = mini_csv_3.iloc[:30,]
	print(top_30[top_30.PRED>=10])
	#
	#top_30.to_csv(DONE_PATH+'FDA_top30_{}.csv'.format(CELL), sep = '\t', index= False)
	#return (mini_csv_3)



CVCL_files_raw = glob.glob(DONE_PATH+'/PRED_*.FINAL.csv')
tissues = ['_'.join(a.split('/')[-1].split('.')[0].split('PRED_')[1].split('_')[2:]) for a in CVCL_files_raw]
strips = [a.split('/')[-1].split('.')[0].split('PRED_')[1].split('_')[1] for a in CVCL_files_raw]
ccle = [a.split('/')[-1].split('.')[0].split('PRED_CELL_')[1] for a in CVCL_files_raw]

for cell in ccle :
	top_bot_30(cell)

