
# 내가 만든거에 Graph 추가하는 방법 고안 

import rdkit
import os 
import os.path as osp
from math import ceil 
import pandas as pd 
import numpy as np
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

seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

ID_GENE_ORDER_mini = ID_G.nodes()
IAS_PPI = nx.adjacency_matrix(ID_G)

JY_GRAPH = ID_G
JY_GRAPH_ORDER = ID_G.nodes()
JY_ADJ = nx.adjacency_matrix(ID_G)

JY_ADJ_tmp = torch.LongTensor(JY_ADJ.toarray())
JY_ADJ_IDX = JY_ADJ_tmp.to_sparse().indices() 

A_B_C_S = CELLO_DC_BETA.reset_index()
A_B_C_S_SET = A_B_C_S[['BETA_sig_id_x','BETA_sig_id_y','DrugCombCello']].drop_duplicates()
A_B_C_S_SET = A_B_C_S_SET.reset_index()


BETA_ORDER_pre =[list(L_matching_list.PPI_name).index(a) for a in JY_GRAPH_ORDER]
BETA_ORDER_DF = L_matching_list.iloc[BETA_ORDER_pre] # 어차피 ref 다르고 같은 애들이라 괜춘 
BETA_ORDER = list(BETA_ORDER_DF.entrez)

def get_LINCS_data(DrugA_SIG, DrugB_SIG):
	DrugA_EXP = BETA_BIND[['id',DrugA_SIG]]
	DrugB_EXP = BETA_BIND[['id',DrugB_SIG]]
	BIND_ORDER =[list(BETA_BIND.id).index(a) for a in BETA_ORDER_DF.entrez] 
	DrugA_EXP_ORD = DrugA_EXP.iloc[BIND_ORDER]
	DrugB_EXP_ORD = DrugB_EXP.iloc[BIND_ORDER]
	#
	ARR = np.array([list(DrugA_EXP_ORD[DrugA_SIG]), list(DrugB_EXP_ORD[DrugB_SIG])])
	SUM = np.sum(ARR, axis = 0)
	return DrugA_EXP_ORD, DrugB_EXP_ORD, SUM


def get_morgan(smiles):
	result = []
	try:
		tmp = Chem.MolFromSmiles(smiles)
		result.append(tmp)
	except ValueError as e :
		tmp.append("NA")
	return result[0]


def get_CHEMI_data(Drug_SIG, bitsize):
	A_SM = BETA_SELEC_SIG[BETA_SELEC_SIG.sig_id == Drug_SIG]['canonical_smiles']
	A_SM = A_SM.values[0]
	#
	A_morgan = get_morgan(A_SM)
	bi = {}
	A_FP = rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect(A_morgan, radius=2, nBits = bitsize , bitInfo=bi)
	if len(A_FP)==bitsize :
		A_FP_LIST = list(A_FP.ToBitString())
	else : 
		A_FP_LIST = ['0']*bitsize
	#
	return A_FP_LIST


def get_synergy_data(DrugA_SIG, DrugB_SIG, Cell):
	ABCS1 = A_B_C_S[A_B_C_S.BETA_sig_id_x == DrugA_SIG]
	ABCS2 = ABCS1[ABCS1.BETA_sig_id_y == DrugB_SIG]
	ABCS3 = ABCS2[ABCS2.DrugCombCello == Cell]
	synergy_score = np.median(ABCS3.synergy_loewe)
	return synergy_score






MY_chem1 = []
MY_chem2 = []
MY_exp = []
MY_syn = []



for IND in range(A_B_C_S_SET.shape[0]):
	DrugA_SIG = A_B_C_S_SET.iloc[IND,]['BETA_sig_id_x']
	DrugB_SIG = A_B_C_S_SET.iloc[IND,]['BETA_sig_id_y']
	Cell = A_B_C_S_SET.iloc[IND,]['DrugCombCello']
	#
	bitsize = 256
	DrugA_FP = [int(a) for a in get_CHEMI_data(DrugA_SIG, bitsize)]
	DrugB_FP = [int(a) for a in get_CHEMI_data(DrugB_SIG, bitsize)]
	#
	_, _, LINCS = get_LINCS_data(DrugA_SIG, DrugB_SIG)
	#
	AB_SYN = get_synergy_data(DrugA_SIG, DrugB_SIG, Cell)
	#
	MY_chem1.append(DrugA_FP)
	MY_chem2.append(DrugB_FP)
	MY_exp.append(list(LINCS))
	MY_syn.append(AB_SYN)

	
MY_chem1_tch = torch.tensor(np.array(MY_chem1))
MY_chem2_tch = torch.tensor(np.array(MY_chem2))
# MY_exp_tch = torch.tensor(np.array(MY_exp))
MY_syn_tch = torch.tensor(np.array(MY_syn))

# MY_chem1 = np.load('/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/MY_chem1.npy')
# MY_chem2 = np.load('/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/MY_chem2.npy')
# MY_exp = np.load('/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/MY_exp.npy')
# MY_syn = np.load('/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/MY_syn.npy')




# make exp feature
for IND in range(A_B_C_S_SET.shape[0]):
	DrugA_SIG = A_B_C_S_SET.iloc[IND,]['BETA_sig_id_x']
	DrugB_SIG = A_B_C_S_SET.iloc[IND,]['BETA_sig_id_y']
	Cell = A_B_C_S_SET.iloc[IND,]['DrugCombCello']
	#
	#
	FEAT1, FEAT2, LINCS = get_LINCS_data(DrugA_SIG, DrugB_SIG)
	#
	AB_SYN = get_synergy_data(DrugA_SIG, DrugB_SIG, Cell)
	#
	MY_chem1.append(DrugA_FP)
	MY_chem2.append(DrugB_FP)
	MY_exp.append(list(LINCS))
	MY_syn.append(AB_SYN)

DrugA_SIG = A_B_C_S_SET.iloc[0,]['BETA_sig_id_x']
DrugB_SIG = A_B_C_S_SET.iloc[0,]['BETA_sig_id_y']
Cell = A_B_C_S_SET.iloc[0,]['DrugCombCello']
FEAT1, FEAT2, LINCS_SUM = get_LINCS_data(DrugA_SIG, DrugB_SIG)
FEAT = pd.merge(FEAT1, FEAT2) # 어차피 같은 순서 
T_FEAT = torch.Tensor(np.array(FEAT[[DrugA_SIG, DrugB_SIG]]))

    


class GCN_edge_weight(Dataset): 
		def __init__(self, list_feature, list_adj, list_adj_weight, list_ans):
				self.list_feature = list_feature # input 1
				self.list_adj = list_adj # input 2
				self.list_adj_weight = list_adj_weight
				self.list_ans = list_ans # output 1 
#
		def __len__(self): 
				return len(self.list_feature)
#
		def __getitem__(self, index): 
				return self.list_feature[index], self.list_adj[0], self.list_adj_weight[0], self.list_ans[index]



def graph_collate_fn(batch):
	edge_index_list = []
	edge_weight_list = []
	node_features_list = []
	node_labels_list = []
	num_nodes_seen = 0
	for features_labels_edge_index_tuple in batch:
		node_features_list.append(features_labels_edge_index_tuple[0])
		edge_weight_list.append(features_labels_edge_index_tuple[2])
		node_labels_list.append(features_labels_edge_index_tuple[3])
		edge_index = features_labels_edge_index_tuple[1]  # all of the components are in the [0, N] range
		edge_index_list.append(edge_index + num_nodes_seen)  # very important! translate the range of this component
		num_nodes_seen += features_labels_edge_index_tuple[0].shape[0]  # update the number of nodes we've seen so far
	node_features = torch.cat(node_features_list, 0)
	edge_weights = torch.cat(edge_weight_list, 0)
	node_labels = torch.cat(node_labels_list, 0).view(-1,6)
	edge_index = torch.cat(edge_index_list, 1)
	return node_features, edge_index, edge_weights, node_labels


class GraphDataLoader(torch.utils.data.dataloader.DataLoader):
	def __init__(self, node_features_list, edge_index_list, edge_weight_list, node_labels_list, batch_size=1, shuffle=False):
		graph_dataset = GCN_edge_weight(node_features_list, edge_index_list, edge_weight_list, node_labels_list)
		super().__init__( graph_dataset, batch_size, shuffle, collate_fn=graph_collate_fn)


def load_graph_data():
	data_loader_train = GraphDataLoader(
					X_train,
					MINI_ONLY_INDEX,
					EDGE_WEIGHT,
					y_train,
					batch_size=100,
					shuffle=False
				)
	data_loader_val = GraphDataLoader(
					X_val,
					MINI_ONLY_INDEX,
					EDGE_WEIGHT,
					y_val,
					batch_size=100,
					shuffle=False  # no need to shuffle the validation and test graphs
				)
	data_loader_test = GraphDataLoader(
					X_test,
					MINI_ONLY_INDEX[80000:],
					EDGE_WEIGHT,
					y_test,
					batch_size=100,
					shuffle=False
				)
	return data_loader_train, data_loader_val, data_loader_test


data_loader_train, data_loader_val, data_loader_test = load_graph_data()









class MY_expGCN_parallel_model(torch.nn.Module):
	def __init__(self, drug1_indim, drug2_indim, layers_1, layers_2, layers_3, out_dim, inDrop, drop):
		super(MY_parallel_model, self).__init__()
		self.layers_1 = [int(a) for a in layers_1]
		self.layers_2 = [int(a) for a in layers_2]
		self.layers_3 = [int(a) for a in layers_3]
		self.input_dim1 = drug1_indim
		self.input_dim2 = drug2_indim
		self.out_dim = out_dim
		self.inDrop = inDrop
		self.drop = drop
		#
		self.Convs_1 = nn.ModuleList([torch.nn.Linear(self.input_dim1, self.layers_1[0] )])
		self.Convs_1.extend([torch.nn.Linear(self.layers_1[a], self.layers_1[a+1]) for a in range(len(self.layers_1)-1)])
		#
		self.Convs_2 = nn.ModuleList([torch.nn.Linear(self.input_dim2, self.layers_2[0] )])
		self.Convs_2.extend([torch.nn.Linear(self.layers_2[a], self.layers_2[a+1]) for a in range(len(self.layers_2)-1)])
		#
		self.SNPs = nn.ModuleList([torch.nn.Linear(self.layers_1[-1]+self.layers_2[-1], self.layers_3[0] )])
		self.SNPs.extend([torch.nn.Linear(self.layers_3[a], self.layers_3[a+1]) for a in range(len(self.layers_3)-1)])
		self.SNPs.extend([torch.nn.Linear(self.layers_3[-1], self.out_dim)])
		#
		self.reset_parameters()
	#
	def reset_parameters(self): 
		for conv in self.Convs_1 :
			conv.reset_parameters()
		for conv in self.Convs_2:
			conv.reset_parameters()
		for conv in self.SNPs:
			conv.reset_parameters()
	#
	def forward(self, input_drug1, input_drug2 ):
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
		X = torch.cat((input_drug1,input_drug2),1)
		for L3 in range(len(self.SNPs)):
			if L3 != len(self.SNPs)-1 :
				X = self.SNPs[L3](X)
				X = F.dropout(X, p=self.drop, training = self.training)
				X = F.relu(X)
			else : 
				X = self.SNPs[L3](X)
		return X
	






















