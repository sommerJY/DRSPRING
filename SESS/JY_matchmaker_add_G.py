
# 내가 만든거에 Graph 추가하는 방법 고안 

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


# graph 마다 확인 
ID_GENE_ORDER_mini = ID_G.nodes()
IAS_PPI = nx.adjacency_matrix(ID_G)

JY_GRAPH = ID_G
JY_GRAPH_ORDER = ID_G.nodes()
JY_ADJ = nx.adjacency_matrix(ID_G)

JY_ADJ_tmp = torch.LongTensor(JY_ADJ.toarray())
JY_ADJ_IDX = JY_ADJ_tmp.to_sparse().indices() 
JY_IDX_WEIGHT = ID_WEIGHT_SCORE



# DC set 확인 
A_B_C_S = CELLO_DC_BETA.reset_index()
A_B_C_S_SET = A_B_C_S[['BETA_sig_id_x','BETA_sig_id_y','DrugCombCello']].drop_duplicates()
A_B_C_S_SET = A_B_C_S_SET.reset_index()


# LINCS 확인 
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
	return DrugA_EXP_ORD, DrugB_EXP_ORD, ARR.T, SUM


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




MY_chem_A = torch.empty(size=(A_B_C_S_SET.shape[0], 256))
MY_chem_B= torch.empty(size=(A_B_C_S_SET.shape[0], 256))
MY_exp_A = torch.empty(size=(A_B_C_S_SET.shape[0], 978))
MY_exp_B = torch.empty(size=(A_B_C_S_SET.shape[0], 978))
MY_exp_AB = torch.empty(size=(A_B_C_S_SET.shape[0], 978, 2))
MY_syn =  torch.empty(size=(A_B_C_S_SET.shape[0],1))



for IND in range(A_B_C_S_SET.shape[0]):
	DrugA_SIG = A_B_C_S_SET.iloc[IND,]['BETA_sig_id_x']
	DrugB_SIG = A_B_C_S_SET.iloc[IND,]['BETA_sig_id_y']
	Cell = A_B_C_S_SET.iloc[IND,]['DrugCombCello']
	#
	bitsize = 256
	DrugA_FP = [int(a) for a in get_CHEMI_data(DrugA_SIG, bitsize)]
	DrugB_FP = [int(a) for a in get_CHEMI_data(DrugB_SIG, bitsize)]
	#
	EXP_A, EXP_B, EXP_AB, LINCS = get_LINCS_data(DrugA_SIG, DrugB_SIG)
	#
	AB_SYN = get_synergy_data(DrugA_SIG, DrugB_SIG, Cell)
	#
	MY_chem_A[IND] = torch.Tensor(DrugA_FP)
	MY_chem_B[IND] = torch.Tensor(DrugB_FP)
	MY_exp_A[IND] = torch.Tensor(EXP_A.iloc[:,1])
	MY_exp_B[IND] = torch.Tensor(EXP_B.iloc[:,1])
	MY_exp_AB[IND] = torch.Tensor(EXP_AB).unsqueeze(0)
	MY_syn[IND] = torch.Tensor([AB_SYN])


MY_chem_A_tch = torch.tensor(np.array(MY_chem_A))
MY_chem_B_tch = torch.tensor(np.array(MY_chem_B))
MY_exp_A_tch = torch.tensor(np.array(MY_exp_A))
MY_exp_B_tch = torch.tensor(np.array(MY_exp_B))
MY_exp_AB_tch = torch.tensor(np.array(MY_exp_AB))
MY_syn_tch = torch.tensor(np.array(MY_syn))

torch.save(MY_chem_A, '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/0610.MY_chem_A.pt')
torch.save(MY_chem_B, '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/0610.MY_chem_B.pt')
torch.save(MY_exp_A, '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/0610.MY_exp_A.pt')
torch.save(MY_exp_B, '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/0610.MY_exp_B.pt')
torch.save(MY_exp_AB, '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/0610.MY_exp_AB.pt')
torch.save(MY_syn, '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/0610.MY_syn.pt')

# MY_chem_A = torch.load('/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/0610.MY_chem_A.pt')
# MY_chem_B = torch.load('/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/0610.MY_chem_B.pt')
# MY_exp_A = torch.load('/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/0610.MY_exp_A.pt')
# MY_exp_B = torch.load('/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/0610.MY_exp_B.pt')
# MY_exp_AB = torch.load('/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/0610.MY_exp_AB.pt')
# MY_syn = torch.load('/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/0610.MY_syn.pt')

# MY_chem_A = torch.load(PRJ_PATH+'0614.MY_chem_A.pt')
# MY_chem_B = torch.load(PRJ_PATH+'0614.MY_chem_B.pt')
# MY_exp_A = torch.load(PRJ_PATH+'0614.MY_exp_A.pt')
# MY_exp_B = torch.load(PRJ_PATH+'0614.MY_exp_B.pt')
# MY_exp_AB = torch.load(PRJ_PATH+'0614.MY_exp_AB.pt')
# MY_syn = torch.load(PRJ_PATH+'0614.MY_syn.pt')


							# 실험용 
							DrugA_SIG = A_B_C_S_SET.iloc[0,]['BETA_sig_id_x']
							DrugB_SIG = A_B_C_S_SET.iloc[0,]['BETA_sig_id_y']
							Cell = A_B_C_S_SET.iloc[0,]['DrugCombCello']
							FEAT1, FEAT2, LINCS_SUM = get_LINCS_data(DrugA_SIG, DrugB_SIG)
							FEAT = pd.merge(FEAT1, FEAT2) # 어차피 같은 순서 
							T_FEAT = torch.Tensor(np.array(FEAT[[DrugA_SIG, DrugB_SIG]]))


						# 어쨌건 train / val / test 는 나눠야함 

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



def prepare_data_GCN(MY_chem1_tch, MY_chem2_tch, MY_exp_tch, MY_syn_tch, norm ) :
	chem_A_train, chem_A_tv, chem_B_train, chem_B_tv, exp_AB_train, exp_AB_tv, syn_train, syn_tv = sklearn.model_selection.train_test_split(
		MY_chem_A_tch, MY_chem_B_tch, MY_exp_AB, MY_syn_tch, 
		test_size= A_B_C_S_SET.shape[0]-6000 , random_state=42 )
	chem_A_val, chem_A_test, chem_B_val, chem_B_test, exp_AB_val, exp_AB_test, syn_val, syn_test  = sklearn.model_selection.train_test_split(
		chem_A_tv, chem_B_tv, exp_AB_tv, syn_tv, 
		test_size=A_B_C_S_SET.shape[0]-8000, random_state=42 )
	#
	train_data = {}
	val_data = {}
	test_data = {}
	#
	train1 = np.concatenate((chem_A_train, chem_B_train),axis=0) 
	train_data['drug1'], mean1, std1, mean2, std2, feat_filt = normalize(train1, norm=norm)
	val_data['drug1'], mmean1, sstd1, mmean2, sstd2, feat_filtt = normalize(chem_A_val, mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
	test_data['drug1'], mean1, std1, mean2, std2, feat_filt = normalize(chem_A_test, mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
	#
	train2 = np.concatenate((chem_B_train, chem_A_train),axis=0)
	train_data['drug2'], mean1, std1, mean2, std2, feat_filt = normalize(train2, norm=norm)
	val_data['drug2'], mmean1, sstd1, mmean2, sstd2, feat_filtt = normalize(chem_B_val, mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
	test_data['drug2'], mean1, std1, mean2, std2, feat_filt = normalize(chem_B_test, mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
	#
	#
	train_data['EXP_AB'] = torch.concat([exp_AB_train, exp_AB_train], axis = 0)
	val_data['EXP_AB'] = exp_AB_val
	test_data['EXP_AB'] = exp_AB_test
	#		
	train_data['y'] = np.concatenate((syn_train,syn_train), axis=0)
	val_data['y'] = syn_val
	test_data['y'] = syn_test
	print(test_data['drug1'].shape)
	print(test_data['drug2'].shape)
	return train_data, val_data, test_data



norm = 'tanh_norm'
train_data, val_data, test_data = prepare_data_GCN(MY_chem_A_tch, MY_chem_B_tch, MY_exp_AB, MY_syn_tch, norm)
# 6000 vs 2000 vs 1230



(1) check without collate 

class DATASET_GCN_W_F(Dataset): 
	def __init__(self, drug1_F, drug2_F, gcn_exp, gcn_adj, gcn_adj_weight, syn_ans):
			self.drug1_F = drug1_F
			self.drug2_F = drug2_F
			self.gcn_exp = gcn_exp 
			self.gcn_adj = gcn_adj 
			self.gcn_adj_weight = gcn_adj_weight
			self.syn_ans = syn_ans 
		#
	def __len__(self): 
			return len(self.drug1_F)
		#
	def __getitem__(self, index): 
			return self.drug1_F[index], self.drug2_F[index], self.gcn_exp[index], self.gcn_adj, self.gcn_adj_weight, self.syn_ans[index]

ys = train_data['y'].squeeze().tolist()

min_s = np.amin(ys)
loss_weight = np.log(train_data['y'] - min_s + np.e)


# JY_ADJ_IDX_T = torch.Tensor(JY_ADJ_IDX)
JY_IDX_WEIGHT_T = torch.Tensor(JY_IDX_WEIGHT).view(1,-1) 


T_train = DATASET_GCN_W_F(torch.Tensor(train_data['drug1']), torch.Tensor(train_data['drug2']), train_data['EXP_AB'], JY_ADJ_IDX, JY_IDX_WEIGHT_T, torch.Tensor(train_data['y']))
T_val = DATASET_GCN_W_F(torch.Tensor(val_data['drug1']), torch.Tensor(val_data['drug2']), val_data['EXP_AB'], JY_ADJ_IDX, JY_IDX_WEIGHT_T, torch.Tensor(val_data['y']))
T_test = DATASET_GCN_W_F(torch.Tensor(test_data['drug1']), torch.Tensor(test_data['drug2']), test_data['EXP_AB'], JY_ADJ_IDX, JY_IDX_WEIGHT_T, torch.Tensor(test_data['y']))


RAY_train = ray.put(T_train)
RAY_val = ray.put(T_val)
RAY_test = ray.put(T_test)
RAY_loss_weight = ray.put(loss_weight)


(2) collate ver 
def graph_collate_fn(batch):
	drug1_list = []
	drug2_list = []
	expAB_list = []
	adj_list = []
	adj_w_list = []
	y_list = []
	num_nodes_seen = 0
	for drug1, drug2, expAB, adj, adj_w, y in batch : 
		drug1_list.append(drug1)
		drug2_list.append(drug2)
		expAB_list.append(expAB)
		adj_list.append(adj+num_nodes_seen)
		adj_w_list.append(adj_w)
		y_list.append(y)
		num_nodes_seen += expAB.shape[0]
	drug1_new = torch.stack(drug1_list, 0)
	drug2_new = torch.stack(drug2_list, 0)
	expAB_new = torch.cat(expAB_list, 0)
	adj_new = 	torch.cat(adj_list, 1)
	adj_w_new = torch.cat(adj_w_list, 1)
	y_new = torch.stack(y_list, 0)
	return drug1_new, drug2_new, expAB_new, adj_new, adj_w_new, y_new
	




class MY_expGCN_parallel_model(torch.nn.Module):
	def __init__(self, G_layer, G_indim, G_hiddim, drug1_indim, drug2_indim, layers_1, layers_2, layers_3, out_dim, inDrop, drop):
		super(MY_expGCN_parallel_model, self).__init__()
		self.G_layer = G_layer
		self.G_indim = G_indim
		self.G_hiddim = G_hiddim
		self.layers_1 = [int(a) for a in layers_1]
		self.layers_2 = [int(a) for a in layers_2]
		self.layers_3 = [int(a) for a in layers_3]
		self.input_dim1 = drug1_indim
		self.input_dim2 = drug2_indim
		self.out_dim = out_dim
		self.inDrop = inDrop
		self.drop = drop
		self.tanh = torch.nn.Tanh()
		self.pool = pyg_nn.global_mean_pool
		#
		self.G_convs = torch.nn.ModuleList([pyg_nn.GCNConv(self.G_indim, self.G_hiddim)])
		self.G_convs.extend([pyg_nn.GCNConv(self.G_hiddim, self.G_hiddim) for i in range(self.G_layer-2)])
		self.G_convs.extend([pyg_nn.GCNConv(self.G_hiddim, self.G_hiddim)])
		self.G_bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(self.G_hiddim) for i in range(self.G_layer-1)])
		#
		self.Convs_1 = torch.nn.ModuleList([torch.nn.Linear(self.input_dim1+self.G_hiddim, self.layers_1[0] )])
		self.Convs_1.extend([torch.nn.Linear(self.layers_1[a], self.layers_1[a+1]) for a in range(len(self.layers_1)-1)])
		#
		self.Convs_2 = torch.nn.ModuleList([torch.nn.Linear(self.input_dim2+self.G_hiddim, self.layers_2[0] )])
		self.Convs_2.extend([torch.nn.Linear(self.layers_2[a], self.layers_2[a+1]) for a in range(len(self.layers_2)-1)])
		#
		self.SNPs = torch.nn.ModuleList([torch.nn.Linear(self.layers_1[-1]+self.layers_2[-1], self.layers_3[0] )])
		self.SNPs.extend([torch.nn.Linear(self.layers_3[a], self.layers_3[a+1]) for a in range(len(self.layers_3)-1)])
		self.SNPs.extend([torch.nn.Linear(self.layers_3[-1], self.out_dim)])
		#
		self.reset_parameters()
	#
	def reset_parameters(self): 
		for conv in self.G_convs :
			conv.reset_parameters()
		for bns in self.G_bns :
			bns.reset_parameters()
		for conv in self.Convs_1 :
			conv.reset_parameters()
		for conv in self.Convs_2:
			conv.reset_parameters()
		for conv in self.SNPs:
			conv.reset_parameters()
	#
	def forward(self, Drug1, Drug2, PPI, ADJ ):
		Bat_info = Drug1.shape[0]
		Node_info = PPI.shape[0]/Bat_info
		Num = [a for a in range(Bat_info)]
		Rep = np.repeat(Num, Node_info)
		batch_labels = torch.Tensor(Rep).long()
		if torch.cuda.is_available():
			batch_labels = batch_labels.cuda()
		#
		for G1 in range(len(self.G_convs)):
			if G1 == len(self.G_convs)-1 :
				PPI = self.G_convs[G1](x=PPI, edge_index=ADJ)
				PPI = F.dropout(PPI, p=self.inDrop, training=self.training)
				PPI = self.pool(PPI, batch_labels )
				PPI = self.tanh(PPI)
				G_out = PPI
			else : 
				PPI = self.G_convs[G1](x=PPI, edge_index=ADJ)
				PPI = self.G_bns[G1](PPI)
				PPI = F.elu(PPI)
		#
		input_drug1 = torch.concat( (Drug1, G_out), 1 ) # normalization 추가 해야할것 같은데 
		input_drug2 = torch.concat( (Drug2, G_out), 1 )
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
		X = torch.cat((input_drug1,input_drug2),1)
		for L3 in range(len(self.SNPs)):
			if L3 != len(self.SNPs)-1 :
				X = self.SNPs[L3](X)
				X = F.dropout(X, p=self.drop, training = self.training)
				X = F.relu(X)
			else : 
				X = self.SNPs[L3](X)
		return X
	


def weighted_mse_loss(input, target, weight):
		return (weight * (input - target) ** 2).mean()


def RAY_MY_train(config, checkpoint_dir=None):
	n_epochs = 1000
	criterion = weighted_mse_loss
	use_cuda = False
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
		config["G_layer"], 	2 , config["G_hiddim"],
		256, 256, 
		dsn1_layers, dsn2_layers, snp_layers, 1, 
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
		for batch_idx_t, (drug1, drug2, expAB, adj, adj_w, y) in enumerate(loaders['train']):
			# move to GPU
			if use_cuda:
				drug1, drug2, expAB, adj, adj_w, y  = drug1.cuda(), drug2.cuda(), expAB.cuda(), adj.cuda(), adj_w.cuda(), y.cuda()
			## find the loss and update the model parameters accordingly
			# clear the gradients of all optimized variables
			optimizer.zero_grad()
			output = MM_MODEL(drug1, drug2, expAB, adj)
			wc = torch.Tensor(batch_cut_weight[batch_idx_t]).view(-1,1)
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
		for batch_idx_v, (drug1, drug2, expAB, adj, adj_w, y) in enumerate(loaders['eval']):
			# move to GPU
			if use_cuda:
				drug1, drug2, expAB, adj, adj_w, y = drug1.cuda(), drug2.cuda(), expAB.cuda(), adj.cuda(), adj_w.cuda(), y.cuda()
			## update the average validation loss
			output = MM_MODEL(drug1, drug2, expAB, adj)
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
		done = datetime.now()
		time_spent = done-now
		#
		with tune.checkpoint_dir(step = epoch) as checkpoint_dir:
			path = os.path.join(checkpoint_dir, "checkpoint")
			torch.save((MM_MODEL.state_dict(), optimizer.state_dict()), path)
			torch.save(MM_MODEL.state_dict(), './model.pth')
		tune.report(TrainLoss= TRAIN_LOSS,  ValLoss=VAL_LOSS )
	#
	print("Finished Training")
	# plot_loss(train_loss_all ,valid_loss_all, '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/my_trial_1', 'MY1' )
	# return trained model
	# return model


def RAY_TEST_MODEL(best_trial):
	dsn1_layers = [best_trial.config["feat_size_0"], best_trial.config["feat_size_1"], best_trial.config["feat_size_2"] ]
	dsn2_layers = [best_trial.config["feat_size_0"], best_trial.config["feat_size_1"], best_trial.config["feat_size_2"] ] 
	snp_layers = [best_trial.config["feat_size_3"], best_trial.config["feat_size_4"]]
	inDrop = best_trial.config["dropout_1"] 
	Drop = best_trial.config["dropout_2"] 
	#	
	best_trained_model = MY_expGCN_parallel_model(
		best_trial.config["G_layer"], 2, best_trial.config["G_hiddim"],
		256, 256, 
		dsn1_layers, dsn2_layers, snp_layers, 1, 
		inDrop, Drop
	)
	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	best_trained_model.to(device)
	checkpoint_path = os.path.join(best_trial.checkpoint.value, "checkpoint")
	model_state, optimizer_state = torch.load(checkpoint_path)
	best_trained_model.load_state_dict(model_state)
	#
	T_test = ray.get(RAY_test)
	Test_loader = torch.utils.data.DataLoader(T_test, batch_size = best_trial.config["batch_size"], collate_fn = graph_collate_fn, shuffle =False, num_workers=best_trial.config['n_workers'])
	#
	test_loss = 0.0
	PRED_list = []
	Y_list = test_data['y'].squeeze().tolist()
	with torch.no_grad():
		best_trained_model.eval()
		for batch_idx_t, (drug1, drug2, expAB, adj, adj_w, y) in enumerate(Test_loader):
			drug1, drug2, expAB, adj, adj_w, y = drug1.cuda(), drug2.cuda(), expAB.cuda(), adj.cuda(), adj_w.cuda(), y.cuda()
			output = best_trained_model(drug1, drug2, expAB, adj)
			MSE = torch.nn.MSELoss()
			loss = MSE(output, y)
			test_loss = test_loss + loss.item()
			outputs = output.squeeze().tolist()
			PRED_list = PRED_list+outputs
	TEST_LOSS = test_loss/(batch_idx_t+1)
	result_pearson(PRED_list, Y_list)
	result_spearman(PRED_list, Y_list)
	print("Best model TEST loss: {}".format(TEST_LOSS))

	

def result_pearson(y, pred):
	pear = stats.pearsonr(y, pred)
	pear_value = pear[0]
	pear_p_val = pear[1]
	print("Pearson correlation is {} and related p_value is {}".format(pear_value, pear_p_val))

def result_spearman(y, pred):
	spear = stats.spearmanr(y, pred)
	spear_value = spear[0]
	spear_p_val = spear[1]
	print("Spearman correlation is {} and related p_value is {}".format(spear_value, spear_p_val))
	



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




def MAIN(num_samples= 10, max_num_epochs=1000, grace_period = 200, cpus_per_trial = 16, gpus_per_trial = 1):
	# OPTUNA CONFIGURE 
	CONFIG={
		'n_workers' : tune.choice([cpus_per_trial]),
		"epoch" : tune.choice([max_num_epochs]),
		"G_layer" : tune.choice([2, 3, 4]),
		"G_hiddim" : tune.choice([512, 256, 128, 64, 32]),
		"batch_size" : tune.choice([128, 64, 32]), # The number of batch sizes should be a power of 2 to take full advantage of the GPUs processing (근데 256 은 큰가봉가. 메모리 에러 남)
		"feat_size_0" : tune.choice([4096, 2048, 1024, 512, 256, 128, 64, 32]),
		"feat_size_1" : tune.choice([4096, 2048, 1024, 512, 256, 128, 64, 32]),
		"feat_size_2" : tune.choice([4096, 2048, 1024, 512, 256, 128, 64, 32]),
		"feat_size_3" : tune.choice([4096, 2048, 1024, 512, 256, 128, 64, 32]),
		"feat_size_4" : tune.choice([4096, 2048, 1024, 512, 256, 128, 64, 32]),
		"dropout_1" : tune.choice([0.01, 0.2, 0.5, 0.8]),
		"dropout_2" : tune.choice([0.01, 0.2, 0.5, 0.8]),
		"lr" : tune.choice([0.00001, 0.0001, 0.001]),
	}
	#
	reporter = CLIReporter(
			metric_columns=["TrainLoss", "ValLoss",  "training_iteration"])
	#
	optuna_search = OptunaSearch(metric="ValLoss", mode="min")
	#
	ASHA_scheduler = tune.schedulers.ASHAScheduler(
		time_attr='training_iteration', metric="ValLoss", mode="min", max_t= max_num_epochs, grace_period = grace_period)
	#
	#
	ANALYSIS = tune.run( # 끝내지 않음 
		tune.with_parameters(RAY_MY_train),  
		name = '22.06.15.PRJ01.TRIAL.2_1.re',
		num_samples=num_samples, 
		config=CONFIG, 
		resources_per_trial={'cpu': cpus_per_trial }, # 'gpu' : gpus_per_trial
		progress_reporter = reporter,
		search_alg = optuna_search,
		scheduler = ASHA_scheduler
	)
	best_trial = ANALYSIS.get_best_trial("ValLoss", "min", "last")
	print("Best trial config: {}".format(best_trial.config))
	print("Best trial final validation loss: {}".format(
		best_trial.last_result["ValLoss"]))
	#
	ALL_DF = ANALYSIS.trial_dataframes
	TMP_DF = ALL_DF[best_trial.logdir]
	plot_loss(list(TMP_DF.TrainLoss), list(TMP_DF.ValLoss), '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/my_trial_2/PRJ.01.2_1', 'MM_GCNexp_IDK')
	# 
	if ray.util.client.ray.is_connected():
		from ray.util.ml_utils.node import force_on_current_node
		remote_fn = force_on_current_node(ray.remote(test_best_model))
		ray.get(remote_fn.remote(best_trial))
	else:
		RAY_TEST_MODEL(best_trial)


# for test 
MAIN(6, 10, 2, 4, 0.5)

# for real 
MAIN(100, 1000, 100, 32, 0.5)
한시간 제한 제거해주기 



# ANALYSIS = tune.run( # 끝내지 않음 
# 	tune.with_parameters(RAY_MY_train),  
# 	name = '22.06.13.MM_trial_2_1',
# 	num_samples=10, 
# 	config=CONFIG, 
# 	resources_per_trial={'cpu': 5},
# 	progress_reporter = reporter,
# 	search_alg = optuna_search,
# 	scheduler = ASHA_scheduler
# 	)














### GPU server 버전은 다시 써보기 
### GPU server 버전은 다시 써보기 




def RAY_MY_train(config, checkpoint_dir=None):
	n_epochs = 1000
	criterion = weighted_mse_loss
	use_cuda = True
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
		config["G_layer"], 	2 , config["G_hiddim"],
		256, 256, 
		dsn1_layers, dsn2_layers, snp_layers, 1, 
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
		for batch_idx_t, (drug1, drug2, expAB, adj, adj_w, y) in enumerate(loaders['train']):
			# move to GPU
			if use_cuda:
				drug1, drug2, expAB, adj, adj_w, y  = drug1.cuda(), drug2.cuda(), expAB.cuda(), adj.cuda(), adj_w.cuda(), y.cuda()
			## find the loss and update the model parameters accordingly
			# clear the gradients of all optimized variables
			optimizer.zero_grad()
			output = MM_MODEL(drug1, drug2, expAB, adj)
			wc = torch.Tensor(batch_cut_weight[batch_idx_t]).view(-1,1)
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
		for batch_idx_v, (drug1, drug2, expAB, adj, adj_w, y) in enumerate(loaders['eval']):
			# move to GPU
			if use_cuda:
				drug1, drug2, expAB, adj, adj_w, y = drug1.cuda(), drug2.cuda(), expAB.cuda(), adj.cuda(), adj_w.cuda(), y.cuda()
			## update the average validation loss
			output = MM_MODEL(drug1, drug2, expAB, adj)
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
		done = datetime.now()
		time_spent = done-now
		#
		with tune.checkpoint_dir(step = epoch) as checkpoint_dir:
			path = os.path.join(checkpoint_dir, "checkpoint")
			torch.save((MM_MODEL.state_dict(), optimizer.state_dict()), path)
			torch.save(MM_MODEL.state_dict(), './model.pth')
		tune.report(TrainLoss= TRAIN_LOSS,  ValLoss=VAL_LOSS )
	#
	print("Finished Training")
	# plot_loss(train_loss_all ,valid_loss_all, '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/my_trial_1', 'MY1' )
	# return trained model
	# return model


def RAY_TEST_MODEL(best_trial):
	dsn1_layers = [best_trial.config["feat_size_0"], best_trial.config["feat_size_1"], best_trial.config["feat_size_2"] ]
	dsn2_layers = [best_trial.config["feat_size_0"], best_trial.config["feat_size_1"], best_trial.config["feat_size_2"] ] 
	snp_layers = [best_trial.config["feat_size_3"], best_trial.config["feat_size_4"]]
	inDrop = best_trial.config["dropout_1"] 
	Drop = best_trial.config["dropout_2"] 
	#	
	best_trained_model = MY_expGCN_parallel_model(
		best_trial.config["G_layer"], 2, best_trial.config["G_hiddim"],
		256, 256, 
		dsn1_layers, dsn2_layers, snp_layers, 1, 
		inDrop, Drop
	)
	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	best_trained_model.to(device)
	checkpoint_path = os.path.join(best_trial.checkpoint.value, "checkpoint")
	model_state, optimizer_state = torch.load(checkpoint_path)
	best_trained_model.load_state_dict(model_state)
	#
	T_test = ray.get(RAY_test)
	Test_loader = torch.utils.data.DataLoader(T_test, batch_size = best_trial.config["batch_size"], collate_fn = graph_collate_fn, shuffle =False, best_trial.config['n_workers'])
	#
	test_loss = 0.0
	PRED_list = []
	Y_list = test_data['y'].squeeze().tolist()
	with torch.no_grad():
		best_trained_model.eval()
		for batch_idx_t, (drug1, drug2, expAB, adj, adj_w, y) in enumerate(Test_loader):
			drug1, drug2, expAB, adj, adj_w, y = drug1.cuda(), drug2.cuda(), expAB.cuda(), adj.cuda(), adj_w.cuda(), y.cuda()
			output = best_trained_model(drug1, drug2, expAB, adj)
			MSE = torch.nn.MSELoss()
			loss = MSE(output, y)
			test_loss = test_loss + loss.item()
			outputs = output.squeeze().tolist()
			PRED_list = PRED_list+outputs
	TEST_LOSS = test_loss/(batch_idx_t+1)
	result_pearson(PRED_list, Y_list)
	result_spearman(PRED_list, Y_list)
	print("Best model TEST loss: {}".format(TEST_LOSS))

	

def result_pearson(y, pred):
	pear = stats.pearsonr(y, pred)
	pear_value = pear[0]
	pear_p_val = pear[1]
	print("Pearson correlation is {} and related p_value is {}".format(pear_value, pear_p_val))

def result_spearman(y, pred):
	spear = stats.spearmanr(y, pred)
	spear_value = spear[0]
	spear_p_val = spear[1]
	print("Spearman correlation is {} and related p_value is {}".format(spear_value, spear_p_val))
	



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




def MAIN(num_samples= 10, max_num_epochs=1000, grace_period = 200, cpus_per_trial = 16, gpus_per_trial = 1):
	# OPTUNA CONFIGURE 
	CONFIG={
		'n_workers' : tune.choice([cpus_per_trial]),
		"epoch" : tune.choice([max_num_epochs]),
		"G_layer" : tune.choice([2, 3, 4]),
		"G_hiddim" : tune.choice([512, 256, 128, 64, 32]),
		"batch_size" : tune.choice([128, 64, 32]), # The number of batch sizes should be a power of 2 to take full advantage of the GPUs processing (근데 256 은 큰가봉가. 메모리 에러 남)
		"feat_size_0" : tune.choice([4096, 2048, 1024, 512, 256, 128, 64, 32]),
		"feat_size_1" : tune.choice([4096, 2048, 1024, 512, 256, 128, 64, 32]),
		"feat_size_2" : tune.choice([4096, 2048, 1024, 512, 256, 128, 64, 32]),
		"feat_size_3" : tune.choice([4096, 2048, 1024, 512, 256, 128, 64, 32]),
		"feat_size_4" : tune.choice([4096, 2048, 1024, 512, 256, 128, 64, 32]),
		"dropout_1" : tune.choice([0.01, 0.2, 0.5, 0.8]),
		"dropout_2" : tune.choice([0.01, 0.2, 0.5, 0.8]),
		"lr" : tune.choice([0.00001, 0.0001, 0.001]),
	}
	#
	reporter = CLIReporter(
			metric_columns=["TrainLoss", "ValLoss",  "training_iteration"])
	#
	optuna_search = OptunaSearch(metric="ValLoss", mode="min")
	#
	ASHA_scheduler = tune.schedulers.ASHAScheduler(
		time_attr='training_iteration', metric="ValLoss", mode="min", max_t= max_num_epochs, grace_period = grace_period)
	#
	#
	ANALYSIS = tune.run( # 끝내지 않음 
		tune.with_parameters(RAY_MY_train),  
		name = '22.06.14.PRJ01.TRIAL2',
		num_samples=num_samples, 
		config=CONFIG, 
		resources_per_trial={'cpu': cpus_per_trial, 'gpu' : gpus_per_trial},
		progress_reporter = reporter,
		search_alg = optuna_search,
		scheduler = ASHA_scheduler
	)
	best_trial = ANALYSIS.get_best_trial("ValLoss", "min", "last")
	print("Best trial config: {}".format(best_trial.config))
	print("Best trial final validation loss: {}".format(
		best_trial.last_result["ValLoss"]))
	#
	ALL_DF = ANALYSIS.trial_dataframes
	TMP_DF = ALL_DF[best_trial.logdir]
	plot_loss(list(TMP_DF.TrainLoss), list(TMP_DF.ValLoss), '/home01/k006a01/PRJ.01/TRIAL_2', 'MM_GCNexp_IDK')
	# 
	if ray.util.client.ray.is_connected():
		from ray.util.ml_utils.node import force_on_current_node
		remote_fn = force_on_current_node(ray.remote(test_best_model))
		ray.get(remote_fn.remote(best_trial))
	else:
		RAY_TEST_MODEL(best_trial)


# for test 
MAIN(6, 10, 2, 16, 0.5)

# for real 
MAIN(100, 1000, 100, 32, 0.5)
한시간 제한 제거해주기 





# 모델 확인하기 



from ray.tune import ExperimentAnalysis
#anal_df = ExperimentAnalysis("~/ray_results/22.06.14.PRJ01.TRIAL2")
ANA_DF = anal_df.dataframe()
ANA_ALL_DF = anal_df.trial_dataframes


#ANA_DF.to_csv('/home01/k006a01/PRJ.01/TRIAL_2/RAY_ANA_DF.P01.2_1.csv')
#import pickle
#with open("/home01/k006a01/PRJ.01/TRIAL_2/RAY_ANA_ALL_DF.P01.2_1.pickle", "wb") as fp:
#    pickle.dump(ANA_ALL_DF,fp) 

PRJ_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_GPU/PJ01.TRIAL.2_1/'
ANA_DF = pd.read_csv('/st06/jiyeonH/11.TOX/MY_TRIAL_GPU/PJ01.TRIAL.2_1/RAY_ANA_DF.P01.2_1.csv')
with open('/st06/jiyeonH/11.TOX/MY_TRIAL_GPU/PJ01.TRIAL.2_1/RAY_ANA_ALL_DF.P01.2_1.pickle', 'rb') as f:
	ANA_ALL_DF = pickle.load(f)


DF_KEY = list(ANA_DF.sort_values('ValLoss')['logdir'])[0]
#  /home01/k006a01/ray_results/22.06.14.PRJ01.TRIAL2/RAY_MY_train_4db301c6_54_G_hiddim=32,G_layer=2,batch_size=128,dropout_1=0.5000,dropout_2=0.0100,epoch=1000,feat_size_0=128,feat_si_2022-06-19_20-58-56

mini_df = ANA_ALL_DF[DF_KEY]
plot_loss(list(mini_df.TrainLoss), list(mini_df.ValLoss), 
'/st06/jiyeonH/11.TOX/MY_TRIAL_GPU/PJ01.TRIAL.2_1', 'TRIAL_2_1.BEST.loss' )




(1) 마지막 모델 확인 

TOPVAL_PATH = DF_KEY
my_config = ANA_DF[ANA_DF.logdir==DF_KEY]

G_layer = my_config["config/G_layer"].item()
G_hiddim = my_config["config/G_hiddim"].item()
dsn1_layers = [my_config['config/feat_size_0'].item(), my_config['config/feat_size_1'].item(), my_config['config/feat_size_2'].item()]
dsn2_layers = [my_config['config/feat_size_0'].item(), my_config['config/feat_size_1'].item(), my_config['config/feat_size_2'].item()] 
snp_layers = [my_config['config/feat_size_3'].item() , my_config['config/feat_size_4'].item()]
inDrop = my_config['config/dropout_1'].item()
Drop = my_config['config/dropout_2'].item()


best_model = MY_expGCN_parallel_model(
		G_layer, 2, G_hiddim,
		256, 256, 
		dsn1_layers, dsn2_layers, snp_layers, 1, 
		inDrop, Drop
		)

state_dict = torch.load(os.path.join(PRJ_PATH, "M1_model.pth"),map_location=torch.device('cpu'))
best_model.load_state_dict(state_dict)

T_test = ray.get(RAY_test)
Test_loader = torch.utils.data.DataLoader(T_test, collate_fn = graph_collate_fn, batch_size = my_config['config/batch_size'].item(), shuffle =False)


best_model.eval()
test_loss = 0.0
PRED_list = []
Y_list = test_data['y'].squeeze().tolist()
with torch.no_grad():
	best_model.eval()
	for batch_idx_t, (drug1, drug2, expAB, adj, adj_w, y) in enumerate(Test_loader):
		output = best_model(drug1, drug2, expAB, adj)
		MSE = torch.nn.MSELoss()
		loss = MSE(output, y)
		test_loss = test_loss + loss.item()
		outputs = output.squeeze().tolist()
		PRED_list = PRED_list+outputs


TEST_LOSS = test_loss/(batch_idx_t+1)
print("Best model TEST loss: {}".format(TEST_LOSS))
result_pearson(PRED_list, Y_list)
result_spearman(PRED_list, Y_list)






(2) 중간 체크포인트 확인 

cck_num =mini_df[mini_df.ValLoss==min(mini_df.ValLoss)].index.item()
checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
TOPVAL_PATH = DF_KEY + checkpoint
# /home01/k006a01/ray_results/22.06.14.PRJ01.TRIAL2/RAY_MY_train_4db301c6_54_G_hiddim=32,G_layer=2,batch_size=128,dropout_1=0.5000,dropout_2=0.0100,epoch=1000,feat_size_0=128,feat_si_2022-06-19_20-58-56/checkpoint_000216



state_dict = torch.load(os.path.join(PRJ_PATH, "M2_checkpoint"),map_location=torch.device('cpu'))
best_model.load_state_dict(state_dict[0])

best_model.eval()
test_loss = 0.0
PRED_list = []
Y_list = test_data['y'].squeeze().tolist()
with torch.no_grad():
	best_model.eval()
	for batch_idx_t, (drug1, drug2, expAB, adj, adj_w, y) in enumerate(Test_loader):
		output = best_model(drug1, drug2, expAB, adj)
		MSE = torch.nn.MSELoss()
		loss = MSE(output, y)
		test_loss = test_loss + loss.item()
		outputs = output.squeeze().tolist()
		PRED_list = PRED_list+outputs


TEST_LOSS = test_loss/(batch_idx_t+1)
print("Best model TEST loss: {}".format(TEST_LOSS))
result_pearson(PRED_list, Y_list)
result_spearman(PRED_list, Y_list)




# 최저를 찾으려면 
import numpy as np
TOT_min = np.Inf
TOT_key = ""
for key in ANA_ALL_DF.keys():
	trial_min = min(ANA_ALL_DF[key]['ValLoss'])
	if trial_min < TOT_min :
		TOT_min = trial_min
		TOT_key = key

# /home01/k006a01/ray_results/22.06.14.PRJ01.TRIAL2/RAY_MY_train_68d928fe_50_G_hiddim=32,G_layer=2,batch_size=128,dropout_1=0.5000,dropout_2=0.0100,epoch=1000,feat_size_0=128,feat_si_2022-06-19_13-10-25

mini_df = ANA_ALL_DF[TOT_key]
plot_loss(list(mini_df.TrainLoss), list(mini_df.ValLoss), 
'/st06/jiyeonH/11.TOX/MY_TRIAL_GPU/PJ01.TRIAL.2_1', 'TRIAL_2_1.MIN.loss' )


TOPVAL_PATH = TOT_key
my_config = ANA_DF[ANA_DF.logdir==TOT_key]

G_layer = my_config["config/G_layer"].item()
G_hiddim = my_config["config/G_hiddim"].item()
dsn1_layers = [my_config['config/feat_size_0'].item(), my_config['config/feat_size_1'].item(), my_config['config/feat_size_2'].item()]
dsn2_layers = [my_config['config/feat_size_0'].item(), my_config['config/feat_size_1'].item(), my_config['config/feat_size_2'].item()] 
snp_layers = [my_config['config/feat_size_3'].item() , my_config['config/feat_size_4'].item()]
inDrop = my_config['config/dropout_1'].item()
Drop = my_config['config/dropout_2'].item()

best_model = MY_expGCN_parallel_model(
		G_layer, 2, G_hiddim,
		256, 256, 
		dsn1_layers, dsn2_layers, snp_layers, 1, 
		inDrop, Drop
		)


cck_num =mini_df[mini_df.ValLoss==min(mini_df.ValLoss)].index.item()
checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
TOPVAL_PATH = TOT_key + checkpoint
# /home01/k006a01/ray_results/22.06.14.PRJ01.TRIAL2/RAY_MY_train_68d928fe_50_G_hiddim=32,G_layer=2,batch_size=128,dropout_1=0.5000,dropout_2=0.0100,epoch=1000,feat_size_0=128,feat_si_2022-06-19_13-10-25/checkpoint_000195


state_dict = torch.load(os.path.join(PRJ_PATH, "M4_checkpoint"),map_location=torch.device('cpu'))
best_model.load_state_dict(state_dict[0])

T_test = ray.get(RAY_test)
Test_loader = torch.utils.data.DataLoader(T_test, collate_fn = graph_collate_fn, batch_size = my_config['config/batch_size'].item(), shuffle =False)

best_model.eval()
test_loss = 0.0
PRED_list = []
Y_list = test_data['y'].squeeze().tolist()
with torch.no_grad():
	best_model.eval()
	for batch_idx_t, (drug1, drug2, expAB, adj, adj_w, y) in enumerate(Test_loader):
		output = best_model(drug1, drug2, expAB, adj)
		MSE = torch.nn.MSELoss()
		loss = MSE(output, y)
		test_loss = test_loss + loss.item()
		outputs = output.squeeze().tolist()
		PRED_list = PRED_list+outputs

TEST_LOSS = test_loss/(batch_idx_t+1)
print("Best model TEST loss: {}".format(TEST_LOSS))
result_pearson(PRED_list, Y_list)
result_spearman(PRED_list, Y_list)
































































# 테스트용
loaders = {
	'train' : torch.utils.data.DataLoader(T_train, batch_size = 100, collate_fn = graph_collate_fn, shuffle =False),
	'eval' : torch.utils.data.DataLoader(T_val, batch_size = 100),
	'test' : torch.utils.data.DataLoader(T_test, batch_size = 100),
	}


pyg_loaders = {
	'train' : torch_geometric.loader.DataLoader(T_train, batch_size = 100),
	'eval' : torch_geometric.loader.DataLoader(T_val, batch_size = 100),
	'test' : torch_geometric.loader.DataLoader(T_test, batch_size = 100),
}


MY_MODEL = MY_expGCN_parallel_model(3, 2, 200, 256, 256, [2048,100,10], [2048,100,10], [20,10], 1, 0.3, 0.5)

for batch_idx_t, (drug1, drug2, expAB, adj, adj_w, y) in enumerate(loaders['train']):
	output = MY_MODEL(drug1, drug2, expAB, adj)
	wc = torch.Tensor(loss_weight2[batch_idx_t]).view(-1,1)
	loss = weighted_mse_loss(output, y, wc )


for batch_idx_t, (drug1, drug2, expAB, adj, adj_w, y) in enumerate(loaders['train']):
	batch_idx_t

