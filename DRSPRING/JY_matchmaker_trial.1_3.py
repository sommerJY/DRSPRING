
prj 1 : 1-3


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

import tensorflow as tf
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

# original exp 같이 가져오기 
MM_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/'
MM_CEL_DATA = pd.read_csv(MM_PATH+'E-MTAB-3610.sdrf.txt', sep = '\t')
RMA_DATA_GDSC = pd.read_csv(MM_PATH+'Cell_line_RMA_proc_basalExp.txt', sep = '\t')
RMA_COL_DATA_GDSC = pd.read_csv(MM_PATH+'cell_lines_details.csv', sep = '\t')
CELLO_TABLE = pd.read_csv(MM_PATH+'cellosaurus.txt', skiprows = 54, na_values='NaN', prefix='V', header = None, sep = '   ' )

CELLO_TABLE_tmp = pd.DataFrame({'NUM':[],'ID':[], 'AC':[], 'AS':[], 'SY':[]})


NUM=0
tmp = pd.DataFrame({'NUM':[],'ID':[], 'AC':[], 'AS':[], 'SY':[]})
for c_i in range(CELLO_TABLE.shape[0]):
	col1 = CELLO_TABLE.loc[c_i]['V0']
	col2 = CELLO_TABLE.loc[c_i]['V1']
	if col1 == 'ID':
		tmp['NUM']=[NUM]
		tmp['ID'] = [col2]
		NUM+=1
	if col1 == 'AC':
		tmp['AC'] = [col2]
	if col1 == 'AS':
		tmp['AS'] = [col2]
	if col1 == 'SY':
		tmp['SY'] = [col2]        
	if col1 == '//' :
		CELLO_TABLE_tmp = pd.concat([CELLO_TABLE_tmp, tmp], ignore_index = True)
		tmp = pd.DataFrame({'NUM':[],'ID':[], 'AC':[], 'AS':[], 'SY':[]})

ALT =[a.split('; ') if type(a)==str else "" for a in list(CELLO_TABLE_tmp.SY)]
CELLO_TABLE_tmp['ALT'] = ALT

CELLO_TABLE_tmp.to_csv(MM_PATH+'CELLO_TABLE.csv', sep = '\t')
CELLO_TABLE_tmp=pd.read_csv(MM_PATH+'CELLO_TABLE.csv', sep = '\t')


# 일단 원래 E-MTAB-3610 랑 매칭 
GDSC_re = RMA_DATA.T.iloc[2:, :]
GDSC_re.columns = list(RMA_DATA.GENE_SYMBOLS)
GDSC_re['COSMIC'] = [int(a.split('.')[1]) for a in GDSC_re.index]

RMA_COL_DATA_GDSC = RMA_COL_DATA_GDSC.iloc[:1001, :]
RMA_COL_DATA_GDSC = RMA_COL_DATA_GDSC[['Sample Name', 'COSMIC identifier', 'GDSC Tissue descriptor 1', 'GDSC Tissue descriptor 2']]

GDSC_MERGE = pd.merge(GDSC_re, RMA_COL_DATA_GDSC, left_on = 'COSMIC', right_on = 'COSMIC identifier', how = 'left')

trial_MM_DATA = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/DrugCombinationData.tsv'
trial_drug1 = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/data/drug1_chem.csv'
trial_drug2 = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/data/drug2_chem.csv'
trial_CEL_gex = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/data/cell_line_gex.csv'

mm_comb_data = pd.read_csv(trial_MM_DATA, sep="\t")
mm_cell_line = pd.read_csv(trial_CEL_gex,header=None)
mm_chem1 = pd.read_csv(trial_drug1,header=None)
mm_chem2 = pd.read_csv(trial_drug2,header=None)
mm_synergies = np.array(comb_data["synergy_loewe"])

mm_cell_exp = pd.concat( [mm_comb_data['cell_line_name'] , mm_cell_line], axis= 1 )

LINCS_CELLO = pd.merge(A_B_C_S_SET, CELLO_TABLE_tmp, left_on = 'DrugCombCello', right_on = 'AC', how = 'left')
# LINCS 에 에 있는 cell line 들은 그래도 CELLO table 에 다 매칭이 됨. 
# 그러면 cello table 에 있는 alt 들에 MM cell 들이 다 맵핑 되는지만 보면 됨 

mm_cell_names = list(set(mm_cell_exp['cell_line_name']))

check_alt = []
for ind in range(LINCS_CELLO.shape[0]):
	ALT = LINCS_CELLO.loc[ind]['ALT']
	this_alt = []
	for alt in ALT :
		if alt in mm_cell_names :
			this_alt.append(alt)
	if len(this_alt) > 1 :  # 2 개 맞붙는 애들은 없는듯 
		print(ind)
		this_alt = ['']
	elif len(this_alt) == 0 :
		this_alt = ['']
	check_alt.append(this_alt)

check_alt_re = sum(check_alt,[])

LINCS_CELLO['ALT_pick']=check_alt_re
# 매칭 안되는거 10개 
MSSING = LINCS_CELLO[LINCS_CELLO['ALT_pick']=='']


CELLO_ALL_ALT = [a for a in CELLO_TABLE_tmp['ALT'] if type(a)==list]
CELLO_ALL_ALT = list(set(sum(CELLO_ALL_ALT, [])))

아... 실패.... 고민.....
그냥 이어붙이기 먼저 하는게 1-2  






# CCLE data 가져와서 보기 
CCLE_PATH = '/home/jiyeon/Dropbox/Lab_computer/09.SE_SS/01.DBs/CCLE/'
CCLE_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/'
CCLE_exp = pd.read_csv(CCLE_PATH+'CCLE_expression.csv')
CCLE_info = pd.read_csv(CCLE_PATH+'sample_info.csv')

SET_MATCH = ONLY_CELLO_CCLE[ONLY_CELLO_CCLE.cellosaurus_accession.isin(A_B_C_S_SET.DrugCombCello)]

ABC_CELL = list(set(A_B_C_S_SET.DrugCombCello))
ABC_CCLE = CCLE_info[CCLE_info.RRID.isin(ABC_CELL)][['DepMap_ID','RRID']]

CCLE_COLs = list(CCLE_exp.columns)[1:] # gene : entrez 

CCLE_COLs_DF = pd.DataFrame({'COL':[],'NAME':[], 'ENT':[]})
for CO in CCLE_COLs : 
	NAME = CO.split(' ')[0]
	ENT = int(CO.split(' ')[1].split('(')[1].split(')')[0])
	tmp = pd.DataFrame({'COL':[CO],'NAME':[NAME], 'ENT':[ENT]})
	CCLE_COLs_DF = pd.concat([CCLE_COLs_DF, tmp], ignore_index = True)


CCLE_filter = CCLE_COLs_DF[CCLE_COLs_DF.ENT.isin(BETA_ORDER)]
CCLE_filter = CCLE_filter.reset_index(drop=True)
CCLE_ORDER_pre =[list(CCLE_filter.ENT).index(a) for a in BETA_ORDER]
CCLE_ORDER_DF = CCLE_filter.iloc[CCLE_ORDER_pre] # 어차피 ref 다르고 같은 애들이라 괜춘 
CCLE_COL_ORDER = list(CCLE_ORDER_DF.COL)

CCLE_exp.columns=['DepMap_ID'] + list(CCLE_exp.columns)[1:]
CCLE_exp2 = pd.merge(ABC_CCLE, CCLE_exp, on ='DepMap_ID', how = 'left' )

CCLE_exp_re = CCLE_exp2[CCLE_COL_ORDER]
CCLE_exp_re.index = list(CCLE_exp2.RRID)


# CCLE 로 ABC 다시 filtering 해서 vector 잇기에 넣는방법 고안하기(근데 어떻게 붙일지가 좀 고민)
A_B_C_S_SET2 = A_B_C_S_SET[A_B_C_S_SET.DrugCombCello.isin(list(CCLE_exp2.RRID))]




def get_B_cell_data(cello_id):
	basal_exp = CCLE_exp_re.loc[cello_id]
	return list(basal_exp)


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



MY_chem_A = torch.empty(size=(A_B_C_S_SET2.shape[0], 256))
MY_chem_B = torch.empty(size=(A_B_C_S_SET2.shape[0], 256))
MY_exp_0 = torch.empty(size=(A_B_C_S_SET2.shape[0], 978))
MY_exp_A = torch.empty(size=(A_B_C_S_SET2.shape[0], 978))
MY_exp_B = torch.empty(size=(A_B_C_S_SET2.shape[0], 978))
MY_exp_AB = torch.empty(size=(A_B_C_S_SET2.shape[0], 978, 2))
MY_syn =  torch.empty(size=(A_B_C_S_SET2.shape[0],1))



for IND in range(A_B_C_S_SET2.shape[0]):
	DrugA_SIG = A_B_C_S_SET2.iloc[IND,]['BETA_sig_id_x']
	DrugB_SIG = A_B_C_S_SET2.iloc[IND,]['BETA_sig_id_y']
	Cell = A_B_C_S_SET2.iloc[IND,]['DrugCombCello']
	print(Cell)
	#
	bitsize = 256
	DrugA_FP = [int(a) for a in get_CHEMI_data(DrugA_SIG, bitsize)]
	DrugB_FP = [int(a) for a in get_CHEMI_data(DrugB_SIG, bitsize)]
	#
	EXP_0 = get_B_cell_data(Cell)
	#
	EXP_A, EXP_B, EXP_AB, LINCS = get_LINCS_data(DrugA_SIG, DrugB_SIG)
	#
	AB_SYN = get_synergy_data(DrugA_SIG, DrugB_SIG, Cell)
	#
	MY_chem_A[IND] = torch.Tensor(DrugA_FP)
	MY_chem_B[IND] = torch.Tensor(DrugB_FP)
	MY_exp_0[IND] = torch.Tensor(EXP_0)
	MY_exp_A[IND] = torch.Tensor(EXP_A.iloc[:,1])
	MY_exp_B[IND] = torch.Tensor(EXP_B.iloc[:,1])
	MY_exp_AB[IND] = torch.Tensor(EXP_AB).unsqueeze(0)
	MY_syn[IND] = torch.Tensor([AB_SYN])



WORK_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/my_trial_1_3/'
torch.save(MY_chem_A, WORK_PATH+'0622.MY_chem_A.pt')
torch.save(MY_chem_B, WORK_PATH+'0622.MY_chem_B.pt')
torch.save(MY_exp_0, WORK_PATH+'0622.MY_exp_0.pt')
torch.save(MY_exp_A, WORK_PATH+'0622.MY_exp_A.pt')
torch.save(MY_exp_B, WORK_PATH+'0622.MY_exp_B.pt')
torch.save(MY_exp_AB, WORK_PATH+'0622.MY_exp_AB.pt')
torch.save(MY_syn, WORK_PATH+'0622.MY_syn.pt')


MY_chem_A_tch = torch.tensor(np.array(MY_chem_A))
MY_chem_B_tch = torch.tensor(np.array(MY_chem_B))
MY_exp_0_tch = torch.tensor(np.array(MY_exp_0))
MY_exp_A_tch = torch.tensor(np.array(MY_exp_A))
MY_exp_B_tch = torch.tensor(np.array(MY_exp_B))
MY_exp_AB_tch = torch.tensor(np.array(MY_exp_AB))
MY_syn_tch = torch.tensor(np.array(MY_syn))








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



def prepare_data(MY_chem1_tch, MY_chem2_tch, MY_exp_0_tch, MY_exp_A_tch, MY_exp_B_tch, MY_syn_tch, norm ) :
	chem_A_train, chem_A_tv, chem_B_train, chem_B_tv, exp_0_train, exp_0_tv, exp_A_train, exp_A_tv, exp_B_train, exp_B_tv, syn_train, syn_tv  = sklearn.model_selection.train_test_split(
		MY_chem_A_tch, MY_chem_B_tch, MY_exp_0_tch, MY_exp_A_tch, MY_exp_B_tch, MY_syn_tch, 
		test_size=MY_chem_A_tch.shape[0]-6000, random_state=42 )
	chem_A_val, chem_A_test, chem_B_val, chem_B_test, exp_0_val, exp_0_test, exp_A_val, exp_A_test, exp_B_val, exp_B_test, syn_val, syn_test = sklearn.model_selection.train_test_split(
		chem_A_tv, chem_B_tv, exp_0_tv, exp_A_tv, exp_B_tv, syn_tv, 
		test_size=MY_chem_A_tch.shape[0]-8000, random_state=42 )
	#
	train_data = {}
	val_data = {}
	test_data = {}
	#
	train1 = np.concatenate((chem_A_train, chem_B_train),axis=0) 
	train_data['drug1'], mean1, std1, mean2, std2, feat_filt = normalize(train1, norm=norm)
	val_data['drug1'], mmean1, sstd1, mmean2, sstd2, feat_filtt = normalize(chem_A_val, mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
	test_data['drug1'], mean1, std1, mean2, std2, feat_filt = normalize(chem_A_test,mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
	#
	train2 = np.concatenate((chem_B_train, chem_A_train),axis=0)
	train_data['drug2'], mean1, std1, mean2, std2, feat_filt = normalize(train2, norm=norm)
	val_data['drug2'], mmean1, sstd1, mmean2, sstd2, feat_filtt = normalize(chem_B_val, mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
	test_data['drug2'], mean1, std1, mean2, std2, feat_filt = normalize(chem_B_test, mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
	#
	train3 = np.concatenate((exp_A_train, exp_B_train), axis=0) 
	train_data['exp1'], mean1, std1, mean2, std2, feat_filt = normalize(train3, norm=norm)
	val_data['exp1'], mmean1, sstd1, mmean2, sstd2, feat_filtt = normalize(exp_A_val, mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
	test_data['exp1'], mean1, std1, mean2, std2, feat_filt = normalize(exp_A_test, mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
	#
	train4 = np.concatenate((exp_B_train, exp_A_train), axis=0)
	train_data['exp2'], mean1, std1, mean2, std2, feat_filt = normalize(train4, norm=norm)
	val_data['exp2'], mmean1, sstd1, mmean2, sstd2, feat_filtt = normalize(exp_B_val, mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
	test_data['exp2'], mean1, std1, mean2, std2, feat_filt = normalize(exp_B_test, mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
	#
	train5 = np.concatenate((exp_0_train, exp_0_train), axis=0)
	train_data['exp0'], mean1, std1, mean2, std2, feat_filt = normalize(train5, norm=norm)
	val_data['exp0'], mmean1, sstd1, mmean2, sstd2, feat_filtt = normalize(exp_0_val, mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
	test_data['exp0'], mean1, std1, mean2, std2, feat_filt = normalize(exp_0_test, mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
	#
	train_data['drug1'] = np.concatenate((train_data['drug1'],train_data['exp1'], train_data['exp0']),axis=1)
	train_data['drug2'] = np.concatenate((train_data['drug2'],train_data['exp2'], train_data['exp0']),axis=1)
	#
	val_data['drug1'] = np.concatenate((val_data['drug1'],val_data['exp1'],val_data['exp0']),axis=1)
	val_data['drug2'] = np.concatenate((val_data['drug2'],val_data['exp2'],val_data['exp0']),axis=1)
	#
	test_data['drug1'] = np.concatenate((test_data['drug1'],test_data['exp1'],test_data['exp0']),axis=1)
	test_data['drug2'] = np.concatenate((test_data['drug2'],test_data['exp2'],test_data['exp0']),axis=1)
	#		
	train_data['y'] = np.concatenate((syn_train , syn_train),axis=0)
	val_data['y'] = syn_val
	test_data['y'] = syn_test
	print(train_data['drug1'].shape)
	print(val_data['drug1'].shape)
	print(test_data['drug1'].shape)
	return train_data, val_data, test_data


class MY_parallel_model(torch.nn.Module):
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
		self.Convs_1 = torch.nn.ModuleList([torch.nn.Linear(self.input_dim1, self.layers_1[0] )])
		self.Convs_1.extend([torch.nn.Linear(self.layers_1[a], self.layers_1[a+1]) for a in range(len(self.layers_1)-1)])
		#
		self.Convs_2 = torch.nn.ModuleList([torch.nn.Linear(self.input_dim2, self.layers_2[0] )])
		self.Convs_2.extend([torch.nn.Linear(self.layers_2[a], self.layers_2[a+1]) for a in range(len(self.layers_2)-1)])
		#
		self.SNPs = torch.nn.ModuleList([torch.nn.Linear(self.layers_1[-1]+self.layers_2[-1], self.layers_3[0] )])
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


def weighted_mse_loss(input, target, weight):
		return (weight * (input - target) ** 2).mean()


class MM_DATASET(Dataset): 
	def __init__(self, drug1, drug2, y ):
			self.drug1 = torch.Tensor(drug1) # input 1
			self.drug2 = torch.Tensor(drug2) # input 2
			self.y = y
		#
	def __len__(self): 
			return len(self.drug1)
		#
	def __getitem__(self, index): 
			return self.drug1[index], self.drug2[index], self.y[index]

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





norm = 'tanh_norm'
train_data, val_data, test_data = prepare_data(
	MY_chem_A_tch, MY_chem_B_tch, MY_exp_0_tch, MY_exp_A_tch, MY_exp_B_tch, MY_syn_tch, norm)
# 6000 vs 2000 vs 1230


criterion = weighted_mse_loss
use_cuda = False

# weight
ys = train_data['y'].squeeze().tolist()
min_s = np.amin(ys)
loss_weight = np.log(train_data['y'] - min_s + np.e)


T_train = MM_DATASET(train_data['drug1'],train_data['drug2'],train_data['y'])
T_val = MM_DATASET(val_data['drug1'],val_data['drug2'],val_data['y'])
T_test = MM_DATASET(test_data['drug1'],test_data['drug2'],test_data['y'])
T_loss_weight = torch.Tensor(loss_weight)

RAY_train = ray.put(T_train)
RAY_val = ray.put(T_val)
RAY_test = ray.put(T_test)
RAY_loss_weight = ray.put(loss_weight)





def RAY_MY_train(config, checkpoint_dir=None):
	n_epochs = config["epoch"]
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
		'train' : torch.utils.data.DataLoader(T_train, batch_size = config["batch_size"], shuffle =False, num_workers=config['n_workers']),
		'eval' : torch.utils.data.DataLoader(T_val, batch_size = config["batch_size"],  shuffle =False, num_workers=config['n_workers']),
		'test' : torch.utils.data.DataLoader(T_test, batch_size = config["batch_size"], shuffle =False, num_workers=config['n_workers']),
	}
	#
	dsn1_layers = [config["feat_size_0"], config["feat_size_1"] , config["feat_size_2"] ]
	dsn2_layers = [config["feat_size_0"], config["feat_size_1"] , config["feat_size_2"] ] 
	snp_layers = [config["feat_size_3"] ,config["feat_size_4"] ,]
	inDrop = config["dropout_1"] 
	Drop = config["dropout_2"] 
	#
	# 
	MM_MODEL = MY_parallel_model(
		2212, 2212, 
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
		for batch_idx_t, (drug1, drug2, y) in enumerate(loaders['train']):
			# move to GPU
			if use_cuda:
				drug1, drug2, y  = drug1.cuda(), drug2.cuda(), y.cuda()
			## find the loss and update the model parameters accordingly
			# clear the gradients of all optimized variables
			optimizer.zero_grad()
			output = MM_MODEL(drug1, drug2)
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
		for batch_idx_v, (drug1, drug2, y) in enumerate(loaders['eval']):
			# move to GPU
			if use_cuda:
				drug1, drug2, y = drug1.cuda(), drug2.cuda(), y.cuda()
			## update the average validation loss
			output = MM_MODEL(drug1, drug2)
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
	


def RAY_TEST_MODEL(best_trial):
	use_cuda = False
	dsn1_layers = [best_trial.config["feat_size_0"], best_trial.config["feat_size_1"], best_trial.config["feat_size_2"] ]
	dsn2_layers = [best_trial.config["feat_size_0"], best_trial.config["feat_size_1"], best_trial.config["feat_size_2"] ] 
	snp_layers = [best_trial.config["feat_size_3"], best_trial.config["feat_size_4"]]
	inDrop = best_trial.config["dropout_1"] 
	Drop = best_trial.config["dropout_2"] 
	#	
	best_trained_model = MY_parallel_model(
		2212, 2212, 
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
	Test_loader = torch.utils.data.DataLoader(T_test, batch_size = best_trial.config["batch_size"], shuffle =False, num_workers=best_trial.config['n_workers'])
	#
	test_loss = 0.0
	PRED_list = []
	Y_list = test_data['y'].squeeze().tolist()
	with torch.no_grad():
		best_trained_model.eval()
		for batch_idx_t, (drug1, drug2, y) in enumerate(Test_loader):
			if torch.cuda.is_available():
				drug1, drug2, y = drug1.cuda(), drug2.cuda(), y.cuda()
			output = best_trained_model(drug1, drug2)
			MSE = torch.nn.MSELoss()
			loss = MSE(output, y)
			test_loss = test_loss + loss.item()
			outputs = output.squeeze().tolist()
			PRED_list = PRED_list+outputs
	TEST_LOSS = test_loss/(batch_idx_t+1)
	result_pearson(PRED_list, Y_list)
	result_spearman(PRED_list, Y_list)
	print("Best model TEST loss: {}".format(TEST_LOSS))

	




def MAIN(num_samples= 10, max_num_epochs=1000, grace_period = 200, cpus_per_trial = 16, gpus_per_trial = 1):
	# OPTUNA CONFIGURE 
	CONFIG={
		'n_workers' : tune.choice([cpus_per_trial]),
		"epoch" : tune.choice([max_num_epochs]),
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
		name = '22.06.22.PRJ01.TRIAL.1_3',
		num_samples=num_samples, 
		config=CONFIG, 
		resources_per_trial={'cpu': cpus_per_trial}, # , 'gpu' : gpus_per_trial
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
	plot_loss(list(TMP_DF.TrainLoss), list(TMP_DF.ValLoss), '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/my_trial_1_3/', 'MM_GCNexp_IDK')
	# 
	if ray.util.client.ray.is_connected():
		from ray.util.ml_utils.node import force_on_current_node
		remote_fn = force_on_current_node(ray.remote(test_best_model))
		ray.get(remote_fn.remote(best_trial))
	else:
		RAY_TEST_MODEL(best_trial)



# num_samples= 10, max_num_epochs=1000, grace_period = 200, cpus_per_trial = 16, gpus_per_trial = 1
MAIN(6, 10, 2, 4, 1)
MAIN(100, 1000, 200, 8, 1)















# 직접진행 경우 -I


num_samples= 100
max_num_epochs=1000
grace_period = 150 
cpus_per_trial = 8


CONFIG={
		'n_workers' : tune.choice([cpus_per_trial]),
		"epoch" : tune.choice([max_num_epochs]),
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
	name = '22.06.22.PRJ01.TRIAL.1_3',
	num_samples=num_samples, 
	config=CONFIG, 
	resources_per_trial={'cpu': cpus_per_trial}, # , 'gpu' : gpus_per_trial
	progress_reporter = reporter,
	search_alg = optuna_search,
	scheduler = ASHA_scheduler
)


best_trial = ANALYSIS.get_best_trial("ValLoss", "min", "last")
print("Best trial config: {}".format(best_trial.config))
print("Best trial final validation loss: {}".format(
	best_trial.last_result["ValLoss"]))
#
from ray.tune import ExperimentAnalysis
#anal_df = ExperimentAnalysis("/home/jiyeonH/ray_results/22.06.15.PRJ01.TRIAL.1_2_re")
ANA_DF = ANALYSIS.dataframe()
ANA_ALL_DF = ANALYSIS.trial_dataframes

DF_KEY = list(ANA_DF.sort_values('ValLoss')['logdir'])[0]
# /home/jiyeonH/ray_results/22.06.15.PRJ01.TRIAL.1_2_re/RAY_MY_train_9c8cd2d4_99_G_hiddim=128,G_layer=3,batch_size=32,dropout_1=0.2,dropout_2=0.01,epoch=1000,feat_size_0=64,feat_size_1=4_2022-06-21_19-52-08

mini_df = ANA_ALL_DF[DF_KEY]
plot_loss(list(mini_df.TrainLoss), list(mini_df.ValLoss), 
'/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/my_trial_1_3', 
'TRIAL_1_3.BEST' )




(1) 마지막 모델 확인 

TOPVAL_PATH = DF_KEY
my_config = ANA_DF[ANA_DF.logdir==DF_KEY]

dsn1_layers = [my_config['config/feat_size_0'].item(), my_config['config/feat_size_1'].item(), my_config['config/feat_size_2'].item()]
dsn2_layers = [my_config['config/feat_size_0'].item(), my_config['config/feat_size_1'].item(), my_config['config/feat_size_2'].item()] 
snp_layers = [my_config['config/feat_size_3'].item() , my_config['config/feat_size_4'].item()]
inDrop = my_config['config/dropout_1'].item()
Drop = my_config['config/dropout_2'].item()

best_model = MY_parallel_model(
		2212,2212, dsn1_layers, dsn2_layers, snp_layers, 1, inDrop, Drop
		)

state_dict = torch.load(os.path.join(TOPVAL_PATH, "model.pth"))
best_model.load_state_dict(state_dict)

T_test = ray.get(RAY_test)
Test_loader = torch.utils.data.DataLoader(T_test, batch_size = my_config['config/batch_size'].item(), shuffle =False)


best_model.eval()
test_loss = 0.0
PRED_list = []
Y_list = test_data['y'].squeeze().tolist()
with torch.no_grad():
	best_model.eval()
	for batch_idx_t, (drug1, drug2, y) in enumerate(Test_loader):
		if torch.cuda.is_available():
			drug1, drug2, y = drug1.cuda(), drug2.cuda(), y.cuda()
		output = best_model(drug1, drug2)
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

state_dict = torch.load(os.path.join(TOPVAL_PATH, "checkpoint"))
best_model.load_state_dict(state_dict[0])

best_model.eval()
test_loss = 0.0
PRED_list = []
Y_list = test_data['y'].squeeze().tolist()
with torch.no_grad():
	best_model.eval()
	for batch_idx_t, (drug1, drug2, y) in enumerate(Test_loader):
		if torch.cuda.is_available():
			drug1, drug2, y = drug1.cuda(), drug2.cuda(), y.cuda()
		output = best_model(drug1, drug2)
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

TOT_min = np.Inf
TOT_key = ""
for key in ANA_ALL_DF.keys():
	trial_min = min(ANA_ALL_DF[key]['ValLoss'])
	if trial_min < TOT_min :
		TOT_min = trial_min
		TOT_key = key



mini_df = ANA_ALL_DF[TOT_key]
plot_loss(list(mini_df.TrainLoss), list(mini_df.ValLoss), 
'/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/my_trial_1_3', 
'TRIAL_1_3.TOT' )


TOPVAL_PATH = TOT_key
my_config = ANA_DF[ANA_DF.logdir==TOT_key]

dsn1_layers = [my_config['config/feat_size_0'].item(), my_config['config/feat_size_1'].item(), my_config['config/feat_size_2'].item()]
dsn2_layers = [my_config['config/feat_size_0'].item(), my_config['config/feat_size_1'].item(), my_config['config/feat_size_2'].item()] 
snp_layers = [my_config['config/feat_size_3'].item() , my_config['config/feat_size_4'].item()]
inDrop = my_config['config/dropout_1'].item()
Drop = my_config['config/dropout_2'].item()

best_model = MY_parallel_model(
		2212,2212, dsn1_layers, dsn2_layers, snp_layers, 1, inDrop, Drop
		)


cck_num =mini_df[mini_df.ValLoss==min(mini_df.ValLoss)].index.item()
checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
TOPVAL_PATH = TOT_key + checkpoint

state_dict = torch.load(os.path.join(TOPVAL_PATH, "checkpoint"))
best_model.load_state_dict(state_dict[0])

T_test = ray.get(RAY_test)
Test_loader = torch.utils.data.DataLoader(T_test, batch_size = my_config['config/batch_size'].item(), shuffle =False)

best_model.eval()
test_loss = 0.0
PRED_list = []
Y_list = test_data['y'].squeeze().tolist()
with torch.no_grad():
	best_model.eval()
	for batch_idx_t, (drug1, drug2, y) in enumerate(Test_loader):
		if torch.cuda.is_available():
			drug1, drug2, y = drug1.cuda(), drug2.cuda(), y.cuda()
		output = best_model(drug1, drug2)
		MSE = torch.nn.MSELoss()
		loss = MSE(output, y)
		test_loss = test_loss + loss.item()
		outputs = output.squeeze().tolist()
		PRED_list = PRED_list+outputs

TEST_LOSS = test_loss/(batch_idx_t+1)
print("Best model TEST loss: {}".format(TEST_LOSS))
result_pearson(PRED_list, Y_list)
result_spearman(PRED_list, Y_list)







##############################################################
##############################################################
##############################################################
##############################################################






























































































def RAY_MY_train(config, checkpoint_dir=None):
	# start_epochs = 1
	n_epochs = 200
	# valid_loss_min_input = np.Inf
	criterion = weighted_mse_loss
	use_cuda = False
	# checkpoint_path = Check_PATH
	# best_model_path = Best_PATH
	# batch_size = 100
	#
	#
	#
	T_train = ray.get(RAY_train)
	T_val = ray.get(RAY_val)
	T_test = ray.get(RAY_test)
	T_loss_weight = ray.get(RAY_loss_weight)
	batch_cut_weight = torch.Tensor([T_loss_weight[i:i+config["batch_size"]] for i in range(0,len(T_loss_weight), config["batch_size"])])
	# batch cut 수정해야함 딱 안떨어지는 batch 있을수도 있음 
	#
	loaders = {
		'train' : torch.utils.data.DataLoader(T_train, batch_size = config["batch_size"]),
		'eval' : torch.utils.data.DataLoader(T_val, batch_size = config["batch_size"]),
		'test' : torch.utils.data.DataLoader(T_test, batch_size = config["batch_size"]),
	}
	#
	#
	dsn1_layers = [2048, config["feat_size_1"] , config["feat_size_2"] ]
	dsn2_layers = [2048, config["feat_size_1"] , config["feat_size_2"] ] 
	snp_layers = [config["feat_size_3"] ,config["feat_size_4"] ,]
	inDrop = config["dropout_1"] 
	Drop = config["dropout_2"] 
	#
	#
	MM_MODEL = MY_parallel_model(
		1234,1234, dsn1_layers, dsn2_layers, snp_layers, 1, inDrop, Drop
		)	
	#
	optimizer = torch.optim.Adam(MM_MODEL.parameters(), lr = config["lr"] )
	if checkpoint_dir :
		checkpoint = os.path.join(checkpoint_dir, "checkpoint")
		model_state, optimizer_state = torch.load(checkpoint)
		model.load_state_dict(model_state)
		optimizer.load_state_dict(optimizer_state)
	#
	#
	# valid_loss_min = valid_loss_min_input # initialize tracker for minimum validation loss
	train_loss_all = []
	valid_loss_all = []
	#
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
		for batch_idx_t, (drug1, drug2, y) in enumerate(loaders['train']):
			# move to GPU
			if use_cuda:
				data, target = data.cuda(), target.cuda()
			## find the loss and update the model parameters accordingly
			# clear the gradients of all optimized variables
			optimizer.zero_grad()
			output = MM_MODEL(drug1, drug2)
			y = y.view(-1,1)
			wc = batch_cut_weight[batch_idx_t].view(-1,1)
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
		for batch_idx_v, (drug1, drug2, y) in enumerate(loaders['eval']):
			# move to GPU
			if use_cuda:
				data, target = data.cuda(), target.cuda()
			## update the average validation loss
			output = MM_MODEL(drug1, drug2)
			y = y.view(-1,1)
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
	# plot_loss(train_loss_all ,valid_loss_all, '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/my_trial_1', 'MY1' )
	# return trained model
	# return model


reporter = CLIReporter(
		metric_columns=["TrainLoss", "ValLoss",  "training_iteration"])

optuna_search = OptunaSearch(metric="ValLoss", mode="min")

ASHA_scheduler = tune.schedulers.ASHAScheduler(
	time_attr='training_iteration', metric="ValLoss", mode="min", max_t= 200, grace_period = 20)


ANALYSIS = tune.run( # 끝내지 않음 
	tune.with_parameters(RAY_MY_train),  
	name = '22.05.24.MM_trial_1',
	num_samples=100, 
	config=CONFIG, 
	resources_per_trial={'cpu': 5},
	progress_reporter = reporter,
	search_alg = optuna_search,
	scheduler = ASHA_scheduler
	)

,
	resume = True





from ray.tune import ExperimentAnalysis
#anal_df = ExperimentAnalysis("/home/jiyeonH/ray_results/22.05.24.MM_trial_1")
ANA_DF = ANALYSIS.dataframe()
ANA_ALL_DF = ANALYSIS.trial_dataframes


DF_KEY = list(ANA_DF.sort_values('ValLoss')['logdir'])[0]
# /home/jiyeonH/ray_results/22.05.24.MM_trial_1/RAY_MY_train_51c5f85e_1_batch_size=100,dropout_1=0.8,dropout_2=0.8,feat_size_1=512,feat_size_2=2048,feat_size_3=1024,feat_size_4=1_2022-05-25_11-56-48

# 190

mini_df = ANA_ALL_DF[DF_KEY]
plot_loss(list(mini_df.TrainLoss), list(mini_df.ValLoss), '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/my_trial_1', 'MM_trial_1.no1' )


checkpoint = '/checkpoint_000199'
TOPVAL_PATH = DF_KEY + checkpoint

dsn1_layers = [2048, 512 , 2048 ]
dsn2_layers = [2048, 512 , 2048 ] 
snp_layers = [1024 , 1024]
inDrop = 0.8
Drop = 0.8
#
#
best_model = MY_parallel_model(
	1234,1234, dsn1_layers, dsn2_layers, snp_layers, 1, inDrop, Drop
	)	

state_dict = torch.load(os.path.join(TOPVAL_PATH, "checkpoint"))
best_model.load_state_dict(state_dict[0])


test_loss = 0
pred = []
ans = []

best_model.eval()

for batch_idx_v, (drug1, drug2, y) in enumerate(loaders['test']):
	with torch.no_grad() :
		output = best_model(drug1, drug2)
		y = y.view(-1,1)
		MSE = torch.nn.MSELoss()
		loss = MSE(output, y) 
		OUT = output.squeeze().tolist()
		Y = y.squeeze().tolist()
		pred = pred+OUT
		ans = ans + Y
		test_loss = test_loss + loss.item()

test_loss/(batch_idx_v+1)
pearson(ans, pred)
spearman(ans, pred)








# 최저를 찾으려면 

TOT_min = np.Inf
TOT_key = ""
for key in ANA_ALL_DF.keys():
	trial_min = min(ANA_ALL_DF[key]['ValLoss'])
	if trial_min < TOT_min :
		TOT_min = trial_min
		TOT_key = key

# /home/jiyeonH/ray_results/22.05.24.MM_trial_1/RAY_MY_train_52efc2d2_3_batch_size=100,dropout_1=0.01,dropout_2=0.2,feat_size_1=4096,feat_size_2=4096,feat_size_3=512,feat_size_4=_2022-05-25_11-56-50
# 170

mini_df = ANA_ALL_DF[TOT_key]

plot_loss(list(mini_df.TrainLoss), list(mini_df.ValLoss), '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/my_trial_1', 'MM_trial_1.no2' )



checkpoint = "/checkpoint_"+str(mini_df[mini_df.ValLoss == TOT_min].index.item()).zfill(6)
TOPVAL_PATH = TOT_key + checkpoint

dsn1_layers = [2048, 4096 , 4096 ]
dsn2_layers = [2048, 4096 , 4096 ] 
snp_layers = [512 , 4096]
inDrop = 0.01
Drop = 0.2
#
#
best_model = MY_parallel_model(
	1234,1234, dsn1_layers, dsn2_layers, snp_layers, 1, inDrop, Drop
	)	

state_dict = torch.load(os.path.join(TOPVAL_PATH, "checkpoint"))
best_model.load_state_dict(state_dict[0])


test_loss = 0
pred = []
ans = []

best_model.eval()

for batch_idx_v, (drug1, drug2, y) in enumerate(loaders['test']):
	with torch.no_grad() :
		output = best_model(drug1, drug2)
		y = y.view(-1,1)
		MSE = torch.nn.MSELoss()
		loss = MSE(output, y) 
		OUT = output.squeeze().tolist()
		Y = y.squeeze().tolist()
		pred = pred+OUT
		ans = ans + Y
		test_loss = test_loss + loss.item()

test_loss/(batch_idx_v+1)
pearson(ans, pred)
spearman(ans, pred)









def draw_stack(NAME):
	correct = 0
	total = len(data_loader_test.dataset)
	pred_list= []
	ans_list = []
	with torch.no_grad():
		for step, data in enumerate(data_loader_test):
			# get the inputs; data is a list of [inputs, labels]
			X = data[0]
			ADJ = data[1]
			ANS = data[2].flatten().long()
			t = [a for a in range(data_loader_test.batch_size)]
			rr = np.repeat(t, data_loader_test.dataset.list_feature.shape[1])
			batch_labels = torch.Tensor(rr).long()
			#
			pred = best_model(X, ADJ, batch_labels).max(dim=1)[1]
			#pred = best_model(data[0:2])[0].max(dim=1)[1]
			pred_list += pred.tolist()
			ans_list += ANS.tolist()
			correct += pred.eq(ANS.view(-1)).sum().item()
	#	
	accuracy = correct/total
	fpr, tpr, thresholds = metrics.roc_curve(ans_list, pred_list)
	roc_auc = metrics.auc(fpr, tpr)
	metrics.confusion_matrix(ans_list, pred_list)
	sns.heatmap(metrics.confusion_matrix(ans_list, pred_list), annot=True)
	print("Accuracy : {}".format(accuracy))
	print("ROC_AUC : {}".format(roc_auc))
	#
	plt.savefig("/st06/jiyeonH/11.TOX/MY_TRIAL_4/{}.png".format(NAME))
	plt.close()




####### 내가 만든 checkpoint 인데, ray 에서는 별로 필요 ㄴ 




		# create checkpoint variable and add important data
		checkpoint = {
			'epoch': epoch,
			'valid_loss_min': valid_loss,
			'state_dict': model.state_dict(),
			'optimizer': optimizer.state_dict(),
		}
		#
		save_ckp(checkpoint, False, checkpoint_path, best_model_path)
		#
		if valid_loss <= valid_loss_min:
			print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
			# save checkpoint as best model
			save_ckp(checkpoint, True, checkpoint_path, best_model_path)
			valid_loss_min = valid_loss

		print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} in {:2f}sec'.format(
			epoch, 
			TRAIN_LOSS,
			VALID_LOSS,
			time_spent.total_seconds()
			))
		#















