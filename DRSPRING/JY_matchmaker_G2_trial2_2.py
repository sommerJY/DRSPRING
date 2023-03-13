# matchmaker 민지 말한 버전 돌려보기 


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


WORK_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/my_trial_2_2/'
DC_PATH = '/st06/jiyeonH/13.DD_SESS/DrugComb.1.5/'
IDK_PATH = '/st06/jiyeonH/13.DD_SESS/ideker/'
LINCS_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/' 
TARGET_PATH = '/st06/jiyeonH/13.DD_SESS/merged_target/'



WORK_PATH='/home01/k006a01/PRJ.01/TRIAL_2.2'
Tch_PATH='/home01/k006a01/PRJ.01/TRIAL_2'
DC_PATH='/home01/k006a01/01.DATA/DrugComb/'
IDK_PATH='/home01/k006a01/01.DATA/IDK/'
LINCS_PATH='/home01/k006a01/01.DATA/LINCS/'


print("filtering")
DC_tmp_DF2=pd.read_csv(DC_PATH+'TOTAL_BLOCK.csv', low_memory=False)
DC_DATA_filter = DC_tmp_DF2[['drug_row_id','drug_col_id','cell_line_id','synergy_loewe']]
DC_DATA_filter2 = DC_DATA_filter.drop_duplicates()
DC_DATA_filter3 = DC_DATA_filter2[DC_DATA_filter2.drug_col_id>0]
DC_DATA_filter4 = DC_DATA_filter3[DC_DATA_filter3.drug_row_id>0]
DC_DATA_filter4.cell_line_id # unique 295


# Drug DATA
with open(DC_PATH+'drugs.json') as json_file :
	DC_DRUG =json.load(json_file)


DC_DRUG_K = list(DC_DRUG[0].keys())
DC_DRUG_DF = pd.DataFrame(columns=DC_DRUG_K)


for DD in range(1,len(DC_DRUG)):
	tmpdf = pd.DataFrame({k:[DC_DRUG[DD][k]] for k in DC_DRUG_K})
	DC_DRUG_DF = pd.concat([DC_DRUG_DF, tmpdf], axis = 0)


DC_DRUG_DF2 = DC_DRUG_DF[['id','dname','cid']]
# dnames = list(DC_DRUG_DF2['dname']) # 그래 이부분때문에 다시 다운로드 받은거였음. 이게 문제가 drug 마다 CID 가 여러개 붙는 애들이 있어서
DC_DRUG_DF2.columns = ['drug_row_id','drug_row','drug_row_cid']
DC_DATA5 = pd.merge(DC_DATA_filter4, DC_DRUG_DF2, on ='drug_row_id', how='left' )

DC_DRUG_DF2.columns = ['drug_col_id','drug_col','drug_col_cid']
DC_DATA6 = pd.merge(DC_DATA5, DC_DRUG_DF2, on ='drug_col_id', how='left')

# Cell DATA
with open(DC_PATH+'cell_lines.json') as json_file :
	DC_CELL =json.load(json_file)


DC_CELL_K = list(DC_CELL[0].keys())
DC_CELL_DF = pd.DataFrame(columns=DC_CELL_K)

for DD in range(1,len(DC_CELL)):
	tmpdf = pd.DataFrame({k:[DC_CELL[DD][k]] for k in DC_CELL_K})
	DC_CELL_DF = pd.concat([DC_CELL_DF, tmpdf], axis = 0)

DC_CELL_DF2 = DC_CELL_DF[['id','name','cellosaurus_accession', 'ccle_name']] # 751450
DC_CELL_DF2.columns = ['cell_line_id', 'DC_cellname','DrugCombCello', 'DrugCombCCLE']

# check DC triads (DC drug, cell line data )
DC_DATA7_1 = pd.merge(DC_DATA6, DC_CELL_DF2 , on= 'cell_line_id', how = 'left')
DC_DATA7_2 = DC_DATA7_1[DC_DATA7_1.drug_row_cid>0]
DC_DATA7_3 = DC_DATA7_2[DC_DATA7_2.drug_col_cid>0] # 735595
cello_t=[True if type(a)==str else False for a in list(DC_DATA7_3.DrugCombCello)]
ccle_t=[True if type(a)==str else False for a in list(DC_DATA7_3.DrugCombCCLE)]

DC_DATA7_4_cello = DC_DATA7_3[cello_t] # 730348
DC_cello_final = DC_DATA7_4_cello[['drug_row_cid','drug_col_cid','DrugCombCello']].drop_duplicates() # 563367
DC_cello_final_dup = DC_DATA7_4_cello[['drug_row_cid','drug_col_cid','DrugCombCello', 'synergy_loewe']].drop_duplicates() # 730348

DC_DATA7_4_ccle = DC_DATA7_3[ccle_t] # 730348
DC_DATA7_4_ccle = DC_DATA7_4_ccle[DC_DATA7_4_ccle.DrugCombCCLE != 'NA'] # 540037
DC_ccle_final = DC_DATA7_4_ccle[['drug_row_cid','drug_col_cid','DrugCombCCLE']].drop_duplicates() # 464137
DC_ccle_final_dup = DC_DATA7_4_ccle[['drug_row_cid','drug_col_cid','DrugCombCCLE', 'synergy_loewe']].drop_duplicates() # 540037


# LINCS DATA
print("LINCS")

BETA_BIND = pd.read_csv(LINCS_PATH+"BETA_DATA_for_SS_df.978.csv")
BETA_SELEC_SIG = pd.read_csv(LINCS_PATH+'SIG_INFO.220405') # cell 58가지, 129116, cid  25589
BETA_CP_info = pd.read_table(LINCS_PATH+'compoundinfo_beta.txt')
BETA_CEL_info = pd.read_table(LINCS_PATH+'cellinfo_beta.txt')
BETA_SIG_info = pd.read_table(LINCS_PATH+'siginfo_beta.txt', low_memory = False)

BETA_GENE = pd.read_table(LINCS_PATH+'geneinfo_beta.txt')
BETA_lm_genes = BETA_GENE[BETA_GENE.feature_space=='landmark'] # 978
BETA_lm_genes = BETA_lm_genes.reset_index()

BETA_CEL_info2 = BETA_CEL_info[['cell_iname','cellosaurus_id','ccle_name']] # 240 
BETA_SELEC_SIG_wCell = pd.merge(BETA_SELEC_SIG, BETA_CEL_info2, on = 'cell_iname', how = 'left') # 129116
BETA_SELEC_SIG_wCell2 = BETA_SELEC_SIG_wCell[['pubchem_cid','cellosaurus_id','ccle_name','sig_id']]
BETA_SELEC_SIG_wCell2 = BETA_SELEC_SIG_wCell2.drop_duplicates() # 129116

cello_tt=[True if type(a)==str else False for a in list(BETA_SELEC_SIG_wCell2.cellosaurus_id)]
BETA_CID_CELLO_SIG = BETA_SELEC_SIG_wCell2[cello_tt][['pubchem_cid','cellosaurus_id','sig_id']].drop_duplicates() # 111916
beta_cid_cello_sig_tf = [ True if a>0 else False for a in list(BETA_CID_CELLO_SIG.pubchem_cid)]
BETA_CID_CELLO_SIG = BETA_CID_CELLO_SIG[beta_cid_cello_sig_tf]

ccle_tt=[True if type(a)==str else False for a in list(BETA_SELEC_SIG_wCell2.ccle_name)]
BETA_CID_CCLE_SIG = BETA_SELEC_SIG_wCell2[ccle_tt][['pubchem_cid','ccle_name','sig_id']].drop_duplicates() # 110620
BETA_CID_CCLE_SIG = BETA_CID_CCLE_SIG[BETA_CID_CCLE_SIG.ccle_name!='NA']
beta_cid_ccle_sig_tf = [ True if a>0 else False for a in list(BETA_CID_CCLE_SIG.pubchem_cid)]
BETA_CID_CCLE_SIG = BETA_CID_CCLE_SIG[beta_cid_ccle_sig_tf]



# cello
BETA_CID_CELLO_SIG.columns=['drug_row_cid', 'DrugCombCello', 'BETA_sig_id']
CELLO_DC_BETA_1 = pd.merge(DC_cello_final_dup, BETA_CID_CELLO_SIG, on = ['drug_row_cid','DrugCombCello'], how = 'left') # 731051

BETA_CID_CELLO_SIG.columns=['drug_col_cid', 'DrugCombCello', 'BETA_sig_id']
CELLO_DC_BETA_2 = pd.merge(CELLO_DC_BETA_1, BETA_CID_CELLO_SIG, on = ['drug_col_cid','DrugCombCello'], how = 'left') # 731644

FILTER = [a for a in range(CELLO_DC_BETA_2.shape[0]) if (type(CELLO_DC_BETA_2.BETA_sig_id_x[a]) == str) & (type(CELLO_DC_BETA_2.BETA_sig_id_y[a]) == str)]
CELLO_DC_BETA = CELLO_DC_BETA_2.loc[FILTER] # 11742
FILTER2 = [True if type(a)==float else False for a in CELLO_DC_BETA.synergy_loewe]
CELLO_DC_BETA = CELLO_DC_BETA.loc[FILTER2] # 11742 ??? 
FILTER3 = [True if np.isnan(a)==False else False for a in CELLO_DC_BETA.synergy_loewe]
CELLO_DC_BETA = CELLO_DC_BETA.loc[FILTER3] # 11701 
CELLO_DC_BETA[['BETA_sig_id_x','BETA_sig_id_y','DrugCombCello']].drop_duplicates() # 9230
CELLO_DC_BETA_cids = list(set(list(CELLO_DC_BETA.drug_row_cid) + list(CELLO_DC_BETA.drug_col_cid))) # 176 





# NETWORK
print('NETWORK')
IDEKER_IAS = pd.read_csv(IDK_PATH+'IAS_score.tsv', sep = '\t')
IDEKER_TOT_GS = list(set(list(IDEKER_IAS['Protein 1'])+list(IDEKER_IAS['Protein 2']))) # 16840
L_matching_list = pd.read_csv(IDK_PATH+'L_12_string.csv', sep = '\t')

IDEKER_IAS_L1 = IDEKER_IAS[IDEKER_IAS['Protein 1'].isin(L_matching_list.PPI_name)]
IDEKER_IAS_L2 = IDEKER_IAS_L1[IDEKER_IAS_L1['Protein 2'].isin(L_matching_list.PPI_name)] # 20232

len(set(list(IDEKER_IAS_L2['Protein 1']) + list(IDEKER_IAS_L2['Protein 2'])))
ID_G = nx.from_pandas_edgelist(IDEKER_IAS_L2, 'Protein 1', 'Protein 2')

ESSEN_NAMES = list(set(L_matching_list['PPI_name']))

MSSNG = [a for a in ESSEN_NAMES if a not in list(ID_G.nodes)]
for nn in list(MSSNG):
		ID_G.add_node(nn)

ID_GENE_ORDER_mini = list(ID_G.nodes()) # 20232
ID_ADJ = nx.adjacency_matrix(ID_G)
ID_ADJ_tmp = torch.LongTensor(ID_ADJ.toarray())
ID_ADJ_IDX = ID_ADJ_tmp.to_sparse().indices()  # [2, 40464]
ID_WEIGHT = [] # len : 20232 -> 40464

ID_ADJ_IDX_T = pd.DataFrame(ID_ADJ_IDX.T.detach().tolist())
ID_ADJ_IDX_T.columns = ['Protein 1','Protein 2']
ID_ADJ_IDX_T['NodeA'] = [list(ID_GENE_ORDER_mini)[A] for A in list(ID_ADJ_IDX_T['Protein 1'])]
ID_ADJ_IDX_T['NodeB'] = [list(ID_GENE_ORDER_mini)[B] for B in list(ID_ADJ_IDX_T['Protein 2'])]
ID_ADJ_IDX_T['NAMESUM'] = ID_ADJ_IDX_T['NodeA'] +'__'+ID_ADJ_IDX_T['NodeB']

IAS_FILTER = IDEKER_IAS[['Protein 1', 'Protein 2', 'Integrated score']]
IAS_FILTER['NAMESUM_1'] = IAS_FILTER['Protein 1']+'__'+IAS_FILTER['Protein 2']
IAS_FILTER['NAMESUM_2'] = IAS_FILTER['Protein 2']+'__'+IAS_FILTER['Protein 1']
IAS_FILTER = IAS_FILTER[['NAMESUM_1','NAMESUM_2','Integrated score']]
IAS_FILTER1= IAS_FILTER[['NAMESUM_1', 'Integrated score']]
IAS_FILTER2= IAS_FILTER[['NAMESUM_2', 'Integrated score']]
IAS_FILTER1.columns = ['NAMESUM', 'Integrated score']
IAS_FILTER2.columns = ['NAMESUM', 'Integrated score']
IAS_FILTER = pd.concat([IAS_FILTER1, IAS_FILTER2],axis = 0)

ID_WEIGHT = pd.merge(ID_ADJ_IDX_T, IAS_FILTER, on = 'NAMESUM', how = 'left' )

ID_WEIGHT_SCORE = list(ID_WEIGHT['Integrated score'])


###########################################################################################
###########################################################################################
###########################################################################################
print("LEARNING")

# Graph 확인 
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

#MY_chem_A = torch.empty(size=(A_B_C_S_SET.shape[0], 256))
#MY_chem_B= torch.empty(size=(A_B_C_S_SET.shape[0], 256))
#MY_exp_A = torch.empty(size=(A_B_C_S_SET.shape[0], 978))
#MY_exp_B = torch.empty(size=(A_B_C_S_SET.shape[0], 978))
#MY_exp_AB = torch.empty(size=(A_B_C_S_SET.shape[0], 978, 2))
#MY_syn =  torch.empty(size=(A_B_C_S_SET.shape[0],1))


#for IND in range(A_B_C_S_SET.shape[0]):
#       DrugA_SIG = A_B_C_S_SET.iloc[IND,]['BETA_sig_id_x']
#       DrugB_SIG = A_B_C_S_SET.iloc[IND,]['BETA_sig_id_y']
#       Cell = A_B_C_S_SET.iloc[IND,]['DrugCombCello']
#       #
#       bitsize = 256
#       DrugA_FP = [int(a) for a in get_CHEMI_data(DrugA_SIG, bitsize)]
#       DrugB_FP = [int(a) for a in get_CHEMI_data(DrugB_SIG, bitsize)]
#       #
#       EXP_A, EXP_B, EXP_AB, LINCS = get_LINCS_data(DrugA_SIG, DrugB_SIG)
#       #
#       AB_SYN = get_synergy_data(DrugA_SIG, DrugB_SIG, Cell)
#       #
#       MY_chem_A[IND] = torch.Tensor(DrugA_FP)
#       MY_chem_B[IND] = torch.Tensor(DrugB_FP)
#       MY_exp_A[IND] = torch.Tensor(EXP_A.iloc[:,1])
#       MY_exp_B[IND] = torch.Tensor(EXP_B.iloc[:,1])
#       MY_exp_AB[IND] = torch.Tensor(EXP_AB).unsqueeze(0)
#       MY_syn[IND] = torch.Tensor([AB_SYN])



#MY_chem_A_tch = torch.tensor(np.array(MY_chem_A))
#MY_chem_B_tch = torch.tensor(np.array(MY_chem_B))
#MY_exp_A_tch = torch.tensor(np.array(MY_exp_A))
#MY_exp_B_tch = torch.tensor(np.array(MY_exp_B))
#MY_exp_AB_tch = torch.tensor(np.array(MY_exp_AB))
#MY_syn_tch = torch.tensor(np.array(MY_syn))

#torch.save(MY_chem_A, WORK_PATH+'0614.MY_chem_A.pt')
#torch.save(MY_chem_B, WORK_PATH+'0614.MY_chem_B.pt')
#torch.save(MY_exp_A, WORK_PATH+'0614.MY_exp_A.pt')
#torch.save(MY_exp_B, WORK_PATH+'0614.MY_exp_B.pt')
#torch.save(MY_exp_AB, WORK_PATH+'0614.MY_exp_AB.pt')
#torch.save(MY_syn, WORK_PATH+'0614.MY_syn.pt')

Tch_PATH='/st06/jiyeonH/11.TOX/MY_TRIAL_GPU/PJ01.TRIAL.2_2/'
MY_chem_A_tch = torch.load(Tch_PATH+'0623.MY_chem_A.pt')
MY_chem_B_tch = torch.load(Tch_PATH+'0623.MY_chem_B.pt')
MY_exp_A_tch = torch.load(Tch_PATH+'0623.MY_exp_A.pt')
MY_exp_B_tch = torch.load(Tch_PATH+'0623.MY_exp_B.pt')
MY_exp_AB_tch = torch.load(Tch_PATH+'0623.MY_exp_AB.pt')
MY_syn_tch = torch.load(Tch_PATH+'0623.MY_syn.pt')

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



def prepare_data_GCN(MY_chem_A_tch, MY_chem_B_tch, MY_exp_A_tch, MY_exp_B_tch, MY_syn_tch, norm ) :
	chem_A_train, chem_A_tv, chem_B_train, chem_B_tv, exp_A_train, exp_A_tv, exp_B_train, exp_B_tv, syn_train, syn_tv = sklearn.model_selection.train_test_split(
			MY_chem_A_tch, MY_chem_B_tch, MY_exp_A_tch, MY_exp_B_tch, MY_syn_tch,
			test_size= A_B_C_S_SET.shape[0]-6000 , random_state=42 )
	chem_A_val, chem_A_test, chem_B_val, chem_B_test, exp_A_val, exp_A_test, exp_B_val, exp_B_test, syn_val, syn_test  = sklearn.model_selection.train_test_split(
			chem_A_tv, chem_B_tv, exp_A_tv, exp_B_tv, syn_tv,
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
	train_data['EXP_A'] = torch.concat([exp_A_train, exp_B_train], axis = 0)
	val_data['EXP_A'] = exp_A_val
	test_data['EXP_A'] = exp_A_test
	#
	train_data['EXP_B'] = torch.concat([exp_B_train, exp_A_train], axis = 0)
	val_data['EXP_B'] = exp_B_val
	test_data['EXP_B'] = exp_B_test
	#               
	train_data['y'] = np.concatenate((syn_train,syn_train), axis=0)
	val_data['y'] = syn_val
	test_data['y'] = syn_test
	#
	print(train_data['drug1'].shape)
	print(val_data['drug1'].shape)
	print(test_data['drug1'].shape)
	return train_data, val_data, test_data



class DATASET_GCN_W_F(Dataset):
	def __init__(self, drug1_F, drug2_F, gcn_exp_A, gcn_exp_B, gcn_adj, gcn_adj_weight, syn_ans):
		self.drug1_F = drug1_F
		self.drug2_F = drug2_F
		self.gcn_exp_A = gcn_exp_A
		self.gcn_exp_B = gcn_exp_B
		self.gcn_adj = gcn_adj
		self.gcn_adj_weight = gcn_adj_weight
		self.syn_ans = syn_ans
	#
	def __len__(self):
		return len(self.drug1_F)
			#
	def __getitem__(self, index):
		return self.drug1_F[index], self.drug2_F[index], self.gcn_exp_A[index], self.gcn_exp_B[index], self.gcn_adj, self.gcn_adj_weight, self.syn_ans[index]



def graph_collate_fn(batch):
	drug1_list = []
	drug2_list = []
	expA_list = []
	expB_list = []
	adj_list = []
	adj_w_list = []
	y_list = []
	num_nodes_seen = 0
	for drug1, drug2, expA, expB, adj, adj_w, y in batch :
		drug1_list.append(drug1)
		drug2_list.append(drug2)
		expA_list.append(expA)
		expB_list.append(expB)
		adj_list.append(adj+num_nodes_seen)
		adj_w_list.append(adj_w)
		y_list.append(y)
		num_nodes_seen += expA.shape[0]
	drug1_new = torch.stack(drug1_list, 0)
	drug2_new = torch.stack(drug2_list, 0)
	expA_new = torch.cat(expA_list, 0)
	expB_new = torch.cat(expB_list, 0)
	adj_new = torch.cat(adj_list, 1)
	adj_w_new = torch.cat(adj_w_list, 1)
	y_new = torch.stack(y_list, 0)
	return drug1_new, drug2_new, expA_new, expB_new, adj_new, adj_w_new, y_new


def weighted_mse_loss(input, target, weight):
	return (weight * (input - target) ** 2).mean()


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
train_data, val_data, test_data = prepare_data_GCN(MY_chem_A_tch, MY_chem_B_tch, MY_exp_A_tch, MY_exp_B_tch, MY_syn_tch, norm)


# WEIGHT 
ys = train_data['y'].squeeze().tolist()
min_s = np.amin(ys)
loss_weight = np.log(train_data['y'] - min_s + np.e)
JY_IDX_WEIGHT_T = torch.Tensor(JY_IDX_WEIGHT).view(1,-1)


# DATA check 
T_train = DATASET_GCN_W_F(torch.Tensor(train_data['drug1']), torch.Tensor(train_data['drug2']), train_data['EXP_A'], train_data['EXP_B'], JY_ADJ_IDX, JY_IDX_WEIGHT_T, torch.Tensor(train_data['y']))
T_val = DATASET_GCN_W_F(torch.Tensor(val_data['drug1']), torch.Tensor(val_data['drug2']), val_data['EXP_A'], val_data['EXP_B'], JY_ADJ_IDX, JY_IDX_WEIGHT_T, torch.Tensor(val_data['y']))
T_test = DATASET_GCN_W_F(torch.Tensor(test_data['drug1']), torch.Tensor(test_data['drug2']), test_data['EXP_A'], test_data['EXP_B'], JY_ADJ_IDX, JY_IDX_WEIGHT_T, torch.Tensor(test_data['y']))

RAY_train = ray.put(T_train)
RAY_val = ray.put(T_val)
RAY_test = ray.put(T_test)
RAY_loss_weight = ray.put(loss_weight)

############################ MAIN
print('MAIN')

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
		self.G_convs_1 = torch.nn.ModuleList([pyg_nn.GCNConv(self.G_indim, self.G_hiddim)])
		self.G_convs_1.extend([pyg_nn.GCNConv(self.G_hiddim, self.G_hiddim) for i in range(self.G_layer-2)])
		self.G_convs_1.extend([pyg_nn.GCNConv(self.G_hiddim, self.G_hiddim)])
		self.G_bns_1 = torch.nn.ModuleList([torch.nn.BatchNorm1d(self.G_hiddim) for i in range(self.G_layer-1)])
		#
		self.G_convs_2 = torch.nn.ModuleList([pyg_nn.GCNConv(self.G_indim, self.G_hiddim)])
		self.G_convs_2.extend([pyg_nn.GCNConv(self.G_hiddim, self.G_hiddim) for i in range(self.G_layer-2)])
		self.G_convs_2.extend([pyg_nn.GCNConv(self.G_hiddim, self.G_hiddim)])
		self.G_bns_2 = torch.nn.ModuleList([torch.nn.BatchNorm1d(self.G_hiddim) for i in range(self.G_layer-1)])
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
		for conv in self.G_convs_1 :
			conv.reset_parameters()
		for bns in self.G_bns_1 :
			bns.reset_parameters()
		for conv in self.G_convs_2 :
			conv.reset_parameters()
		for bns in self.G_bns_2 :
			bns.reset_parameters()
		for conv in self.Convs_1 :
			conv.reset_parameters()
		for conv in self.Convs_2:
			conv.reset_parameters()
		for conv in self.SNPs:
			conv.reset_parameters()
	#
	def forward(self, Drug1, Drug2, EXP1, EXP2, ADJ ):
		Bat_info = Drug1.shape[0]
		Node_info = EXP1.shape[0]/Bat_info
		Num = [a for a in range(Bat_info)]
		Rep = np.repeat(Num, Node_info)
		batch_labels = torch.Tensor(Rep).long()
		if torch.cuda.is_available():
			batch_labels = batch_labels.cuda()
		#
		for G1 in range(len(self.G_convs_1)):
			if G1 == len(self.G_convs_1)-1 :
				EXP1 = self.G_convs_1[G1](x=EXP1, edge_index=ADJ)
				EXP1 = F.dropout(EXP1, p=self.inDrop, training=self.training)
				EXP1 = self.pool(EXP1, batch_labels )
				EXP1 = self.tanh(EXP1)
				G_out_1 = EXP1
			else :
				EXP1 = self.G_convs_1[G1](x=EXP1, edge_index=ADJ)
				EXP1 = self.G_bns_1[G1](EXP1)
				EXP1 = F.elu(EXP1)
		#
		for G2 in range(len(self.G_convs_2)):
			if G2 == len(self.G_convs_2)-1 :
				EXP2 = self.G_convs_2[G2](x=EXP2, edge_index=ADJ)
				EXP2 = F.dropout(EXP2, p=self.inDrop, training=self.training)
				EXP2 = self.pool(EXP2, batch_labels )
				EXP2 = self.tanh(EXP2)
				G_out_2 = EXP2
			else :
				EXP2 = self.G_convs_2[G2](x=EXP2, edge_index=ADJ)
				EXP2 = self.G_bns_2[G2](EXP2)
				EXP2 = F.elu(EXP2)
		#
		input_drug1 = torch.concat( (Drug1, G_out_1), 1 ) # normalization 추가 해야할것 같은데 
		input_drug2 = torch.concat( (Drug2, G_out_2), 1 )
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



def RAY_MY_train(config, checkpoint_dir=None):
	n_epochs = config["epoch"]
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
			config["G_layer"], 1 , config["G_hiddim"],
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
		for batch_idx_t, (drug1, drug2, expA, expB, adj, adj_w, y) in enumerate(loaders['train']):
			expA = expA.view(-1,1)#### 다른점 
			expB = expB.view(-1,1)#### 다른점 
			# move to GPU
			if use_cuda:
				drug1, drug2, expA, expB, adj, adj_w, y = drug1.cuda(), drug2.cuda(), expA.cuda(), expB.cuda(), adj.cuda(), adj_w.cuda(), y.cuda()
			## find the loss and update the model parameters accordingly
			# clear the gradients of all optimized variables
			optimizer.zero_grad()
			output = MM_MODEL(drug1, drug2, expA, expB, adj)
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
		for batch_idx_v, (drug1, drug2, expA, expB, adj, adj_w, y) in enumerate(loaders['eval']):
			expA = expA.view(-1,1)#### 다른점 
			expB = expB.view(-1,1)#### 다른점 
			# move to GPU
			if use_cuda:
				drug1, drug2, expA, expB, adj, adj_w, y = drug1.cuda(), drug2.cuda(), expA.cuda(), expB.cuda(), adj.cuda(), adj_w.cuda(), y.cuda()
			## update the average validation loss
			output = MM_MODEL(drug1, drug2, expA, expB, adj)
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
	best_trained_model = MY_expGCN_parallel_model(
		best_trial.config["G_layer"], 1, best_trial.config["G_hiddim"],
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
	for batch_idx_t, (drug1, drug2, expA, expB, adj, adj_w, y) in enumerate(Test_loader):
		expA = expA.view(-1,1)#### 다른점 
		expB = expB.view(-1,1)#### 다른점 
		if use_cuda:
			drug1, drug2, expA, expB, adj, adj_w, y = drug1.cuda(), drug2.cuda(), expA.cuda(), expB.cuda(), adj.cuda(), adj_w.cuda(), y.cuda()
		output = best_trained_model(drug1, drug2, expA, expB, adj)
		MSE = torch.nn.MSELoss()
		loss = MSE(output, y)
		test_loss = test_loss + loss.item()
		outputs = output.squeeze().tolist()
		PRED_list = PRED_list+outputs
	TEST_LOSS = test_loss/(batch_idx_t+1)
	result_pearson(PRED_list, Y_list)
	result_spearman(PRED_list, Y_list)
	print("Best model TEST loss: {}".format(TEST_LOSS))





def MAIN(WORK_PATH, num_samples= 10, max_num_epochs=1000, grace_period = 150, cpus_per_trial = 16, gpus_per_trial = 1):
	# OPTUNA CONFIGURE 
	CONFIG={
		'n_workers' : tune.choice([cpus_per_trial]),
		"epoch" : tune.choice([max_num_epochs]),
		"G_layer" : tune.choice([2, 3, 4]),
		"G_hiddim" : tune.choice([512, 256, 128, 64, 32]),
		"batch_size" : tune.choice([32]), # The number of batch sizes should be a power of 2 to take full advantage of the GPUs processing # 128, 64, 
		"feat_size_0" : tune.choice([32]), # 4096, 2048, 1024, 512, 256, 128, 64, 
		"feat_size_1" : tune.choice([32]),# 4096, 2048, 1024, 512, 256, 128, 64, 
		"feat_size_2" : tune.choice([32]),# 4096, 2048, 1024, 512, 256, 128, 64, 
		"feat_size_3" : tune.choice([32]),# 4096, 2048, 1024, 512, 256, 128, 64, 
		"feat_size_4" : tune.choice([32]),# 4096, 2048, 1024, 512, 256, 128, 64, 
		"dropout_1" : tune.choice([0.01, 0.2, 0.5, 0.8]),
		"dropout_2" : tune.choice([0.01, 0.2, 0.5, 0.8]),
		"lr" : tune.choice([0.001]),#0.00001, 0.0001, 
	}
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
		name = '22.06.23.PRJ01.TRIAL2_2',
		num_samples=num_samples,
		config=CONFIG,
		resources_per_trial={'cpu': cpus_per_trial},#, 'gpu' : gpus_per_trial
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
	plot_loss(list(TMP_DF.TrainLoss), list(TMP_DF.ValLoss), WORK_PATH, 'MM_GCNexp_IDK')
	#
	print('start test with best model')
	if ray.util.client.ray.is_connected():
		from ray.util.ml_utils.node import force_on_current_node
		remote_fn = force_on_current_node(ray.remote(test_best_model))
		ray.get(remote_fn.remote(best_trial))
	else:
		RAY_TEST_MODEL(best_trial)



MAIN(WORK_PATH, 100, 1000, 150, 64, 1)
###MAIN(WORK_PATH, 3, 3, 1, 16, 0.5)




#WORK_PATH='/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/my_trial_2_2'
#
MAIN(WORK_PATH, 1, 2, 1, 64, 0.5)
num_samples= 10, max_num_epochs=1000, grace_period = 150, cpus_per_trial = 16, gpus_per_trial = 1








####3 test 

test_model = MY_expGCN_parallel_model(3 , 1 , 100,
			256, 256,
			[100, 200 , 5 ], [100, 200 , 5 ], [100,150], 
			1,0.5, 0.3)



loaders = {
	'train' : torch.utils.data.DataLoader(T_train, batch_size = 32, collate_fn = graph_collate_fn, shuffle =False, num_workers=1),
}

for batch_idx_t, (drug1, drug2, expA, expB, adj, adj_w, y) in enumerate(loaders['train']):
	if batch_idx_t < 10:
		output = test_model(drug1, drug2, expA, expB, adj)













#############
결과 확인 
#############


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


from ray.tune import ExperimentAnalysis
#anal_df = ExperimentAnalysis("~/ray_results/22.06.23.PRJ01.TRIAL2_2")
ANA_DF = anal_df.dataframe()
ANA_ALL_DF = anal_df.trial_dataframes

#ANA_DF.to_csv('/home01/k006a01/PRJ.01/TRIAL_2.2/RAY_ANA_DF.P01.2_2.csv')
#import pickle
#with open("/home01/k006a01/PRJ.01/TRIAL_2.2/RAY_ANA_ALL_DF.P01.2_2.pickle", "wb") as fp:
#   pickle.dump(ANA_ALL_DF,fp) 

PRJ_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_GPU/PJ01.TRIAL.2_2/'
ANA_DF = pd.read_csv(PRJ_PATH+'RAY_ANA_DF.P01.2_2.csv')
with open(PRJ_PATH+'RAY_ANA_ALL_DF.P01.2_2.pickle', 'rb') as f:
	ANA_ALL_DF = pickle.load(f)


DF_KEY = list(ANA_DF.sort_values('ValLoss')['logdir'])[0]
#  /home01/k006a01/ray_results/22.06.23.PRJ01.TRIAL2_2/RAY_MY_train_e8da756c_2_G_hiddim=64,G_layer=4,batch_size=128,dropout_1=0.0100,dropout_2=0.8000,epoch=1000,feat_size_0=4096,feat_si_2022-06-23_11-33-47

mini_df = ANA_ALL_DF[DF_KEY]
plot_loss(list(mini_df.TrainLoss), list(mini_df.ValLoss), 
PRJ_PATH, 'TRIAL_2_2.BEST.loss' )



(1) 마지막 모델 확인 

TOPVAL_PATH = DF_KEY
my_config = ANA_DF[ANA_DF.logdir==DF_KEY]

G_layer = my_config['config/G_layer'].item()
G_hiddim = my_config['config/G_hiddim'].item()
dsn1_layers = [my_config['config/feat_size_0'].item(), my_config['config/feat_size_1'].item(), my_config['config/feat_size_2'].item()]
dsn2_layers = [my_config['config/feat_size_0'].item(), my_config['config/feat_size_1'].item(), my_config['config/feat_size_2'].item()] 
snp_layers = [my_config['config/feat_size_3'].item() , my_config['config/feat_size_4'].item()]
inDrop = my_config['config/dropout_1'].item()
Drop = my_config['config/dropout_2'].item()


best_model = MY_expGCN_parallel_model(
		G_layer, 1, G_hiddim,
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
	for batch_idx_t, (drug1, drug2, expA, expB, adj, adj_w, y) in enumerate(Test_loader):
		expA = expA.view(-1,1)
		expB = expB.view(-1,1)
		output = best_model(drug1, drug2, expA, expB, adj)
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
# /home01/k006a01/ray_results/22.06.23.PRJ01.TRIAL2_2/RAY_MY_train_e8da756c_2_G_hiddim=64,G_layer=4,batch_size=128,dropout_1=0.0100,dropout_2=0.8000,epoch=1000,feat_size_0=4096,feat_si_2022-06-23_11-33-47/checkpoint_000985
min(mini_df.ValLoss)


state_dict = torch.load(os.path.join(PRJ_PATH, "M2_checkpoint"),map_location=torch.device('cpu'))
best_model.load_state_dict(state_dict[0])

best_model.eval()
test_loss = 0.0
PRED_list = []
Y_list = test_data['y'].squeeze().tolist()
with torch.no_grad():
	best_model.eval()
	for batch_idx_t, (drug1, drug2, expA, expB, adj, adj_w, y) in enumerate(Test_loader):
		expA = expA.view(-1,1)
		expB = expB.view(-1,1)
		output = best_model(drug1, drug2, expA, expB, adj)
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

TOT_min
# /home01/k006a01/ray_results/22.06.23.PRJ01.TRIAL2_2/RAY_MY_train_2bf15eac_69_G_hiddim=256,G_layer=3,batch_size=128,dropout_1=0.0100,dropout_2=0.8000,epoch=1000,feat_size_0=512,feat_s_2022-06-26_17-54-00


mini_df = ANA_ALL_DF[TOT_key]
plot_loss(list(mini_df.TrainLoss), list(mini_df.ValLoss), 
PRJ_PATH, 'TRIAL_2_2.MIN.loss' )


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
		G_layer, 1, G_hiddim,
		256, 256,
		dsn1_layers, dsn2_layers, snp_layers, 1,
		inDrop, Drop
)

cck_num =mini_df[mini_df.ValLoss==min(mini_df.ValLoss)].index.item()
checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
TOPVAL_PATH = TOT_key + checkpoint
# /home01/k006a01/ray_results/22.06.23.PRJ01.TRIAL2_2/RAY_MY_train_2bf15eac_69_G_hiddim=256,G_layer=3,batch_size=128,dropout_1=0.0100,dropout_2=0.8000,epoch=1000,feat_size_0=512,feat_s_2022-06-26_17-54-00/checkpoint_000985


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
	for batch_idx_t, (drug1, drug2, expA, expB, adj, adj_w, y) in enumerate(Test_loader):
		expA = expA.view(-1,1)
		expB = expB.view(-1,1)
		output = best_model(drug1, drug2, expA, expB, adj)
		MSE = torch.nn.MSELoss()
		loss = MSE(output, y)
		test_loss = test_loss + loss.item()
		outputs = output.squeeze().tolist()
		PRED_list = PRED_list+outputs


TEST_LOSS = test_loss/(batch_idx_t+1)
print("Best model TEST loss: {}".format(TEST_LOSS))
result_pearson(PRED_list, Y_list)
result_spearman(PRED_list, Y_list)

PRJ_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_GPU/PJ01.TRIAL.2_2/'
jy_corrplot(PRED_list, Y_list, PRJ_PATH,'2_2.M1_model' )














