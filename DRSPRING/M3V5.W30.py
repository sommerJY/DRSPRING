

#생각해보니 5CV 로 돌리지 않고 내꺼 모델에서 그냥 5CV 를 돌리면 될 문제라서 
# 1gpu 버전으로 수정하기로 함 


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


#NETWORK_PATH = '/st06/jiyeonH/13.DD_SESS/HumanNetV3/'
#LINCS_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/' 
#DATA_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V3_FULL/'
#DC_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/'


ray.init()

NETWORK_PATH = '/home01/k020a01/01.Data/HumanNet/'
LINCS_PATH = '/home01/k020a01/01.Data/LINCS/'
DC_PATH = '/home01/k020a01/01.Data/DrugComb/'


						print('NETWORK')
									# 978
						hunet_gsp = pd.read_csv(NETWORK_PATH+'HumanNet-GSP.tsv', sep = '\t', header = None)
						hunet_gsp.columns = ['G_A','G_B']

						LINCS_gene_file = pd.read_csv(LINCS_PATH+'geneinfo_beta.txt', sep = '\t')
						LINCS_978 = LINCS_gene_file[LINCS_gene_file.feature_space == 'landmark']
						LINCS_978 = LINCS_978[['gene_id','gene_symbol']]
						LINCS_978['new_node'] = [str(list(LINCS_978.gene_id)[i]) + "__" + list(LINCS_978.gene_symbol)[i] for i in range(978)]
						LINCS_978 = LINCS_978.reset_index(drop=True)
						lm_entrezs = list(LINCS_978.gene_id)


						hnet_L1 = hunet_gsp[hunet_gsp['G_A'].isin(lm_entrezs)]
						hnet_L2 = hnet_L1[hnet_L1['G_B'].isin(lm_entrezs)] # 3885

						len(set(list(hnet_L2['G_A']) + list(hnet_L2['G_B']))) # 611

						ID_G = nx.from_pandas_edgelist(hnet_L2, 'G_A', 'G_B')

						MSSNG = [a for a in lm_entrezs if a not in list(ID_G.nodes)]

						for nn in list(MSSNG):
							ID_G.add_node(nn)


						# edge 3871
						ID_GENE_ORDER_mini = list(ID_G.nodes()) # 978
						ID_ADJ = nx.adjacency_matrix(ID_G)
						ID_ADJ_tmp = torch.LongTensor(ID_ADJ.toarray())
						ID_ADJ_IDX = ID_ADJ_tmp.to_sparse().indices()  # [2, 7742]
						ID_WEIGHT = [] # len : 3871 -> 7742

						ID_WEIGHT_SCORE = [1 for a in range(ID_ADJ_IDX.shape[1])]






# HS Drug pathway DB 활용 -> 349
print('NETWORK')
# HUMANNET 사용 

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

# MSSNG = [a for a in lm_entrezs if a not in list(ID_G.nodes)]

#for nn in list(MSSNG):
#	ID_G.add_node(nn)

# edge 
ID_GENE_ORDER_mini = list(ID_G.nodes()) # 978
ID_ADJ = nx.adjacency_matrix(ID_G)
ID_ADJ_tmp = torch.LongTensor(ID_ADJ.toarray())
ID_ADJ_IDX = ID_ADJ_tmp.to_sparse().indices()  # [2, 7742]
ID_WEIGHT = [] # len : 3871 -> 7742

# 원래는 edge score 있지만 일단은...
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







MJ_NAME = 'M3V5'
WORK_DATE = '23.05.17' # 349
MISS_NAME = 'MIS2'




# W20 & W21
SAVE_PATH = '/home01/k020a01/02.M3V5/M3V5_349_DATA/'
# SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V5_349_FULL/'
file_name = 'M3V5_349_MISS2_FULL'



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




WORK_NAME = 'WORK_30' # 349 -> leave one drug out 


#MISS_filter = ['AOBO']
#MISS_filter = ['AOBO','AXBO','AOBX']
MISS_filter = ['AOBO','AXBO','AOBX','AXBX']

A_B_C_S_SET = A_B_C_S_SET_ADD[A_B_C_S_SET_ADD.Basal_Exp == 'O']

A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.SYN_OX == 'O']

## A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.T2OX == 'O'] ###################### old targets 
A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.T1OX == 'O'] ####################### new targets 

A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.type.isin(MISS_filter)]



# basal node exclude -> CCLE match 만 사용
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
ccle_names = [a for a in ccle_exp3.DrugCombCCLE if type(a) == str]

A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.DrugCombCCLE.isin(ccle_names)]




data_ind = list(A_B_C_S_SET.index)

MY_chem_A_feat_RE = MY_chem_A_feat[data_ind]
MY_chem_B_feat_RE = MY_chem_B_feat[data_ind]
MY_chem_A_adj_RE = MY_chem_A_adj[data_ind]
MY_chem_B_adj_RE = MY_chem_B_adj[data_ind]
MY_g_EXP_A_RE = MY_g_EXP_A[data_ind]
MY_g_EXP_B_RE = MY_g_EXP_B[data_ind]

# MY_Target_A = copy.deepcopy(MY_Target_2_A)[data_ind] ############## OLD TARGET !!!!!! #####
# MY_Target_B = copy.deepcopy(MY_Target_2_B)[data_ind] ############## OLD TARGET !!!!!! #####

MY_Target_A = copy.deepcopy(MY_Target_1_A)[data_ind] ############## NEW TARGET !!!!!! #####
MY_Target_B = copy.deepcopy(MY_Target_1_B)[data_ind] ############## NEW TARGET !!!!!! #####


MY_CellBase_RE = MY_CellBase[data_ind]
MY_syn_RE = MY_syn[data_ind]


A_B_C_S_SET = A_B_C_S_SET.reset_index(drop = True)




# cell line vector 

DC_CELL_DF2 = pd.read_csv(DC_PATH+'DC_CELL_INFO.csv', sep = '\t')

DC_CELL_info_filt = DC_CELL_DF2[DC_CELL_DF2.DrugCombCCLE.isin(A_B_C_S_SET.DrugCombCCLE)] # 38

A_B_C_S_SET_COH = pd.merge(A_B_C_S_SET, DC_CELL_info_filt[['DrugCombCCLE','DC_cellname']], on = 'DrugCombCCLE', how = 'left'  )



# sample number filter 

# 빈도 확인 

C_names = list(set(A_B_C_S_SET_COH.DC_cellname))

C_freq = [list(A_B_C_S_SET_COH.DC_cellname).count(a) for a in C_names]

C_df = pd.DataFrame({'cell' : C_names, 'freq' : C_freq })
C_df = C_df.sort_values('freq')




CELL_CUT = 200 ############ WORK 20 ##############

C_freq_filter = C_df[C_df.freq > CELL_CUT ] 


A_B_C_S_SET_COH = A_B_C_S_SET_COH[A_B_C_S_SET_COH.DC_cellname.isin(C_freq_filter.cell)]

DC_CELL_info_filt_re = DC_CELL_info_filt[DC_CELL_info_filt.DC_cellname.isin(C_freq_filter.cell)]
DC_CELL_info_filt_re['cell_onehot'] = [a for a in range(len(set(DC_CELL_info_filt_re.DrugCombCCLE)))]

DC_CELL_info_filt_re = DC_CELL_info_filt_re.reset_index(drop = True)


data_ind = list(A_B_C_S_SET_COH.index)

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

# merge 전 후로 index 달라지므로 뒤에 넣어줬음 
A_B_C_S_SET_COH2 = pd.merge(A_B_C_S_SET_COH, DC_CELL_info_filt_re[['DC_cellname','cell_onehot']], on = 'DC_cellname', how='left')
cell_one_hot = torch.nn.functional.one_hot(torch.Tensor(A_B_C_S_SET_COH2['cell_onehot']).long())




###########################################################################################
###########################################################################################
###########################################################################################

print("LEARNING")


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




# 0328 added.... hahahahaha
A_B_C_S_SET_SM = copy.deepcopy(A_B_C_S_SET_COH2)

A_B_C_S_SET_SM['ori_index'] = list(A_B_C_S_SET_SM.index)
aaa = list(A_B_C_S_SET_SM['drug_row_CID'])
bbb = list(A_B_C_S_SET_SM['drug_col_CID'])
aa = list(A_B_C_S_SET_SM['ROW_CAN_SMILES'])
bb = list(A_B_C_S_SET_SM['COL_CAN_SMILES'])
cc = list(A_B_C_S_SET_SM['DrugCombCCLE'])

A_B_C_S_SET_SM['CID_CID_CCLE'] = [str(int(aaa[i])) + '___' + str(int(bbb[i]))+ '___' + cc[i] if aaa[i] < bbb[i] else str(int(bbb[i])) + '___' + str(int(aaa[i]))+ '___' + cc[i] for i in range(A_B_C_S_SET_SM.shape[0])]
A_B_C_S_SET_SM['SM_C_CHECK'] = [aa[i] + '___' + bb[i]+ '___' + cc[i] if aa[i] < bb[i] else bb[i] + '___' + aa[i]+ '___' + cc[i] for i in range(A_B_C_S_SET_SM.shape[0])]

A_B_C_S_SET_SM[['drug_row_CID','drug_col_CID','DrugCombCCLE']].drop_duplicates() # 52152
A_B_C_S_SET_SM[['ROW_CAN_SMILES','COL_CAN_SMILES','DrugCombCCLE']].drop_duplicates() # 52120
len(set(A_B_C_S_SET_SM['CID_CID_CCLE'])) # 51212
len(set(A_B_C_S_SET_SM['SM_C_CHECK'])) # 51160



# get unique values, remove duplicates, but keep original counts
data_no_dup, counts = np.unique(list(A_B_C_S_SET_SM['SM_C_CHECK']), return_counts=True)
data_no_dup_cells = [setset.split('___')[2] for setset in data_no_dup]


# leave drug out 하기 위해서 일단 이걸로 확인하기 

data_nodup_df = pd.DataFrame({'setset' : data_no_dup.tolist(), 'cell' : data_no_dup_cells })
data_nodup_df2 = data_nodup_df.sort_values('cell')
data_nodup_df2 = data_nodup_df2.reset_index(drop =True)

data_nodup_df2['CHEM_A'] = [setset.split('___')[0] for setset in list(data_nodup_df2['setset'])]
data_nodup_df2['CHEM_B'] = [setset.split('___')[1] for setset in list(data_nodup_df2['setset'])]

all_chem_in_df2 = list(set(list(data_nodup_df2['CHEM_A']) + list(data_nodup_df2['CHEM_B'])))
all_chem_in_df2 = sklearn.utils.shuffle(all_chem_in_df2, random_state=42)


# 그래서 확인해보니까 생각보다 10퍼센트만 빼낸다고 해도 데이터의 20 퍼 이상이 잘리는걸 확인함...
# test 안나누고 전체 5CV 로 진행하기로 함 
# 물질 하나만 빼고 하면 무슨기준으로 해야하는지도 명확하지가 않아서...


test_percent = round(len(all_chem_in_df2)*0.2)

bins = []
for ii in list(range(0, len(all_chem_in_df2), test_percent)):
	if len(bins)< 5 :
		bins.append(ii)

bins = bins[1:]

res_list = np.split(all_chem_in_df2, bins)


CV_0_sm_test = res_list[0].tolist() ; CV_0_sm_train = [a for a in all_chem_in_df2 if a not in CV_0_sm_test]
CV_1_sm_test = res_list[1].tolist() ; CV_1_sm_train = [a for a in all_chem_in_df2 if a not in CV_1_sm_test]
CV_2_sm_test = res_list[2].tolist() ; CV_2_sm_train = [a for a in all_chem_in_df2 if a not in CV_2_sm_test]
CV_3_sm_test = res_list[3].tolist() ; CV_3_sm_train = [a for a in all_chem_in_df2 if a not in CV_3_sm_test]
CV_4_sm_test = res_list[4].tolist() ; CV_4_sm_train = [a for a in all_chem_in_df2 if a not in CV_4_sm_test]



# use just index 
# normalizing 과정에 대한 고민 필요 ... GCN 이니까 상관 없지 않을까
def prepare_data_GCN(CV_num, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, 
MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, 
cell_one_hot, MY_syn_RE2, norm ) : 
	#
	# CV_num = 0 
	train_key = 'CV_{}_sm_train'.format(CV_num)
	test_key = 'CV_{}_sm_test'.format(CV_num)
	train_sm = globals()[train_key] # 698
	test_sm = globals()[test_key] # 175
	# 
	bool_check_1 = [(data_nodup_df2.CHEM_A[a] in test_sm) or (data_nodup_df2.CHEM_B[a] in test_sm) for a in range(data_nodup_df2.shape[0])]
	bool_check_2 = [not ele for ele in bool_check_1]
	#
	train_no_dup = data_nodup_df2[bool_check_2] # train val df 
	test_no_dup = data_nodup_df2[bool_check_1] # from test included df 
	#
	ABCS_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(train_no_dup.setset)]
	ABCS_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(test_no_dup.setset)]
	ABCS_train_ind = list(ABCS_train.index)
	random.shuffle(ABCS_train_ind)
	ABCS_train = ABCS_train.loc[ABCS_train_ind]
	#
	train_ind = list(ABCS_train.index)
	test_ind = list(ABCS_test.index)
	# 
	chem_feat_A_train = MY_chem_A_feat_RE2[train_ind]; chem_feat_A_test = MY_chem_A_feat_RE2[test_ind]
	chem_feat_B_train = MY_chem_B_feat_RE2[train_ind]; chem_feat_B_test = MY_chem_B_feat_RE2[test_ind]
	chem_adj_A_train = MY_chem_A_adj_RE2[train_ind]; chem_adj_A_test = MY_chem_A_adj_RE2[test_ind]
	chem_adj_B_train = MY_chem_B_adj_RE2[train_ind];  chem_adj_B_test = MY_chem_B_adj_RE2[test_ind]
	gene_A_train = MY_g_EXP_A_RE2[train_ind];  gene_A_test = MY_g_EXP_A_RE2[test_ind]
	gene_B_train = MY_g_EXP_B_RE2[train_ind]; gene_B_test = MY_g_EXP_B_RE2[test_ind]
	target_A_train = MY_Target_A2[train_ind]; target_A_test = MY_Target_A2[test_ind]
	target_B_train = MY_Target_B2[train_ind]; target_B_test = MY_Target_B2[test_ind]
	cell_basal_train = MY_CellBase_RE2[train_ind]; cell_basal_test = MY_CellBase_RE2[test_ind]
	cell_train = cell_one_hot[train_ind];  cell_test = cell_one_hot[test_ind]
	syn_train = MY_syn_RE2[train_ind]; syn_test = MY_syn_RE2[test_ind]
	#
	train_data = {}
	test_data = {}
	#
	train_data['drug1_feat'] = torch.concat([chem_feat_A_train, chem_feat_B_train], axis = 0)
	test_data['drug1_feat'] = chem_feat_A_test
	#
	train_data['drug2_feat'] = torch.concat([chem_feat_B_train, chem_feat_A_train], axis = 0)
	test_data['drug2_feat'] = chem_feat_B_test
	#
	train_data['drug1_adj'] = torch.concat([chem_adj_A_train, chem_adj_B_train], axis = 0)
	test_data['drug1_adj'] = chem_adj_A_test
	#
	train_data['drug2_adj'] = torch.concat([chem_adj_B_train, chem_adj_A_train], axis = 0)
	test_data['drug2_adj'] = chem_adj_B_test
	#
	train_data['GENE_A'] = torch.concat([gene_A_train, gene_B_train], axis = 0)
	test_data['GENE_A'] = gene_A_test
	#
	train_data['GENE_B'] = torch.concat([gene_B_train, gene_A_train], axis = 0)
	test_data['GENE_B'] = gene_B_test
	#
	train_data['TARGET_A'] = torch.concat([target_A_train, target_B_train], axis = 0)
	test_data['TARGET_A'] = target_A_test
	#
	train_data['TARGET_B'] = torch.concat([target_B_train, target_A_train], axis = 0)
	test_data['TARGET_B'] = target_B_test
	#   #
	train_data['cell_BASAL'] = torch.concat((cell_basal_train, cell_basal_train), axis=0)
	test_data['cell_BASAL'] = cell_basal_test
	##
	train_data['cell'] = torch.concat((cell_train, cell_train), axis=0)
	test_data['cell'] = cell_test
	#            
	train_data['y'] = torch.concat((syn_train, syn_train), axis=0)
	test_data['y'] = syn_test
	#
	print(train_data['drug1_feat'].shape, flush=True)
	print(test_data['drug1_feat'].shape, flush=True)
	return train_data, test_data, ABCS_train, ABCS_test






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


				def plot_loss(train_loss, train_max, train_min,  valid_loss, valid_max, valid_min, path, plotname):
					# fig = plt.figure(figsize=(10,8))
					fig, ax = plt.subplots(figsize = (10,8))
					ax.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss', color = 'Blue')
					ax.fill_between(range(1,len(train_loss)+1), train_min, train_max, alpha = 0.3, edgecolor = 'Blue', facecolor = 'Blue' )
					ax.plot(range(1,len(valid_loss)+1),valid_loss, label='Validation Loss', color= 'Red' )
					ax.fill_between(range(1,len(valid_loss)+1), valid_min, valid_max, alpha = 0.3, edgecolor ='Red', facecolor =  'Red')
					ax.xlabel('epochs')
					ax.ylabel('loss')
					ax.ylim(0, math.ceil(max(train_loss+valid_loss))) # 일정한 scale
					ax.xlim(0, len(train_loss)+1) # 일정한 scale
					ax.grid(True)
					ax.legend()
					ax.tight_layout()
					fig.savefig('{}/{}.loss_plot.png'.format(path, plotname), bbox_inches = 'tight')
					plt.close()




seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)



# gcn_drug1_F, gcn_drug2_F, gcn_drug1_ADJ, gcn_drug2_ADJ, gcn_exp_A, gcn_exp_B, gcn_tgt_A, gcn_tgt_B, gcn_adj, gcn_adj_weight, syn_ans, cell_info
norm = 'tanh_norm'

# CV_0
train_data_0, test_data_0, abcs_train_0, abcs_test_0 = prepare_data_GCN(0, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, 
MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, 
cell_one_hot, MY_syn_RE2, norm)

# CV_1
train_data_1, test_data_1, abcs_train_1, abcs_test_1 = prepare_data_GCN(1, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, 
MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, 
cell_one_hot, MY_syn_RE2, norm)

# CV_2
train_data_2, test_data_2, abcs_train_2, abcs_test_2 = prepare_data_GCN(2, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, 
MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, 
cell_one_hot, MY_syn_RE2, norm)

# CV_3
train_data_3, test_data_3, abcs_train_3, abcs_test_3 = prepare_data_GCN(3, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, 
MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, 
cell_one_hot, MY_syn_RE2, norm)

# CV_4
train_data_4, test_data_4, abcs_train_4, abcs_test_4 = prepare_data_GCN(4, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, 
MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, 
cell_one_hot, MY_syn_RE2, norm)



# WEIGHT 
def get_loss_weight(CV) :
	train_data = globals()['train_data_'+str(CV)]
	ys = train_data['y'].squeeze().tolist()
	min_s = np.amin(ys)
	loss_weight = np.log(train_data['y'] - min_s + np.e)
	return loss_weight


LOSS_WEIGHT_0 = get_loss_weight(0)
LOSS_WEIGHT_1 = get_loss_weight(1)
LOSS_WEIGHT_2 = get_loss_weight(2)
LOSS_WEIGHT_3 = get_loss_weight(3)
LOSS_WEIGHT_4 = get_loss_weight(4)


JY_IDX_WEIGHT_T = torch.Tensor(JY_IDX_WEIGHT).view(1,-1)






# DATA check  
def make_merged_data(CV) :
	train_data = globals()['train_data_'+str(CV)]
	test_data = globals()['test_data_'+str(CV)]
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
	return T_train, T_test





# CV 0 
T_train_0, T_test_0 = make_merged_data(0)
RAY_train_0 = ray.put(T_train_0)
RAY_test_0 = ray.put(T_test_0)
RAY_loss_weight_0 = ray.put(LOSS_WEIGHT_0)


# CV 1
T_train_1, T_test_1 = make_merged_data(1)
RAY_train_1 = ray.put(T_train_1)
RAY_test_1 = ray.put(T_test_1)
RAY_loss_weight_1 = ray.put(LOSS_WEIGHT_1)


# CV 2 
T_train_2,  T_test_2 = make_merged_data(2)
RAY_train_2 = ray.put(T_train_2)
RAY_test_2 = ray.put(T_test_2)
RAY_loss_weight_2 = ray.put(LOSS_WEIGHT_2)


# CV 3
T_train_3, T_test_3 = make_merged_data(3)
RAY_train_3 = ray.put(T_train_3)
RAY_test_3 = ray.put(T_test_3)
RAY_loss_weight_3 = ray.put(LOSS_WEIGHT_3)


# CV 4
T_train_4, T_test_4 = make_merged_data(4)
RAY_train_4 = ray.put(T_train_4)
RAY_test_4 = ray.put(T_test_4)
RAY_loss_weight_4 = ray.put(LOSS_WEIGHT_4)





							# DATA check  
							def make_merged_data(CV) :
								train_data = globals()['train_data_'+str(CV)]
								test_data = globals()['test_data_'+str(CV)]
								#
								T_train = DATASET_GCN_W_FT(
									torch.Tensor(train_data['drug1_feat'])[0:256], torch.Tensor(train_data['drug2_feat'])[0:256], 
									torch.Tensor(train_data['drug1_adj'])[0:256], torch.Tensor(train_data['drug2_adj'])[0:256],
									torch.Tensor(train_data['GENE_A'])[0:256], torch.Tensor(train_data['GENE_B'])[0:256], 
									torch.Tensor(train_data['TARGET_A'])[0:256], torch.Tensor(train_data['TARGET_B'])[0:256], torch.Tensor(train_data['cell_BASAL'])[0:256], 
									JY_ADJ_IDX, JY_IDX_WEIGHT_T, 
									train_data['cell'][0:256].float(),
									torch.Tensor(train_data['y'])[0:256]
									)
								#
								T_test = DATASET_GCN_W_FT(
									torch.Tensor(test_data['drug1_feat'])[0:256], torch.Tensor(test_data['drug2_feat'])[0:256], 
									torch.Tensor(test_data['drug1_adj'])[0:256], torch.Tensor(test_data['drug2_adj'])[0:256],
									torch.Tensor(test_data['GENE_A'])[0:256], torch.Tensor(test_data['GENE_B'])[0:256], 
									torch.Tensor(test_data['TARGET_A'])[0:256], torch.Tensor(test_data['TARGET_B'])[0:256], torch.Tensor(test_data['cell_BASAL'])[0:256], 
									JY_ADJ_IDX, JY_IDX_WEIGHT_T, 
									test_data['cell'][0:256].float(),
									torch.Tensor(test_data['y'])[0:256]
									)
								#
								return T_train, T_test





							# CV 0 
							T_train_0, T_test_0 = make_merged_data(0)
							RAY_train_0 = ray.put(T_train_0)
							RAY_test_0 = ray.put(T_test_0)
							RAY_loss_weight_0 = ray.put(torch.Tensor([1 for a in range(256)]).view(256,-1))


							# CV 1
							T_train_1, T_test_1 = make_merged_data(1)
							RAY_train_1 = ray.put(T_train_1)
							RAY_test_1 = ray.put(T_test_1)
							RAY_loss_weight_1 = ray.put(torch.Tensor([1 for a in range(256)]).view(256,-1))


							# CV 2 
							T_train_2,  T_test_2 = make_merged_data(2)
							RAY_train_2 = ray.put(T_train_2)
							RAY_test_2 = ray.put(T_test_2)
							RAY_loss_weight_2 = ray.put(torch.Tensor([1 for a in range(256)]).view(256,-1))

							# CV 3
							T_train_3, T_test_3 = make_merged_data(3)
							RAY_train_3 = ray.put(T_train_3)
							RAY_test_3 = ray.put(T_test_3)
							RAY_loss_weight_3 = ray.put(torch.Tensor([1 for a in range(256)]).view(256,-1))


							# CV 4
							T_train_4, T_test_4 = make_merged_data(4)
							RAY_train_4 = ray.put(T_train_4)
							RAY_test_4 = ray.put(T_test_4)
							RAY_loss_weight_4 = ray.put(torch.Tensor([1 for a in range(256)]).view(256,-1))










def inner_train( LOADER_DICT, THIS_MODEL, THIS_OPTIMIZER , use_cuda=False) :
	criterion = weighted_mse_loss
	THIS_MODEL.train()
	running_loss = 0
	last_loss = 0 
	#
	ans_list = []
	pred_list = []
	batch_cut_weight = LOADER_DICT['loss_weight']
	#
	for batch_idx_t, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y) in enumerate(LOADER_DICT['train']) :
		expA = expA.view(-1,3)#### 다른점 
		expB = expB.view(-1,3)#### 다른점 
		adj_w = adj_w.squeeze()
		# move to GPU
		if use_cuda:
			drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell = drug1_f.cuda(), drug2_f.cuda(), drug1_a.cuda(), drug2_a.cuda(), expA.cuda(), expB.cuda(), adj.cuda(), adj_w.cuda(), y.cuda(), cell.cuda() 
		## find the loss and update the model parameters accordingly
		# clear the gradients of all optimized variables
		THIS_OPTIMIZER.zero_grad()
		output = THIS_MODEL(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y)
		wc = torch.Tensor(batch_cut_weight[batch_idx_t]).view(-1,1)
		if torch.cuda.is_available():
			wc = wc.cuda()
		loss = weighted_mse_loss(output, y, wc ) # weight 더해주기 
		loss.backward()
		THIS_OPTIMIZER.step()
		#
		running_loss = running_loss + loss.item()
		pred_list = pred_list + output.squeeze().tolist()
		ans_list = ans_list + y.squeeze().tolist()
	#
	last_loss = running_loss / (batch_idx_t+1)
	train_sc, _ = stats.spearmanr(pred_list, ans_list)
	train_pc, _ = stats.pearsonr(pred_list, ans_list)
	return last_loss, train_pc, train_sc, THIS_MODEL, THIS_OPTIMIZER     


def inner_val( LOADER_DICT, THIS_MODEL , use_cuda = False) :
	THIS_MODEL.eval()
	#
	running_loss = 0
	last_loss = 0 
	#
	ans_list = []
	pred_list = []
	with torch.no_grad() :
		for batch_idx_v, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y) in enumerate(LOADER_DICT['eval']) :
			expA = expA.view(-1,3)#### 다른점 
			expB = expB.view(-1,3)#### 다른점 
			adj_w = adj_w.squeeze()
			# move to GPU
			if use_cuda:
				drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell = drug1_f.cuda(), drug2_f.cuda(), drug1_a.cuda(), drug2_a.cuda(), expA.cuda(), expB.cuda(), adj.cuda(), adj_w.cuda(), y.cuda(), cell.cuda()
			## update the average validation loss
			output = THIS_MODEL(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y)
			MSE = torch.nn.MSELoss()
			loss = MSE(output, y) # train 이 아니라서 weight 안넣어줌. 그냥 nn.MSE 넣어주기 
			# update average validation loss 
			running_loss = running_loss + loss.item()
			pred_list = pred_list + output.squeeze().tolist()
			ans_list = ans_list + y.squeeze().tolist()
		#
	last_loss = running_loss / (batch_idx_v+1)
	val_sc, _ = stats.spearmanr(pred_list, ans_list)
	val_pc, _ = stats.pearsonr(pred_list, ans_list)
	return last_loss, val_pc, val_sc, THIS_MODEL     






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



def pickling(this, where, name) :
	with open(os.path.join(where, name), 'wb') as pp :
		pickle.dump(this, pp)




def LEARN_MODEL(PRJ_PATH, my_config, n_epoch, use_cuda = True):
	n_epochs = 1000
	criterion = weighted_mse_loss
	use_cuda = use_cuda 
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
	#
	print('CV0')
	# CV 0 
	CV_0_train = ray.get(RAY_train_0)
	CV_0_val = ray.get(RAY_test_0)
	CV_0_loss_weight = ray.get(RAY_loss_weight_0)
	CV_0_batch_cut_weight = [CV_0_loss_weight[i:i+my_config["config/batch_size"].item()] for i in range(0,len(CV_0_loss_weight), my_config["config/batch_size"].item())]
	#
	CV_0_loaders = {
			'train' : torch.utils.data.DataLoader(CV_0_train, batch_size = my_config["config/batch_size"].item(), collate_fn = graph_collate_fn, shuffle =False, num_workers=my_config['config/n_workers'].item()), # 
			'eval' : torch.utils.data.DataLoader(CV_0_val, batch_size = my_config["config/batch_size"].item(), collate_fn = graph_collate_fn, shuffle =False, num_workers=my_config['config/n_workers'].item()), # 
			'loss_weight' : CV_0_batch_cut_weight
	}
	#
	#  
	CV_0_MODEL = MY_expGCN_parallel_model(
			G_chem_layer, CV_0_train.gcn_drug1_F.shape[-1] , G_chem_hdim,      # G_layer_chem, G_indim_chem, G_hiddim_chem, 
			G_exp_layer, 3 , G_exp_hdim,      # G_layer_exp, G_indim_exp, G_hiddim_exp, 
			dsn1_layers, dsn2_layers, snp_layers,      # drug 1 layers, drug 2 layers, merged layer, 
			len(set(A_B_C_S_SET_SM.DrugCombCCLE)), 1,      # cell_dim ,out_dim,
			inDrop, Drop      # inDrop, drop
			)
	# 
	# CV 1 
	CV_1_train = ray.get(RAY_train_1)
	CV_1_val = ray.get(RAY_test_1)
	CV_1_loss_weight = ray.get(RAY_loss_weight_1)
	CV_1_batch_cut_weight = [CV_1_loss_weight[i:i+my_config["config/batch_size"].item()] for i in range(0,len(CV_1_loss_weight), my_config["config/batch_size"].item())]
	#
	CV_1_loaders = {
			'train' : torch.utils.data.DataLoader(CV_1_train, batch_size = my_config["config/batch_size"].item(), collate_fn = graph_collate_fn, shuffle =False, num_workers=my_config['config/n_workers'].item()), # 
			'eval' : torch.utils.data.DataLoader(CV_1_val, batch_size = my_config["config/batch_size"].item(), collate_fn = graph_collate_fn, shuffle =False, num_workers=my_config['config/n_workers'].item()), # 
			'loss_weight' : CV_1_batch_cut_weight
	}
	#
	print('CV1')
	#  
	CV_1_MODEL = MY_expGCN_parallel_model(
			G_chem_layer, CV_1_train.gcn_drug1_F.shape[-1] , G_chem_hdim,      # G_layer_chem, G_indim_chem, G_hiddim_chem, 
			G_exp_layer, 3 , G_exp_hdim,      # G_layer_exp, G_indim_exp, G_hiddim_exp, 
			dsn1_layers, dsn2_layers, snp_layers,      # drug 1 layers, drug 2 layers, merged layer, 
			len(set(A_B_C_S_SET_SM.DrugCombCCLE)), 1,      # cell_dim ,out_dim,
			inDrop, Drop      # inDrop, drop
			)
	# 
	print('CV2')
	# CV 2 
	CV_2_train = ray.get(RAY_train_2)
	CV_2_val = ray.get(RAY_test_2)
	CV_2_loss_weight = ray.get(RAY_loss_weight_2)
	CV_2_batch_cut_weight = [CV_2_loss_weight[i:i+my_config["config/batch_size"].item()] for i in range(0,len(CV_2_loss_weight), my_config["config/batch_size"].item())]
	#
	CV_2_loaders = {
		'train' : torch.utils.data.DataLoader(CV_2_train, batch_size = my_config["config/batch_size"].item(), collate_fn = graph_collate_fn, shuffle =False, num_workers=my_config['config/n_workers'].item()), # 
		'eval' : torch.utils.data.DataLoader(CV_2_val, batch_size = my_config["config/batch_size"].item(), collate_fn = graph_collate_fn, shuffle =False, num_workers=my_config['config/n_workers'].item()), # 
		'loss_weight' : CV_2_batch_cut_weight
	}
	#
	#  
	CV_2_MODEL = MY_expGCN_parallel_model(
			G_chem_layer, CV_2_train.gcn_drug1_F.shape[-1] , G_chem_hdim,      # G_layer_chem, G_indim_chem, G_hiddim_chem, 
			G_exp_layer, 3 , G_exp_hdim,      # G_layer_exp, G_indim_exp, G_hiddim_exp, 
			dsn1_layers, dsn2_layers, snp_layers,      # drug 1 layers, drug 2 layers, merged layer, 
			len(set(A_B_C_S_SET_SM.DrugCombCCLE)), 1,      # cell_dim ,out_dim,
			inDrop, Drop      # inDrop, drop
			)
	# 
	print('CV3')
	# CV 3
	CV_3_train = ray.get(RAY_train_3)
	CV_3_val = ray.get(RAY_test_3)
	CV_3_loss_weight = ray.get(RAY_loss_weight_3)
	CV_3_batch_cut_weight = [CV_3_loss_weight[i:i+my_config["config/batch_size"].item()] for i in range(0,len(CV_3_loss_weight), my_config["config/batch_size"].item())]
	#
	CV_3_loaders = {
		'train' : torch.utils.data.DataLoader(CV_3_train, batch_size = my_config["config/batch_size"].item(), collate_fn = graph_collate_fn, shuffle =False, num_workers=my_config['config/n_workers'].item()), # 
		'eval' : torch.utils.data.DataLoader(CV_3_val, batch_size = my_config["config/batch_size"].item(), collate_fn = graph_collate_fn, shuffle =False, num_workers=my_config['config/n_workers'].item()), # 
		'loss_weight' : CV_3_batch_cut_weight
	}
	#
	#  
	CV_3_MODEL = MY_expGCN_parallel_model(
		G_chem_layer, CV_3_train.gcn_drug1_F.shape[-1] , G_chem_hdim,      # G_layer_chem, G_indim_chem, G_hiddim_chem, 
		G_exp_layer, 3 , G_exp_hdim,      # G_layer_exp, G_indim_exp, G_hiddim_exp, 
		dsn1_layers, dsn2_layers, snp_layers,      # drug 1 layers, drug 2 layers, merged layer, 
		len(set(A_B_C_S_SET_SM.DrugCombCCLE)), 1,      # cell_dim ,out_dim,
		inDrop, Drop      # inDrop, drop
		)
	# 
	print('CV4')
	# CV 4
	CV_4_train = ray.get(RAY_train_4)
	CV_4_val = ray.get(RAY_test_4)
	CV_4_loss_weight = ray.get(RAY_loss_weight_4)
	CV_4_batch_cut_weight = [CV_4_loss_weight[i:i+my_config["config/batch_size"].item()] for i in range(0,len(CV_4_loss_weight), my_config["config/batch_size"].item())]
	#
	CV_4_loaders = {
		'train' : torch.utils.data.DataLoader(CV_4_train, batch_size = my_config["config/batch_size"].item(), collate_fn = graph_collate_fn, shuffle =False, num_workers=my_config['config/n_workers'].item()),# 
		'eval' : torch.utils.data.DataLoader(CV_4_val, batch_size = my_config["config/batch_size"].item(), collate_fn = graph_collate_fn, shuffle =False, num_workers=my_config['config/n_workers'].item()), # 
		'loss_weight' : CV_4_batch_cut_weight
	}
	#
	#  
	CV_4_MODEL = MY_expGCN_parallel_model(
		G_chem_layer, CV_4_train.gcn_drug1_F.shape[-1] , G_chem_hdim,      # G_layer_chem, G_indim_chem, G_hiddim_chem, 
		G_exp_layer, 3 , G_exp_hdim,      # G_layer_exp, G_indim_exp, G_hiddim_exp, 
		dsn1_layers, dsn2_layers, snp_layers,      # drug 1 layers, drug 2 layers, merged layer, 
		len(set(A_B_C_S_SET_SM.DrugCombCCLE)), 1,      # cell_dim ,out_dim,
		inDrop, Drop      # inDrop, drop
		)
	# 
	#
	if torch.cuda.is_available():
		CV_0_MODEL = CV_0_MODEL.cuda()
		CV_1_MODEL = CV_1_MODEL.cuda()
		CV_2_MODEL = CV_2_MODEL.cuda()
		CV_3_MODEL = CV_3_MODEL.cuda()
		CV_4_MODEL = CV_4_MODEL.cuda()
		if torch.cuda.device_count() > 1 :
			CV_0_MODEL = torch.nn.DataParallel(CV_0_MODEL)
			CV_1_MODEL = torch.nn.DataParallel(CV_1_MODEL)
			CV_2_MODEL = torch.nn.DataParallel(CV_2_MODEL)
			CV_3_MODEL = torch.nn.DataParallel(CV_3_MODEL)
			CV_4_MODEL = torch.nn.DataParallel(CV_4_MODEL)
	#       
	CV_0_optimizer = torch.optim.Adam(CV_0_MODEL.parameters(), lr = my_config["config/lr"].item() )
	CV_1_optimizer = torch.optim.Adam(CV_1_MODEL.parameters(), lr = my_config["config/lr"].item() )
	CV_2_optimizer = torch.optim.Adam(CV_2_MODEL.parameters(), lr = my_config["config/lr"].item() )
	CV_3_optimizer = torch.optim.Adam(CV_3_MODEL.parameters(), lr = my_config["config/lr"].item() )
	CV_4_optimizer = torch.optim.Adam(CV_4_MODEL.parameters(), lr = my_config["config/lr"].item() )
	#
	#
	key_list = ['CV_0','CV_1','CV_2','CV_3','CV_4']
	train_loss_all = {}
	valid_loss_all = {}
	train_pearson_corr_all = {}
	train_spearman_corr_all = {}
	val_pearson_corr_all = {}
	val_spearman_corr_all = {}
	#
	for key in key_list :
		train_loss_all[key] = []
		valid_loss_all[key] = []
		train_pearson_corr_all[key]=[]
		train_spearman_corr_all[key]=[]
		val_pearson_corr_all[key] = []
		val_spearman_corr_all[key] = []
	#
	#
	#
	for epoch in range(n_epochs):
		now = datetime.now()
		print(now)
		train_loss = 0.0
		valid_loss = 0.0
		#
		###################
		# train the model #
		###################
		cv_0_t_loss, cv_0_t_pc, cv_0_t_sc, CV_0_MODEL, CV_0_optimizer  = inner_train(CV_0_loaders, CV_0_MODEL, CV_0_optimizer, use_cuda)
		train_loss_all['CV_0'].append(cv_0_t_loss)
		train_pearson_corr_all['CV_0'].append(cv_0_t_pc)
		train_spearman_corr_all['CV_0'].append(cv_0_t_sc)	
		#
		cv_1_t_loss, cv_1_t_pc, cv_1_t_sc, CV_1_MODEL, CV_1_optimizer  = inner_train(CV_1_loaders, CV_1_MODEL, CV_1_optimizer, use_cuda)
		train_loss_all['CV_1'].append(cv_1_t_loss)
		train_pearson_corr_all['CV_1'].append(cv_1_t_pc)
		train_spearman_corr_all['CV_1'].append(cv_1_t_sc)
		# 
		cv_2_t_loss, cv_2_t_pc, cv_2_t_sc, CV_2_MODEL, CV_2_optimizer  = inner_train(CV_2_loaders, CV_2_MODEL, CV_2_optimizer, use_cuda)
		train_loss_all['CV_2'].append(cv_2_t_loss)
		train_pearson_corr_all['CV_2'].append(cv_2_t_pc)
		train_spearman_corr_all['CV_2'].append(cv_2_t_sc)
		# 
		cv_3_t_loss, cv_3_t_pc, cv_3_t_sc, CV_3_MODEL, CV_3_optimizer  = inner_train(CV_3_loaders, CV_3_MODEL, CV_3_optimizer, use_cuda)
		train_loss_all['CV_3'].append(cv_3_t_loss)
		train_pearson_corr_all['CV_3'].append(cv_3_t_pc)
		train_spearman_corr_all['CV_3'].append(cv_3_t_sc)
		# 
		cv_4_t_loss, cv_4_t_pc, cv_4_t_sc, CV_4_MODEL, CV_4_optimizer  = inner_train(CV_4_loaders, CV_4_MODEL, CV_4_optimizer, use_cuda)
		train_loss_all['CV_4'].append(cv_4_t_loss)
		train_pearson_corr_all['CV_4'].append(cv_4_t_pc)
		train_spearman_corr_all['CV_4'].append(cv_4_t_sc)
		# 
		#
		######################    
		# validate the model #
		######################
		cv_0_v_loss, cv_0_v_pc, cv_0_v_sc, CV_0_MODEL  = inner_val(CV_0_loaders, CV_0_MODEL, use_cuda)
		valid_loss_all['CV_0'].append(cv_0_v_loss)
		val_pearson_corr_all['CV_0'].append(cv_0_v_pc)
		val_spearman_corr_all['CV_0'].append(cv_0_v_sc) 
		#
		cv_1_v_loss, cv_1_v_pc, cv_1_v_sc, CV_1_MODEL  = inner_val(CV_1_loaders, CV_1_MODEL, use_cuda)
		valid_loss_all['CV_1'].append(cv_1_v_loss)
		val_pearson_corr_all['CV_1'].append(cv_1_v_pc)
		val_spearman_corr_all['CV_1'].append(cv_1_v_sc)
		# 
		cv_2_v_loss, cv_2_v_pc, cv_2_v_sc, CV_2_MODEL  = inner_val(CV_2_loaders, CV_2_MODEL, use_cuda)
		valid_loss_all['CV_2'].append(cv_2_v_loss)
		val_pearson_corr_all['CV_2'].append(cv_2_v_pc)
		val_spearman_corr_all['CV_2'].append(cv_2_v_sc)
		# 
		cv_3_v_loss, cv_3_v_pc, cv_3_v_sc, CV_3_MODEL  = inner_val(CV_3_loaders, CV_3_MODEL, use_cuda)
		valid_loss_all['CV_3'].append(cv_3_v_loss)
		val_pearson_corr_all['CV_3'].append(cv_3_v_pc)
		val_spearman_corr_all['CV_3'].append(cv_3_v_sc)
		# 
		cv_4_v_loss, cv_4_v_pc, cv_4_v_sc, CV_4_MODEL  = inner_val(CV_4_loaders, CV_4_MODEL, use_cuda)
		valid_loss_all['CV_4'].append(cv_4_v_loss)
		val_pearson_corr_all['CV_4'].append(cv_4_v_pc)
		val_spearman_corr_all['CV_4'].append(cv_4_v_sc)
		#
		#
		AVG_TRAIN_LOSS = np.mean([cv_0_t_loss, cv_1_t_loss, cv_2_t_loss, cv_3_t_loss, cv_4_t_loss])
		AVG_T_PC = np.mean([cv_0_t_pc, cv_1_t_pc, cv_2_t_pc, cv_3_t_pc, cv_4_t_pc])
		AVG_T_SC = np.mean([cv_0_t_sc, cv_1_t_sc, cv_2_t_sc, cv_3_t_sc, cv_4_t_sc])
		AVG_VAL_LOSS = np.mean([cv_0_v_loss, cv_1_v_loss, cv_2_v_loss, cv_3_v_loss, cv_4_v_loss])
		AVG_V_PC = np.mean([cv_0_v_pc, cv_1_v_pc, cv_2_v_pc, cv_3_v_pc, cv_4_v_pc])
		AVG_V_SC = np.mean([cv_0_v_sc, cv_1_v_sc, cv_2_v_sc, cv_3_v_sc, cv_4_v_sc])
		#
		done = datetime.now()
		time_spent = done-now
		print('epoch:{}/1000, AVGtrainLoss:{:.2f}, AVGtrainPC:{:.2f}, AVGtrainSC:{:.2f}, AVGtestLoss:{:.2f}, AVGtestPC:{:.2f}, AVGtestSC:{:.2f}'.format(
			epoch, AVG_TRAIN_LOSS, AVG_T_PC, AVG_T_SC, AVG_VAL_LOSS, AVG_V_PC, AVG_V_SC
			), flush=True)
		#
		cv_0_path = os.path.join(PRJ_PATH, "CV_0_checkpoint")
		cv_1_path = os.path.join(PRJ_PATH, "CV_1_checkpoint")
		cv_2_path = os.path.join(PRJ_PATH, "CV_2_checkpoint")
		cv_3_path = os.path.join(PRJ_PATH, "CV_3_checkpoint")
		cv_4_path = os.path.join(PRJ_PATH, "CV_4_checkpoint")
		os.makedirs( os.path.join(cv_0_path,'CV{}'.format(epoch)), exist_ok = True)
		os.makedirs( os.path.join(cv_1_path,'CV{}'.format(epoch)), exist_ok = True)
		os.makedirs( os.path.join(cv_2_path,'CV{}'.format(epoch)), exist_ok = True)
		os.makedirs( os.path.join(cv_3_path,'CV{}'.format(epoch)), exist_ok = True)
		os.makedirs( os.path.join(cv_4_path,'CV{}'.format(epoch)), exist_ok = True)
		torch.save((CV_0_MODEL.state_dict(), CV_0_optimizer.state_dict()), os.path.join(cv_0_path,'CV{}'.format(epoch), 'model'))
		torch.save((CV_1_MODEL.state_dict(), CV_1_optimizer.state_dict()), os.path.join(cv_1_path,'CV{}'.format(epoch), 'model'))
		torch.save((CV_2_MODEL.state_dict(), CV_2_optimizer.state_dict()), os.path.join(cv_2_path,'CV{}'.format(epoch), 'model'))
		torch.save((CV_3_MODEL.state_dict(), CV_3_optimizer.state_dict()), os.path.join(cv_3_path,'CV{}'.format(epoch), 'model'))
		torch.save((CV_4_MODEL.state_dict(), CV_4_optimizer.state_dict()), os.path.join(cv_4_path,'CV{}'.format(epoch), 'model'))
		torch.save(CV_0_MODEL.state_dict(), PRJ_PATH+'/CV_0_model.pth')
		torch.save(CV_1_MODEL.state_dict(), PRJ_PATH+'/CV_1_model.pth')
		torch.save(CV_2_MODEL.state_dict(), PRJ_PATH+'/CV_2_model.pth')
		torch.save(CV_3_MODEL.state_dict(), PRJ_PATH+'/CV_3_model.pth')
		torch.save(CV_4_MODEL.state_dict(), PRJ_PATH+'/CV_4_model.pth')
		#
		pickling(train_loss_all, PRJ_PATH, "TRAIN_LOSS")
		pickling(valid_loss_all, PRJ_PATH, "VALID_LOSS")
		pickling(train_pearson_corr_all, PRJ_PATH, "TRAIN_PC")
		pickling(train_spearman_corr_all, PRJ_PATH, "TRAIN_SC")
		pickling(val_pearson_corr_all, PRJ_PATH, "VALID_PC")
		pickling(val_spearman_corr_all, PRJ_PATH, "VALID_SC")
		#
	#
	print("Finished Training")




W_NAME = 'W30'
PRJ_PATH = '/home01/k020a01/02.M3V5/M3V5_W30_349/'
#PRJ_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V5_W30_349/'


model_path = '/home01/k020a01/02.M3V5/M3V5_W21_349_MIS2/'
#model_path = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V5_W21_MIS2_349/'

ANA_DF_CSV = pd.read_csv(os.path.join( model_path ,'RAY_ANA_DF.{}.csv'.format('M3V5_W21_349_MIS2')))

my_config = ANA_DF_CSV[ANA_DF_CSV.trial_id=='28b3f0dc'] # 349 

# LEARN_MODEL (PRJ_PATH, my_config, model_path, model_name, model_num, n_epoch, use_cuda = True)
LEARN_MODEL (PRJ_PATH, my_config, 394 , use_cuda = True)



sbatch gpu1.any M3V5_WORK30.349.py

tail ~/logs/M3V5W30_GPU1_11458.log

tail ~/02.M3V5/M3V5_W30_349/RESULT.G1.txt -n 100



1epoch 에 5분 이상은 에바길래 고쳐줌 128 worker 로 








#########################################
################# GPU ###################
################# GPU ###################
################# GPU ###################
#########################################



import pandas as pd 
import matplotlib.pyplot as plt
from ray.tune import ExperimentAnalysis
from ray.tune import Analysis
import pickle
import math
import torch
import os 
import copy


with open('/home01/k020a01/02.M3V5/M3V5_W30_349/TRAIN_LOSS', 'rb') as f:
	TRAIN_LOSS = pickle.load(f)

with open('/home01/k020a01/02.M3V5/M3V5_W30_349/TRAIN_PC', 'rb') as f:
	TRAIN_PC = pickle.load(f)

with open('/home01/k020a01/02.M3V5/M3V5_W30_349/TRAIN_SC', 'rb') as f:
	TRAIN_SC = pickle.load(f)



with open('/home01/k020a01/02.M3V5/M3V5_W30_349/VALID_LOSS', 'rb') as f:
	VALID_LOSS = pickle.load(f)

with open('/home01/k020a01/02.M3V5/M3V5_W30_349/VALID_PC', 'rb') as f:
	VALID_PC = pickle.load(f)

with open('/home01/k020a01/02.M3V5/M3V5_W30_349/VALID_SC', 'rb') as f:
	VALID_SC = pickle.load(f)


TRAIN_LOSS = pd.DataFrame(TRAIN_LOSS)
TRAIN_PC = pd.DataFrame(TRAIN_PC)
TRAIN_SC = pd.DataFrame(TRAIN_SC)

VALID_LOSS = pd.DataFrame(VALID_LOSS)
VALID_PC = pd.DataFrame(VALID_PC)
VALID_SC = pd.DataFrame(VALID_SC)



################# CPU ###################
################# CPU ###################
################# CPU ###################
################# CPU ###################
################# CPU ###################
################# CPU ###################
################# CPU ###################


PRJ_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V5_W30_349/'


with open('/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V5_W30_349/TRAIN_LOSS', 'rb') as f:
	TRAIN_LOSS = pickle.load(f)

with open('/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V5_W30_349/TRAIN_PC', 'rb') as f:
	TRAIN_PC = pickle.load(f)

with open('/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V5_W30_349/TRAIN_SC', 'rb') as f:
	TRAIN_SC = pickle.load(f)



with open('/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V5_W30_349/VALID_LOSS', 'rb') as f:
	VALID_LOSS = pickle.load(f)

with open('/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V5_W30_349/VALID_PC', 'rb') as f:
	VALID_PC = pickle.load(f)

with open('/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V5_W30_349/VALID_SC', 'rb') as f:
	VALID_SC = pickle.load(f)


TRAIN_LOSS = pd.DataFrame(TRAIN_LOSS)
TRAIN_PC = pd.DataFrame(TRAIN_PC)
TRAIN_SC = pd.DataFrame(TRAIN_SC)

VALID_LOSS = pd.DataFrame(VALID_LOSS)
VALID_PC = pd.DataFrame(VALID_PC)
VALID_SC = pd.DataFrame(VALID_SC)


epoch = 394


round(np.mean(TRAIN_LOSS.loc[epoch]), 4)
round(np.std(TRAIN_LOSS.loc[epoch]), 4)

round(np.mean(TRAIN_PC.loc[epoch]), 4)
round(np.std(TRAIN_PC.loc[epoch]), 4)

round(np.mean(TRAIN_SC.loc[epoch]), 4)
round(np.std(TRAIN_SC.loc[epoch]), 4)

round(np.mean(VALID_LOSS.loc[epoch]), 4)
round(np.std(VALID_LOSS.loc[epoch]), 4)

round(np.mean(VALID_PC.loc[epoch]), 4)
round(np.std(VALID_PC.loc[epoch]), 4)

round(np.mean(VALID_SC.loc[epoch]), 4)
round(np.std(VALID_SC.loc[epoch]), 4)
























