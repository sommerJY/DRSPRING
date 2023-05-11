
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


ray.init()

NETWORK_PATH = '/home01/k020a01/01.Data/HumanNet/'
LINCS_PATH = '/home01/k020a01/01.Data/LINCS/'
DC_PATH = '/home01/k020a01/01.Data/DrugComb/'


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
WORK_DATE = '23.04.16' # 349
MISS_NAME = 'MIS2'

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



# A_B_C_S SET filter check
WORK_NAME = 'WORK_20v1' # 349


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
data_nodup_df = pd.DataFrame({'setset' : data_no_dup.tolist(), 'cell' : data_no_dup_cells })
data_nodup_df2 = data_nodup_df.sort_values('cell')
data_nodup_df2 = data_nodup_df2.reset_index(drop =True)

grouped_df = data_nodup_df2.groupby('cell')


TrainVal_list = []; Test_list =[]

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



check_index_list = TrainVal_list+Test_list
len(TrainVal_list) # 51182
len(Test_list) # 51182
len(TrainVal_list)+len(Test_list) # 51182

set(data_nodup_df2.index)-set(check_index_list)

len(data_nodup_df2.loc[Test_list]['setset'])
len(set(data_nodup_df2.loc[Test_list]['setset']))

len(data_nodup_df2.loc[TrainVal_list]['setset'])
len(set(data_nodup_df2.loc[TrainVal_list]['setset']))




# CV split 
data_nodup_df3 = data_nodup_df2.loc[TrainVal_list]
data_nodup_df3 = data_nodup_df3.reset_index(drop=True)

grouped_df2 = data_nodup_df3.groupby('cell')

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
	print(i)
	print(len(g2))
	len(set(sum([list(ii.setset) for ii in res],[])))
	len(set(sum([list(ii.setset) for ii in res],[]))) == len(sum([list(ii.setset) for ii in res],[]))
	CV_1_list = CV_1_list + res[0].index.tolist()
	CV_2_list = CV_2_list + res[1].index.tolist()
	CV_3_list = CV_3_list + res[2].index.tolist()
	CV_4_list = CV_4_list + res[3].index.tolist()
	CV_5_list = CV_5_list + res[4].index.tolist()



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

len(CV_5_list + CV_1_list + CV_2_list + CV_3_list + CV_4_list)
len(set(CV_5_list + CV_1_list + CV_2_list + CV_3_list + CV_4_list ))


len(data_nodup_df3.loc[CV_1_list]['setset'])
len(set(data_nodup_df3.loc[CV_1_list]['setset']))

len(data_nodup_df3.loc[CV_2_list]['setset'])
len(set(data_nodup_df3.loc[CV_2_list]['setset']))

len(data_nodup_df3.loc[CV_3_list]['setset'])
len(set(data_nodup_df3.loc[CV_3_list]['setset']))

len(data_nodup_df3.loc[CV_4_list]['setset'])
len(set(data_nodup_df3.loc[CV_4_list]['setset']))

len(data_nodup_df3.loc[CV_5_list]['setset'])
len(set(data_nodup_df3.loc[CV_5_list]['setset']))

len(data_nodup_df3.loc[CV_5_list + CV_1_list + CV_2_list + CV_3_list + CV_4_list]['setset'])
len(set(data_nodup_df3.loc[CV_5_list + CV_1_list + CV_2_list + CV_3_list + CV_4_list]['setset']))



# use just index 
# normalizing 과정에 대한 고민 필요 ... GCN 이니까 상관 없지 않을까
def prepare_data_GCN(CV_num, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, 
MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, 
cell_one_hot, MY_syn_RE2, norm ) : 
	#        
	# CV_num = 0
	train_key = 'CV{}_train'.format(CV_num)
	val_key = 'CV{}_val'.format(CV_num)
	test_key = 'CV{}_test'.format(CV_num)
	#
	train_no_dup = data_nodup_df3.loc[CV_ND_INDS[train_key]] # train val df 
	val_no_dup = data_nodup_df3.loc[CV_ND_INDS[val_key]] # train val df 
	tv_no_dup = pd.concat([train_no_dup, val_no_dup])
	test_no_dup = data_nodup_df2.loc[CV_ND_INDS[test_key]] # from test included df 
	#
	# ABCS_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(train_no_dup.setset)]
	# ABCS_val = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(val_no_dup.setset)]
	ABCS_tv = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(tv_no_dup.setset)]
	ABCS_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(test_no_dup.setset)]
	#
	#train_ind = list(ABCS_train.index)
	#val_ind = list(ABCS_val.index)
	tv_ind = list(ABCS_tv.index)
	random.shuffle(tv_ind)
	test_ind = list(ABCS_test.index)
	# 
	chem_feat_A_tv = MY_chem_A_feat_RE2[tv_ind]; chem_feat_A_test = MY_chem_A_feat_RE2[test_ind]
	chem_feat_B_tv = MY_chem_B_feat_RE2[tv_ind]; chem_feat_B_test = MY_chem_B_feat_RE2[test_ind]
	chem_adj_A_tv = MY_chem_A_adj_RE2[tv_ind]; chem_adj_A_test = MY_chem_A_adj_RE2[test_ind]
	chem_adj_B_tv = MY_chem_B_adj_RE2[tv_ind]; chem_adj_B_test = MY_chem_B_adj_RE2[test_ind]
	gene_A_tv = MY_g_EXP_A_RE2[tv_ind];  gene_A_test = MY_g_EXP_A_RE2[test_ind]
	gene_B_tv = MY_g_EXP_B_RE2[tv_ind];  gene_B_test = MY_g_EXP_B_RE2[test_ind]
	target_A_tv = MY_Target_A2[tv_ind];  target_A_test = MY_Target_A2[test_ind]
	target_B_tv = MY_Target_B2[tv_ind];  target_B_test = MY_Target_B2[test_ind]
	cell_basal_tv = MY_CellBase_RE2[tv_ind];  cell_basal_test = MY_CellBase_RE2[test_ind]
	cell_tv = cell_one_hot[tv_ind];  cell_test = cell_one_hot[test_ind]
	syn_tv = MY_syn_RE2[tv_ind];  syn_test = MY_syn_RE2[test_ind]
	#
	tv_data = {}
	test_data = {}
	#
	tv_data['drug1_feat'] = torch.concat([chem_feat_A_tv, chem_feat_B_tv], axis = 0)
	test_data['drug1_feat'] = chem_feat_A_test
	#
	tv_data['drug2_feat'] = torch.concat([chem_feat_B_tv, chem_feat_A_tv], axis = 0)
	test_data['drug2_feat'] = chem_feat_B_test
	#
	tv_data['drug1_adj'] = torch.concat([chem_adj_A_tv, chem_adj_B_tv], axis = 0)
	test_data['drug1_adj'] = chem_adj_A_test
	#
	tv_data['drug2_adj'] = torch.concat([chem_adj_B_tv, chem_adj_A_tv], axis = 0)
	test_data['drug2_adj'] = chem_adj_B_test
	#
	tv_data['GENE_A'] = torch.concat([gene_A_tv, gene_B_tv], axis = 0)
	test_data['GENE_A'] = gene_A_test
	#
	tv_data['GENE_B'] = torch.concat([gene_B_tv, gene_A_tv], axis = 0)
	test_data['GENE_B'] = gene_B_test
	#
	tv_data['TARGET_A'] = torch.concat([target_A_tv, target_B_tv], axis = 0)
	test_data['TARGET_A'] = target_A_test
	#
	tv_data['TARGET_B'] = torch.concat([target_B_tv, target_A_tv], axis = 0)
	test_data['TARGET_B'] = target_B_test
	#   #
	tv_data['cell_BASAL'] = torch.concat((cell_basal_tv, cell_basal_tv), axis=0)
	test_data['cell_BASAL'] = cell_basal_test
	##
	tv_data['cell'] = torch.concat((cell_tv, cell_tv), axis=0)
	test_data['cell'] = cell_test
	#            
	tv_data['y'] = torch.concat((syn_tv, syn_tv), axis=0)
	test_data['y'] = syn_test
	#
	print(tv_data['drug1_feat'].shape, flush=True)
	print(test_data['drug1_feat'].shape, flush=True)
	return tv_data, test_data





# just to check if it's going well 
CV_num = 0
train_key = 'CV{}_train'.format(CV_num)
val_key = 'CV{}_val'.format(CV_num)
test_key = 'CV{}_test'.format(CV_num)
#
train_no_dup = data_nodup_df3.loc[CV_ND_INDS[train_key]] # train val df 
val_no_dup = data_nodup_df3.loc[CV_ND_INDS[val_key]] # train val df 
tv_no_dup = pd.concat([train_no_dup, val_no_dup])
test_no_dup = data_nodup_df2.loc[CV_ND_INDS[test_key]] # from test included df 
#
ABCS_tv_ch = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(list(train_no_dup.setset) + list(val_no_dup.setset))]
ABCS_test_ch = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(test_no_dup.setset)]

ch1 = ABCS_tv_ch.reset_index(drop=True)
ch1_ind = list(ch1.index)
aa = list(ch1['ROW_CAN_SMILES'])
bb = list(ch1['COL_CAN_SMILES'])
cc = list(ch1['DrugCombCCLE'])
SM_C_CHECK_1 = [aa[i] + '___' + bb[i]+ '___' + cc[i] for i in ch1_ind]
SM_C_CHECK_2 = [bb[i] + '___' + aa[i]+ '___' + cc[i] for i in ch1_ind]
ch1_checklist = list(set(SM_C_CHECK_1+SM_C_CHECK_2)) # 65794 -> 65361

ch2 = ABCS_test_ch.reset_index(drop=True)
ch2_ind = list(ch2.index)
aa = list(ch2['ROW_CAN_SMILES'])
bb = list(ch2['COL_CAN_SMILES'])
cc = list(ch2['DrugCombCCLE'])
SM_C_CHECK_1 = [aa[i] + '___' + bb[i]+ '___' + cc[i] for i in ch2_ind]
SM_C_CHECK_2 = [bb[i] + '___' + aa[i]+ '___' + cc[i] for i in ch2_ind]
ch2_checklist = list(set(SM_C_CHECK_1+SM_C_CHECK_2)) #20733 -> 20526

print('trainval : test')
print(len(set(ch1_checklist)))
print(len(set(ch2_checklist)))
print(len(set(ch2_checklist) - set(ch1_checklist))) # 20514

#sample_prob1 = 'CC1=C(C=C(C=C1)C(=O)NC2=CC(=CC(=C2)C(F)(F)F)N3C=C(N=C3)C)NC4=NC=CC(=N4)C5=CN=CC=C5___C1=NC2=C(N=C(N=C2N1C3C(C(C(O3)CO)O)O)F)N___SKMEL5_SKIN'
#sample_prob2 = 'C1=NC2=C(N=C(N=C2N1C3C(C(C(O3)CO)O)O)F)N___CC1=C(C=C(C=C1)C(=O)NC2=CC(=CC(=C2)C(F)(F)F)N3C=C(N=C3)C)NC4=NC=CC(=N4)C5=CN=CC=C5___SKMEL5_SKIN'


 


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
	plt.close()



seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)



# gcn_drug1_F, gcn_drug2_F, gcn_drug1_ADJ, gcn_drug2_ADJ, gcn_exp_A, gcn_exp_B, gcn_tgt_A, gcn_tgt_B, gcn_adj, gcn_adj_weight, syn_ans, cell_info
norm = 'tanh_norm'

# CV_0 -> there's no CV now 
train_data_0, test_data_0 = prepare_data_GCN(0, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, 
MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, 
cell_one_hot, MY_syn_RE2, norm)



# WEIGHT 
def get_loss_weight(CV) :
	train_data = globals()['train_data_'+str(CV)]
	ys = train_data['y'].squeeze().tolist()
	min_s = np.amin(ys)
	loss_weight = np.log(train_data['y'] - min_s + np.e)
	return loss_weight


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





########################################################


# CV 0 

LOSS_WEIGHT_0 = get_loss_weight(0)

JY_IDX_WEIGHT_T = torch.Tensor(JY_IDX_WEIGHT).view(1,-1)

T_train_0, T_test_0 = make_merged_data(0)




def CUDA_TRAIN(LOADER_DICT, THIS_MODEL, THIS_OPTIMIZER, use_cuda = False) :
	criterion = weighted_mse_loss
	#
	train_loss_all = []
	running_loss = 0
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
		## update the average validation loss
		THIS_OPTIMIZER.zero_grad()
		output = THIS_MODEL(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y)
		wc = torch.Tensor(batch_cut_weight[batch_idx_t]).view(-1,1)
		if torch.cuda.is_available():
			wc = wc.cuda()
		loss = weighted_mse_loss(output, y, wc ) 
		loss.backward()
		THIS_OPTIMIZER.step()
		#
		# update average validation loss 
		running_loss = running_loss + loss.item()
		pred_list = pred_list + output.squeeze().tolist()
		ans_list = ans_list + y.squeeze().tolist()
	#
	last_loss = running_loss / (batch_idx_t+1)
	train_sc, _ = stats.spearmanr(pred_list, ans_list)
	train_pc, _ = stats.pearsonr(pred_list, ans_list)
	return last_loss, train_pc, train_sc, THIS_MODEL, THIS_OPTIMIZER     




def CUDA_TEST( LOADER_DICT, THIS_MODEL , use_cuda = False) :
	THIS_MODEL.eval()
	MSE = torch.nn.MSELoss()
	#
	running_loss = 0
	last_loss = 0 
	#
	ans_list = []
	pred_list = []
	with torch.no_grad() :
		for batch_idx_v, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y) in enumerate(LOADER_DICT['test']) :
			expA = expA.view(-1,3)#### 다른점 
			expB = expB.view(-1,3)#### 다른점 
			adj_w = adj_w.squeeze()
			# move to GPU
			if use_cuda:
				drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell = drug1_f.cuda(), drug2_f.cuda(), drug1_a.cuda(), drug2_a.cuda(), expA.cuda(), expB.cuda(), adj.cuda(), adj_w.cuda(), y.cuda(), cell.cuda()
			## update the average validation loss
			output = THIS_MODEL(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y)
			loss = MSE(output, y) # train 이 아니라서 weight 안넣어줌. 그냥 nn.MSE 넣어주기 
			# update average validation loss 
			running_loss = running_loss + loss.item()
			pred_list = pred_list + output.squeeze().tolist()
			ans_list = ans_list + y.squeeze().tolist()
		#
	last_loss = running_loss / (batch_idx_v+1)
	val_sc, _ = stats.spearmanr(pred_list, ans_list)
	val_pc, _ = stats.pearsonr(pred_list, ans_list)
	return last_loss, val_pc, val_sc, pred_list, ans_list    





def LEARN_MODEL (PRJ_PATH, my_config, n_epoch, use_cuda = True) :
	#
	ABCS_tv_ch.to_csv(os.path.join(PRJ_PATH,'RETRAIN.tvlist.csv'), index=False)
	ABCS_test_ch.to_csv(os.path.join(PRJ_PATH,'RETRAIN.testlist.csv'), index=False)
	#
	train_loader = torch.utils.data.DataLoader(T_train_0, batch_size = my_config["config/batch_size"].item(), collate_fn = graph_collate_fn, shuffle = True, num_workers=my_config['config/n_workers'].item()) # 
	test_loader = torch.utils.data.DataLoader(T_test_0, batch_size = my_config["config/batch_size"].item(), collate_fn = graph_collate_fn, shuffle =False, num_workers=my_config['config/n_workers'].item()) # 
	batch_cut_weight = [LOSS_WEIGHT_0[i:i+my_config["config/batch_size"].item()] for i in range(0,len(LOSS_WEIGHT_0), my_config["config/batch_size"].item())]
	#
	LOADER_DICT = { 
		'train': train_loader, 
		'test' : test_loader, 
		'loss_weight' : batch_cut_weight
		}
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
	best_model = MY_expGCN_parallel_model(
				G_chem_layer, T_test_0.gcn_drug1_F.shape[-1] , G_chem_hdim,
				G_exp_layer, 3, G_exp_hdim,
				dsn1_layers, dsn2_layers, snp_layers, 
				len(set(A_B_C_S_SET_SM.DrugCombCCLE)), 1,
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
	#
	THIS_OPT = torch.optim.Adam(best_model.parameters(), lr = my_config["config/lr"].item() )
	#
	# n_epoch = 3 ; use_cuda = False
	train_df = pd.DataFrame(columns =['epoch','train_loss','train_pcor','train_scor'])
	pred_result_1 = pd.DataFrame(columns =list(range(ABCS_test_ch.shape[0])))
	pred_result_2 = pd.DataFrame(columns =['last_loss','test_PC','test_SC'])
	print(train_df)
	print(pred_result_1)
	print(pred_result_2)
	#
	train_df.to_csv(os.path.join(PRJ_PATH,'RETRAIN.tv_RESULT.csv'), index=False)
	pred_result_1.to_csv(os.path.join(PRJ_PATH,'RETRAIN.test_RESULT1.csv'), index=False)
	pred_result_2.to_csv(os.path.join(PRJ_PATH,'RETRAIN.test_RESULT2.csv'), index=False)
	#
	train_loss_all = []
	train_pearson_corr_all = []
	train_spearman_corr_all = []
	for epoch in range(n_epoch) :
		now = datetime.now()
		print(now)
		best_model.train()
		last_loss, train_pc, train_sc, best_model, THIS_OPT = CUDA_TRAIN(LOADER_DICT, best_model, THIS_OPT, use_cuda)
		train_loss_all.append(last_loss)
		train_pearson_corr_all.append(train_pc)
		train_spearman_corr_all.append(train_sc)
		#
		print('epoch : {}/1000, TrainLoss : {}, TrainPcor : {}, TrainScor : {}'.format(epoch, last_loss, train_pc, train_sc ), flush=True)
		checkpoint_path = os.path.join(PRJ_PATH, 'RETRAIN', "checkpoint_{}".format(epoch))
		torch.save((best_model.state_dict(), THIS_OPT.state_dict()), checkpoint_path)
		torch.save(best_model.state_dict(), os.path.join(PRJ_PATH, 'model.pth'))
		#
		train_df = pd.DataFrame({
		'epoch' : [epoch],
		'train_loss' : [last_loss],
		'train_pcor' : [train_pc],
		'train_scor' : [train_sc]
		})
		print(train_df)
		train_df.to_csv(os.path.join(PRJ_PATH,'RETRAIN.tv_RESULT.csv'), mode='a', index=False, header = False)
		#
		###############################################
		#
		print('start TEST', flush = True)
		best_model.eval()
		last_loss, test_pc, test_sc, pred_list, ans_list = CUDA_TEST( LOADER_DICT, best_model , use_cuda)
		pred_result_1 = pd.DataFrame({
			'test_pred_{}'.format(epoch): pred_list
		}).T
		print(pred_result_1)
		pred_result_2 = pd.DataFrame({
			'last_loss' : [last_loss],
			'test_PC' : [test_pc],
			'test_SC' : [test_sc]
		})
		print(pred_result_2)
		pred_result_1.to_csv(os.path.join(PRJ_PATH,'RETRAIN.test_RESULT1.csv'), mode='a', index=True, header = False)
		pred_result_2.to_csv(os.path.join(PRJ_PATH,'RETRAIN.test_RESULT2.csv'), mode='a', index=False, header = False)
		pred_result_1 = pd.DataFrame({'test_y' : ans_list}).T
		pred_result_1.to_csv(os.path.join(PRJ_PATH,'RETRAIN.test_RESULT1.csv'), mode='a', index=True, header = False)
		print('epoch : {}/1000, Test Loss : {}, Test Pcor : {}, Test Scor : {}'.format(epoch, last_loss, test_pc, test_sc ), flush=True)
	
	# 
	
	




W_NAME = 'W20v1'


PRJ_PATH = '/home01/k020a01/02.M3V5/M3V5_349_MIS2/'
os.makedirs( os.path.join(PRJ_PATH,'RETRAIN'), exist_ok = True)
# ANA_DF_CSV = pd.read_csv(os.path.join(PRJ_PATH,'RAY_ANA_DF.{}_{}_{}.csv'.format(MJ_NAME, MISS_NAME, W_NAME)))
ANA_DF_CSV = pd.read_csv(os.path.join(PRJ_PATH,'RAY_ANA_DF.{}.csv'.format('M3V5_349_MIS2')))

my_config = ANA_DF_CSV[ANA_DF_CSV.trial_id=='dc5b32ee'] # 349 


# LEARN_MODEL (PRJ_PATH, my_config, model_path, model_name, model_num, n_epoch, use_cuda = True)
LEARN_MODEL (PRJ_PATH, my_config, 1000 , use_cuda = True)
LEARN_MODEL (PRJ_PATH, my_config, 3 , use_cuda = True)





tail ~/logs/M3V5W20_GPU1_10590.log
tail ~/02.M3V5/M3V5_349_MIS2/RETRAIN.G1.txt 
tail ~/02.M3V5/M3V5_349_MIS2/RETRAIN.test_RESULT1.csv
tail ~/02.M3V5/M3V5_349_MIS2/RETRAIN.tv_RESULT.csv
tail ~/02.M3V5/M3V5_349_MIS2/RETRAIN.test_RESULT2.csv

진즉에 이렇게 할걸 


#############################################################################


RETRAIN 결과 확인 



진행할때는 163 말고 500대로 하기 

# W20 
get /home01/k020a01/02.M3V5/M3V5_349_MIS2/RETRAIN/checkpoint_598 RETRAIN_CCK
get /home01/k020a01/02.M3V5/M3V5_349_MIS2/RETRAIN.t* 

# W21
get /home01/k020a01/02.M3V5/M3V5_W21_349_MIS2/RETRAIN/checkpoint_394 RETRAIN_CCK
get /home01/k020a01/02.M3V5/M3V5_W21_349_MIS2/RETRAIN.t*  


# W20 
PRJ_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V5_W20_MIS2_349/'

# W21 
PRJ_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V5_W21_MIS2_349/'
PRJ_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V5_W21_MIS2_978/RETRAIN/'


RET_tv_result = pd.read_csv(os.path.join(PRJ_PATH, 'RETRAIN.tv_RESULT.csv' ))
RET_test_result = pd.read_csv(os.path.join(PRJ_PATH, 'RETRAIN.test_RESULT2.csv' ))
RET_test_value = pd.read_csv(os.path.join(PRJ_PATH, 'RETRAIN.test_RESULT1.csv' )) 
RET_test_data = pd.read_csv(os.path.join(PRJ_PATH, 'RETRAIN.testlist.csv' ))  

RET_test_value = RET_test_value.drop_duplicates()
cck_answer = RET_test_value.loc['test_y']

# W21 349
# cck_result = RET_test_value.loc['test_pred_598']
cck_result = RET_test_value.loc['test_pred_394']

# W21 978
cck_result = RET_test_value.loc['test_pred_332']



cell_tissue = [ '_'.join(a.split('_')[1:]) if type(a) == str else 'NA' for a in list(RET_test_data['DrugCombCCLE'])]

result_merge_df = pd.DataFrame({
	'y_data' : cck_answer, 'pred_result' : cck_result,
	'cell_name' : list(RET_test_data.DC_cellname), 'cell_tissue' : cell_tissue,
	'drug_row_CID' : list(RET_test_data.drug_row_CID), 'drug_col_CID': list(RET_test_data.drug_col_CID),
	'type' : list(RET_test_data.type)
	})







def plot_three(big_title, train_loss, valid_loss, train_Pcorr, valid_Pcorr, train_Scorr, valid_Scorr, path, plotname, epoch = 0 ):
	fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(30, 8))
	#
	# loss plot 
	ax1.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss', color = 'Blue', linewidth=4 )
	ax1.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss', color = 'Red', linewidth=4)
	ax1.set_xlabel('epochs', fontsize=20)
	ax1.set_ylabel('loss', fontsize=20)
	ax1.tick_params(axis='both', which='major', labelsize=20 )
	ax1.set_ylim(0, math.ceil(max(train_loss+valid_loss))) # 일정한 scale
	ax1.set_xlim(0, len(train_loss)+1) # 일정한 scale
	ax1.grid(True)
	if epoch > 0 : 
		ax1.axvline(x = epoch, color = 'green', linestyle = '--', linewidth = 3)
	ax1.set_title('5CV Average Loss', fontsize=20)
	#
	# Pearson Corr 
	ax2.plot(range(1,len(train_Pcorr)+1), train_Pcorr, label='Training PCorr', color = 'Blue', linewidth=4 )
	ax2.plot(range(1,len(valid_Pcorr)+1),valid_Pcorr,label='Validation PCorr', color = 'Red', linewidth=4)
	ax2.set_xlabel('epochs', fontsize=20)
	ax2.set_ylabel('PCor', fontsize=20)
	ax2.tick_params(axis='both', which='major', labelsize=20 )
	ax2.set_ylim(0, math.ceil(max(train_Pcorr+valid_Pcorr))) # 일정한 scale
	ax2.set_xlim(0, len(train_Pcorr)+1) # 일정한 scale
	ax2.grid(True)
	if epoch > 0 : 
		ax2.axvline(x = epoch, color = 'green', linestyle = '--', linewidth = 3)
	#
	ax2.set_title('5CV Average Pearson', fontsize=20)
	#
	# Spearman Corr 
	ax3.plot(range(1,len(train_Scorr)+1), train_Scorr, label='Training SCorr', color = 'Blue', linewidth=4 )
	ax3.plot(range(1,len(valid_Scorr)+1),valid_Scorr,label='Validation SCorr', color = 'Red', linewidth=4)
	ax3.set_xlabel('epochs', fontsize=20)
	ax3.set_ylabel('SCor', fontsize=20)
	ax3.tick_params(axis='both', which='major', labelsize=20 )
	ax3.set_ylim(0, math.ceil(max(train_Scorr+valid_Scorr))) # 일정한 scale
	ax3.set_xlim(0, len(train_Scorr)+1) # 일정한 scale
	ax3.grid(True)
	if epoch > 0 : 
		ax3.axvline(x = epoch, color = 'green', linestyle = '--', linewidth = 3)
	#
	ax3.set_title('5CV Average Spearman', fontsize=20)
	#
	fig.suptitle(big_title, fontsize=18)
	plt.tight_layout()
	fig.savefig('{}/{}.three_plot.png'.format(path, plotname), bbox_inches = 'tight')



####### W 20 
# 아놔 test 비율 바꿨어야 하는데 시불
WORK_DATE = '23.04.10'
PRJ_NAME = 'M3V5'
MISS_NAME = 'MIS2'
WORK_NAME = 'WORK_20'
W_NAME = 'W20'
PPI_NAME = '349' # 349
MJ_NAME = 'M3V5'

plot_three(
	"Model F2 Retrain version",
	list(RET_tv_result.train_loss), list(RET_test_result.last_loss), 
	list(RET_tv_result.train_pcor), list(RET_test_result.test_PC), 
	list(RET_tv_result.train_scor), list(RET_test_result.test_SC), 
	PRJ_PATH, '{}_{}_{}__F_RET'.format(MJ_NAME, MISS_NAME, WORK_NAME), epoch = 598 )



####### W 21
# 아놔 test 비율 바꿨어야 하는데 시불
PRJ_NAME = 'M3V5'
MISS_NAME = 'MIS2'
WORK_NAME = 'WORK_21'
W_NAME = 'W21'
PPI_NAME = '349' # 349
MJ_NAME = 'M3V5'


# 버전 이름 바꿔줘야해 
plot_three(
	"Model 5 Retrain version",
	list(RET_tv_result.train_loss), list(RET_test_result.last_loss), 
	list(RET_tv_result.train_pcor), list(RET_test_result.test_PC), 
	list(RET_tv_result.train_scor), list(RET_test_result.test_SC), 
	PRJ_PATH, '{}_{}_{}__5_RET'.format(MJ_NAME, MISS_NAME, WORK_NAME), epoch = 394 )





####### W 21 978
# 아놔 test 비율 바꿨어야 하는데 시불
PRJ_NAME = 'M3V5'
MISS_NAME = 'MIS2'
WORK_NAME = 'WORK_21'
W_NAME = 'W21'
PPI_NAME = '978' #
MJ_NAME = 'M3V5'


# 버전 이름 바꿔줘야해 
plot_three(
	"Model 5 Retrain version",
	list(RET_tv_result.train_loss), list(RET_test_result.last_loss), 
	list(RET_tv_result.train_pcor), list(RET_test_result.test_PC), 
	list(RET_tv_result.train_scor), list(RET_test_result.test_SC), 
	PRJ_PATH, '{}_{}_{}__5_RET'.format(MJ_NAME, MISS_NAME, WORK_NAME), epoch = 332 )











tissue_set = ['CENTRAL_NERVOUS_SYSTEM', 'LUNG', 'BREAST', 'BONE', 'OVARY', 'PROSTATE', 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'LARGE_INTESTINE', 'SKIN' ] # list(set(test_cell_df['tissue']))
color_set = ['#CF3476','#FF7514','#025669','#308446','#84C3BE','#D53032','#4ddcfd','#ffcd36','#ac8cff'] # "#20603D","#828282","#1E1E1E"
color_dict = {a : color_set[tissue_set.index(a)] for a in tissue_set}

result_merge_df['tissue_oh'] = [color_dict[a] for a in list(result_merge_df['cell_tissue'])]

tmptmp = result_merge_df[['cell_name','tissue_oh']].drop_duplicates()
tmptmp.columns = ['DC_cellname','tissue_oh']



test_cell_df = pd.DataFrame({'DC_cellname' : list(set(result_merge_df.cell_name))})

cell_P = []
cell_S = []
cell_num = []

for cell in list(test_cell_df.DC_cellname) :
	tmp_test_re = result_merge_df[result_merge_df.cell_name == cell]
	cell_P_corr, _ = stats.pearsonr(tmp_test_re.y_data, tmp_test_re.pred_result)
	cell_S_corr, _ = stats.spearmanr(tmp_test_re.y_data, tmp_test_re.pred_result)
	cell_nums = tmp_test_re.shape[0]
	cell_P.append(cell_P_corr)
	cell_S.append(cell_S_corr)
	cell_num.append(cell_nums)


test_cell_df['P_COR'] = cell_P
test_cell_df['S_COR'] = cell_S
test_cell_df['cell_num'] = cell_num

test_cell_df = pd.merge(test_cell_df, tmptmp, on = 'DC_cellname', how='left')


# Pearson corr
test_cell_df = test_cell_df.sort_values('P_COR')

fig = plt.figure(figsize=(15,8))
x_pos = [a*2 for a in range(test_cell_df.shape[0])]

plt.bar(x_pos, test_cell_df['P_COR'], color=test_cell_df['tissue_oh']) # 
plt.xticks(x_pos,list(test_cell_df['DC_cellname']), rotation=90, fontsize= 18)
plt.yticks(np.arange(0, 1, step=0.2),np.round(np.arange(0, 1, step=0.2),2), fontsize= 18)
for i in range(test_cell_df.shape[0]):
	#plt.annotate(str(list(test_cell_df['cell_num'])[i]), xy=(x_pos[i],list(test_cell_df['P_COR'])[i]), ha='center', va='bottom', fontsize= 18)
	plt.annotate(str(list(np.round(test_cell_df['P_COR'],1))[i]), xy=(x_pos[i],list(test_cell_df['P_COR'])[i]), ha='center', va='bottom', fontsize= 18)

plt.legend(loc = 'upper left')
plt.tight_layout()
fig.savefig('{}/{}.png'.format(PRJ_PATH, 'new_plot_pearson_RET'), bbox_inches = 'tight')
plt.close()









# 추가적으로 궁금해져서 
# AXBX 의 비율을 살펴보기로 함 
# total 10552

5CV 때 애를 살펴봐야함 


TEST_ALL_O = result_merge_df[result_merge_df.type.isin(['AOBO'])] # 1348
TEST_HALF_O = result_merge_df[result_merge_df.type.isin(['AXBO','AOBX'])] # 2179
TEST_NO_O = result_merge_df[result_merge_df.type.isin(['AXBX'])] # 7025



def give_test_result_corDF (MISS_DF) : 
	test_cell_df = pd.DataFrame({'cell_name' : list(set(MISS_DF.cell_name))})
	#
	cell_P = []
	cell_S = []
	cell_num = []
	#
	for cell in list(test_cell_df.cell_name) :
		tmp_test_re = MISS_DF[MISS_DF.cell_name == cell]
		if tmp_test_re.shape[0] > 1 :
			cell_P_corr, _ = stats.pearsonr(tmp_test_re.y_data, tmp_test_re.pred_result)
			cell_S_corr, _ = stats.spearmanr(tmp_test_re.y_data, tmp_test_re.pred_result)
			cell_nums = tmp_test_re.shape[0]
			cell_P.append(cell_P_corr)
			cell_S.append(cell_S_corr)
			cell_num.append(cell_nums)
		else :
			cell_nums = tmp_test_re.shape[0]
			cell_P.append(0)
			cell_S.append(0)
			cell_num.append(cell_nums)
	#
	test_cell_df['P_COR'] = cell_P
	test_cell_df['S_COR'] = cell_S
	test_cell_df['cell_num'] = cell_num
	#
	test_cell_df = pd.merge(test_cell_df, result_merge_df[['cell_name','cell_tissue']].drop_duplicates(), on = 'cell_name', how = 'left'  )
	#
	test_cell_df = test_cell_df.sort_values('S_COR')
	return test_cell_df


TEST_ALL_O_RESULT = give_test_result_corDF(TEST_ALL_O)
TEST_HALF_O_RESULT = give_test_result_corDF(TEST_HALF_O)
TEST_NO_O_RESULT = give_test_result_corDF(TEST_NO_O)








##### 

애초에 원래 예측하려고 했던 데이터의 분포가 어떤데? 

result_merge_df

for_split_vio_1 = result_merge_df[['cell_name','cell_tissue','y_data']]
for_split_vio_1.columns = ['cell_name','cell_tissue','value'] 
for_split_vio_1['label'] = 'y_data'

for_split_vio_2 = result_merge_df[['cell_name','cell_tissue','pred_result']]
for_split_vio_2.columns = ['cell_name','cell_tissue','value'] 
for_split_vio_2['label'] = 'pred_result'

for_split_vio = pd.concat([for_split_vio_1, for_split_vio_2])


fig, ax = plt.subplots(figsize=(20, 8))
sns.violinplot(
	ax =ax, data  = for_split_vio, 
	x = 'cell_name', y = 'value', 
	split = True, hue = 'label',
	inner = 'quart',
	palette = {'y_data' : '0.85', 'pred_result' : 'b'},
	linewidth=1,  edgecolor="dimgrey")

ax.set_xlabel('cell names', fontsize=20)
ax.set_ylabel('value', fontsize=20)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
ax.tick_params(axis='y', which='major', labelsize=20 )
plt.tight_layout()
plt.savefig(os.path.join(PRJ_PATH,'cell_compare.png'), dpi = 300)
plt.close()









for cell_name in list(set(result_merge_df.cell_name)) :
	cell_name
	min(result_merge_df[result_merge_df.cell_name==cell_name]['y_data'])
	max(result_merge_df[result_merge_df.cell_name==cell_name]['y_data'])
	np.std(result_merge_df[result_merge_df.cell_name==cell_name]['y_data'])






