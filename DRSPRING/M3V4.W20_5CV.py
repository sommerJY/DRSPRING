
# checkpoint dir 설정 확인 

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






					# HS Functional 활용 -> 845
					print('NETWORK')
					# HUMANNET 사용 

					hunet_gsp = pd.read_csv(NETWORK_PATH+'HumanNet-FN.tsv', sep = '\t', header = None)
					hunet_gsp.columns = ['G_A','G_B','SC']

					LINCS_gene_file = pd.read_csv(LINCS_PATH+'geneinfo_beta.txt', sep = '\t')
					LINCS_978 = LINCS_gene_file[LINCS_gene_file.feature_space == 'landmark']
					LINCS_978 = LINCS_978[['gene_id','gene_symbol']]
					LINCS_978['new_node'] = [str(list(LINCS_978.gene_id)[i]) + "__" + list(LINCS_978.gene_symbol)[i] for i in range(978)]
					LINCS_978 = LINCS_978.reset_index(drop=True)
					lm_entrezs = list(LINCS_978.gene_id)

					hnet_L1 = hunet_gsp[hunet_gsp['G_A'].isin(lm_entrezs)]
					hnet_L2 = hnet_L1[hnet_L1['G_B'].isin(lm_entrezs)] # 3885
					hnet_L3 = hnet_L2[hnet_L2.SC >= 2.0]

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


MJ_NAME = 'M3V4'
WORK_DATE = '23.03.19'
MISS_NAME = 'MIS2'


# SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V4_CCLE_FULL/'

SAVE_PATH = '/home01/k020a01/02.VER3/M3V4_349_DATA/'
# SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V4_349_FULL/'

SAVE_PATH = '/home01/k020a01/02.VER3/M3V4_978_DATA/'
# SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V4_978_FULL/'

SAVE_PATH = '/home01/k020a01/02.VER3/M3V4_845_DATA/'
# SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V4_845_FULL/'


file_name = 'M3V4_349_MISS2_FULL'
file_name = 'M3V4_978_MISS2_FULL'
file_name = 'M3V4_845_MISS2_FULL'


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
WORK_NAME = 'WORK_20v2' # 978
WORK_NAME = 'WORK_20v3' # 845




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




					fig, ax = plt.subplots(figsize=(30, 15))
					## fig, ax = plt.subplots(figsize=(40, 15))

					x_pos = [a*3 for a in range(C_df.shape[0])]
					ax.bar(x_pos, list(C_df['freq']))

					plt.xticks(x_pos, list(C_df['cell']), rotation=90, fontsize=18)

					for i in range(C_df.shape[0]):
						plt.annotate(str(int(list(C_df['freq'])[i])), xy=(x_pos[i], list(C_df['freq'])[i]), ha='center', va='bottom', fontsize=18)

					ax.set_ylabel('cell nums')
					ax.set_title('used cells')
					plt.tight_layout()

					plotname = 'total_cells'
					path = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V4_MIS2_W20v1'
					fig.savefig('{}/{}.png'.format(path, plotname), bbox_inches = 'tight')
					plt.close()

						# C_MJ = pd.merge(C_df, DC_CELL_info_filt[['DC_cellname','DrugCombCello','DrugCombCCLE']], left_on = 'cell', right_on = 'DC_cellname', how = 'left')
						# C_MJ.to_csv('/st06/jiyeonH/13.DD_SESS/01.PRJ2/cell_cut_example.csv')
















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
	test_no_dup = data_nodup_df2.loc[CV_ND_INDS[test_key]] # from test included df 
	#
	ABCS_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(train_no_dup.setset)]
	ABCS_val = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(val_no_dup.setset)]
	ABCS_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(test_no_dup.setset)]
	#
	train_ind = list(ABCS_train.index)
	val_ind = list(ABCS_val.index)
	test_ind = list(ABCS_test.index)
	# 
	chem_feat_A_train = MY_chem_A_feat_RE2[train_ind]; chem_feat_A_val = MY_chem_A_feat_RE2[val_ind]; chem_feat_A_test = MY_chem_A_feat_RE2[test_ind]
	chem_feat_B_train = MY_chem_B_feat_RE2[train_ind]; chem_feat_B_val = MY_chem_B_feat_RE2[val_ind]; chem_feat_B_test = MY_chem_B_feat_RE2[test_ind]
	chem_adj_A_train = MY_chem_A_adj_RE2[train_ind]; chem_adj_A_val = MY_chem_A_adj_RE2[val_ind]; chem_adj_A_test = MY_chem_A_adj_RE2[test_ind]
	chem_adj_B_train = MY_chem_B_adj_RE2[train_ind]; chem_adj_B_val = MY_chem_B_adj_RE2[val_ind]; chem_adj_B_test = MY_chem_B_adj_RE2[test_ind]
	gene_A_train = MY_g_EXP_A_RE2[train_ind]; gene_A_val = MY_g_EXP_A_RE2[val_ind]; gene_A_test = MY_g_EXP_A_RE2[test_ind]
	gene_B_train = MY_g_EXP_B_RE2[train_ind]; gene_B_val = MY_g_EXP_B_RE2[val_ind]; gene_B_test = MY_g_EXP_B_RE2[test_ind]
	target_A_train = MY_Target_A2[train_ind]; target_A_val = MY_Target_A2[val_ind]; target_A_test = MY_Target_A2[test_ind]
	target_B_train = MY_Target_B2[train_ind]; target_B_val = MY_Target_B2[val_ind]; target_B_test = MY_Target_B2[test_ind]
	cell_basal_train = MY_CellBase_RE2[train_ind]; cell_basal_val = MY_CellBase_RE2[val_ind]; cell_basal_test = MY_CellBase_RE2[test_ind]
	cell_train = cell_one_hot[train_ind]; cell_val = cell_one_hot[val_ind]; cell_test = cell_one_hot[test_ind]
	syn_train = MY_syn_RE2[train_ind]; syn_val = MY_syn_RE2[val_ind]; syn_test = MY_syn_RE2[test_ind]
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
	train_data['GENE_A'] = torch.concat([gene_A_train, gene_B_train], axis = 0)
	val_data['GENE_A'] = gene_A_val
	test_data['GENE_A'] = gene_A_test
	#
	train_data['GENE_B'] = torch.concat([gene_B_train, gene_A_train], axis = 0)
	val_data['GENE_B'] = gene_B_val
	test_data['GENE_B'] = gene_B_test
	#
	train_data['TARGET_A'] = torch.concat([target_A_train, target_B_train], axis = 0)
	val_data['TARGET_A'] = target_A_val
	test_data['TARGET_A'] = target_A_test
	#
	train_data['TARGET_B'] = torch.concat([target_B_train, target_A_train], axis = 0)
	val_data['TARGET_B'] = target_B_val
	test_data['TARGET_B'] = target_B_test
	#   #
	train_data['cell_BASAL'] = torch.concat((cell_basal_train, cell_basal_train), axis=0)
	val_data['cell_BASAL'] = cell_basal_val
	test_data['cell_BASAL'] = cell_basal_test
	##
	train_data['cell'] = torch.concat((cell_train, cell_train), axis=0)
	val_data['cell'] = cell_val
	test_data['cell'] = cell_test
	#            
	train_data['y'] = torch.concat((syn_train, syn_train), axis=0)
	val_data['y'] = syn_val
	test_data['y'] = syn_test
	#
	print(train_data['drug1_feat'].shape, flush=True)
	print(val_data['drug1_feat'].shape, flush=True)
	print(test_data['drug1_feat'].shape, flush=True)
	return train_data, val_data, test_data



ch1 = ABCS_train.reset_index(drop=True)
ch1_ind = list(ch1.index)
aa = list(ch1['ROW_CAN_SMILES'])
bb = list(ch1['COL_CAN_SMILES'])
cc = list(ch1['DrugCombCCLE'])
SM_C_CHECK_1 = [aa[i] + '___' + bb[i]+ '___' + cc[i] for i in ch1_ind]
SM_C_CHECK_2 = [bb[i] + '___' + aa[i]+ '___' + cc[i] for i in ch1_ind]
ch1_checklist = list(set(SM_C_CHECK_1+SM_C_CHECK_2)) # 65794 -> 65361

ch2 = ABCS_test.reset_index(drop=True)
ch2_ind = list(ch2.index)
aa = list(ch2['ROW_CAN_SMILES'])
bb = list(ch2['COL_CAN_SMILES'])
cc = list(ch2['DrugCombCCLE'])
SM_C_CHECK_1 = [aa[i] + '___' + bb[i]+ '___' + cc[i] for i in ch2_ind]
SM_C_CHECK_2 = [bb[i] + '___' + aa[i]+ '___' + cc[i] for i in ch2_ind]
ch2_checklist = list(set(SM_C_CHECK_1+SM_C_CHECK_2)) #20733 -> 20526

print('trainval : test')
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

# CV_0
train_data_0, val_data_0, test_data_0 = prepare_data_GCN(0, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, 
MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, 
cell_one_hot, MY_syn_RE2, norm)

# CV_1
train_data_1, val_data_1, test_data_1 = prepare_data_GCN(1, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, 
MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, 
cell_one_hot, MY_syn_RE2, norm)

# CV_2
train_data_2, val_data_2, test_data_2 = prepare_data_GCN(2, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, 
MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, 
cell_one_hot, MY_syn_RE2, norm)

# CV_3
train_data_3, val_data_3, test_data_3 = prepare_data_GCN(3, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, 
MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, 
cell_one_hot, MY_syn_RE2, norm)

# CV_4
train_data_4, val_data_4, test_data_4 = prepare_data_GCN(4, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, 
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
	val_data = globals()['val_data_'+str(CV)]
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
	return T_train, T_val, T_test





# CV 0 
T_train_0, T_val_0, T_test_0 = make_merged_data(0)
RAY_train_0 = ray.put(T_train_0)
RAY_val_0 = ray.put(T_val_0)
RAY_test_0 = ray.put(T_test_0)
RAY_loss_weight_0 = ray.put(LOSS_WEIGHT_0)


# CV 1
T_train_1, T_val_1, T_test_1 = make_merged_data(1)
RAY_train_1 = ray.put(T_train_1)
RAY_val_1 = ray.put(T_val_1)
RAY_test_1 = ray.put(T_test_1)
RAY_loss_weight_1 = ray.put(LOSS_WEIGHT_1)


# CV 2 
T_train_2, T_val_2, T_test_2 = make_merged_data(2)
RAY_train_2 = ray.put(T_train_2)
RAY_val_2 = ray.put(T_val_2)
RAY_test_2 = ray.put(T_test_2)
RAY_loss_weight_2 = ray.put(LOSS_WEIGHT_2)


# CV 3
T_train_3, T_val_3, T_test_3 = make_merged_data(3)
RAY_train_3 = ray.put(T_train_3)
RAY_val_3 = ray.put(T_val_3)
RAY_test_3 = ray.put(T_test_3)
RAY_loss_weight_3 = ray.put(LOSS_WEIGHT_3)


# CV 4
T_train_4, T_val_4, T_test_4 = make_merged_data(4)
RAY_train_4 = ray.put(T_train_4)
RAY_val_4 = ray.put(T_val_4)
RAY_test_4 = ray.put(T_test_4)
RAY_loss_weight_4 = ray.put(LOSS_WEIGHT_4)




def inner_train( LOADER_DICT, THIS_MODEL, THIS_OPTIMIZER , use_cuda=False) :
	THIS_MODEL.train()
	running_loss = 0
	last_loss = 0 
	#
	ans_list = []
	pred_list = []
	batch_cut_weight = LOADER_DICT['loss_weight']
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






def RAY_MY_train(config, checkpoint_dir=None):
	n_epochs = 1000
	criterion = weighted_mse_loss
	use_cuda = False  #  #  #  #  #  #  # True
	#
	dsn1_layers = [config["feat_size_0"], config["feat_size_1"] , config["feat_size_2"] ]
	dsn2_layers = [config["feat_size_0"], config["feat_size_1"] , config["feat_size_2"] ]
	snp_layers = [config["feat_size_3"] ,config["feat_size_4"]]
	inDrop = config["dropout_1"]
	Drop = config["dropout_2"]
	#
	# CV 0 
	CV_0_train = ray.get(RAY_train_0)
	CV_0_val = ray.get(RAY_val_0)
	CV_0_test = ray.get(RAY_test_0)
	CV_0_loss_weight = ray.get(RAY_loss_weight_0)
	CV_0_batch_cut_weight = [CV_0_loss_weight[i:i+config["batch_size"]] for i in range(0,len(CV_0_loss_weight), config["batch_size"])]
	#
	CV_0_loaders = {
			'train' : torch.utils.data.DataLoader(CV_0_train, batch_size = config["batch_size"], collate_fn = graph_collate_fn, shuffle =False, num_workers=config['n_workers']),
			'eval' : torch.utils.data.DataLoader(CV_0_val, batch_size = config["batch_size"], collate_fn = graph_collate_fn, shuffle =False, num_workers=config['n_workers']),
			'test' : torch.utils.data.DataLoader(CV_0_test, batch_size = config["batch_size"], collate_fn = graph_collate_fn, shuffle =False, num_workers=config['n_workers']),
			'loss_weight' : CV_0_batch_cut_weight
	}
	#
	#
	#  
	CV_0_MODEL = MY_expGCN_parallel_model(
			config["G_chem_layer"], CV_0_train.gcn_drug1_F.shape[-1] , config["G_chem_hdim"],      # G_layer_chem, G_indim_chem, G_hiddim_chem, 
			config["G_exp_layer"], 3 , config["G_exp_hdim"],      # G_layer_exp, G_indim_exp, G_hiddim_exp, 
			dsn1_layers, dsn2_layers, snp_layers,      # drug 1 layers, drug 2 layers, merged layer, 
			len(set(A_B_C_S_SET_SM.DrugCombCCLE)), 1,      # cell_dim ,out_dim,
			inDrop, Drop      # inDrop, drop
			)
	# 
	# CV 1 
	CV_1_train = ray.get(RAY_train_1)
	CV_1_val = ray.get(RAY_val_1)
	CV_1_test = ray.get(RAY_test_1)
	CV_1_loss_weight = ray.get(RAY_loss_weight_1)
	CV_1_batch_cut_weight = [CV_1_loss_weight[i:i+config["batch_size"]] for i in range(0,len(CV_1_loss_weight), config["batch_size"])]
	#
	CV_1_loaders = {
			'train' : torch.utils.data.DataLoader(CV_1_train, batch_size = config["batch_size"], collate_fn = graph_collate_fn, shuffle =False, num_workers=config['n_workers']),
			'eval' : torch.utils.data.DataLoader(CV_1_val, batch_size = config["batch_size"], collate_fn = graph_collate_fn, shuffle =False, num_workers=config['n_workers']),
			'test' : torch.utils.data.DataLoader(CV_1_test, batch_size = config["batch_size"], collate_fn = graph_collate_fn, shuffle =False, num_workers=config['n_workers']),
			'loss_weight' : CV_1_batch_cut_weight
	}
	#
	#  
	CV_1_MODEL = MY_expGCN_parallel_model(
			config["G_chem_layer"], CV_1_train.gcn_drug1_F.shape[-1] , config["G_chem_hdim"],      # G_layer_chem, G_indim_chem, G_hiddim_chem, 
			config["G_exp_layer"], 3 , config["G_exp_hdim"],      # G_layer_exp, G_indim_exp, G_hiddim_exp, 
			dsn1_layers, dsn2_layers, snp_layers,      # drug 1 layers, drug 2 layers, merged layer, 
			len(set(A_B_C_S_SET_SM.DrugCombCCLE)), 1,      # cell_dim ,out_dim,
			inDrop, Drop      # inDrop, drop
			)
	# # CV 2 
	CV_2_train = ray.get(RAY_train_2)
	CV_2_val = ray.get(RAY_val_2)
	CV_2_test = ray.get(RAY_test_2)
	CV_2_loss_weight = ray.get(RAY_loss_weight_2)
	CV_2_batch_cut_weight = [CV_2_loss_weight[i:i+config["batch_size"]] for i in range(0,len(CV_2_loss_weight), config["batch_size"])]
	#
	CV_2_loaders = {
		'train' : torch.utils.data.DataLoader(CV_2_train, batch_size = config["batch_size"], collate_fn = graph_collate_fn, shuffle =False, num_workers=config['n_workers']),
		'eval' : torch.utils.data.DataLoader(CV_2_val, batch_size = config["batch_size"], collate_fn = graph_collate_fn, shuffle =False, num_workers=config['n_workers']),
		'test' : torch.utils.data.DataLoader(CV_2_test, batch_size = config["batch_size"], collate_fn = graph_collate_fn, shuffle =False, num_workers=config['n_workers']),
		'loss_weight' : CV_2_batch_cut_weight
	}
	#
	#  
	CV_2_MODEL = MY_expGCN_parallel_model(
		config["G_chem_layer"], CV_2_train.gcn_drug1_F.shape[-1] , config["G_chem_hdim"],      # G_layer_chem, G_indim_chem, G_hiddim_chem, 
		config["G_exp_layer"], 3 , config["G_exp_hdim"],      # G_layer_exp, G_indim_exp, G_hiddim_exp, 
		dsn1_layers, dsn2_layers, snp_layers,      # drug 1 layers, drug 2 layers, merged layer, 
		len(set(A_B_C_S_SET_SM.DrugCombCCLE)), 1,      # cell_dim ,out_dim,
		inDrop, Drop      # inDrop, drop
		)
	# 
	# CV 3
	CV_3_train = ray.get(RAY_train_3)
	CV_3_val = ray.get(RAY_val_3)
	CV_3_test = ray.get(RAY_test_3)
	CV_3_loss_weight = ray.get(RAY_loss_weight_3)
	CV_3_batch_cut_weight = [CV_3_loss_weight[i:i+config["batch_size"]] for i in range(0,len(CV_3_loss_weight), config["batch_size"])]
	#
	CV_3_loaders = {
		'train' : torch.utils.data.DataLoader(CV_3_train, batch_size = config["batch_size"], collate_fn = graph_collate_fn, shuffle =False, num_workers=config['n_workers']),
		'eval' : torch.utils.data.DataLoader(CV_3_val, batch_size = config["batch_size"], collate_fn = graph_collate_fn, shuffle =False, num_workers=config['n_workers']),
		'test' : torch.utils.data.DataLoader(CV_3_test, batch_size = config["batch_size"], collate_fn = graph_collate_fn, shuffle =False, num_workers=config['n_workers']),
		'loss_weight' : CV_3_batch_cut_weight
	}
	#
	#  
	CV_3_MODEL = MY_expGCN_parallel_model(
		config["G_chem_layer"], CV_3_train.gcn_drug1_F.shape[-1] , config["G_chem_hdim"],      # G_layer_chem, G_indim_chem, G_hiddim_chem, 
		config["G_exp_layer"], 3 , config["G_exp_hdim"],      # G_layer_exp, G_indim_exp, G_hiddim_exp, 
		dsn1_layers, dsn2_layers, snp_layers,      # drug 1 layers, drug 2 layers, merged layer, 
		len(set(A_B_C_S_SET_SM.DrugCombCCLE)), 1,      # cell_dim ,out_dim,
		inDrop, Drop      # inDrop, drop
		)
	# 
	# CV 4
	CV_4_train = ray.get(RAY_train_4)
	CV_4_val = ray.get(RAY_val_4)
	CV_4_test = ray.get(RAY_test_4)
	CV_4_loss_weight = ray.get(RAY_loss_weight_4)
	CV_4_batch_cut_weight = [CV_4_loss_weight[i:i+config["batch_size"]] for i in range(0,len(CV_4_loss_weight), config["batch_size"])]
	#
	CV_4_loaders = {
		'train' : torch.utils.data.DataLoader(CV_4_train, batch_size = config["batch_size"], collate_fn = graph_collate_fn, shuffle =False, num_workers=config['n_workers']),
		'eval' : torch.utils.data.DataLoader(CV_4_val, batch_size = config["batch_size"], collate_fn = graph_collate_fn, shuffle =False, num_workers=config['n_workers']),
		'test' : torch.utils.data.DataLoader(CV_4_test, batch_size = config["batch_size"], collate_fn = graph_collate_fn, shuffle =False, num_workers=config['n_workers']),
		'loss_weight' : CV_4_batch_cut_weight
	}
	#
	#  
	CV_4_MODEL = MY_expGCN_parallel_model(
		config["G_chem_layer"], CV_4_train.gcn_drug1_F.shape[-1] , config["G_chem_hdim"],      # G_layer_chem, G_indim_chem, G_hiddim_chem, 
		config["G_exp_layer"], 3 , config["G_exp_hdim"],      # G_layer_exp, G_indim_exp, G_hiddim_exp, 
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
	CV_0_optimizer = torch.optim.Adam(CV_0_MODEL.parameters(), lr = config["lr"] )
	CV_1_optimizer = torch.optim.Adam(CV_1_MODEL.parameters(), lr = config["lr"] )
	CV_2_optimizer = torch.optim.Adam(CV_2_MODEL.parameters(), lr = config["lr"] )
	CV_3_optimizer = torch.optim.Adam(CV_3_MODEL.parameters(), lr = config["lr"] )
	CV_4_optimizer = torch.optim.Adam(CV_4_MODEL.parameters(), lr = config["lr"] )
	#
	#
	key_list = ['CV_0','CV_1','CV_2','CV_3','CV_4']
	train_loss_all = {}
	valid_loss_all = {}
	train_pearson_corr_all = {}
	train_spearman_corr_all = {}
	val_pearson_corr_all = {}
	val_spearman_corr_all = {}
	for key in key_list :
		train_loss_all[key] = []
		valid_loss_all[key] = []
		train_pearson_corr_all[key]=[]
		train_spearman_corr_all[key]=[]
		val_pearson_corr_all[key] = []
		val_spearman_corr_all[key] = []
	#
	for epoch in range(n_epochs):
		now = datetime.now()
		train_loss = 0.0
		valid_loss = 0.0
		#
		###################
		# train the model #
		###################
		cv_0_t_loss, cv_0_t_pc, cv_0_t_sc, CV_0_MODEL, CV_0_optimizer  = inner_train(CV_0_loaders, CV_0_MODEL, CV_0_optimizer, True)
		train_loss_all['CV_0'].append(cv_0_t_loss)
		train_pearson_corr_all['CV_0'].append(cv_0_t_pc)
		train_spearman_corr_all['CV_0'].append(cv_0_t_sc)	
		#
		cv_1_t_loss, cv_1_t_pc, cv_1_t_sc, CV_1_MODEL, CV_1_optimizer  = inner_train(CV_1_loaders, CV_1_MODEL, CV_1_optimizer, True)
		train_loss_all['CV_1'].append(cv_1_t_loss)
		train_pearson_corr_all['CV_1'].append(cv_1_t_pc)
		train_spearman_corr_all['CV_1'].append(cv_1_t_sc)
		# 
		cv_2_t_loss, cv_2_t_pc, cv_2_t_sc, CV_2_MODEL, CV_2_optimizer  = inner_train(CV_2_loaders, CV_2_MODEL, CV_2_optimizer, True)
		train_loss_all['CV_2'].append(cv_2_t_loss)
		train_pearson_corr_all['CV_2'].append(cv_2_t_pc)
		train_spearman_corr_all['CV_2'].append(cv_2_t_sc)
		# 
		cv_3_t_loss, cv_3_t_pc, cv_3_t_sc, CV_3_MODEL, CV_3_optimizer  = inner_train(CV_3_loaders, CV_3_MODEL, CV_3_optimizer, True)
		train_loss_all['CV_3'].append(cv_3_t_loss)
		train_pearson_corr_all['CV_3'].append(cv_3_t_pc)
		train_spearman_corr_all['CV_3'].append(cv_3_t_sc)
		# 
		cv_4_t_loss, cv_4_t_pc, cv_4_t_sc, CV_4_MODEL, CV_4_optimizer  = inner_train(CV_4_loaders, CV_4_MODEL, CV_4_optimizer, True)
		train_loss_all['CV_4'].append(cv_4_t_loss)
		train_pearson_corr_all['CV_4'].append(cv_4_t_pc)
		train_spearman_corr_all['CV_4'].append(cv_4_t_sc)
		# 
		######################    
		# validate the model #
		######################
		cv_0_v_loss, cv_0_v_pc, cv_0_v_sc, CV_0_MODEL  = inner_val(CV_0_loaders, CV_0_MODEL, True)
		valid_loss_all['CV_0'].append(cv_0_v_loss)
		val_pearson_corr_all['CV_0'].append(cv_0_v_pc)
		val_spearman_corr_all['CV_0'].append(cv_0_v_sc) 
		#
		cv_1_v_loss, cv_1_v_pc, cv_1_v_sc, CV_1_MODEL  = inner_val(CV_1_loaders, CV_1_MODEL, True)
		valid_loss_all['CV_1'].append(cv_1_v_loss)
		val_pearson_corr_all['CV_1'].append(cv_1_v_pc)
		val_spearman_corr_all['CV_1'].append(cv_1_v_sc)
		# 
		cv_2_v_loss, cv_2_v_pc, cv_2_v_sc, CV_2_MODEL  = inner_val(CV_2_loaders, CV_2_MODEL, True)
		valid_loss_all['CV_2'].append(cv_2_v_loss)
		val_pearson_corr_all['CV_2'].append(cv_2_v_pc)
		val_spearman_corr_all['CV_2'].append(cv_2_v_sc)
		# 
		cv_3_v_loss, cv_3_v_pc, cv_3_v_sc, CV_3_MODEL  = inner_val(CV_3_loaders, CV_3_MODEL, True)
		valid_loss_all['CV_3'].append(cv_3_v_loss)
		val_pearson_corr_all['CV_3'].append(cv_3_v_pc)
		val_spearman_corr_all['CV_3'].append(cv_3_v_sc)
		# 
		cv_4_v_loss, cv_4_v_pc, cv_4_v_sc, CV_4_MODEL  = inner_val(CV_4_loaders, CV_4_MODEL, True)
		valid_loss_all['CV_4'].append(cv_4_v_loss)
		val_pearson_corr_all['CV_4'].append(cv_4_v_pc)
		val_spearman_corr_all['CV_4'].append(cv_4_v_sc)
		#
		AVG_TRAIN_LOSS = np.mean([cv_0_t_loss, cv_1_t_loss, cv_2_t_loss, cv_3_t_loss, cv_4_t_loss])
		AVG_T_PC = np.mean([cv_0_t_pc, cv_1_t_pc, cv_2_t_pc, cv_3_t_pc, cv_4_t_pc])
		AVG_T_SC = np.mean([cv_0_t_sc, cv_1_t_sc, cv_2_t_sc, cv_3_t_sc, cv_4_t_sc])
		AVG_VAL_LOSS = np.mean([cv_0_v_loss, cv_1_v_loss, cv_2_v_loss, cv_3_v_loss, cv_4_v_loss])
		AVG_V_PC = np.mean([cv_0_v_pc, cv_1_v_pc, cv_2_v_pc, cv_3_v_pc, cv_4_v_pc])
		AVG_V_SC = np.mean([cv_0_v_sc, cv_1_v_sc, cv_2_v_sc, cv_3_v_sc, cv_4_v_sc])
		#
		#
		done = datetime.now()
		time_spent = done-now
		#
		with tune.checkpoint_dir(step = epoch) as checkpoint_dir:
			trial_name = '_'.join(checkpoint_dir.split('/')[-2].split('_')[0:4])
			print('trial : {}, epoch : {}, TrainLoss : {}, ValLoss : {}'.format(trial_name, epoch, AVG_TRAIN_LOSS, AVG_VAL_LOSS), flush=True)
			cv_0_path = os.path.join(checkpoint_dir, "CV_0_checkpoint")
			cv_1_path = os.path.join(checkpoint_dir, "CV_1_checkpoint")
			cv_2_path = os.path.join(checkpoint_dir, "CV_2_checkpoint")
			cv_3_path = os.path.join(checkpoint_dir, "CV_3_checkpoint")
			cv_4_path = os.path.join(checkpoint_dir, "CV_4_checkpoint")
			torch.save((CV_0_MODEL.state_dict(), CV_0_optimizer.state_dict()), cv_0_path)
			torch.save((CV_1_MODEL.state_dict(), CV_1_optimizer.state_dict()), cv_1_path)
			torch.save((CV_2_MODEL.state_dict(), CV_2_optimizer.state_dict()), cv_2_path)
			torch.save((CV_3_MODEL.state_dict(), CV_3_optimizer.state_dict()), cv_3_path)
			torch.save((CV_4_MODEL.state_dict(), CV_4_optimizer.state_dict()), cv_4_path)
			torch.save(CV_0_MODEL.state_dict(), './CV_0_model.pth')
			torch.save(CV_1_MODEL.state_dict(), './CV_1_model.pth')
			torch.save(CV_2_MODEL.state_dict(), './CV_2_model.pth')
			torch.save(CV_3_MODEL.state_dict(), './CV_3_model.pth')
			torch.save(CV_4_MODEL.state_dict(), './CV_4_model.pth')
		#
		tune.report(AV_T_LS= AVG_TRAIN_LOSS,  AV_T_PC = AVG_T_PC, AV_T_SC = AVG_T_SC, 
		AV_V_LS=AVG_VAL_LOSS, AV_V_PC = AVG_V_PC, AV_V_SC = AVG_V_SC )
	#
	print("Finished Training")



# 이건 테스트 버전임. 생각하고 해 

def MAIN(ANAL_name, WORK_PATH, PRJ_PATH, PRJ_NAME, MISS_NAME, num_samples= 10, max_num_epochs=1000, grace_period = 150, cpus_per_trial = 16, gpus_per_trial = 1):
	# OPTUNA CONFIGURE 
	CONFIG={
		'n_workers' : tune.choice([cpus_per_trial]),
		"epoch" : tune.choice([max_num_epochs]),
		"G_chem_layer" : tune.choice([3]), # 
		"G_exp_layer" : tune.choice([3]), # 
		"G_chem_hdim" : tune.choice([32, 16, 8]), # 
		"G_exp_hdim" : tune.choice([32, 16, 8]), # 
		"batch_size" : tune.choice([ 16, 32 ]), # CPU 니까 # 256, 
		"feat_size_0" : tune.choice([ 256, 128, 64, 32  ]), # 4096, 2048, 1024, 512, 256, 128, 64, 32
		"feat_size_1" : tune.choice([ 256, 128, 64, 32 ]),# 4096, 2048, 1024, 512, 256, 128, 64, 32
		"feat_size_2" : tune.choice([ 256, 128, 64, 32 ]),# 4096, 2048, 1024, 512, 256, 128, 64, 32
		"feat_size_3" : tune.choice([ 256, 128, 64, 32 ]),# 4096, 2048, 1024, 512, 256, 128, 64, 32
		"feat_size_4" : tune.choice([ 256, 128, 64, 32 ]),# 4096, 2048, 1024, 512, 256, 128, 64, 32
		"dropout_1" : tune.choice([0.2]), # 0.01, 0.2, 0.5, 0.8
		"dropout_2" : tune.choice([0.2]), # 0.01, 0.2, 0.5, 0.8
		"lr" : tune.choice([ 0.0001]),# 0.00001, 0.0001, 0.001
	}
	#
	#pickle.dumps(trainable)
	reporter = CLIReporter(
		metric_columns=["AV_T_LS", "AV_V_LS", 'AV_V_PC','AV_V_SC', "training_iteration"])
	#
	optuna_search = OptunaSearch(metric="AV_V_LS", mode="min")
	#
	ASHA_scheduler = tune.schedulers.ASHAScheduler(
		time_attr='training_iteration', metric="AV_V_SC", mode="max", max_t= max_num_epochs, grace_period = grace_period )
	#
	#
	ANALYSIS = tune.run( # 끝내지 않음 
		tune.with_parameters(RAY_MY_train),
		name = ANAL_name,
		num_samples=num_samples,
		config=CONFIG,
		resources_per_trial={'cpu': cpus_per_trial,'gpu' : gpus_per_trial }, # , 
		progress_reporter = reporter,
		search_alg = optuna_search,
		scheduler = ASHA_scheduler,
		#resume = True
	)
	best_trial = ANALYSIS.get_best_trial("AV_V_LS", "min", "last")
	print("Best trial config: {}".format(best_trial.config), flush=True)
	print("Best trial final validation loss: {}".format(
	best_trial.last_result["AV_V_LS"]), flush=True)
	#
	return ANALYSIS


W_NAME = 'W20'
#WORK_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/{}_{}_{}/'.format(MJ_NAME, MISS_NAME, W_NAME )
WORK_PATH = '/home01/k020a01/02.VER3/M3V4_MIS2_W20v1/'


# cpu
MAIN('PRJ02.{}.{}.{}.{}'.format(WORK_DATE, MJ_NAME, MISS_NAME, WORK_NAME), WORK_PATH, WORK_PATH, WORK_NAME, MISS_NAME, 4, 10, 1, 16, 1)


#8gpu
MAIN('PRJ02.{}.{}.{}.{}'.format(WORK_DATE, MJ_NAME, MISS_NAME, WORK_NAME), WORK_PATH, WORK_PATH, WORK_NAME, MISS_NAME, 50, 1000, 150, 16, 1)






그래서 성공하면 다시 W20 으로 해서 넣어야 하는데 trial 하나 당 지금 5CV 를 해야하는거니까
범위를 많이 못줄것 같음 
음 그럼 최소한의 범위에서 알고리즘 비교용으로 해보고
마지막에 괜찮게 나온 애를 hyper parameter tuning 해보는게 맞는것 같다는 생각이 듬
그러면 5CV 에서 다시 random split 을 해서 learning 하게 하면 될것 같음..! 


그래서 우리꺼에서 해보니까 대충 64개 데이터에 대한 5CV 평균내는 내용이 대충 
batch 16 에서 120~300 초대가 나오는것 같음 


node 8개 잡는 방법 연구를 해야하는데 이거를 ray 버전 안높이고 그냥 일단 해볼까 
그러고 나서 같은 내용이 어떻게 적용될 수 있는지를 생각해봐야할듯 

-> 일단 민지가 다시 보내주는 내용 기반으로 해봐야함 
300대로 줄인거 preprocessing 중 





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

WORK_DATE = '23.03.19'
PRJ_NAME = 'M3V4'
MISS_NAME = 'MIS2'
WORK_NAME = 'WORK_20v1'
W_NAME = 'W20v1' # 349


WORK_DATE = '23.03.19'
PRJ_NAME = 'M3V4'
MISS_NAME = 'MIS2'
WORK_NAME = 'WORK_20v2'
W_NAME = 'W20v2' # 978

# 아놔 날짜 바꿨어야하는데 시불 
WORK_DATE = '23.03.19'
PRJ_NAME = 'M3V4'
MISS_NAME = 'MIS2'
WORK_NAME = 'WORK_20v1'
W_NAME = 'W20v1' # 349






anal_dir = "/home01/k020a01/ray_results/PRJ02.{}.{}.{}.{}/".format(WORK_DATE, PRJ_NAME, MISS_NAME, WORK_NAME)
list_dir = os.listdir(anal_dir)
exp_json = [a for a in list_dir if 'experiment_state' in a]
exp_json
# anal_df = ExperimentAnalysis(anal_dir+exp_json[2])
anal_df = Analysis(anal_dir)


ANA_DF = anal_df.dataframe()
ANA_ALL_DF = anal_df.trial_dataframes


# ORIGINAL VER
ANA_DF.to_csv('/home01/k020a01/02.VER3/{}_{}_{}/RAY_ANA_DF.{}_{}_{}.csv'.format(PRJ_NAME, MISS_NAME, W_NAME, PRJ_NAME, MISS_NAME, W_NAME))
import pickle
with open("/home01/k020a01/02.VER3/{}_{}_{}/RAY_ANA_DF.{}_{}_{}.pickle".format(PRJ_NAME, MISS_NAME, W_NAME, PRJ_NAME, MISS_NAME, W_NAME), "wb") as fp:
	pickle.dump(ANA_ALL_DF,fp) 

'/home01/k020a01/02.VER3/{}_{}_{}/RAY_ANA_DF.{}_{}_{}.csv'.format(PRJ_NAME, MISS_NAME, W_NAME, PRJ_NAME, MISS_NAME, W_NAME)
"/home01/k020a01/02.VER3/{}_{}_{}/RAY_ANA_DF.{}_{}_{}.pickle".format(PRJ_NAME, MISS_NAME, W_NAME, PRJ_NAME, MISS_NAME, W_NAME)


# 다시돌린거 확인을 위해서는 아예 GPU 에서 새로 matrix 저장해주는게 맞는것 같음 
ANA_TF_CHECK = [a.split('_')[0]>'2023-03-27' for a in ANA_DF['date'] ] # 문제는 중간에 버린애들도 포함.. 알아서 걸릴듯
ANA_DF_RE = ANA_DF[ANA_TF_CHECK]
new_key = list(ANA_DF_RE.logdir)
ANA_ALL_DF_RE = {k : v for k,v in ANA_ALL_DF.items() if k in new_key}

ANA_DF_RE.to_csv('/home01/k020a01/02.VER3/{}_{}_{}/RAY_ANA_DF.{}_{}_{}.csv'.format(PRJ_NAME, MISS_NAME, W_NAME, PRJ_NAME, MISS_NAME, W_NAME))
import pickle
with open("/home01/k020a01/02.VER3/{}_{}_{}/RAY_ANA_DF.{}_{}_{}.pickle".format(PRJ_NAME, MISS_NAME, W_NAME, PRJ_NAME, MISS_NAME, W_NAME), "wb") as fp:
	pickle.dump(ANA_ALL_DF_RE,fp) 

'/home01/k020a01/02.VER3/{}_{}_{}/RAY_ANA_DF.{}_{}_{}.csv'.format(PRJ_NAME, MISS_NAME, W_NAME, PRJ_NAME, MISS_NAME, W_NAME)
"/home01/k020a01/02.VER3/{}_{}_{}/RAY_ANA_DF.{}_{}_{}.pickle".format(PRJ_NAME, MISS_NAME, W_NAME, PRJ_NAME, MISS_NAME, W_NAME)

ANA_DF = copy.deepcopy(ANA_DF_RE)
ANA_ALL_DF = copy.deepcopy(ANA_ALL_DF_RE)




# 1) best final loss 
min(ANA_DF.sort_values('AV_V_LS')['AV_V_LS'])
DF_KEY = list(ANA_DF.sort_values('AV_V_LS')['logdir'])[0]
DF_KEY

# get /model.pth M1_model.pth


#  2) best final's best chck 
mini_df = ANA_ALL_DF[DF_KEY]
cck_num = mini_df[mini_df.AV_V_LS==min(mini_df.AV_V_LS)].index.item()
checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
TOPVAL_PATH = DF_KEY + checkpoint
TOPVAL_PATH

# get /checkpoint M2_model




# 3) total checkpoint best 
import numpy as np
TOT_min = np.Inf
TOT_key = ""
for key in ANA_ALL_DF.keys():
	trial_min = min(ANA_ALL_DF[key]['AV_V_LS'])
	if trial_min < TOT_min :
		TOT_min = trial_min
		TOT_key = key

mini_df = ANA_ALL_DF[TOT_key]
cck_num =mini_df[mini_df.AV_V_LS==min(mini_df.AV_V_LS)].index.item()
checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
TOPVAL_PATH = TOT_key + checkpoint
TOPVAL_PATH

# get /checkpoint M4_model



# 4) correlation best 
max_cor = max(ANA_DF.sort_values('AV_V_SC')['AV_V_SC'])
DF_KEY = ANA_DF[ANA_DF.AV_V_SC == max_cor]['logdir'].item()
print('best SCOR final', flush=True)
print(DF_KEY, flush=True)

# get /model.pth C1_model.pth



# 5) correlation best's best corr 
mini_df = ANA_ALL_DF[DF_KEY]
cck_num = mini_df[mini_df.AV_V_SC==max(mini_df.AV_V_SC)].index.item()
checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
TOPCOR_PATH = DF_KEY + checkpoint
TOPCOR_PATH

# get /checkpoint C2_model.pth



# 6) correlation best of all 
import numpy as np
TOT_max = -np.Inf
TOT_key = ""
for key in ANA_ALL_DF.keys():
	trial_max = max(ANA_ALL_DF[key]['AV_V_SC'])
	if trial_max > TOT_max :
		TOT_max = trial_max
		TOT_key = key
#

mini_df = ANA_ALL_DF[TOT_key]
cck_num =mini_df[mini_df.AV_V_SC==max(mini_df.AV_V_SC)].index.item()
checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
TOPCOR_PATH = TOT_key + checkpoint
TOPCOR_PATH

# get /checkpoint C4_model.pth











################## LOCAL ###################
################## LOCAL ###################
################## LOCAL ###################
################## LOCAL ###################

from ray.tune import ExperimentAnalysis


def plot_Pcorr(train_corr, valid_corr, path, plotname):
	fig = plt.figure(figsize=(10,8))
	plt.plot(range(1,len(train_corr)+1),train_corr, label='Training Corr')
	plt.plot(range(1,len(valid_corr)+1),valid_corr,label='Validation Corr')
	plt.xlabel('epochs')
	plt.ylabel('corr')
	plt.ylim(0, math.ceil(max(train_corr+valid_corr))) # 일정한 scale
	plt.xlim(0, len(train_corr)+1) # 일정한 scale
	plt.grid(True)
	plt.legend()
	plt.tight_layout()
	fig.savefig('{}/{}.Pcorr_plot.png'.format(path, plotname), bbox_inches = 'tight')
	plt.close()

def plot_Scorr(train_corr, valid_corr, path, plotname):
	fig = plt.figure(figsize=(10,8))
	plt.plot(range(1,len(train_corr)+1),train_corr, label='Training Corr')
	plt.plot(range(1,len(valid_corr)+1),valid_corr,label='Validation Corr')
	plt.xlabel('epochs')
	plt.ylabel('corr')
	plt.ylim(0, math.ceil(max(train_corr+valid_corr))) # 일정한 scale
	plt.xlim(0, len(train_corr)+1) # 일정한 scale
	plt.grid(True)
	plt.legend()
	plt.tight_layout()
	fig.savefig('{}/{}.Scorr_plot.png'.format(path, plotname), bbox_inches = 'tight')
	plt.close()



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








def inner_test( TEST_DATA, THIS_MODEL , use_cuda = False) :
	THIS_MODEL.eval()
	#
	running_loss = 0
	last_loss = 0 
	#
	ans_list = []
	pred_list = []
	with torch.no_grad() :
		for batch_idx_v, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y) in enumerate(TEST_DATA) :
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
	return last_loss, val_pc, val_sc, pred_list, ans_list    



def TEST_CPU (PRJ_PATH, CV_num, my_config, model_path, model_name, model_num) :
	use_cuda = False
	#
	CV_test_dict = { 
		'CV_0': T_test_0, 'CV_1' : T_test_1, 'CV_2' : T_test_2,
		'CV_3' : T_test_3, 'CV_4': T_test_4 }
	#
	T_test = CV_test_dict[CV_num]
	test_loader = torch.utils.data.DataLoader(T_test, batch_size = my_config["config/batch_size"].item(), collate_fn = graph_collate_fn, shuffle =False, num_workers=my_config['config/n_workers'].item())
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
				G_chem_layer, T_test.gcn_drug1_F.shape[-1] , G_chem_hdim,
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
	if torch.cuda.is_available():
		state_dict = torch.load(os.path.join(model_path, model_name))
	else:
		state_dict = torch.load(os.path.join(model_path, model_name), map_location=torch.device('cpu'))
	#
	print("state_dict_done", flush = True)
	if type(state_dict) == tuple:
		best_model.load_state_dict(state_dict[0])
	else : 
		best_model.load_state_dict(state_dict)	#
	print("state_load_done", flush = True)
	#
	#
	last_loss, val_pc, val_sc, pred_list, ans_list = inner_test(test_loader, best_model)
	R__1 , R__2 = jy_corrplot(pred_list, ans_list, PRJ_PATH, 'P.{}.{}.{}.{}_model'.format(PRJ_NAME, MISS_NAME, CV_num, model_num) )
	return  last_loss, R__1, R__2, pred_list, ans_list





WORK_DATE = '23.03.19'
PRJ_NAME = 'M3V4'
MISS_NAME = 'MIS2'
WORK_NAME = 'WORK_20v1'
W_NAME = 'W20v1' # 349



WORK_DATE = '23.03.19'
PRJ_NAME = 'M3V4'
MISS_NAME = 'MIS2'
WORK_NAME = 'WORK_20v2'
W_NAME = 'W20v2' # 978



PRJ_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/{}_{}_{}/'.format(PRJ_NAME, MISS_NAME, W_NAME)
ANA_DF = pd.read_csv(PRJ_PATH+'RAY_ANA_DF.{}_{}_{}.csv'.format(PRJ_NAME, MISS_NAME, W_NAME))
with open(PRJ_PATH+'RAY_ANA_DF.{}_{}_{}.pickle'.format(PRJ_NAME, MISS_NAME, W_NAME), 'rb') as f:
	ANA_ALL_DF = pickle.load(f)


TOPVAL_PATH = PRJ_PATH


# (1) MSE Min
#
#
min(ANA_DF.sort_values('AV_V_LS')['AV_V_LS'])
DF_KEY = list(ANA_DF.sort_values('AV_V_LS')['logdir'])[0]
print("MS MIN DF_KEY : ")
print(DF_KEY, flush=True)
mini_df = ANA_ALL_DF[DF_KEY]
my_config = ANA_DF[ANA_DF.logdir==DF_KEY]
R_1_V = list(ANA_DF.sort_values('AV_V_LS')['AV_V_LS'])[0]
R_1_T_CV0, R_1_1_CV0, R_1_2_CV0, pred_1_CV0, pred_1_CV0 = TEST_CPU(PRJ_PATH, 'CV_0', my_config, PRJ_PATH, 'M1_CV_0_model.pth', 'M1')
R_1_T_CV1, R_1_1_CV1, R_1_2_CV1, pred_1_CV1, pred_1_CV1 = TEST_CPU(PRJ_PATH, 'CV_1', my_config, PRJ_PATH, 'M1_CV_1_model.pth', 'M1')
R_1_T_CV2, R_1_1_CV2, R_1_2_CV2, pred_1_CV2, pred_1_CV2 = TEST_CPU(PRJ_PATH, 'CV_2', my_config, PRJ_PATH, 'M1_CV_2_model.pth', 'M1')
R_1_T_CV3, R_1_1_CV3, R_1_2_CV3, pred_1_CV3, pred_1_CV3 = TEST_CPU(PRJ_PATH, 'CV_3', my_config, PRJ_PATH, 'M1_CV_3_model.pth', 'M1')
R_1_T_CV4, R_1_1_CV4, R_1_2_CV4, pred_1_CV4, pred_1_CV4 = TEST_CPU(PRJ_PATH, 'CV_4', my_config, PRJ_PATH, 'M1_CV_4_model.pth', 'M1')

plot_loss(list(mini_df.AV_T_LS), list(mini_df.AV_V_LS), PRJ_PATH, '{}_{}_{}_VAL_BestLast'.format(MJ_NAME, MISS_NAME, WORK_NAME)  )
plot_Pcorr(list(mini_df.AV_T_PC), list(mini_df.AV_V_PC), PRJ_PATH, '{}_{}_{}_VAL_BestLast'.format(MJ_NAME, MISS_NAME, WORK_NAME))
plot_Scorr(list(mini_df.AV_T_SC), list(mini_df.AV_V_SC), PRJ_PATH, '{}_{}_{}_VAL_BestLast'.format(MJ_NAME, MISS_NAME, WORK_NAME))


# (2) best final's checkpoint
#
cck_num = mini_df[mini_df.AV_V_LS==min(mini_df.AV_V_LS)].index.item()
checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
TOPVAL_PATH = DF_KEY + checkpoint
print('best final check', flush=True)
print(TOPVAL_PATH, flush=True)
R_2_V = min(mini_df.AV_V_LS)
R_2_V
R_2_T_CV0, R_2_1_CV0, R_2_2_CV0, pred_2_CV0, pred_2_CV0 = TEST_CPU(PRJ_PATH, 'CV_0', my_config, PRJ_PATH, 'M2_CV_0_model.pth', 'M2')
R_2_T_CV1, R_2_1_CV1, R_2_2_CV1, pred_2_CV1, pred_2_CV1 = TEST_CPU(PRJ_PATH, 'CV_1', my_config, PRJ_PATH, 'M2_CV_1_model.pth', 'M2')
R_2_T_CV2, R_2_1_CV2, R_2_2_CV2, pred_2_CV2, pred_2_CV2 = TEST_CPU(PRJ_PATH, 'CV_2', my_config, PRJ_PATH, 'M2_CV_2_model.pth', 'M2')
R_2_T_CV3, R_2_1_CV3, R_2_2_CV3, pred_2_CV3, pred_2_CV3 = TEST_CPU(PRJ_PATH, 'CV_3', my_config, PRJ_PATH, 'M2_CV_3_model.pth', 'M2')
R_2_T_CV4, R_2_1_CV4, R_2_2_CV4, pred_2_CV4, pred_2_CV4 = TEST_CPU(PRJ_PATH, 'CV_4', my_config, PRJ_PATH, 'M2_CV_4_model.pth', 'M2')



#
# 3) total checkpoint best 
#	
#
import numpy as np
TOT_min = np.Inf
TOT_key = ""
for key in ANA_ALL_DF.keys():
	trial_min = min(ANA_ALL_DF[key]['AV_V_LS'])
	if trial_min < TOT_min :
		TOT_min = trial_min
		TOT_key = key
#

print('best val', flush=True)
print(TOT_key, flush=True)
mini_df = ANA_ALL_DF[TOT_key]
my_config = ANA_DF[ANA_DF.logdir==TOT_key]
cck_num =mini_df[mini_df.AV_V_LS==min(mini_df.AV_V_LS)].index.item()
checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
TOPVAL_PATH = TOT_key + checkpoint
print('best val check', flush=True)
print(TOPVAL_PATH, flush=True)
R_3_V = min(mini_df.AV_V_LS)
R_3_V
R_3_T_CV0, R_3_1_CV0, R_3_2_CV0, pred_3_CV0, pred_3_CV0 = TEST_CPU(PRJ_PATH, 'CV_0', my_config, PRJ_PATH, 'M4_CV_0_model.pth', 'M4')
R_3_T_CV1, R_3_1_CV1, R_3_2_CV1, pred_3_CV1, pred_3_CV1 = TEST_CPU(PRJ_PATH, 'CV_1', my_config, PRJ_PATH, 'M4_CV_1_model.pth', 'M4')
R_3_T_CV2, R_3_1_CV2, R_3_2_CV2, pred_3_CV2, pred_3_CV2 = TEST_CPU(PRJ_PATH, 'CV_2', my_config, PRJ_PATH, 'M4_CV_2_model.pth', 'M4')
R_3_T_CV3, R_3_1_CV3, R_3_2_CV3, pred_3_CV3, pred_3_CV3 = TEST_CPU(PRJ_PATH, 'CV_3', my_config, PRJ_PATH, 'M4_CV_3_model.pth', 'M4')
R_3_T_CV4, R_3_1_CV4, R_3_2_CV4, pred_3_CV4, pred_3_CV4 = TEST_CPU(PRJ_PATH, 'CV_4', my_config, PRJ_PATH, 'M4_CV_4_model.pth', 'M4')

plot_loss(list(mini_df.AV_T_LS), list(mini_df.AV_V_LS), PRJ_PATH, '{}_{}_{}_VAL_TOT'.format(MJ_NAME, MISS_NAME, WORK_NAME)  )
plot_Pcorr(list(mini_df.AV_T_PC), list(mini_df.AV_V_PC), PRJ_PATH, '{}_{}_{}_VAL_TOT'.format(MJ_NAME, MISS_NAME, WORK_NAME))
plot_Scorr(list(mini_df.AV_T_SC), list(mini_df.AV_V_SC), PRJ_PATH, '{}_{}_{}_VAL_TOT'.format(MJ_NAME, MISS_NAME, WORK_NAME))



#
# (4) best SCOR final
#	
#
max_cor = max(ANA_DF.sort_values('AV_V_SC')['AV_V_SC'])
DF_KEY = ANA_DF[ANA_DF.AV_V_SC == max_cor]['logdir'].item()
print('best SCOR final', flush=True)
print(DF_KEY, flush=True)
mini_df = ANA_ALL_DF[DF_KEY]
my_config = ANA_DF[ANA_DF.logdir==DF_KEY]
R_4_V = max_cor
R_4_V
R_4_T_CV0, R_4_1_CV0, R_4_2_CV0, pred_4_CV0, pred_4_CV0 = TEST_CPU(PRJ_PATH, 'CV_0', my_config, PRJ_PATH, 'C1_CV_0_model.pth', 'C1')
R_4_T_CV1, R_4_1_CV1, R_4_2_CV1, pred_4_CV1, pred_4_CV1 = TEST_CPU(PRJ_PATH, 'CV_1', my_config, PRJ_PATH, 'C1_CV_1_model.pth', 'C1')
R_4_T_CV2, R_4_1_CV2, R_4_2_CV2, pred_4_CV2, pred_4_CV2 = TEST_CPU(PRJ_PATH, 'CV_2', my_config, PRJ_PATH, 'C1_CV_2_model.pth', 'C1')
R_4_T_CV3, R_4_1_CV3, R_4_2_CV3, pred_4_CV3, pred_4_CV3 = TEST_CPU(PRJ_PATH, 'CV_3', my_config, PRJ_PATH, 'C1_CV_3_model.pth', 'C1')
R_4_T_CV4, R_4_1_CV4, R_4_2_CV4, pred_4_CV4, pred_4_CV4 = TEST_CPU(PRJ_PATH, 'CV_4', my_config, PRJ_PATH, 'C1_CV_4_model.pth', 'C1')



plot_loss(list(mini_df.AV_T_LS), list(mini_df.AV_V_LS), PRJ_PATH, '{}_{}_{}_SCOR_BestLast'.format(MJ_NAME, MISS_NAME, WORK_NAME)  )
plot_Pcorr(list(mini_df.AV_T_PC), list(mini_df.AV_V_PC), PRJ_PATH, '{}_{}_{}_SCOR_BestLast'.format(MJ_NAME, MISS_NAME, WORK_NAME))
plot_Scorr(list(mini_df.AV_T_SC), list(mini_df.AV_V_SC), PRJ_PATH, '{}_{}_{}_SCOR_BestLast'.format(MJ_NAME, MISS_NAME, WORK_NAME))


#
#
# (5) BEST cor final 내에서의 max cor 
#	
#
cck_num =mini_df[mini_df.AV_V_SC==max(mini_df.AV_V_SC)].index.item()
checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
TOPCOR_PATH = DF_KEY + checkpoint
print('best final check', flush=True)
print(TOPCOR_PATH, flush=True)
R_5_V = max(mini_df.AV_V_SC)
R_5_V
R_5_T_CV0, R_5_1_CV0, R_5_2_CV0, pred_5_CV0, pred_5_CV0 = TEST_CPU(PRJ_PATH, 'CV_0', my_config, PRJ_PATH, 'C2_CV_0_model.pth', 'C2')
R_5_T_CV1, R_5_1_CV1, R_5_2_CV1, pred_5_CV1, pred_5_CV1 = TEST_CPU(PRJ_PATH, 'CV_1', my_config, PRJ_PATH, 'C2_CV_1_model.pth', 'C2')
R_5_T_CV2, R_5_1_CV2, R_5_2_CV2, pred_5_CV2, pred_5_CV2 = TEST_CPU(PRJ_PATH, 'CV_2', my_config, PRJ_PATH, 'C2_CV_2_model.pth', 'C2')
R_5_T_CV3, R_5_1_CV3, R_5_2_CV3, pred_5_CV3, pred_5_CV3 = TEST_CPU(PRJ_PATH, 'CV_3', my_config, PRJ_PATH, 'C2_CV_3_model.pth', 'C2')
R_5_T_CV4, R_5_1_CV4, R_5_2_CV4, pred_5_CV4, pred_5_CV4 = TEST_CPU(PRJ_PATH, 'CV_4', my_config, PRJ_PATH, 'C2_CV_4_model.pth', 'C2')

#
#
# (6) 그냥 최고 corr 
#	
#
import numpy as np
TOT_max = -np.Inf
TOT_key = ""
for key in ANA_ALL_DF.keys():
	trial_max = max(ANA_ALL_DF[key]['AV_V_SC'])
	if trial_max > TOT_max :
		TOT_max = trial_max
		TOT_key = key
#

print('best cor', flush=True)
print(TOT_key, flush=True)
mini_df = ANA_ALL_DF[TOT_key]
my_config = ANA_DF[ANA_DF.logdir==TOT_key]
cck_num =mini_df[mini_df.AV_V_SC==max(mini_df.AV_V_SC)].index.item()
checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
TOPCOR_PATH = TOT_key + checkpoint
print('best cor check', flush=True)
print(TOPCOR_PATH, flush=True)
R_6_V = max(mini_df.AV_V_SC)
R_6_V
R_6_T_CV0, R_6_1_CV0, R_6_2_CV0, pred_6_CV0, pred_6_CV0 = TEST_CPU(PRJ_PATH, 'CV_0', my_config, PRJ_PATH, 'C4_CV_0_model.pth', 'C4')
R_6_T_CV1, R_6_1_CV1, R_6_2_CV1, pred_6_CV1, pred_6_CV1 = TEST_CPU(PRJ_PATH, 'CV_1', my_config, PRJ_PATH, 'C4_CV_1_model.pth', 'C4')
R_6_T_CV2, R_6_1_CV2, R_6_2_CV2, pred_6_CV2, pred_6_CV2 = TEST_CPU(PRJ_PATH, 'CV_2', my_config, PRJ_PATH, 'C4_CV_2_model.pth', 'C4')
R_6_T_CV3, R_6_1_CV3, R_6_2_CV3, pred_6_CV3, pred_6_CV3 = TEST_CPU(PRJ_PATH, 'CV_3', my_config, PRJ_PATH, 'C4_CV_3_model.pth', 'C4')
R_6_T_CV4, R_6_1_CV4, R_6_2_CV4, pred_6_CV4, pred_6_CV4 = TEST_CPU(PRJ_PATH, 'CV_4', my_config, PRJ_PATH, 'C4_CV_4_model.pth', 'C4')
#

plot_loss(list(mini_df.AV_T_LS), list(mini_df.AV_V_LS), PRJ_PATH, '{}_{}_{}_SCOR_TOT'.format(MJ_NAME, MISS_NAME, WORK_NAME)  )
plot_Pcorr(list(mini_df.AV_T_PC), list(mini_df.AV_V_PC), PRJ_PATH, '{}_{}_{}_SCOR_TOT'.format(MJ_NAME, MISS_NAME, WORK_NAME))
plot_Scorr(list(mini_df.AV_T_SC), list(mini_df.AV_V_SC), PRJ_PATH, '{}_{}_{}_SCOR_TOT'.format(MJ_NAME, MISS_NAME, WORK_NAME))


#final_result(R_1_V, R_1_T, R_1_1, R_1_2, R_2_V, R_2_T, R_2_1, R_2_2, R_3_V, R_3_T, R_3_1, R_3_2,R_4_V, R_4_T, R_4_1, R_4_2, R_5_V, R_5_T, R_5_1, R_5_2, R_6_V, R_6_T, R_6_1, R_6_2)

final_result(R_1_V, R_1_T_CV0, R_1_T_CV1, R_1_T_CV2, R_1_T_CV3, R_1_T_CV4, R_1_1_CV0, R_1_1_CV1, R_1_1_CV2, R_1_1_CV3, R_1_1_CV4, R_1_2_CV0, R_1_2_CV1, R_1_2_CV2, R_1_2_CV3, R_1_2_CV4,
	R_2_V, R_2_T_CV0, R_2_T_CV1, R_2_T_CV2, R_2_T_CV3, R_2_T_CV4, R_2_1_CV0, R_2_1_CV1, R_2_1_CV2, R_2_1_CV3, R_2_1_CV4, R_2_2_CV0, R_2_2_CV1, R_2_2_CV2, R_2_2_CV3, R_2_2_CV4,
	R_3_V, R_3_T_CV0, R_3_T_CV1, R_3_T_CV2, R_3_T_CV3, R_3_T_CV4, R_3_1_CV0, R_3_1_CV1, R_3_1_CV2, R_3_1_CV3, R_3_1_CV4, R_3_2_CV0, R_3_2_CV1, R_3_2_CV2, R_3_2_CV3, R_3_2_CV4,
	R_4_V, R_4_T_CV0, R_4_T_CV1, R_4_T_CV2, R_4_T_CV3, R_4_T_CV4, R_4_1_CV0, R_4_1_CV1, R_4_1_CV2, R_4_1_CV3, R_4_1_CV4, R_4_2_CV0, R_4_2_CV1, R_4_2_CV2, R_4_2_CV3, R_4_2_CV4,
	R_5_V, R_5_T_CV0, R_5_T_CV1, R_5_T_CV2, R_5_T_CV3, R_5_T_CV4, R_5_1_CV0, R_5_1_CV1, R_5_1_CV2, R_5_1_CV3, R_5_1_CV4, R_5_2_CV0, R_5_2_CV1, R_5_2_CV2, R_5_2_CV3, R_5_2_CV4,
	R_6_V, R_6_T_CV0, R_6_T_CV1, R_6_T_CV2, R_6_T_CV3, R_6_T_CV4, R_6_1_CV0, R_6_1_CV1, R_6_1_CV2, R_6_1_CV3, R_6_1_CV4, R_6_2_CV0, R_6_2_CV1, R_6_2_CV2, R_6_2_CV3, R_6_2_CV4)









def final_result(
	R_1_V, R_1_T_CV0, R_1_T_CV1, R_1_T_CV2, R_1_T_CV3, R_1_T_CV4, R_1_1_CV0, R_1_1_CV1, R_1_1_CV2, R_1_1_CV3, R_1_1_CV4, R_1_2_CV0, R_1_2_CV1, R_1_2_CV2, R_1_2_CV3, R_1_2_CV4,
	R_2_V, R_2_T_CV0, R_2_T_CV1, R_2_T_CV2, R_2_T_CV3, R_2_T_CV4, R_2_1_CV0, R_2_1_CV1, R_2_1_CV2, R_2_1_CV3, R_2_1_CV4, R_2_2_CV0, R_2_2_CV1, R_2_2_CV2, R_2_2_CV3, R_2_2_CV4,
	R_3_V, R_3_T_CV0, R_3_T_CV1, R_3_T_CV2, R_3_T_CV3, R_3_T_CV4, R_3_1_CV0, R_3_1_CV1, R_3_1_CV2, R_3_1_CV3, R_3_1_CV4, R_3_2_CV0, R_3_2_CV1, R_3_2_CV2, R_3_2_CV3, R_3_2_CV4,
	R_4_V, R_4_T_CV0, R_4_T_CV1, R_4_T_CV2, R_4_T_CV3, R_4_T_CV4, R_4_1_CV0, R_4_1_CV1, R_4_1_CV2, R_4_1_CV3, R_4_1_CV4, R_4_2_CV0, R_4_2_CV1, R_4_2_CV2, R_4_2_CV3, R_4_2_CV4,
	R_5_V, R_5_T_CV0, R_5_T_CV1, R_5_T_CV2, R_5_T_CV3, R_5_T_CV4, R_5_1_CV0, R_5_1_CV1, R_5_1_CV2, R_5_1_CV3, R_5_1_CV4, R_5_2_CV0, R_5_2_CV1, R_5_2_CV2, R_5_2_CV3, R_5_2_CV4,
	R_6_V, R_6_T_CV0, R_6_T_CV1, R_6_T_CV2, R_6_T_CV3, R_6_T_CV4, R_6_1_CV0, R_6_1_CV1, R_6_1_CV2, R_6_1_CV3, R_6_1_CV4, R_6_2_CV0, R_6_2_CV1, R_6_2_CV2, R_6_2_CV3, R_6_2_CV4) :
	print('---1---', flush=True)
	print('- Val MSE : {:.4f}'.format(R_1_V), flush=True)
	print('- Test MSE MEAN : {:.4f}'.format(np.mean([R_1_T_CV0, R_1_T_CV1, R_1_T_CV2, R_1_T_CV3, R_1_T_CV4])), flush=True)
	print('- Test MSE STD : {:.4f}'.format(np.std([R_1_T_CV0, R_1_T_CV1, R_1_T_CV2, R_1_T_CV3, R_1_T_CV4])), flush=True)
	print('- Test Pearson MEAN : {:.4f}'.format(np.mean([R_1_1_CV0, R_1_1_CV1, R_1_1_CV2, R_1_1_CV3, R_1_1_CV4])), flush=True)
	print('- Test Pearson STD : {:.4f}'.format(np.std([R_1_1_CV0, R_1_1_CV1, R_1_1_CV2, R_1_1_CV3, R_1_1_CV4])), flush=True)
	print('- Test Spearman : {:.4f}'.format(np.mean([R_1_2_CV0, R_1_2_CV1, R_1_2_CV2, R_1_2_CV3, R_1_2_CV4])), flush=True)
	print('- Test Spearman : {:.4f}'.format(np.std([R_1_2_CV0, R_1_2_CV1, R_1_2_CV2, R_1_2_CV3, R_1_2_CV4])), flush=True)
	print('---2---', flush=True)
	print('- Val MSE : {:.4f}'.format(R_2_V), flush=True)
	print('- Test MSE MEAN : {:.4f}'.format(np.mean([R_2_T_CV0, R_2_T_CV1, R_2_T_CV2, R_2_T_CV3, R_2_T_CV4])), flush=True)
	print('- Test MSE STD : {:.4f}'.format(np.std([R_2_T_CV0, R_2_T_CV1, R_2_T_CV2, R_2_T_CV3, R_2_T_CV4])), flush=True)
	print('- Test Pearson MEAN : {:.4f}'.format(np.mean([R_2_1_CV0, R_2_1_CV1, R_2_1_CV2, R_2_1_CV3, R_2_1_CV4])), flush=True)
	print('- Test Pearson STD : {:.4f}'.format(np.std([R_2_1_CV0, R_2_1_CV1, R_2_1_CV2, R_2_1_CV3, R_2_1_CV4])), flush=True)
	print('- Test Spearman : {:.4f}'.format(np.mean([R_2_2_CV0, R_2_2_CV1, R_2_2_CV2, R_2_2_CV3, R_2_2_CV4])), flush=True)
	print('- Test Spearman : {:.4f}'.format(np.std([R_2_2_CV0, R_2_2_CV1, R_2_2_CV2, R_2_2_CV3, R_2_2_CV4])), flush=True)
	print('---3---', flush=True)
	print('- Val MSE : {:.4f}'.format(R_3_V), flush=True)
	print('- Test MSE MEAN : {:.4f}'.format(np.mean([R_3_T_CV0, R_3_T_CV1, R_3_T_CV2, R_3_T_CV3, R_3_T_CV4])), flush=True)
	print('- Test MSE STD : {:.4f}'.format(np.std([R_3_T_CV0, R_3_T_CV1, R_3_T_CV2, R_3_T_CV3, R_3_T_CV4])), flush=True)
	print('- Test Pearson MEAN : {:.4f}'.format(np.mean([R_3_1_CV0, R_3_1_CV1, R_3_1_CV2, R_3_1_CV3, R_3_1_CV4])), flush=True)
	print('- Test Pearson STD : {:.4f}'.format(np.std([R_3_1_CV0, R_3_1_CV1, R_3_1_CV2, R_3_1_CV3, R_3_1_CV4])), flush=True)
	print('- Test Spearman : {:.4f}'.format(np.mean([R_3_2_CV0, R_3_2_CV1, R_3_2_CV2, R_3_2_CV3, R_3_2_CV4])), flush=True)
	print('- Test Spearman : {:.4f}'.format(np.std([R_3_2_CV0, R_3_2_CV1, R_3_2_CV2, R_3_2_CV3, R_3_2_CV4])), flush=True)
	print('---4---', flush=True)
	print('- Val MSE : {:.4f}'.format(R_4_V), flush=True)
	print('- Test MSE MEAN : {:.4f}'.format(np.mean([R_4_T_CV0, R_4_T_CV1, R_4_T_CV2, R_4_T_CV3, R_4_T_CV4])), flush=True)
	print('- Test MSE STD : {:.4f}'.format(np.std([R_4_T_CV0, R_4_T_CV1, R_4_T_CV2, R_4_T_CV3, R_4_T_CV4])), flush=True)
	print('- Test Pearson MEAN : {:.4f}'.format(np.mean([R_4_1_CV0, R_4_1_CV1, R_4_1_CV2, R_4_1_CV3, R_4_1_CV4])), flush=True)
	print('- Test Pearson STD : {:.4f}'.format(np.std([R_4_1_CV0, R_4_1_CV1, R_4_1_CV2, R_4_1_CV3, R_4_1_CV4])), flush=True)
	print('- Test Spearman : {:.4f}'.format(np.mean([R_4_2_CV0, R_4_2_CV1, R_4_2_CV2, R_4_2_CV3, R_4_2_CV4])), flush=True)
	print('- Test Spearman : {:.4f}'.format(np.std([R_4_2_CV0, R_4_2_CV1, R_4_2_CV2, R_4_2_CV3, R_4_2_CV4])), flush=True)
	print('---5---', flush=True)
	print('- Val MSE : {:.4f}'.format(R_5_V), flush=True)
	print('- Test MSE MEAN : {:.4f}'.format(np.mean([R_5_T_CV0, R_5_T_CV1, R_5_T_CV2, R_5_T_CV3, R_5_T_CV4])), flush=True)
	print('- Test MSE STD : {:.4f}'.format(np.std([R_5_T_CV0, R_5_T_CV1, R_5_T_CV2, R_5_T_CV3, R_5_T_CV4])), flush=True)
	print('- Test Pearson MEAN : {:.4f}'.format(np.mean([R_5_1_CV0, R_5_1_CV1, R_5_1_CV2, R_5_1_CV3, R_5_1_CV4])), flush=True)
	print('- Test Pearson STD : {:.4f}'.format(np.std([R_5_1_CV0, R_5_1_CV1, R_5_1_CV2, R_5_1_CV3, R_5_1_CV4])), flush=True)
	print('- Test Spearman : {:.4f}'.format(np.mean([R_5_2_CV0, R_5_2_CV1, R_5_2_CV2, R_5_2_CV3, R_5_2_CV4])), flush=True)
	print('- Test Spearman : {:.4f}'.format(np.std([R_5_2_CV0, R_5_2_CV1, R_5_2_CV2, R_5_2_CV3, R_5_2_CV4])), flush=True)
	print('---6---', flush=True)
	print('- Val MSE : {:.4f}'.format(R_6_V), flush=True)
	print('- Test MSE MEAN : {:.4f}'.format(np.mean([R_6_T_CV0, R_6_T_CV1, R_6_T_CV2, R_6_T_CV3, R_6_T_CV4])), flush=True)
	print('- Test MSE STD : {:.4f}'.format(np.std([R_6_T_CV0, R_6_T_CV1, R_6_T_CV2, R_6_T_CV3, R_6_T_CV4])), flush=True)
	print('- Test Pearson MEAN : {:.4f}'.format(np.mean([R_6_1_CV0, R_6_1_CV1, R_6_1_CV2, R_6_1_CV3, R_6_1_CV4])), flush=True)
	print('- Test Pearson STD : {:.4f}'.format(np.std([R_6_1_CV0, R_6_1_CV1, R_6_1_CV2, R_6_1_CV3, R_6_1_CV4])), flush=True)
	print('- Test Spearman : {:.4f}'.format(np.mean([R_6_2_CV0, R_6_2_CV1, R_6_2_CV2, R_6_2_CV3, R_6_2_CV4])), flush=True)
	print('- Test Spearman : {:.4f}'.format(np.std([R_6_2_CV0, R_6_2_CV1, R_6_2_CV2, R_6_2_CV3, R_6_2_CV4])), flush=True)












###### BEST result to plot 
###### BEST result to plot 
###### BEST result to plot 
###### BEST result to plot 
###### BEST result to plot 


#
# 3) total checkpoint best 
#	
#
import numpy as np
TOT_min = np.Inf
TOT_key = ""
for key in ANA_ALL_DF.keys():
	trial_min = min(ANA_ALL_DF[key]['AV_V_LS'])
	if trial_min < TOT_min :
		TOT_min = trial_min
		TOT_key = key
#

print('best val', flush=True)
print(TOT_key, flush=True)
mini_df = ANA_ALL_DF[TOT_key]
my_config = ANA_DF[ANA_DF.logdir==TOT_key]
cck_num =mini_df[mini_df.AV_V_LS==min(mini_df.AV_V_LS)].index.item()
checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
TOPVAL_PATH = TOT_key + checkpoint
print('best val check', flush=True)
print(TOPVAL_PATH, flush=True)
R_3_V = min(mini_df.AV_V_LS)
R_3_V
pred_list_0, ans_list_0 = copy.deepcopy(pred_3_CV0), copy.deepcopy(pred_3_CV0)
pred_list_1, ans_list_1 = copy.deepcopy(pred_3_CV1), copy.deepcopy(pred_3_CV1)
pred_list_2, ans_list_2 = copy.deepcopy(pred_3_CV2), copy.deepcopy(pred_3_CV2)
pred_list_3, ans_list_3 = copy.deepcopy(pred_3_CV3), copy.deepcopy(pred_3_CV3)
pred_list_4, ans_list_4 = copy.deepcopy(pred_3_CV4), copy.deepcopy(pred_3_CV4)



# (5) BEST cor final 내에서의 max cor 
#	
#
cck_num =mini_df[mini_df.AV_V_SC==max(mini_df.AV_V_SC)].index.item()
checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
TOPCOR_PATH = DF_KEY + checkpoint
print('best final check', flush=True)
print(TOPCOR_PATH, flush=True)
R_5_V = max(mini_df.AV_V_SC)
R_5_V
pred_list_0, ans_list_0 = copy.deepcopy(pred_5_CV0), copy.deepcopy(pred_5_CV0)
pred_list_1, ans_list_1 = copy.deepcopy(pred_5_CV1), copy.deepcopy(pred_5_CV1)
pred_list_2, ans_list_2 = copy.deepcopy(pred_5_CV2), copy.deepcopy(pred_5_CV2)
pred_list_3, ans_list_3 = copy.deepcopy(pred_5_CV3), copy.deepcopy(pred_5_CV3)
pred_list_4, ans_list_4 = copy.deepcopy(pred_5_CV4), copy.deepcopy(pred_5_CV4)



#
# (6) 그냥 최고 corr 
#	
#
import numpy as np
TOT_max = -np.Inf
TOT_key = ""
for key in ANA_ALL_DF.keys():
	trial_max = max(ANA_ALL_DF[key]['AV_V_SC'])
	if trial_max > TOT_max :
		TOT_max = trial_max
		TOT_key = key
#

print('best cor', flush=True)
print(TOT_key, flush=True)
mini_df = ANA_ALL_DF[TOT_key]
my_config = ANA_DF[ANA_DF.logdir==TOT_key]
cck_num =mini_df[mini_df.AV_V_SC==max(mini_df.AV_V_SC)].index.item()
checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
TOPCOR_PATH = TOT_key + checkpoint
print('best cor check', flush=True)
print(TOPCOR_PATH, flush=True)
R_6_V = max(mini_df.AV_V_SC)
R_6_V
pred_list_0, ans_list_0 = copy.deepcopy(pred_6_CV0), copy.deepcopy(pred_6_CV0)
pred_list_1, ans_list_1 = copy.deepcopy(pred_6_CV1), copy.deepcopy(pred_6_CV1)
pred_list_2, ans_list_2 = copy.deepcopy(pred_6_CV2), copy.deepcopy(pred_6_CV2)
pred_list_3, ans_list_3 = copy.deepcopy(pred_6_CV3), copy.deepcopy(pred_6_CV3)
pred_list_4, ans_list_4 = copy.deepcopy(pred_6_CV4), copy.deepcopy(pred_6_CV4)














# test 데이터 다시 가져오기 





CV_num = 0
train_key = 'CV{}_train'.format(CV_num)
val_key = 'CV{}_val'.format(CV_num)
test_key = 'CV{}_test'.format(CV_num)
#
train_no_dup = data_nodup_df3.loc[CV_ND_INDS[train_key]] # train val df 
val_no_dup = data_nodup_df3.loc[CV_ND_INDS[val_key]] # train val df 
test_no_dup = data_nodup_df2.loc[CV_ND_INDS[test_key]] # from test included df 
#
ABCS_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(train_no_dup.setset)]
ABCS_val = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(val_no_dup.setset)]
ABCS_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(test_no_dup.setset)]
#
train_ind = list(ABCS_train.index)
val_ind = list(ABCS_val.index)
test_ind = list(ABCS_test.index)


ABCS_test_result = ABCS_test[['DrugCombCCLE','type','cell_onehot' ]]
ABCS_test_result['ANS'] = ans_list_0
ABCS_test_result['PRED_0'] = pred_list_0
ABCS_test_result['PRED_1'] = pred_list_1
ABCS_test_result['PRED_2'] = pred_list_2
ABCS_test_result['PRED_3'] = pred_list_3
ABCS_test_result['PRED_4'] = pred_list_4

ABCS_test_result['PRED_M'] = ( ABCS_test_result[['PRED_0','PRED_1','PRED_2','PRED_3','PRED_4']].sum(axis = 1) )/ 5


DC_CELL_info_filt['tissue'] = [ '_'.join(a.split('_')[1:]) if type(a) == str else 'NA' for a in list(DC_CELL_info_filt['DrugCombCCLE'])]


ABCS_test_re = pd.merge(ABCS_test_result, DC_CELL_info_filt[['DrugCombCCLE','DC_cellname', 'tissue']], on = 'DrugCombCCLE', how = 'left'  )


test_cell_df = pd.DataFrame({'DC_cellname' : list(set(ABCS_test_re.DC_cellname))})

cell_P = []
cell_S = []
cell_num = []

for cell in list(test_cell_df.DC_cellname) :
	tmp_test_re = ABCS_test_re[ABCS_test_re.DC_cellname == cell]
	cell_P_corr, _ = stats.pearsonr(tmp_test_re.ANS, tmp_test_re.PRED_M)
	cell_S_corr, _ = stats.spearmanr(tmp_test_re.ANS, tmp_test_re.PRED_M)
	cell_nums = tmp_test_re.shape[0]
	cell_P.append(cell_P_corr)
	cell_S.append(cell_S_corr)
	cell_num.append(cell_nums)


test_cell_df['P_COR'] = cell_P
test_cell_df['S_COR'] = cell_S
test_cell_df['cell_num'] = cell_num

test_cell_df = pd.merge(test_cell_df, DC_CELL_info_filt[['DC_cellname','tissue']], on = 'DC_cellname', how = 'left'  )

tissue_set = ['CENTRAL_NERVOUS_SYSTEM', 'LUNG', 'BREAST', 'BONE', 'OVARY', 'PROSTATE', 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'LARGE_INTESTINE', 'SKIN' ] # list(set(test_cell_df['tissue']))
color_set = ['#CF3476','#FF7514','#025669','#308446','#84C3BE','#D53032','#4ddcfd','#ffcd36','#ac8cff'] # "#20603D","#828282","#1E1E1E"
color_dict = {a : color_set[tissue_set.index(a)] for a in tissue_set}

test_cell_df['tissue_oh'] = [color_dict[a] for a in list(test_cell_df['tissue'])]





									# test color set 
									구린버전
										tissue_set = ['CENTRAL_NERVOUS_SYSTEM', 'LUNG',  'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE','BREAST','LARGE_INTESTINE', 'BONE',  'SKIN', 'PROSTATE',  'OVARY' ] # list(set(test_cell_df['tissue']))
										color_set = ["#FFA420","#826C34","#D36E70","#705335","#57A639","#434B4D","#C35831","#B32821","#FAD201","#20603D","#828282","#1E1E1E"] # "#20603D","#828282","#1E1E1E"

								이쁜버전 
									color_map = pd.DataFrame({
										'tissue':['CENTRAL_NERVOUS_SYSTEM', 'LUNG', 'BREAST', 'BONE', 'OVARY', 'PROSTATE', 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'LARGE_INTESTINE', 'SKIN'],
										'my_col':['#CF3476','#FF7514','#025669','#308446','#84C3BE','#D53032','#4ddcfd','#ffcd36','#ac8cff']
										})
									Telemagenta / Pastel Orange / Azure Blue / Traffic Green / light green / strawberry red / 옅은 파랑 /노랑 /보라






# Spearman corr
test_cell_df = test_cell_df.sort_values('S_COR')

fig = plt.figure(figsize=(22,15))
x_pos = [a*2 for a in range(test_cell_df.shape[0])]

plt.bar(x_pos, test_cell_df['S_COR'], color=test_cell_df['tissue_oh']) # 
plt.xticks(x_pos,list(test_cell_df['DC_cellname']), rotation=90, fontsize= 18)
for i in range(test_cell_df.shape[0]):
	plt.annotate(str(list(test_cell_df['cell_num'])[i]), xy=(x_pos[i],list(test_cell_df['S_COR'])[i]), ha='center', va='bottom', fontsize= 18)

plt.legend()
plt.tight_layout()
fig.savefig('{}/{}.png'.format(PRJ_PATH, 'test_cell_spearman'), bbox_inches = 'tight')
plt.close()


# Pearson corr
test_cell_df = test_cell_df.sort_values('P_COR')

fig = plt.figure(figsize=(22,15))
x_pos = [a*2 for a in range(test_cell_df.shape[0])]

plt.bar(x_pos, test_cell_df['P_COR'], color=test_cell_df['tissue_oh']) # 
plt.xticks(x_pos,list(test_cell_df['DC_cellname']), rotation=90, fontsize= 18)
for i in range(test_cell_df.shape[0]):
	plt.annotate(str(list(test_cell_df['cell_num'])[i]), xy=(x_pos[i],list(test_cell_df['P_COR'])[i]), ha='center', va='bottom', fontsize= 18)

plt.legend()
plt.tight_layout()
fig.savefig('{}/{}.png'.format(PRJ_PATH, 'test_cell_pearson'), bbox_inches = 'tight')
plt.close()


test_cell_df


# 이쁜 그림을 위한 func 

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
fig.savefig('{}/{}.png'.format(PRJ_PATH, 'new_plot_pearson'), bbox_inches = 'tight')
plt.close()











# 추가적으로 궁금해져서 
# AXBX 의 비율을 살펴보기로 함 
# total 10552
TEST_ALL_O = ABCS_test_result[ABCS_test_result.type.isin(['AOBO'])] # 1348
TEST_HALF_O = ABCS_test_result[ABCS_test_result.type.isin(['AXBO','AOBX'])] # 2179
TEST_NO_O = ABCS_test_result[ABCS_test_result.type.isin(['AXBX'])] # 7025


DC_CELL_info_filt['tissue'] = [ '_'.join(a.split('_')[1:]) if type(a) == str else 'NA' for a in list(DC_CELL_info_filt['DrugCombCCLE'])]



def give_test_result_corDF (ABCS_test_result) : 
	THIS_TEST = pd.merge(ABCS_test_result, DC_CELL_info_filt[['DrugCombCCLE','DC_cellname', 'tissue']], on = 'DrugCombCCLE', how = 'left'  )
	test_cell_df = pd.DataFrame({'DC_cellname' : list(set(THIS_TEST.DC_cellname))})
	#
	cell_P = []
	cell_S = []
	cell_num = []
	#
	for cell in list(test_cell_df.DC_cellname) :
		tmp_test_re = THIS_TEST[THIS_TEST.DC_cellname == cell]
		if tmp_test_re.shape[0] > 1 :
			cell_P_corr, _ = stats.pearsonr(tmp_test_re.ANS, tmp_test_re.PRED_M)
			cell_S_corr, _ = stats.spearmanr(tmp_test_re.ANS, tmp_test_re.PRED_M)
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
	test_cell_df = pd.merge(test_cell_df, DC_CELL_info_filt[['DC_cellname','tissue']], on = 'DC_cellname', how = 'left'  )
	#
	tissue_set = ['CENTRAL_NERVOUS_SYSTEM', 'LUNG',  'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE','BREAST','LARGE_INTESTINE', 'BONE',  'SKIN', 'PROSTATE',  'OVARY' ] # list(set(test_cell_df['tissue']))
	color_set = ["#FFA420","#826C34","#D36E70","#705335","#57A639","#434B4D","#C35831","#B32821","#FAD201","#20603D","#828282","#1E1E1E"] # "#20603D","#828282","#1E1E1E"
	color_dict = {a : color_set[tissue_set.index(a)] for a in tissue_set}
	#
	test_cell_df['tissue_oh'] = [color_dict[a] for a in list(test_cell_df['tissue'])]
	test_cell_df = test_cell_df.sort_values('S_COR')
	return test_cell_df


TEST_ALL_O_RESULT = give_test_result_corDF(TEST_ALL_O)
TEST_HALF_O_RESULT = give_test_result_corDF(TEST_HALF_O)
TEST_NO_O_RESULT = give_test_result_corDF(TEST_NO_O)














######## cell line rank check (from P02.cellline_ABC.py)


avail_cell_list = list(set(A_B_C_S_SET_SM.DrugCombCCLE))

# total drugcomb data 확인 
DC_PATH = '/st06/jiyeonH/13.DD_SESS/DrugComb.1.5/' 

DC_tmp_DF1 = pd.read_csv(DC_PATH+'TOTAL_BLOCK.csv', low_memory=False) # 1469182
DC_tmp_DF1_re = DC_tmp_DF1[['drug_row_id','drug_col_id','cell_line_id','synergy_loewe','quality']]
DC_tmp_DF1_re['drug_row_id_re'] = [float(a) for a in list(DC_tmp_DF1_re['drug_row_id'])]
DC_tmp_DF1_re['drug_col_id_re'] = [float(a) for a in list(DC_tmp_DF1_re['drug_col_id'])]

DC_tmp_DF2 = DC_tmp_DF1_re[DC_tmp_DF1_re['quality'] != 'bad'] # 1457561
DC_tmp_DF3 = DC_tmp_DF2[(DC_tmp_DF2.drug_row_id_re > 0 ) & (DC_tmp_DF2.drug_col_id_re > 0 )] # 740932
DC_tmp_DF4 = DC_tmp_DF3[DC_tmp_DF3.cell_line_id>0].drop_duplicates() # 740884


# Drug info 
with open(DC_PATH+'drugs.json') as json_file :
	DC_DRUG =json.load(json_file)

DC_DRUG_K = list(DC_DRUG[0].keys())
DC_DRUG_DF = pd.DataFrame(columns=DC_DRUG_K)

for DD in range(1,len(DC_DRUG)):
	tmpdf = pd.DataFrame({k:[DC_DRUG[DD][k]] for k in DC_DRUG_K})
	DC_DRUG_DF = pd.concat([DC_DRUG_DF, tmpdf], axis = 0)

DC_DRUG_DF['id_re'] = [float(a) for a in list(DC_DRUG_DF['id'])]

DC_DRUG_DF_FULL = pd.read_csv('/st06/jiyeonH/11.TOX/MY_TRIAL_6/DC_DRUG_DF_PC.csv', sep ='\t')

DC_lengs = list(DC_DRUG_DF_FULL.leng)
DC_lengs2 = [int(a) for a in DC_lengs if a!= 'error']





# Cell info 
with open(DC_PATH+'cell_lines.json') as json_file :
	DC_CELL =json.load(json_file)

DC_CELL_K = list(DC_CELL[0].keys())
DC_CELL_DF = pd.DataFrame(columns=DC_CELL_K)

for DD in range(1,len(DC_CELL)):
	tmpdf = pd.DataFrame({k:[DC_CELL[DD][k]] for k in DC_CELL_K})
	DC_CELL_DF = pd.concat([DC_CELL_DF, tmpdf], axis = 0)

DC_CELL_DF2 = DC_CELL_DF[['id','name','cellosaurus_accession', 'ccle_name']] # 2319
DC_CELL_DF2.columns = ['cell_line_id', 'DC_cellname','DrugCombCello', 'DrugCombCCLE']






print("DC filtering")

DC_DATA_filter = DC_tmp_DF2[['drug_row_id_re','drug_col_id_re','cell_line_id','synergy_loewe']] # 1457561
DC_DATA_filter2 = DC_DATA_filter.drop_duplicates() # 1374958 -> 1363698

DC_DATA_filter3 = DC_DATA_filter2[DC_DATA_filter2.drug_row_id_re > 0] # 1374958 -> 1363698
DC_DATA_filter4 = DC_DATA_filter3[DC_DATA_filter3.drug_col_id_re > 0] # 751450 -> 740884
DC_DATA_filter4.cell_line_id # unique 295
DC_DATA_filter4[['drug_row_id_re','drug_col_id_re','cell_line_id']].drop_duplicates() # 648516
DC_DATA_filter4[['drug_row_id_re','drug_col_id_re']].drop_duplicates() # 75174
len(list(set(list(DC_DATA_filter4.drug_row_id_re) + list(DC_DATA_filter4.drug_col_id_re)))) # 4327

DC_DATA_filter4 = DC_DATA_filter4.reset_index(drop = False)

DC_DATA_filter5 = pd.merge(DC_DATA_filter4, DC_CELL_DF2[['cell_line_id','DrugCombCCLE']], on = 'cell_line_id', how ='left' )



# 그래서 drugcomb 에서 일단 사용된 내용들 CID 기준 
DC_DATA_filter6 = DC_DATA_filter5[DC_DATA_filter5.DrugCombCCLE.isin(avail_cell_list)]

good_ind = [a for a in range(DC_DRUG_DF_FULL.shape[0]) if type(DC_DRUG_DF_FULL.CAN_SMILES[a]) == str ]
DC_DRUG_DF_FULL_filt = DC_DRUG_DF_FULL.loc[good_ind]

DC_DRUG_DF_FULL_filt['leng2'] = [int(a) for a in list(DC_DRUG_DF_FULL_filt.leng)]
DC_DRUG_DF_FULL_filt = DC_DRUG_DF_FULL_filt[DC_DRUG_DF_FULL_filt.leng2 <=50] # 7775

DC_DRUG_DF_FULL_cut = DC_DRUG_DF_FULL_filt[['id','CID','CAN_SMILES']] # DrugComb 에서 combi 할 수 있는 총 CID : 7775개 cid 
DC_DRUG_DF_FULL_cut.columns = ['drug_row_id_re','ROW_CID','ROW_CAN_SMILES']

# 있는 combi 에 대한 CID 붙이기 
DC_re_1 = pd.merge(DC_DATA_filter6, DC_DRUG_DF_FULL_cut, on = 'drug_row_id_re', how = 'left') # 146942

DC_DRUG_DF_FULL_cut.columns = ['drug_col_id_re','COL_CID', 'COL_CAN_SMILES']
DC_re_2 = pd.merge(DC_re_1, DC_DRUG_DF_FULL_cut, on = 'drug_col_id_re', how = 'left')

DC_DRUG_DF_FULL_cut.columns = ['id','CID','CAN_SMILES']

DC_re_3 = DC_re_2[['ROW_CID','COL_CID','DrugCombCCLE']].drop_duplicates()
DC_re_4 = DC_re_3.reset_index(drop = True)




목적 : training 에 활용한 DC 애들 말고, 그 CID 들에 해당하는 모든 조합중에서 우리 train 에 안쓰인 애들. 맞나? 

from itertools import combinations
from itertools import product
from itertools import permutations

DC_all_cids = list(set(DC_DRUG_DF_FULL_cut[DC_DRUG_DF_FULL_cut.CID>0]['CID'])) # 7775개 (SM 있고, 50 이하에 leng 붙는 애들 )
DC_pairs = list(combinations(DC_all_cids, 2)) 
# permutation : 모든 cid - cid 양면 
# combination : unique 한 cid - cid 


# 그러고 나서 DC 안에 있는 모든 CID - CID - Cello triads 조사
IN_DC_pairs_1 = [(DC_re_4.ROW_CID[a] ,DC_re_4.COL_CID[a], DC_re_4.DrugCombCCLE[a]) for a in range(DC_re_4.shape[0])]
IN_DC_pairs_2 = [(DC_re_4.COL_CID[a] ,DC_re_4.ROW_CID[a], DC_re_4.DrugCombCCLE[a]) for a in range(DC_re_4.shape[0])]
IN_DC_pairs = IN_DC_pairs_1 + IN_DC_pairs_2 # 239,044

# 혹시 내가 썼던 모델에 대한 파일에 관련된 애들은 빠져있나? 
# A_B_C_S_SET_ADD_CH  = A_B_C_S_SET_ADD[A_B_C_S_SET_ADD.DrugCombCCLE.isin(avail_cell_list)]
# A_B_C_S_SET_ADD_CH = A_B_C_S_SET_ADD_CH.reset_index(drop = True)
# ADD_CHECK_1 = [(A_B_C_S_SET_ADD_CH.drug_row_CID[a] ,A_B_C_S_SET_ADD_CH.drug_col_CID[a], A_B_C_S_SET_ADD_CH.DrugCombCCLE[a]) for a in range(A_B_C_S_SET_ADD_CH.shape[0])]
# ADD_CHECK_2 = [(A_B_C_S_SET_ADD_CH.drug_col_CID[a] ,A_B_C_S_SET_ADD_CH.drug_row_CID[a], A_B_C_S_SET_ADD_CH.DrugCombCCLE[a]) for a in range(A_B_C_S_SET_ADD_CH.shape[0])]
# ADD_CHECK = ADD_CHECK_1 + ADD_CHECK_2
# che = list(set(ADD_CHECK) - set(IN_DC_pairs) )






# 사용하는 cell line 별로 test 대상 선별해서 저장하기 
# 오래걸려! 
# c = 'CVCL_0031' # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

import json

# mkdir PRJ_PATH+'VAL/'
CELVAL_PATH = PRJ_PATH + 'VAL/'
os.makedirs(CELVAL_PATH, exist_ok=True)

def save_cell_json (cell_name) :
	this_list = [(a,b,cell_name) for a,b in DC_pairs]
	NOT_in_DC_pairs = set(this_list) - set(IN_DC_pairs)
	VAL_LIST = list(NOT_in_DC_pairs)
	with open(CELVAL_PATH+'{}.json'.format(cell_name), 'w') as f:
		json.dump(VAL_LIST, f)


for cell_name in avail_cell_list :
	save_cell_json(cell_name)






# 1) 그거에 맞게 drug feature 저장하기 -> DC 에 있는 전체 CID 에 대해서 그냥 진행한거니까 
# 1) 그거에 맞게 drug feature 저장하기 -> 이제 다시 안만들어도 됨 그냥 복사하셈 
# 1) 그거에 맞게 drug feature 저장하기 -> 전체 7775 개 
# 1) 그거에 맞게 drug feature 저장하기 

PC_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/'

for_CAN_smiles = pd.read_csv(PC_PATH+'CID_SMILES.csv', low_memory = False)



def get_CHEM(cid, k=1):
	maxNumAtoms = max_len
	smiles = for_CAN_smiles[for_CAN_smiles.CID == cid]['CAN_SMILES'].item()
	iMol = Chem.MolFromSmiles(smiles.strip())
	iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(iMol) 
	iFeature = np.zeros((maxNumAtoms, 64)) ################## feature 크기 고정 
	iFeatureTmp = []
	for atom in iMol.GetAtoms():
		iFeatureTmp.append( atom_feature(atom) )### atom features only
	iFeature[0:len(iFeatureTmp), 0:64] = iFeatureTmp ### 0 padding for feature-set
	iAdj = np.zeros((maxNumAtoms, maxNumAtoms))
	iAdj[0:len(iFeatureTmp), 0:len(iFeatureTmp)] = iAdjTmp + np.eye(len(iFeatureTmp))
	ADJ = adj_k(np.asarray(iAdj), k)
	return iFeature, ADJ


def atom_feature(atom):
	ar = np.array(one_of_k_encoding_unk(atom.GetSymbol(),
									  ['C', 'N', 'O', 'S', 'F', 'Cl', 'P', 'F', 'Br', 'I',
									   'Na', 'Fe', 'B', 'Mg', 'Al', 'Si', 'K', 'H', 'Se', 'Ca',
									   'Zn', 'As', 'Mo', 'V', 'Cu', 'Hg', 'Cr', 'Co', 'Bi','Tc',
									   'Sb', 'Gd', 'Li', 'Ag', 'Au', 'Unknown']) +
					one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7]) +
					one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
					one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4]) +
					one_of_k_encoding_unk(atom.GetFormalCharge(), [-3, -2, -1, 0, 1, 2, 3, 4, 5]) +
					[atom.GetIsAromatic()])    # (36, 8, 5, 5, 9, 1) -> 64 
	return ar



def one_of_k_encoding(x, allowable_set):
	if x not in allowable_set:
		raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
	#print list((map(lambda s: x == s, allowable_set)))
	return list(map(lambda s: x == s, allowable_set))



def one_of_k_encoding_unk(x, allowable_set):
	"""Maps inputs not in the allowable set to the last element."""
	if x not in allowable_set:
		x = allowable_set[-1]
	return list(map(lambda s: x == s, allowable_set))

def convertAdj(adj):
	dim = len(adj)
	a = adj.flatten()
	b = np.zeros(dim*dim)
	c = (np.ones(dim*dim)-np.equal(a,b)).astype('float64')
	d = c.reshape((dim, dim))
	return d



def adj_k(adj, k): # 근데 k 가 왜 필요한거지 -> 민지 말에 의하면 
	ret = adj
	for i in range(0, k-1):
		ret = np.dot(ret, adj)  
	return convertAdj(ret)



DC_DRUG_DF_FULL_filt2 = DC_DRUG_DF_FULL_filt.reset_index(drop = True)

max_len = 50
MY_chem_A_feat = torch.empty(size=(DC_DRUG_DF_FULL_filt2.shape[0], max_len, 64))
MY_chem_A_adj = torch.empty(size=(DC_DRUG_DF_FULL_filt2.shape[0], max_len, max_len))


for IND in range(DC_DRUG_DF_FULL_filt2.shape[0]): #  
	if IND%100 == 0 : 
		print(str(IND)+'/'+str(DC_DRUG_DF_FULL_filt2.shape[0]) )
		datetime.now()
	#
	DrugA_CID = DC_DRUG_DF_FULL_filt2['CID'][IND]
	#
	k=1
	DrugA_Feat, DrugA_ADJ = get_CHEM(DrugA_CID, k)
	# 
	MY_chem_A_feat[IND] = torch.Tensor(DrugA_Feat)
	MY_chem_A_adj[IND] = torch.Tensor(DrugA_ADJ)


SAVE_PATH = CELVAL_PATH

torch.save(MY_chem_A_feat, SAVE_PATH+'DC_ALL.MY_chem_feat.pt')
torch.save(MY_chem_A_adj, SAVE_PATH+'DC_ALL.MY_chem_adj.pt')

DC_DRUG_DF_FULL_filt2.to_csv(SAVE_PATH+'DC_ALL_7555_ORDER.csv')



# 2) 그거에 맞게 MJ EXP feauture 저장하기 # 읽는데 한세월 
# 2) 그거에 맞게 MJ EXP feauture 저장하기 
# 2) 그거에 맞게 MJ EXP feauture 저장하기 
# 2) 그거에 맞게 MJ EXP feauture 저장하기 


MJ_DIR = '/st06/jiyeonH/13.DD_SESS/01.PRJ2/'

# train 에서 활용해서 주는거
MJ_request_ANS_for_train = pd.read_csv(MJ_DIR+'PRJ2_EXP_ccle_fugcn_hst1.csv') 
# 내가 요청한 전부 
MJ_request_ANS_FULL = pd.read_csv(MJ_DIR+'PRJ2ver2_EXP_ccle_fugcn_hst1.csv') # 349 



set(MJ_request_ANS_FULL.columns) - set(MJ_request_ANS_for_train.columns)
set(MJ_request_ANS_for_train.columns) - set(MJ_request_ANS_FULL.columns)


ORD = [list(MJ_request_ANS_FULL.entrez_id).index(a) for a in BETA_ENTREZ_ORDER]
MJ_request_ANS_FULL = MJ_request_ANS_FULL.iloc[ORD]


colnana = list(MJ_request_ANS_FULL.columns)[3:]
MJ_tuples_1 = [a for a in colnana if '__' in a]
MJ_tuples_2 = [(a.split('__')[0], a.split('__')[1]) for a in colnana ]

MJ_tup_df = pd.DataFrame()

MJ_tup_df['sample'] = MJ_tuples_1 
MJ_tup_df['tuple'] = MJ_tuples_2

MJ_exp_list = []

for IND in range(MJ_tup_df.shape[0]) :
	if IND%100 == 0 :
		print(str(IND)+'/'+str(MJ_tup_df.shape[0]) )
		datetime.now()
	Column = MJ_tup_df['sample'][IND]
	MJ_vector = MJ_request_ANS_FULL[Column].values.tolist()
	MJ_exp_list.append(MJ_vector)


MJ_TENSOR = torch.Tensor(MJ_exp_list)

SAVE_PATH = CELVAL_PATH

torch.save(MJ_TENSOR, SAVE_PATH+'AVAIL_EXP_TOT.pt')

MJ_tup_df.to_csv(SAVE_PATH+'AVAIL_EXP_TOT.csv')










# 3) 그거에 맞게 Target 저장하기 # target 종류 따라서 저장해줘야함 
# 3) 그거에 맞게 Target 저장하기 # 그래도 얼마 안걸림 
# 3) 그거에 맞게 Target 저장하기 
# 3) 그거에 맞게 Target 저장하기 




(2) 14 & 15 NEW TARGET 

DC_TOT_CIDS = DC_DRUG_DF_FULL[['CID','leng']]
DC_TOT_CIDS = DC_TOT_CIDS[DC_TOT_CIDS.CID>0]
total_DC_CIDs = set(DC_TOT_CIDS['CID'])
gene_ids = list(BETA_ORDER_DF.gene_id)

DC_TOT_CIDS = DC_TOT_CIDS.reset_index(drop = True)

TARGET_PATH = '/st06/jiyeonH/13.DD_SESS/01.PRJ2/'
TARGET_DB = pd.read_csv(TARGET_PATH+'TARGET_CID_ENTREZ.csv', sep ='\t', index_col = 0)

#A_B_C_S_SET_CIDS = list(set(list(A_B_C_S_SET_ADD.drug_row_CID)+list(A_B_C_S_SET_ADD.drug_col_CID)))
#TARGET_DB_RE = TARGET_DB[TARGET_DB.CID_RE.isin(A_B_C_S_SET_CIDS)] # 없는 애도 데려가야해 
#TARGET_DB_RE = TARGET_DB_RE[TARGET_DB_RE.ENTREZ_RE.isin(gene_ids)]




target_cids = copy.deepcopy(total_DC_CIDs)
gene_ids = list(BETA_ORDER_DF.gene_id)


def get_targets(CID): # 데려 가기로 함 
	if CID in target_cids:
		tmp_df2 = TARGET_DB[TARGET_DB.CID == CID]
		targets = list(set(tmp_df2.EntrezID))
		vec = [1 if a in targets else 0 for a in gene_ids ]
	else :
		#vec = [0] * 978
		vec = [0] * 349
	return vec


TARGETs = []

for IND in range(DC_TOT_CIDS.shape[0]) :
	if IND%100 == 0 :
		print(str(IND)+'/'+str(DC_TOT_CIDS.shape[0]) )
		datetime.now()
	CID = DC_TOT_CIDS['CID'][IND]
	target_vec = get_targets(CID)
	TARGETs.append(target_vec)
	

TARGET_TENSOR = torch.Tensor(TARGETs)

SAVE_PATH = CELVAL_PATH
torch.save(TARGET_TENSOR, SAVE_PATH+'DC_ALL_TARGET.pt')

DC_TOT_CIDS.to_csv(SAVE_PATH+'DC_ALL_TARGET.csv')






# 4) 그거에 맞게 Cell Basal 저장하기 
# 4) 그거에 맞게 Cell Basal 저장하기 
# 4) 그거에 맞게 Cell Basal 저장하기 
# 4) 그거에 맞게 Cell Basal 저장하기 
# 4) 그거에 맞게 Cell Basal 저장하기
# 4) 그거에 맞게 Cell Basal 저장하기 
# avail_cell_list 필요 



CCLE_PATH = '/st06/jiyeonH/13.DD_SESS/CCLE.22Q1/'

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


DC_CELL_DF_ids = set(DC_CELL_DF.ccle_name) # 1659
ccle_cell_ids = set(ccle_cell_info.DrugCombCCLE) # 1672
# DC_CELL_DF_ids - ccle_cell_ids = 205
# ccle_cell_ids - DC_CELL_DF_ids = 218




DC_CELL_DF3 = DC_CELL_DF2[DC_CELL_DF2.DrugCombCCLE.isin(avail_cell_list)]
DC_CELL_DF3 = DC_CELL_DF3.reset_index(drop = True)

cell_df = ccle_exp3[ccle_exp3.DrugCombCCLE.isin(avail_cell_list)]

# ccle supporting cvcl : len(set(cell_df.DrugCombCCLE)) 25! all in

cell_basal_exp_list = []
# give vector 
for i in range(DC_CELL_DF3.shape[0]) :
	if i%100 == 0 :
		print(str(i)+'/'+str(DC_CELL_DF3.shape[0]) )
		datetime.now()
	cello = DC_CELL_DF3['DrugCombCCLE'][i]
	if cello in ccle_cello_names : 
		ccle_exp_df = cell_df[cell_df.DrugCombCCLE==cello][BETA_ENTREZ_ORDER]
		ccle_exp_vector = ccle_exp_df.values[0].tolist()
		cell_basal_exp_list.append(ccle_exp_vector)
	else : # no worries here. 
		#ccle_exp_vector = [0]*978
		ccle_exp_vector = [0]*349
		cell_basal_exp_list.append(ccle_exp_vector)

cell_base_tensor = torch.Tensor(cell_basal_exp_list)

# 
SAVE_PATH = CELVAL_PATH
torch.save(cell_base_tensor, SAVE_PATH+'AVAIL_CLL_MY_CellBase.pt')

DC_CELL_DF3.to_csv(SAVE_PATH + 'AVAIL_CELL_DF.csv')








# 5) 그거에 맞게 LINCS EXP 는 또 따로 저장하기 

BETA_BIND = pd.read_csv('/st06/jiyeonH/11.TOX/MY_TRIAL_5/'+"BETA_DATA_for_20220705_978.csv")
BETA_CP_info = pd.read_table('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+'compoundinfo_beta.txt') # pert 34419 , cansmiles : 
BETA_CEL_info = pd.read_table('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+'cellinfo_beta.txt')
BETA_SIG_info = pd.read_table('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+'siginfo_beta.txt', low_memory = False)

filter1 = BETA_SIG_info[BETA_SIG_info.pert_type.isin(['ctl_vehicle', 'ctl_untrt' ,'trt_cp' ])]
filter2 = filter1[filter1.is_exemplar_sig==1]
BETA_CP_info_filt = BETA_CP_info[['pert_id','canonical_smiles']].drop_duplicates() # 34419
can_sm_re = pd.read_csv('/st06/jiyeonH/11.TOX/DR_SPRING/tmp/can_sm_conv', sep = '\t', header = None)
can_sm_re.columns = ['canonical_smiles','CONV_CID']
can_sm_re = can_sm_re.drop_duplicates()
len(set([a for a in BETA_CP_info['pert_id'] if type(a) == str])) # 34419
len(set([a for a in can_sm_re['canonical_smiles'] if type(a) == str])) # 28575
len(set(can_sm_re[can_sm_re.CONV_CID>0]['CONV_CID'])) # 27841

can_sm_re2 = pd.merge(BETA_CP_info_filt, can_sm_re, on = 'canonical_smiles', how = 'left') # 34419 -> 1 sm 1 cid 확인 
can_sm_re3 = can_sm_re2[['pert_id','canonical_smiles','CONV_CID']].drop_duplicates() # 


BETA_MJ = pd.read_csv('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+"Lincs_pubchem_mj.csv") # SID 필터까지 그렇게 완성한 무엇 

BETA_MJ_RE = BETA_MJ[['pert_id','SMILES_cid','canonical_smiles']].drop_duplicates() # 25903
BETA_MJ_RE_CK = BETA_MJ_RE[['pert_id','SMILES_cid']]
len(set([a for a in BETA_MJ_RE['pert_id'] if type(a) == str])) # 25903
len(set([a for a in BETA_MJ_RE['canonical_smiles'] if type(a) == str])) # 25864
len(set(BETA_MJ_RE_CK[BETA_MJ_RE_CK.SMILES_cid>0]['SMILES_cid'])) # 25642

check = pd.merge(can_sm_re3, BETA_MJ_RE_CK, on = 'pert_id', how = 'left' )
check2 = check[check.CONV_CID != check.SMILES_cid]
check3 = check2[check2.SMILES_cid > 0 ]
check4 = check3[check3.CONV_CID > 0 ] # 순수하게 다른 애들 

pert_id_match = check[check.CONV_CID == check.SMILES_cid][['pert_id','canonical_smiles','CONV_CID']]
conv_win = check2[(check2.CONV_CID >0 ) & ( np.isnan(check2.SMILES_cid)==True)][['pert_id','canonical_smiles','CONV_CID']]
mj_win = check2[(check2.SMILES_cid >0 ) & ( np.isnan(check2.CONV_CID)==True)][['pert_id','canonical_smiles','SMILES_cid']]
nans = check2[(np.isnan(check2.SMILES_cid)==True ) & ( np.isnan(check2.CONV_CID)==True)] # 5995
nans2 = nans[nans.pert_id.isin(filter2.pert_id)]
nans3 = nans2[-nans2.canonical_smiles.isin(['restricted', np.nan])]

pert_id_match.columns = ['pert_id','canonical_smiles','CID'] # 25418,
conv_win.columns = ['pert_id','canonical_smiles','CID'] # 2521,
mj_win.columns =['pert_id','canonical_smiles','CID']

individual_check = check4.reset_index(drop =True)

# 하나하나 결국 살펴봄 
individual_check_conv = individual_check.loc[[0,4,5,6,10,11,12,13,16,17,18,19]+[a for a in range(21,34)]+[36,40,54]][['pert_id','canonical_smiles','CONV_CID']]
individual_check_mj = individual_check.loc[[1,2,3,7,8,9,14,15,20,34,35,37,38,39]+[a for a in range(41,54)]+[55,56,57]][['pert_id','canonical_smiles','SMILES_cid']]
individual_check_conv.columns = ['pert_id','canonical_smiles','CID'] # 28
individual_check_mj.columns = ['pert_id','canonical_smiles','CID'] # 30 

LINCS_PERT_MATCH = pd.concat([pert_id_match, conv_win, mj_win, individual_check_conv,  individual_check_mj]) # 28424
len(set([a for a in LINCS_PERT_MATCH['pert_id'] if type(a) == str])) # 34419 -> 28424
len(set([a for a in LINCS_PERT_MATCH['canonical_smiles'] if type(a) == str])) # 28575 -> 28381
len(set(LINCS_PERT_MATCH[LINCS_PERT_MATCH.CID>0]['CID'])) # 27841 -> 28154
LINCS_PERT_MATCH_cids = list(set(LINCS_PERT_MATCH.CID))

BETA_EXM = pd.merge(filter2, LINCS_PERT_MATCH, on='pert_id', how = 'left')
BETA_EXM2 = BETA_EXM[BETA_EXM.CID > 0] # 128038 # 이건 늘어났음 

BETA_CEL_info2 = BETA_CEL_info[['cell_iname','cellosaurus_id','ccle_name']] # 240 
BETA_SELEC_SIG_wCell = pd.merge(BETA_EXM2, BETA_CEL_info2, on = 'cell_iname', how = 'left') # 128038
BETA_SELEC_SIG_wCell2 = BETA_SELEC_SIG_wCell[['pert_id','CID','cellosaurus_id','ccle_name','sig_id']]
BETA_SELEC_SIG_wCell2 = BETA_SELEC_SIG_wCell2.drop_duplicates() # 128038

cello_tt=[True if type(a)==str else False for a in list(BETA_SELEC_SIG_wCell2.ccle_name)] 
BETA_CID_CELLO_SIG = BETA_SELEC_SIG_wCell2[cello_tt][['pert_id','CID','ccle_name','sig_id']].drop_duplicates() # 111012
beta_cid_cello_sig_tf = [ True if a>0 else False for a in list(BETA_CID_CELLO_SIG.CID)]
BETA_CID_CELLO_SIG = BETA_CID_CELLO_SIG[beta_cid_cello_sig_tf]
BETA_CID_CELLO_SIG['CID'] = [int(a) for a in list(BETA_CID_CELLO_SIG['CID']) ] # 111012 




ORD = [list(BETA_BIND.id).index(a) for a in BETA_ENTREZ_ORDER]
BETA_BIND_ORD = BETA_BIND.iloc[ORD]


BETA_CID_CELLO_SIG_re = BETA_CID_CELLO_SIG.reset_index(drop = True)

BETA_CID_CELLO_SIG_tup = [(BETA_CID_CELLO_SIG_re.CID[a], BETA_CID_CELLO_SIG_re.ccle_name[a]) for a in range(BETA_CID_CELLO_SIG_re.shape[0])]
BETA_CID_CELLO_SIG_re['tuple'] = BETA_CID_CELLO_SIG_tup
BETA_CID_CELLO_SIG_tup_re = [(str(BETA_CID_CELLO_SIG_re.CID[a]), BETA_CID_CELLO_SIG_re.ccle_name[a]) for a in range(BETA_CID_CELLO_SIG_re.shape[0])]
BETA_CID_CELLO_SIG_re['tuple_re'] = BETA_CID_CELLO_SIG_tup_re

BETA_CID_CELLO_SIG_re_re = BETA_CID_CELLO_SIG_re[BETA_CID_CELLO_SIG_re['ccle_name'].isin(avail_cell_list)]

BETA_CID_CELLO_SIG_re_re = BETA_CID_CELLO_SIG_re_re.reset_index(drop=True)

LINCS_exp_list = []

for IND in range(BETA_CID_CELLO_SIG_re_re.shape[0]) :
	if IND%100 == 0 :
		print(str(IND)+'/'+str(BETA_CID_CELLO_SIG_re_re.shape[0]) )
		datetime.now()
	Column = BETA_CID_CELLO_SIG_re_re['sig_id'][IND]
	L_vector = BETA_BIND_ORD[Column].values.tolist()
	LINCS_exp_list.append(L_vector)


L_TENSOR = torch.Tensor(LINCS_exp_list)

SAVE_PATH = CELVAL_PATH

torch.save(L_TENSOR, SAVE_PATH+'AVAIL_LINCS_EXP_cell.pt')

BETA_CID_CELLO_SIG_re_re.to_csv(SAVE_PATH+'AVAIL_LINCS_EXP_cell.csv')















/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V4_MIS2_W20v1/VAL

'A549_LUNG' -> TB01 Batch 128 / work 12 / 
'DLD1_LARGE_INTESTINE' -> INT04 Batch 64 / work 6 / 
'VCAP_PROSTATE'-> AMD05 Batch 64 / work 4 / 
'ES2_OVARY' -> -> INT05 Batch 64 / work 6 / 



'K562_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE'
'A375_SKIN'
'MDAMB231_BREAST'
'HCT116_LARGE_INTESTINE'
'PC3_PROSTATE'
'SKMEL5_SKIN'
'SKMEL28_SKIN'
'SW620_LARGE_INTESTINE'
'RKO_LARGE_INTESTINE'
'LOVO_LARGE_INTESTINE'
'U251MG_CENTRAL_NERVOUS_SYSTEM'
'OVCAR8_OVARY'
'MCF7_BREAST'
'HT29_LARGE_INTESTINE'
'MDAMB468_BREAST'
'BT474_BREAST'
'A673_BONE'
'T47D_BREAST'
'ZR751_BREAST'
'MELHO_SKIN'
'HS578T_BREAST'









############## cell line 하나만 보기 

cell_path = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V4_MIS2_W20v1/VAL/'

import glob

CELL_NAME = 'VCAP_PROSTATE'
CELL_pred_file = pd.read_csv(cell_path+'PRED_{}.FINAL_ing.csv'.format(CELL_NAME)) # ing 파일이라서 의미가 음슴 


CELL_pred_file['pred_mean'] = np.mean(CELL_pred_file[['CV0', 'CV1','CV2','CV3','CV4']], 1)
CELL_pred_file['pred_std'] = np.std(CELL_pred_file[['CV0', 'CV1','CV2','CV3','CV4']], 1)

np.max(CELL_pred_file.pred_std)
np.min(CELL_pred_file.pred_std)

근데 지금 그래서 pred 한게 
pred_row_cid = len(set(CELL_pred_file.ROW_CID))






pred_df = pd.DataFrame(columns = ['CV0', 'CV1','CV2','CV3','CV4','ROW_CID','COL_CID','CCLE','Y'])
for fifi in CVCL_files :
	print(fifi.split('/')[8].split('.')[0])
	tmp_df = pd.read_csv(fifi,  low_memory=False, index_col = False)
	tmp_df2 = tmp_df[['ROW_CID','COL_CID','CELLO', 'PRED_RES', 'Y']]
	pred_df = pd.concat([pred_df, tmp_df2])


merged_CVCL = pred_df.drop_duplicates()

# merged_CVCL.sort_values('PRED_RES')
# merged_CVCL.to_csv(cell_path+'/PRED_RESULT.csv', index=False)
# merged_CVCL = pd.read_csv(cell_path+'/PRED_RESULT.csv')

# 그놈의 색깔 
PRJ_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/{}_{}_{}/'.format(PRJ_NAME, MISS_NAME, W_NAME)
all_CellBase_DF = pd.read_csv(cell_path+'/AVAIL_CELL_DF.csv')
all_CellBase_DF['tissue'] = [ '_'.join(a.split('_')[1:]) if type(a) == str else 'NA' for a in list(all_CellBase_DF['DrugCombCCLE'])]

all_CellBase_DF.at[(all_CellBase_DF[all_CellBase_DF.DrugCombCello=='CVCL_0395']).index.item(), 'tissue'] = 'PROSTATE'
all_CellBase_DF.at[(all_CellBase_DF[all_CellBase_DF.DrugCombCello=='CVCL_A442']).index.item(), 'tissue'] = 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE'
all_CellBase_DF.at[(all_CellBase_DF[all_CellBase_DF.DrugCombCello=='CVCL_0219']).index.item(), 'tissue'] = 'LARGE_INTESTINE'

tissue_set = ['CENTRAL_NERVOUS_SYSTEM', 'LUNG',  'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE','BREAST','LARGE_INTESTINE', 'BONE',  'SKIN', 'PROSTATE',  'OVARY' ] # list(set(test_cell_df['tissue']))
color_set = ["#FFA420","#826C34","#D36E70","#705335","#57A639","#434B4D","#C35831","#B32821","#FAD201","#20603D","#828282","#1E1E1E"]
color_dict = {a : color_set[tissue_set.index(a)] for a in tissue_set}

merged_CVCL_RE = pd.merge(merged_CVCL, all_CellBase_DF[['DC_cellname','DrugCombCello','tissue']], left_on = 'CELLO', right_on='DrugCombCello', how = 'left')


# BOX plot -> 너무 퍼져서 violin plot 이 의미가 없었음 

cell_list = list(set(merged_CVCL_RE.DrugCombCello))

fig, ax = plt.subplots(figsize=(30, 15))
x_pos = [a+1 for a in range(25)]
data_list = []
color_list = []
cell_renames = []
for ind in range(25) : 
	cell = cell_list[ind]
	tmp_per = merged_CVCL_RE[merged_CVCL_RE.DrugCombCello==cell]
	data_list.append(np.array(tmp_per['PRED_RES']))
	color = color_dict[list(set(tmp_per['tissue']))[0]]
	cell_rename = list(set(tmp_per['DC_cellname']))[0]
	cell_renames.append(cell_rename)
	color_list.append(color)

box = ax.boxplot(data_list, patch_artist=True ) # x_pos,
 
for patch, color in zip(box['boxes'], color_list):
	patch.set_facecolor(color)

plt.xticks(x_pos, cell_renames, rotation=90, fontsize=18)

ax.set_xlabel('cell names')
ax.set_ylabel('pred_synergy')
ax.set_title(WORK_NAME)
plt.tight_layout()

plotname = 'FINAL_cell_box'
fig.savefig('{}/{}.png'.format(cell_path, plotname), bbox_inches = 'tight')
plt.close()



############## cell line wise violin plot 

cell_path = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V4_MIS2_W20v1/VAL'

import glob

CVCL_files_raw = glob.glob(cell_path+'/PRED_CVCL*')
CVCL_files = [a for a in CVCL_files_raw if '_ing' not in a]
CVCL_files_0553 = [a for a in CVCL_files if '0553' in a]

pred_df = pd.DataFrame(columns = ['ROW_CID','COL_CID','CELLO', 'PRED_RES', 'Y'])
for fifi in CVCL_files :
	print(fifi.split('/')[8].split('.')[0])
	tmp_df = pd.read_csv(fifi,  low_memory=False, index_col = False)
	tmp_df2 = tmp_df[['ROW_CID','COL_CID','CELLO', 'PRED_RES', 'Y']]
	pred_df = pd.concat([pred_df, tmp_df2])


merged_CVCL = pred_df.drop_duplicates()

# merged_CVCL.sort_values('PRED_RES')
# merged_CVCL.to_csv(cell_path+'/PRED_RESULT.csv', index=False)
# merged_CVCL = pd.read_csv(cell_path+'/PRED_RESULT.csv')

# 그놈의 색깔 
PRJ_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/{}_{}_{}/'.format(PRJ_NAME, MISS_NAME, W_NAME)
all_CellBase_DF = pd.read_csv(cell_path+'/AVAIL_CELL_DF.csv')
all_CellBase_DF['tissue'] = [ '_'.join(a.split('_')[1:]) if type(a) == str else 'NA' for a in list(all_CellBase_DF['DrugCombCCLE'])]

all_CellBase_DF.at[(all_CellBase_DF[all_CellBase_DF.DrugCombCello=='CVCL_0395']).index.item(), 'tissue'] = 'PROSTATE'
all_CellBase_DF.at[(all_CellBase_DF[all_CellBase_DF.DrugCombCello=='CVCL_A442']).index.item(), 'tissue'] = 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE'
all_CellBase_DF.at[(all_CellBase_DF[all_CellBase_DF.DrugCombCello=='CVCL_0219']).index.item(), 'tissue'] = 'LARGE_INTESTINE'

tissue_set = ['CENTRAL_NERVOUS_SYSTEM', 'LUNG',  'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE','BREAST','LARGE_INTESTINE', 'BONE',  'SKIN', 'PROSTATE',  'OVARY' ] # list(set(test_cell_df['tissue']))
color_set = ["#FFA420","#826C34","#D36E70","#705335","#57A639","#434B4D","#C35831","#B32821","#FAD201","#20603D","#828282","#1E1E1E"]
color_dict = {a : color_set[tissue_set.index(a)] for a in tissue_set}

merged_CVCL_RE = pd.merge(merged_CVCL, all_CellBase_DF[['DC_cellname','DrugCombCello','tissue']], left_on = 'CELLO', right_on='DrugCombCello', how = 'left')


# BOX plot -> 너무 퍼져서 violin plot 이 의미가 없었음 

cell_list = list(set(merged_CVCL_RE.DrugCombCello))

fig, ax = plt.subplots(figsize=(30, 15))
x_pos = [a+1 for a in range(25)]
data_list = []
color_list = []
cell_renames = []
for ind in range(25) : 
	cell = cell_list[ind]
	tmp_per = merged_CVCL_RE[merged_CVCL_RE.DrugCombCello==cell]
	data_list.append(np.array(tmp_per['PRED_RES']))
	color = color_dict[list(set(tmp_per['tissue']))[0]]
	cell_rename = list(set(tmp_per['DC_cellname']))[0]
	cell_renames.append(cell_rename)
	color_list.append(color)

box = ax.boxplot(data_list, patch_artist=True ) # x_pos,
 
for patch, color in zip(box['boxes'], color_list):
	patch.set_facecolor(color)

plt.xticks(x_pos, cell_renames, rotation=90, fontsize=18)

ax.set_xlabel('cell names')
ax.set_ylabel('pred_synergy')
ax.set_title(WORK_NAME)
plt.tight_layout()

plotname = 'FINAL_cell_box'
fig.savefig('{}/{}.png'.format(cell_path, plotname), bbox_inches = 'tight')
plt.close()

