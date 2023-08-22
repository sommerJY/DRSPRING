

# 203 : 5CV 돌리기 




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



NETWORK_PATH = '/home01/k040a01/01.Data/HumanNet/'
LINCS_PATH = '/home01/k040a01/01.Data/LINCS/'
DC_PATH = '/home01/k040a01/01.Data/DrugComb/'

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




SAVE_PATH = '/home01/k040a01/02.M3V6/M3V6_349_DATA/'
# SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V6_349_FULL/'

file_name = 'M3V6_349_MISS2_FULL' # 0608

A_B_C_S_SET_ADD = pd.read_csv(SAVE_PATH+'{}.A_B_C_S_SET_ADD.csv'.format(file_name), low_memory=False)
MY_chem_A_feat = torch.load(SAVE_PATH+'{}.MY_chem_A_feat.pt'.format(file_name))
MY_chem_B_feat = torch.load(SAVE_PATH+'{}.MY_chem_B_feat.pt'.format(file_name))
MY_chem_A_adj = torch.load(SAVE_PATH+'{}.MY_chem_A_adj.pt'.format(file_name))
MY_chem_B_adj = torch.load(SAVE_PATH+'{}.MY_chem_B_adj.pt'.format(file_name))
MY_g_EXP_A = torch.load(SAVE_PATH+'{}.MY_g_EXP_A.pt'.format(file_name))
MY_g_EXP_B = torch.load(SAVE_PATH+'{}.MY_g_EXP_B.pt'.format(file_name))
MY_Target_1_A = torch.load(SAVE_PATH+'{}.MY_Target_1_A.pt'.format(file_name))
MY_Target_1_B = torch.load(SAVE_PATH+'{}.MY_Target_1_B.pt'.format(file_name))
MY_CellBase = torch.load(SAVE_PATH+'{}.MY_CellBase.pt'.format(file_name))
MY_syn = torch.load(SAVE_PATH+'{}.MY_syn.pt'.format(file_name))




A_B_C_S_SET_ADD2 = copy.deepcopy(A_B_C_S_SET_ADD)

cid_a = list(A_B_C_S_SET_ADD2['CID_A'])
cid_b = list(A_B_C_S_SET_ADD2['CID_B'])
sm_a = list(A_B_C_S_SET_ADD2['ROW_CAN_SMILES'])
sm_b = list(A_B_C_S_SET_ADD2['COL_CAN_SMILES'])
ccle = list(A_B_C_S_SET_ADD2['CELL'])

A_B_C_S_SET_ADD2['CID_CID'] = [str(int(cid_a[i])) + '___' + str(int(cid_b[i])) if cid_a[i] < cid_b[i] else str(int(cid_b[i])) + '___' + str(int(cid_a[i])) for i in range(A_B_C_S_SET_ADD2.shape[0])]
A_B_C_S_SET_ADD2['SM_C_CHECK'] = [sm_a[i] + '___' + sm_b[i]+ '___' + ccle[i] if sm_a[i] < sm_b[i] else sm_b[i] + '___' + sm_a[i]+ '___' + ccle[i] for i in range(A_B_C_S_SET_ADD2.shape[0])]

A_B_C_S_SET_ADD2['ori_index'] = list(A_B_C_S_SET_ADD2.index)




MISS_filter = ['AOBO','AXBX','AXBO','AOBX'] # 

A_B_C_S_SET = A_B_C_S_SET_ADD2[A_B_C_S_SET_ADD2.Basal_Exp == 'O']

A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.SYN_OX == 'O']

A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.T1OX == 'O'] ####################### new targets 

A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.type.isin(MISS_filter)]




# basal node exclude -> CCLE match 만 사용
CCLE_PATH = '/home01/k040a01/01.Data/CCLE/'
# CCLE_PATH = '/st06/jiyeonH/13.DD_SESS/CCLE.22Q1/'
ccle_exp = pd.read_csv(CCLE_PATH+'CCLE_expression.csv', low_memory=False)
ccle_info= pd.read_csv(CCLE_PATH+'sample_info.csv', low_memory=False)

ccle_cell_info = ccle_info[['DepMap_ID','CCLE_Name']]
ccle_cell_info.columns = ['DepMap_ID','DrugCombCCLE']

ccle_cell_info_filt = ccle_cell_info[ccle_cell_info.DepMap_ID.isin(ccle_exp['Unnamed: 0'])]
ccle_names = [a for a in ccle_cell_info_filt.DrugCombCCLE if type(a) == str]


A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.CELL.isin(ccle_names)]




data_ind = list(A_B_C_S_SET.index)

MY_chem_A_feat_RE = MY_chem_A_feat[data_ind]
MY_chem_B_feat_RE = MY_chem_B_feat[data_ind]
MY_chem_A_adj_RE = MY_chem_A_adj[data_ind]
MY_chem_B_adj_RE = MY_chem_B_adj[data_ind]
MY_g_EXP_A_RE = MY_g_EXP_A[data_ind]
MY_g_EXP_B_RE = MY_g_EXP_B[data_ind]
MY_Target_A = copy.deepcopy(MY_Target_1_A)[data_ind] ############## NEW TARGET !!!!!! #####
MY_Target_B = copy.deepcopy(MY_Target_1_B)[data_ind] ############## NEW TARGET !!!!!! #####

MY_CellBase_RE = MY_CellBase[data_ind]
MY_syn_RE = MY_syn[data_ind]


A_B_C_S_SET = A_B_C_S_SET.reset_index(drop = True)



# cell line vector 

DC_CELL_DF2 = pd.read_csv(DC_PATH+'DC_CELL_INFO.csv', sep = '\t')
DC_CELL_DF2 = pd.concat([
	DC_CELL_DF2, 
	pd.DataFrame({'cell_line_id' : [1],'DC_cellname' : ['786O'],'DrugCombCello' : ['CVCL_1051'],'DrugCombCCLE':['786O_KIDNEY']})])

DC_CELL_info_filt = DC_CELL_DF2[DC_CELL_DF2.DrugCombCCLE.isin(A_B_C_S_SET.CELL)] # 38

DC_CELL_info_filt = DC_CELL_info_filt.drop(['Unnamed: 0'], axis = 1)
DC_CELL_info_filt.columns = ['cell_line_id', 'DC_cellname', 'DrugCombCello', 'CELL']
DC_CELL_info_filt = DC_CELL_info_filt[['CELL','DC_cellname']]

A_B_C_S_SET_COH = pd.merge(A_B_C_S_SET, DC_CELL_info_filt, on = 'CELL', how = 'left'  )



# sample number filter 

# 빈도 확인 

C_names = list(set(A_B_C_S_SET_COH.DC_cellname))
C_names.sort()

C_freq = [list(A_B_C_S_SET_COH.DC_cellname).count(a) for a in C_names]
C_cclename = [list(A_B_C_S_SET_COH[A_B_C_S_SET_COH.DC_cellname==a]['CELL'])[0] for a in C_names]

C_df = pd.DataFrame({'cell' : C_names, 'freq' : C_freq, 'ccle' : C_cclename})
C_df = C_df.sort_values('freq')

CELL_CUT = 200 ####### 이것도 그렇게 되면 바꿔야하지 않을까 ##################################################################

C_freq_filter = C_df[C_df.freq > CELL_CUT ] 


A_B_C_S_SET_COH = A_B_C_S_SET_COH[A_B_C_S_SET_COH.DC_cellname.isin(C_freq_filter.cell)]

DC_CELL_info_filt_re = DC_CELL_info_filt[DC_CELL_info_filt.DC_cellname.isin(C_freq_filter.cell)]
DC_CELL_info_filt_re['cell_onehot'] = [a for a in range(len(set(DC_CELL_info_filt_re.CELL)))]

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

print(A_B_C_S_SET_COH2.shape)


print('CIDs', flush = True)
tmp = list(set(A_B_C_S_SET_COH2.CID_CID))
tmp2 = sum([a.split('___') for a in tmp],[])
print(len(set(tmp2)) , flush = True)

print('CID_CID', flush = True)
print(len(set(A_B_C_S_SET_COH2.CID_CID)), flush = True)


print('CID_CID_CCLE', flush = True)
print(len(set(A_B_C_S_SET_COH2.cid_cid_cell)), flush = True)

print('DrugCombCCLE', flush = True)
print(len(set(A_B_C_S_SET_COH2.CELL)), flush = True)


###########################################################################################
###########################################################################################
###########################################################################################

# 일단 생 5CV


print("LEARNING")

A_B_C_S_SET_SM = copy.deepcopy(A_B_C_S_SET_COH2) # 

# get unique values, remove duplicates, but keep original counts
data_no_dup, counts = np.unique(list(A_B_C_S_SET_SM['SM_C_CHECK']), return_counts=True)
data_no_dup_cells = [setset.split('___')[2] for setset in data_no_dup]
data_no_dup_sm_sm = [setset.split('___')[0]+'___'+setset.split('___')[1] for setset in data_no_dup]
data_nodup_df = pd.DataFrame({
	'setset' : data_no_dup.tolist(), 
	'cell' : data_no_dup_cells,
	'SM_SM' : data_no_dup_sm_sm
	 })




SM_SM_list = list(set(data_nodup_df.SM_SM))
SM_SM_list.sort()
sm_sm_list_1 = sklearn.utils.shuffle(SM_SM_list, random_state=42)

bins = [a for a in range(0, len(sm_sm_list_1), round(len(sm_sm_list_1)*0.2) )]
bins = bins[1:]
res = np.split(sm_sm_list_1, bins)

CV_1_smsm = list(res[0])
CV_2_smsm = list(res[1])
CV_3_smsm = list(res[2])
CV_4_smsm = list(res[3])
CV_5_smsm = list(res[4])
if len(res) > 5 :
	CV_5_smsm = list(res[4]) + list(res[5])

len(sm_sm_list_1)
len(CV_1_smsm) + len(CV_2_smsm) + len(CV_3_smsm) + len(CV_4_smsm) + len(CV_5_smsm)

CV_1_setset = list(data_nodup_df[data_nodup_df.SM_SM.isin(CV_1_smsm)]['setset'])
CV_2_setset = list(data_nodup_df[data_nodup_df.SM_SM.isin(CV_2_smsm)]['setset'])
CV_3_setset = list(data_nodup_df[data_nodup_df.SM_SM.isin(CV_3_smsm)]['setset'])
CV_4_setset = list(data_nodup_df[data_nodup_df.SM_SM.isin(CV_4_smsm)]['setset'])
CV_5_setset = list(data_nodup_df[data_nodup_df.SM_SM.isin(CV_5_smsm)]['setset'])




CV_ND_INDS = {
	'CV0_train' : CV_1_setset + CV_2_setset + CV_3_setset + CV_4_setset, 
	'CV0_test' : CV_5_setset,
	'CV1_train' : CV_2_setset + CV_3_setset + CV_4_setset + CV_5_setset, 
	'CV1_test' : CV_1_setset,
	'CV2_train' : CV_3_setset + CV_4_setset + CV_5_setset + CV_1_setset,
	'CV2_test' : CV_2_setset,
	'CV3_train' : CV_4_setset + CV_5_setset + CV_1_setset + CV_2_setset,
	'CV3_test' : CV_3_setset,
	'CV4_train' : CV_5_setset + CV_1_setset + CV_2_setset + CV_3_setset,
	'CV4_test' : CV_4_setset 
}

print(data_nodup_df.shape)
len( CV_1_setset + CV_2_setset + CV_3_setset + CV_4_setset + CV_5_setset)
len(set( CV_1_setset + CV_2_setset + CV_3_setset + CV_4_setset + CV_5_setset ))





# 주어진 환경 맞춰서 5CV만 돌릴거라서 수정버전 
def prepare_data_GCN(CV_num, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, 
MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, 
cell_one_hot, MY_syn_RE2, norm ) : 
	# 
	# CV_num = 0
	train_key = 'CV{}_train'.format(CV_num)
	test_key = 'CV{}_test'.format(CV_num)
	# 
	#
	ABCS_tv = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS[train_key])]
	ABCS_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS[test_key])]
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
	#
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




seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


# gcn_drug1_F, gcn_drug2_F, gcn_drug1_ADJ, gcn_drug2_ADJ, gcn_exp_A, gcn_exp_B, gcn_tgt_A, gcn_tgt_B, gcn_adj, gcn_adj_weight, syn_ans, cell_info
norm = 'tanh_norm'

# CV_0
train_data_0, test_data_0 = prepare_data_GCN(0, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, 
MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, 
cell_one_hot, MY_syn_RE2, norm)

# CV_1
train_data_1, test_data_1 = prepare_data_GCN(1, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, 
MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, 
cell_one_hot, MY_syn_RE2, norm)

# CV_2
train_data_2, test_data_2 = prepare_data_GCN(2, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, 
MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, 
cell_one_hot, MY_syn_RE2, norm)

# CV_3
train_data_3, test_data_3 = prepare_data_GCN(3, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, 
MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, 
cell_one_hot, MY_syn_RE2, norm)

# CV_4
train_data_4, test_data_4 = prepare_data_GCN(4, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, 
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
T_train_2, T_test_2 = make_merged_data(2)
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
		for batch_idx_v, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y) in enumerate(LOADER_DICT['test']) :
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












def RAY_MY_train(config, checkpoint_dir=None):
	n_epochs = 1000
	criterion = weighted_mse_loss
	use_cuda = True  #  #  #  #  #  #  # True
	#
	dsn_layers = [int(a) for a in config["dsn_layer"].split('-') ]
	snp_layers = [int(a) for a in config["snp_layer"].split('-') ]
	inDrop = config["dropout_1"]
	Drop = config["dropout_2"]
	#
	CV_NUM = config["CV"] 
	RAY_train_list = [RAY_train_0 ,RAY_train_1 ,RAY_train_2 ,RAY_train_3 ,RAY_train_4 ]
	RAY_test_list = [RAY_test_0 ,RAY_test_1 ,RAY_test_2 ,RAY_test_3 ,RAY_test_4 ]
	RAY_loss_weight_list = [RAY_loss_weight_0 ,RAY_loss_weight_1 ,RAY_loss_weight_2 ,RAY_loss_weight_3 ,RAY_loss_weight_4]
	#
	# 
	CV_0_train = ray.get(RAY_train_list[CV_NUM])
	CV_0_test = ray.get(RAY_test_list[CV_NUM])
	CV_0_loss_weight = ray.get(RAY_loss_weight_list[CV_NUM])
	CV_0_batch_cut_weight = [CV_0_loss_weight[i:i+config["batch_size"]] for i in range(0,len(CV_0_loss_weight), config["batch_size"])]
	#
	CV_0_loaders = {
			'train' : torch.utils.data.DataLoader(CV_0_train, batch_size = config["batch_size"], collate_fn = graph_collate_fn, shuffle =False, num_workers=config['n_workers']),
			'test' : torch.utils.data.DataLoader(CV_0_test, batch_size = config["batch_size"], collate_fn = graph_collate_fn, shuffle =False, num_workers=config['n_workers']),
			'loss_weight' : CV_0_batch_cut_weight
	}
	#
	#  
	CV_0_MODEL = MY_expGCN_parallel_model(
			config["G_chem_layer"], CV_0_train.gcn_drug1_F.shape[-1] , config["G_chem_hdim"],      # G_layer_chem, G_indim_chem, G_hiddim_chem, 
			config["G_exp_layer"], 3 , config["G_exp_hdim"],      # G_layer_exp, G_indim_exp, G_hiddim_exp, 
			dsn_layers, dsn_layers, snp_layers,      # drug 1 layers, drug 2 layers, merged layer, 
			 1,      # cell_dim ,out_dim,
			inDrop, Drop      # inDrop, drop
			)
	# 
	#
	if torch.cuda.is_available():
		CV_0_MODEL = CV_0_MODEL.cuda()
		if torch.cuda.device_count() > 1 :
			CV_0_MODEL = torch.nn.DataParallel(CV_0_MODEL)
	#       
	CV_0_optimizer = torch.optim.Adam(CV_0_MODEL.parameters(), lr = config["lr"] )
	#
	#
	train_loss_all = []
	valid_loss_all = []
	train_pearson_corr_all=[]
	train_spearman_corr_all=[]
	val_pearson_corr_all = []
	val_spearman_corr_all = []
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
		train_loss_all.append(cv_0_t_loss)
		train_pearson_corr_all.append(cv_0_t_pc)
		train_spearman_corr_all.append(cv_0_t_sc)	
		#
		#
		######################    
		# validate the model #
		######################
		cv_0_v_loss, cv_0_v_pc, cv_0_v_sc, CV_0_MODEL  = inner_val(CV_0_loaders, CV_0_MODEL, True)
		valid_loss_all.append(cv_0_v_loss)
		val_pearson_corr_all.append(cv_0_v_pc)
		val_spearman_corr_all.append(cv_0_v_sc) 
		#
		#
		done = datetime.now()
		time_spent = done-now
		#
		with tune.checkpoint_dir(step = epoch) as checkpoint_dir:
			trial_name = '_'.join(checkpoint_dir.split('/')[-2].split('_')[0:4])
			cv_0_path = os.path.join(checkpoint_dir, "checkpoint")
			torch.save((CV_0_MODEL.state_dict(), CV_0_optimizer.state_dict()), cv_0_path)
		#
		tune.report(T_LS= cv_0_t_loss,  T_PC = cv_0_t_pc, T_SC = cv_0_t_sc, 
		V_LS=cv_0_v_loss, V_PC = cv_0_v_pc, V_SC = cv_0_v_sc )
	#
	result_dict = {
		'train_loss_all' : train_loss_all, 'valid_loss_all' : valid_loss_all, 
		'train_pearson_corr_all' : train_pearson_corr_all, 'train_spearman_corr_all' : train_spearman_corr_all, 
		'val_pearson_corr_all' : val_pearson_corr_all, 'val_spearman_corr_all' : val_spearman_corr_all, 
	}
	with open(file='RESULT_DICT.{}.pickle'.format(CV_NUM), mode='wb') as f:
		pickle.dump(result_dict, f)
	#
	print("Finished Training")







def MAIN(ANAL_name, my_config, num_samples= 10, max_num_epochs=1000, cpus_per_trial = 16, gpus_per_trial = 1):
	# OPTUNA CONFIGURE 
	CONFIG={
		'n_workers' : tune.grid_search([cpus_per_trial]),
		"epoch" : tune.grid_search([max_num_epochs]),
		"CV" : tune.grid_search([0,1,2,3,4]),
		"G_chem_layer" : tune.grid_search([my_config['config/G_chem_layer'].item()]), # 
		"G_exp_layer" : tune.grid_search([my_config['config/G_exp_layer'].item()]), # 
		"G_chem_hdim" : tune.grid_search([my_config['config/G_chem_hdim'].item()]), # 
		"G_exp_hdim" : tune.grid_search([my_config['config/G_exp_hdim'].item()]), # 
		"batch_size" : tune.grid_search([my_config['config/batch_size'].item() ]), # 
		"dsn_layer" : tune.grid_search([my_config['config/dsn_layer'].item() ]), # 
		"snp_layer" : tune.grid_search([my_config['config/snp_layer'].item() ]), # 
		"dropout_1" : tune.grid_search([ my_config['config/dropout_1'].item() ]),
		"dropout_2" : tune.grid_search([ my_config['config/dropout_2'].item() ]),
		"lr" : tune.grid_search([ my_config['config/lr'].item() ]), 
	}
	#
	#pickle.dumps(trainable)
	reporter = CLIReporter(
		metric_columns=["T_LS",'T_PC','T_SC',"V_LS",'V_PC','V_SC', "training_iteration"])#
	#
	ANALYSIS = tune.run( # 끝내지 않음 
		tune.with_parameters(RAY_MY_train),
		name = ANAL_name,
		num_samples=num_samples,
		config=CONFIG,
		resources_per_trial={'cpu': cpus_per_trial,'gpu' : gpus_per_trial }, # , 
		progress_reporter = reporter
	)
	#
	return ANALYSIS




W_NAME = 'W203' # 고른 내용으로 5CV 다시 
MJ_NAME = 'M3V6'
WORK_DATE = '23.06.22' # 349
MISS_NAME = 'MIS2'
PPI_NAME = '349'
WORK_NAME = 'WORK_203' # 349###################################################################################################


WORK_PATH = '/home01/k040a01/02.{}/{}_{}_{}_{}/'.format(MJ_NAME,MJ_NAME,W_NAME,PPI_NAME,MISS_NAME)


OLD_PATH = '/home01/k040a01/02.M3V6/M3V6_W202_349_MIS2'
ANA_DF_CSV = pd.read_csv(os.path.join(OLD_PATH,'RAY_ANA_DF.{}.csv'.format('M3V6_W202_349_MIS2')))

my_config = ANA_DF_CSV[ANA_DF_CSV.trial_id=='1cf5052a'] # 349 


#4gpu for 5cv 
MAIN('PRJ02.{}.{}.{}.{}.{}'.format(WORK_DATE, MJ_NAME, WORK_NAME, PPI_NAME, MISS_NAME), my_config, 1, 1000, 24, 0.5)


with open(file='{}/CV_SM_list.pickle'.format(WORK_PATH), mode='wb') as f:
	pickle.dump(CV_ND_INDS, f)






sbatch gpu4.W203.CV5.any M3V6_WORK203.349.py
tail -n 100 /home01/k040a01/02.M3V6/M3V6_W203_349_MIS2/RESULT.G4.CV5.txt
tail ~/logs/M3V6W203_GPU4_13062.log







gene exp only 

206_5 : 
sbatch gpu4.W206.CV5.any M3V6_WORK206_5.349.py


207_5 : 
sbatch gpu4.W207.CV5.any M3V6_WORK207_5.349.py


208_5 : 
sbatch gpu4.W208.CV5.any M3V6_WORK208_5.349.py


209_5 :
sbatch gpu4.W209.CV5.any M3V6_WORK209_5.349.py


210_5:
sbatch gpu4.W210.CV5.any M3V6_WORK210_5.349.py



















import pandas as pd 
import matplotlib.pyplot as plt
from ray.tune import ExperimentAnalysis
from ray.tune import Analysis
import pickle
import math
import torch
import os 
import copy
import numpy as np

MJ_NAME = 'M3V6'
PPI_NAME = '349'
MISS_NAME = 'MIS2'

W_NAME = 'W203'
WORK_NAME = 'WORK_203_3' # 349
WORK_DATE = '23.06.23' # 349



anal_dir = "/home01/k040a01/ray_results/PRJ02.{}.{}.{}.{}.{}".format(WORK_DATE, MJ_NAME,  WORK_NAME, PPI_NAME, MISS_NAME)
#     anal_dir = '/home01/k040a01/ray_results/PRJ02.23.06.13.M3V5.WORK_37.349.MIS22/'
list_dir = os.listdir(anal_dir)
exp_json = [a for a in list_dir if 'experiment_state' in a]
exp_json
anal_df = ExperimentAnalysis(os.path.join(anal_dir, exp_json[0]))

ANA_DF_1 = anal_df.dataframe()
ANA_ALL_DF_1 = anal_df.trial_dataframes

ANA_DF = ANA_DF_1

ANA_DF = ANA_DF.sort_values('config/CV')
ANA_DF.index = [0,1,2,3,4]
ANA_ALL_DF = ANA_ALL_DF_1


ANA_DF.to_csv('/home01/k040a01/02.M3V6/{}_{}_{}_{}/RAY_ANA_DF.{}_{}_{}_{}.csv'.format(MJ_NAME, W_NAME, PPI_NAME, MISS_NAME,  MJ_NAME, W_NAME, PPI_NAME, MISS_NAME))
import pickle
with open("/home01/k040a01/02.M3V6/{}_{}_{}_{}/RAY_ANA_DF.{}_{}_{}_{}.pickle".format(MJ_NAME, W_NAME, PPI_NAME, MISS_NAME, MJ_NAME, W_NAME, PPI_NAME, MISS_NAME), "wb") as fp:
	pickle.dump(ANA_ALL_DF,fp) 

'/home01/k040a01/02.M3V6/{}_{}_{}_{}/RAY_ANA_DF.{}_{}_{}_{}.csv'.format(MJ_NAME, W_NAME, PPI_NAME, MISS_NAME,  MJ_NAME, W_NAME, PPI_NAME, MISS_NAME)
"/home01/k040a01/02.M3V6/{}_{}_{}_{}/RAY_ANA_DF.{}_{}_{}_{}.pickle".format(MJ_NAME, W_NAME, PPI_NAME, MISS_NAME, MJ_NAME, W_NAME, PPI_NAME, MISS_NAME)


limit = 1000

cv0_key = ANA_DF['logdir'][0] ;	cv1_key = ANA_DF['logdir'][1]; 	cv2_key = ANA_DF['logdir'][2] ;	cv3_key = ANA_DF['logdir'][3];	cv4_key = ANA_DF['logdir'][4]

epc_T_LS_mean = np.mean([ANA_ALL_DF[cv0_key]['T_LS'][0:limit], ANA_ALL_DF[cv1_key]['T_LS'][0:limit],ANA_ALL_DF[cv2_key]['T_LS'][0:limit], ANA_ALL_DF[cv3_key]['T_LS'][0:limit], ANA_ALL_DF[cv4_key]['T_LS'][0:limit]], axis = 0)
epc_T_LS_std = np.std([ANA_ALL_DF[cv0_key]['T_LS'][0:limit], ANA_ALL_DF[cv1_key]['T_LS'][0:limit],ANA_ALL_DF[cv2_key]['T_LS'][0:limit], ANA_ALL_DF[cv3_key]['T_LS'][0:limit], ANA_ALL_DF[cv4_key]['T_LS'][0:limit]], axis = 0)

epc_T_PC_mean = np.mean([ANA_ALL_DF[cv0_key]['T_PC'][0:limit], ANA_ALL_DF[cv1_key]['T_PC'][0:limit],ANA_ALL_DF[cv2_key]['T_PC'][0:limit], ANA_ALL_DF[cv3_key]['T_PC'][0:limit], ANA_ALL_DF[cv4_key]['T_PC'][0:limit]], axis = 0)
epc_T_PC_std = np.std([ANA_ALL_DF[cv0_key]['T_PC'][0:limit], ANA_ALL_DF[cv1_key]['T_PC'][0:limit],ANA_ALL_DF[cv2_key]['T_PC'][0:limit], ANA_ALL_DF[cv3_key]['T_PC'][0:limit], ANA_ALL_DF[cv4_key]['T_PC'][0:limit]], axis = 0)

epc_T_SC_mean = np.mean([ANA_ALL_DF[cv0_key]['T_SC'][0:limit], ANA_ALL_DF[cv1_key]['T_SC'][0:limit],ANA_ALL_DF[cv2_key]['T_SC'][0:limit], ANA_ALL_DF[cv3_key]['T_SC'][0:limit], ANA_ALL_DF[cv4_key]['T_SC'][0:limit]], axis = 0)
epc_T_SC_std = np.std([ANA_ALL_DF[cv0_key]['T_SC'][0:limit], ANA_ALL_DF[cv1_key]['T_SC'][0:limit],ANA_ALL_DF[cv2_key]['T_SC'][0:limit], ANA_ALL_DF[cv3_key]['T_SC'][0:limit], ANA_ALL_DF[cv4_key]['T_SC'][0:limit]], axis = 0)

epc_V_LS_mean = np.mean([ANA_ALL_DF[cv0_key]['V_LS'][0:limit], ANA_ALL_DF[cv1_key]['V_LS'][0:limit],ANA_ALL_DF[cv2_key]['V_LS'][0:limit], ANA_ALL_DF[cv3_key]['V_LS'][0:limit], ANA_ALL_DF[cv4_key]['V_LS'][0:limit]], axis = 0)
epc_V_LS_std = np.std([ANA_ALL_DF[cv0_key]['V_LS'][0:limit], ANA_ALL_DF[cv1_key]['V_LS'][0:limit],ANA_ALL_DF[cv2_key]['V_LS'][0:limit], ANA_ALL_DF[cv3_key]['V_LS'][0:limit], ANA_ALL_DF[cv4_key]['V_LS'][0:limit]], axis = 0)

epc_V_PC_mean = np.mean([ANA_ALL_DF[cv0_key]['V_PC'][0:limit], ANA_ALL_DF[cv1_key]['V_PC'][0:limit],ANA_ALL_DF[cv2_key]['V_PC'][0:limit], ANA_ALL_DF[cv3_key]['V_PC'][0:limit], ANA_ALL_DF[cv4_key]['V_PC'][0:limit]], axis = 0)
epc_V_PC_std = np.std([ANA_ALL_DF[cv0_key]['V_PC'][0:limit], ANA_ALL_DF[cv1_key]['V_PC'][0:limit],ANA_ALL_DF[cv2_key]['V_PC'][0:limit], ANA_ALL_DF[cv3_key]['V_PC'][0:limit], ANA_ALL_DF[cv4_key]['V_PC'][0:limit]], axis = 0)

epc_V_SC_mean = np.mean([ANA_ALL_DF[cv0_key]['V_SC'][0:limit], ANA_ALL_DF[cv1_key]['V_SC'][0:limit],ANA_ALL_DF[cv2_key]['V_SC'][0:limit], ANA_ALL_DF[cv3_key]['V_SC'][0:limit], ANA_ALL_DF[cv4_key]['V_SC'][0:limit]], axis = 0)
epc_V_SC_std = np.std([ANA_ALL_DF[cv0_key]['V_SC'][0:limit], ANA_ALL_DF[cv1_key]['V_SC'][0:limit],ANA_ALL_DF[cv2_key]['V_SC'][0:limit], ANA_ALL_DF[cv3_key]['V_SC'][0:limit], ANA_ALL_DF[cv4_key]['V_SC'][0:limit]], axis = 0)


epc_result = pd.DataFrame({
	'T_LS_mean' : epc_T_LS_mean, 'T_PC_mean' : epc_T_PC_mean, 'T_SC_mean' : epc_T_SC_mean, 
	'T_LS_std' : epc_T_LS_std, 'T_PC_std' : epc_T_PC_std, 'T_SC_std' : epc_T_SC_std, 
	'V_LS_mean' : epc_V_LS_mean, 'V_PC_mean' : epc_V_PC_mean, 'V_SC_mean' : epc_V_SC_mean, 
	'V_LS_std' : epc_V_LS_std, 'V_PC_std' : epc_V_PC_std, 'V_SC_std' : epc_V_SC_std,
})

epc_result[[
    'T_LS_mean', 'T_LS_std', 'T_PC_mean', 'T_PC_std',
    'T_SC_mean','T_SC_std', 'V_LS_mean', 'V_LS_std', 
    'V_PC_mean', 'V_PC_std','V_SC_mean','V_SC_std']].to_csv("/home01/k040a01/02.M3V6/{}_{}_{}_{}/RAY_ANA_DF.{}_{}_{}_{}.resDF".format(MJ_NAME, W_NAME, PPI_NAME, MISS_NAME, MJ_NAME, W_NAME, PPI_NAME, MISS_NAME))

"/home01/k040a01/02.M3V6/{}_{}_{}_{}/RAY_ANA_DF.{}_{}_{}_{}.resDF".format(MJ_NAME, W_NAME, PPI_NAME, MISS_NAME, MJ_NAME, W_NAME, PPI_NAME, MISS_NAME)
        


1) min loss

min(epc_result.sort_values('V_LS_mean')['V_LS_mean']) ; min_VLS = min(epc_result.sort_values('V_LS_mean')['V_LS_mean'])
KEY_EPC = epc_result[epc_result.V_LS_mean == min_VLS].index.item()
checkpoint = "/checkpoint_"+str(KEY_EPC).zfill(6)
VLS_cv0_PATH = cv0_key + checkpoint
VLS_cv0_PATH
VLS_cv1_PATH = cv1_key + checkpoint
VLS_cv1_PATH
VLS_cv2_PATH = cv2_key + checkpoint
VLS_cv2_PATH
VLS_cv3_PATH = cv3_key + checkpoint
VLS_cv3_PATH
VLS_cv4_PATH = cv4_key + checkpoint
VLS_cv4_PATH


KEY_EPC
round(epc_result.loc[KEY_EPC].V_LS_mean,3)
round(epc_result.loc[KEY_EPC].V_LS_std,3)



2) PC best 

epc_result.sort_values('V_PC_mean', ascending = False) 
max(epc_result['V_PC_mean']); max_VPC = max(epc_result['V_PC_mean'])
KEY_EPC = epc_result[epc_result.V_PC_mean == max_VPC].index.item()
checkpoint = "/checkpoint_"+str(KEY_EPC).zfill(6)
VPC_cv0_PATH = cv0_key + checkpoint
VPC_cv0_PATH
VPC_cv1_PATH = cv1_key + checkpoint
VPC_cv1_PATH
VPC_cv2_PATH = cv2_key + checkpoint
VPC_cv2_PATH
VPC_cv3_PATH = cv3_key + checkpoint
VPC_cv3_PATH
VPC_cv4_PATH = cv4_key + checkpoint
VPC_cv4_PATH


KEY_EPC
round(epc_result.loc[KEY_EPC].V_PC_mean,3)
round(epc_result.loc[KEY_EPC].V_PC_std,3)


3) SC best 

epc_result.sort_values('V_SC_mean', ascending = False) 
max(epc_result['V_SC_mean']); max_VSC = max(epc_result['V_SC_mean'])
KEY_EPC = epc_result[epc_result.V_SC_mean == max_VSC].index.item()
checkpoint = "/checkpoint_"+str(KEY_EPC).zfill(6)
VSC_cv0_PATH = cv0_key + checkpoint
VSC_cv0_PATH
VSC_cv1_PATH = cv1_key + checkpoint
VSC_cv1_PATH
VSC_cv2_PATH = cv2_key + checkpoint
VSC_cv2_PATH
VSC_cv3_PATH = cv3_key + checkpoint
VSC_cv3_PATH
VSC_cv4_PATH = cv4_key + checkpoint
VSC_cv4_PATH

KEY_EPC
round(epc_result.loc[KEY_EPC].V_SC_mean,3)
round(epc_result.loc[KEY_EPC].V_SC_std,3)





epc = 450

ANA_ALL_DF[cv0_key]['V_PC'][epc], ANA_ALL_DF[cv1_key]['V_PC'][epc],ANA_ALL_DF[cv2_key]['V_PC'][epc], ANA_ALL_DF[cv3_key]['V_PC'][epc], ANA_ALL_DF[cv4_key]['V_PC'][epc]








########################################################################
########################################################################
########################################################################
							CPU 
########################################################################
########################################################################
########################################################################





NETWORK_PATH = '/st06/jiyeonH/13.DD_SESS/HumanNetV3/'
LINCS_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/' 
DATA_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V3_FULL/'
DC_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/'





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


SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V6_349_FULL/'

file_name = 'M3V6_349_MISS2_FULL' # 0608

A_B_C_S_SET_ADD = pd.read_csv(SAVE_PATH+'{}.A_B_C_S_SET_ADD.csv'.format(file_name), low_memory=False)
MY_chem_A_feat = torch.load(SAVE_PATH+'{}.MY_chem_A_feat.pt'.format(file_name))
MY_chem_B_feat = torch.load(SAVE_PATH+'{}.MY_chem_B_feat.pt'.format(file_name))
MY_chem_A_adj = torch.load(SAVE_PATH+'{}.MY_chem_A_adj.pt'.format(file_name))
MY_chem_B_adj = torch.load(SAVE_PATH+'{}.MY_chem_B_adj.pt'.format(file_name))
MY_g_EXP_A = torch.load(SAVE_PATH+'{}.MY_g_EXP_A.pt'.format(file_name))
MY_g_EXP_B = torch.load(SAVE_PATH+'{}.MY_g_EXP_B.pt'.format(file_name))
MY_Target_1_A = torch.load(SAVE_PATH+'{}.MY_Target_1_A.pt'.format(file_name))
MY_Target_1_B = torch.load(SAVE_PATH+'{}.MY_Target_1_B.pt'.format(file_name))
MY_CellBase = torch.load(SAVE_PATH+'{}.MY_CellBase.pt'.format(file_name))
MY_syn = torch.load(SAVE_PATH+'{}.MY_syn.pt'.format(file_name))




A_B_C_S_SET_ADD2 = copy.deepcopy(A_B_C_S_SET_ADD)

cid_a = list(A_B_C_S_SET_ADD2['CID_A'])
cid_b = list(A_B_C_S_SET_ADD2['CID_B'])
sm_a = list(A_B_C_S_SET_ADD2['ROW_CAN_SMILES'])
sm_b = list(A_B_C_S_SET_ADD2['COL_CAN_SMILES'])
ccle = list(A_B_C_S_SET_ADD2['CELL'])

A_B_C_S_SET_ADD2['CID_CID'] = [str(int(cid_a[i])) + '___' + str(int(cid_b[i])) if cid_a[i] < cid_b[i] else str(int(cid_b[i])) + '___' + str(int(cid_a[i])) for i in range(A_B_C_S_SET_ADD2.shape[0])]
A_B_C_S_SET_ADD2['SM_C_CHECK'] = [sm_a[i] + '___' + sm_b[i]+ '___' + ccle[i] if sm_a[i] < sm_b[i] else sm_b[i] + '___' + sm_a[i]+ '___' + ccle[i] for i in range(A_B_C_S_SET_ADD2.shape[0])]

A_B_C_S_SET_ADD2['ori_index'] = list(A_B_C_S_SET_ADD2.index)




MISS_filter = ['AOBO','AXBX','AXBO','AOBX'] # 

A_B_C_S_SET = A_B_C_S_SET_ADD2[A_B_C_S_SET_ADD2.Basal_Exp == 'O']

A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.SYN_OX == 'O']

A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.T1OX == 'O'] ####################### new targets 

A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.type.isin(MISS_filter)]




# basal node exclude -> CCLE match 만 사용
CCLE_PATH = '/st06/jiyeonH/13.DD_SESS/CCLE.22Q1/'
ccle_exp = pd.read_csv(CCLE_PATH+'CCLE_expression.csv', low_memory=False)
ccle_info= pd.read_csv(CCLE_PATH+'sample_info.csv', low_memory=False)

ccle_cell_info = ccle_info[['DepMap_ID','CCLE_Name']]
ccle_cell_info.columns = ['DepMap_ID','DrugCombCCLE']

ccle_cell_info_filt = ccle_cell_info[ccle_cell_info.DepMap_ID.isin(ccle_exp['Unnamed: 0'])]
ccle_names = [a for a in ccle_cell_info_filt.DrugCombCCLE if type(a) == str]


A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.CELL.isin(ccle_names)]




data_ind = list(A_B_C_S_SET.index)

MY_chem_A_feat_RE = MY_chem_A_feat[data_ind]
MY_chem_B_feat_RE = MY_chem_B_feat[data_ind]
MY_chem_A_adj_RE = MY_chem_A_adj[data_ind]
MY_chem_B_adj_RE = MY_chem_B_adj[data_ind]
MY_g_EXP_A_RE = MY_g_EXP_A[data_ind]
MY_g_EXP_B_RE = MY_g_EXP_B[data_ind]
MY_Target_A = copy.deepcopy(MY_Target_1_A)[data_ind] ############## NEW TARGET !!!!!! #####
MY_Target_B = copy.deepcopy(MY_Target_1_B)[data_ind] ############## NEW TARGET !!!!!! #####

MY_CellBase_RE = MY_CellBase[data_ind]
MY_syn_RE = MY_syn[data_ind]


A_B_C_S_SET = A_B_C_S_SET.reset_index(drop = True)



# cell line vector 

DC_CELL_DF2 = pd.read_csv(DC_PATH+'DC_CELL_INFO.csv', sep = '\t')
DC_CELL_DF2 = pd.concat([
	DC_CELL_DF2, 
	pd.DataFrame({'cell_line_id' : [1],'DC_cellname' : ['786O'],'DrugCombCello' : ['CVCL_1051'],'DrugCombCCLE':['786O_KIDNEY']})])

DC_CELL_info_filt = DC_CELL_DF2[DC_CELL_DF2.DrugCombCCLE.isin(A_B_C_S_SET.CELL)] # 38

DC_CELL_info_filt = DC_CELL_info_filt.drop(['Unnamed: 0'], axis = 1)
DC_CELL_info_filt.columns = ['cell_line_id', 'DC_cellname', 'DrugCombCello', 'CELL']
DC_CELL_info_filt = DC_CELL_info_filt[['CELL','DC_cellname']]

A_B_C_S_SET_COH = pd.merge(A_B_C_S_SET, DC_CELL_info_filt, on = 'CELL', how = 'left'  )

							# DC_CELL_DF2['OX'] = DC_CELL_DF2.DrugCombCCLE.apply(lambda x : 'O' if type(x) == str else 'X')
							# DC_CELL_info_filt = DC_CELL_DF2[DC_CELL_DF2['OX']=='O']
							# DC_CELL_info_filt = DC_CELL_info_filt.drop(['Unnamed: 0'], axis = 1)
							# DC_CELL_info_filt.columns = ['cell_line_id', 'DC_cellname', 'DrugCombCello', 'CELL', 'OX']
							# DC_CELL_info_filt = DC_CELL_info_filt[['CELL','DC_cellname']]










# sample number filter 

# 빈도 확인 

C_names = list(set(A_B_C_S_SET_COH.DC_cellname))
C_names.sort()

C_freq = [list(A_B_C_S_SET_COH.DC_cellname).count(a) for a in C_names]
C_cclename = [list(A_B_C_S_SET_COH[A_B_C_S_SET_COH.DC_cellname==a]['CELL'])[0] for a in C_names]

C_df = pd.DataFrame({'cell' : C_names, 'freq' : C_freq, 'ccle' : C_cclename})
C_df = C_df.sort_values('freq')

CELL_CUT = 200 ####### 이것도 그렇게 되면 바꿔야하지 않을까 ##################################################################

C_freq_filter = C_df[C_df.freq > CELL_CUT ] 


A_B_C_S_SET_COH = A_B_C_S_SET_COH[A_B_C_S_SET_COH.DC_cellname.isin(C_freq_filter.cell)]

DC_CELL_info_filt_re = DC_CELL_info_filt[DC_CELL_info_filt.DC_cellname.isin(C_freq_filter.cell)]
DC_CELL_info_filt_re['cell_onehot'] = [a for a in range(len(set(DC_CELL_info_filt_re.CELL)))]

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



print('CIDs', flush = True)
tmp = list(set(A_B_C_S_SET_COH2.CID_CID))
tmp2 = sum([a.split('___') for a in tmp],[])
print(len(set(tmp2)) , flush = True)


print('CID_CID', flush = True)
print(len(set(A_B_C_S_SET_COH2.CID_CID)), flush = True)



print('CID_CID_CCLE', flush = True)
print(len(set(A_B_C_S_SET_COH2.cid_cid_cell)), flush = True)

print('DrugCombCCLE', flush = True)
print(len(set(A_B_C_S_SET_COH2.CELL)), flush = True)


###########################################################################################
###########################################################################################
###########################################################################################

# 일단 생 5CV


print("LEARNING")

A_B_C_S_SET_SM = copy.deepcopy(A_B_C_S_SET_COH2) # 

# get unique values, remove duplicates, but keep original counts
data_no_dup, counts = np.unique(list(A_B_C_S_SET_SM['SM_C_CHECK']), return_counts=True)
data_no_dup_cells = [setset.split('___')[2] for setset in data_no_dup]
data_no_dup_sm_sm = [setset.split('___')[0]+'___'+setset.split('___')[1] for setset in data_no_dup]
data_nodup_df = pd.DataFrame({
	'setset' : data_no_dup.tolist(), 
	'cell' : data_no_dup_cells,
	'SM_SM' : data_no_dup_sm_sm
	 })




SM_SM_list = list(set(data_nodup_df.SM_SM))
SM_SM_list.sort()
sm_sm_list_1 = sklearn.utils.shuffle(SM_SM_list, random_state=42)

bins = [a for a in range(0, len(sm_sm_list_1), round(len(sm_sm_list_1)*0.2) )]
bins = bins[1:]
res = np.split(sm_sm_list_1, bins)

CV_1_smsm = list(res[0])
CV_2_smsm = list(res[1])
CV_3_smsm = list(res[2])
CV_4_smsm = list(res[3])
CV_5_smsm = list(res[4])
if len(res) > 5 :
	CV_5_smsm = list(res[4]) + list(res[5])

len(sm_sm_list_1)
len(CV_1_smsm) + len(CV_2_smsm) + len(CV_3_smsm) + len(CV_4_smsm) + len(CV_5_smsm)

CV_1_setset = list(data_nodup_df[data_nodup_df.SM_SM.isin(CV_1_smsm)]['setset'])
CV_2_setset = list(data_nodup_df[data_nodup_df.SM_SM.isin(CV_2_smsm)]['setset'])
CV_3_setset = list(data_nodup_df[data_nodup_df.SM_SM.isin(CV_3_smsm)]['setset'])
CV_4_setset = list(data_nodup_df[data_nodup_df.SM_SM.isin(CV_4_smsm)]['setset'])
CV_5_setset = list(data_nodup_df[data_nodup_df.SM_SM.isin(CV_5_smsm)]['setset'])




CV_ND_INDS = {
	'CV0_train' : CV_1_setset + CV_2_setset + CV_3_setset + CV_4_setset, 
	'CV0_test' : CV_5_setset,
	'CV1_train' : CV_2_setset + CV_3_setset + CV_4_setset + CV_5_setset, 
	'CV1_test' : CV_1_setset,
	'CV2_train' : CV_3_setset + CV_4_setset + CV_5_setset + CV_1_setset,
	'CV2_test' : CV_2_setset,
	'CV3_train' : CV_4_setset + CV_5_setset + CV_1_setset + CV_2_setset,
	'CV3_test' : CV_3_setset,
	'CV4_train' : CV_5_setset + CV_1_setset + CV_2_setset + CV_3_setset,
	'CV4_test' : CV_4_setset 
}

print(data_nodup_df.shape)
len( CV_1_setset + CV_2_setset + CV_3_setset + CV_4_setset + CV_5_setset)
len(set( CV_1_setset + CV_2_setset + CV_3_setset + CV_4_setset + CV_5_setset ))







WORK_NAME = 'WORK_203' # 349
W_NAME = 'W203'
PRJ_NAME = 'M3V6'
MJ_NAME = 'M3V6'
MISS_NAME = 'MIS2'
PPI_NAME = '349'

PRJ_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/{}_{}_{}_{}/'.format(PRJ_NAME, W_NAME, PPI_NAME, MISS_NAME)

# 저장해둔거랑 같은지 확인 
with open('{}/CV_SM_list.pickle'.format(PRJ_PATH), 'rb') as f:
	CV_ND_INDS_ray = pickle.load(f)
 
for kk in ['CV0_train', 'CV0_test', 'CV1_train', 'CV1_test', 'CV2_train', 'CV2_test', 'CV3_train', 'CV3_test', 'CV4_train', 'CV4_test'] :
	CV_ND_INDS[kk] == CV_ND_INDS_ray[kk]

# 모두 true 










# 주어진 환경 맞춰서 5CV만 돌릴거라서 수정버전 
def prepare_data_GCN(CV_num, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, 
MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, 
cell_one_hot, MY_syn_RE2, norm ) : 
	# 
	# CV_num = 0
	train_key = 'CV{}_train'.format(CV_num)
	test_key = 'CV{}_test'.format(CV_num)
	# 
	#
	ABCS_tv = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS[train_key])]
	ABCS_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS[test_key])]
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
	#
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




seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


# gcn_drug1_F, gcn_drug2_F, gcn_drug1_ADJ, gcn_drug2_ADJ, gcn_exp_A, gcn_exp_B, gcn_tgt_A, gcn_tgt_B, gcn_adj, gcn_adj_weight, syn_ans, cell_info
norm = 'tanh_norm'

# CV_0
train_data_0, test_data_0 = prepare_data_GCN(0, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, 
MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, 
cell_one_hot, MY_syn_RE2, norm)

# CV_1
train_data_1, test_data_1 = prepare_data_GCN(1, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, 
MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, 
cell_one_hot, MY_syn_RE2, norm)

# CV_2
train_data_2, test_data_2 = prepare_data_GCN(2, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, 
MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, 
cell_one_hot, MY_syn_RE2, norm)

# CV_3
train_data_3, test_data_3 = prepare_data_GCN(3, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, 
MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, 
cell_one_hot, MY_syn_RE2, norm)

# CV_4
train_data_4, test_data_4 = prepare_data_GCN(4, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, 
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
RAY_test_0 = ray.put(T_test_0)


# CV 1
T_train_1, T_test_1 = make_merged_data(1)
RAY_test_1 = ray.put(T_test_1)


# CV 2 
T_train_2, T_test_2 = make_merged_data(2)
RAY_test_2 = ray.put(T_test_2)


# CV 3
T_train_3, T_test_3 = make_merged_data(3)
RAY_test_3 = ray.put(T_test_3)


# CV 4
T_train_4, T_test_4 = make_merged_data(4)
RAY_test_4 = ray.put(T_test_4)




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
	test_loader = torch.utils.data.DataLoader(T_test, batch_size = my_config["config/batch_size"].item(), collate_fn = graph_collate_fn, shuffle =False, num_workers=16) # my_config['config/n_workers'].item()
	#
	G_chem_layer = my_config['config/G_chem_layer'].item()
	G_chem_hdim = my_config['config/G_chem_hdim'].item()
	G_exp_layer = my_config['config/G_exp_layer'].item()
	G_exp_hdim = my_config['config/G_exp_hdim'].item() 
	dsn_layers = [int(a) for a in my_config["config/dsn_layer"].split('-') ]
	snp_layers = [int(a) for a in my_config["config/snp_layer"].split('-') ]
	inDrop = my_config['config/dropout_1'].item()
	Drop = my_config['config/dropout_2'].item()
	#
	best_model = MY_expGCN_parallel_model(
				G_chem_layer, T_test.gcn_drug1_F.shape[-1] , G_chem_hdim,
				G_exp_layer, 3, G_exp_hdim,
				dsn_layers, dsn_layers, snp_layers, 
				1,
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




PRJ_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/{}_{}_{}_{}/'.format(PRJ_NAME, W_NAME, PPI_NAME, MISS_NAME)
ANA_DF = pd.read_csv(PRJ_PATH+'RAY_ANA_DF.{}_{}_{}_{}.csv'.format(PRJ_NAME, W_NAME, PPI_NAME, MISS_NAME))
with open(PRJ_PATH+'RAY_ANA_DF.{}_{}_{}_{}.pickle'.format(PRJ_NAME, W_NAME, PPI_NAME, MISS_NAME), 'rb') as f:
	ANA_ALL_DF = pickle.load(f)


TOPVAL_PATH = PRJ_PATH





my_config = ANA_DF.loc[0]


# 1) full 

R_1_T_CV0, R_1_1_CV0, R_1_2_CV0, pred_1_CV0, ans_1_CV0 = TEST_CPU(PRJ_PATH, 'CV_0', my_config, PRJ_PATH, 'full_CV_0_model.pth', 'FULL')
R_1_T_CV1, R_1_1_CV1, R_1_2_CV1, pred_1_CV1, ans_1_CV1 = TEST_CPU(PRJ_PATH, 'CV_1', my_config, PRJ_PATH, 'full_CV_1_model.pth', 'FULL')
R_1_T_CV2, R_1_1_CV2, R_1_2_CV2, pred_1_CV2, ans_1_CV2 = TEST_CPU(PRJ_PATH, 'CV_2', my_config, PRJ_PATH, 'full_CV_2_model.pth', 'FULL')
R_1_T_CV3, R_1_1_CV3, R_1_2_CV3, pred_1_CV3, ans_1_CV3 = TEST_CPU(PRJ_PATH, 'CV_3', my_config, PRJ_PATH, 'full_CV_3_model.pth', 'FULL')
R_1_T_CV4, R_1_1_CV4, R_1_2_CV4, pred_1_CV4, ans_1_CV4 = TEST_CPU(PRJ_PATH, 'CV_4', my_config, PRJ_PATH, 'full_CV_4_model.pth', 'FULL')


# 2) min loss 

R_2_T_CV0, R_2_1_CV0, R_2_2_CV0, pred_2_CV0, ans_2_CV0 = TEST_CPU(PRJ_PATH, 'CV_0', my_config, PRJ_PATH, 'VLS_CV_0_model.pth', 'VLS')
R_2_T_CV1, R_2_1_CV1, R_2_2_CV1, pred_2_CV1, ans_2_CV1 = TEST_CPU(PRJ_PATH, 'CV_1', my_config, PRJ_PATH, 'VLS_CV_1_model.pth', 'VLS')
R_2_T_CV2, R_2_1_CV2, R_2_2_CV2, pred_2_CV2, ans_2_CV2 = TEST_CPU(PRJ_PATH, 'CV_2', my_config, PRJ_PATH, 'VLS_CV_2_model.pth', 'VLS')
R_2_T_CV3, R_2_1_CV3, R_2_2_CV3, pred_2_CV3, ans_2_CV3 = TEST_CPU(PRJ_PATH, 'CV_3', my_config, PRJ_PATH, 'VLS_CV_3_model.pth', 'VLS')
R_2_T_CV4, R_2_1_CV4, R_2_2_CV4, pred_2_CV4, ans_2_CV4 = TEST_CPU(PRJ_PATH, 'CV_4', my_config, PRJ_PATH, 'VLS_CV_4_model.pth', 'VLS')



R_3_T_CV0, R_3_1_CV0, R_3_2_CV0, pred_3_CV0, ans_3_CV0 = TEST_CPU(PRJ_PATH, 'CV_0', my_config, PRJ_PATH, 'VPC_CV_0_model.pth', 'VPC')
R_3_T_CV1, R_3_1_CV1, R_3_2_CV1, pred_3_CV1, ans_3_CV1 = TEST_CPU(PRJ_PATH, 'CV_1', my_config, PRJ_PATH, 'VPC_CV_1_model.pth', 'VPC')
R_3_T_CV2, R_3_1_CV2, R_3_2_CV2, pred_3_CV2, ans_3_CV2 = TEST_CPU(PRJ_PATH, 'CV_2', my_config, PRJ_PATH, 'VPC_CV_2_model.pth', 'VPC')
R_3_T_CV3, R_3_1_CV3, R_3_2_CV3, pred_3_CV3, ans_3_CV3 = TEST_CPU(PRJ_PATH, 'CV_3', my_config, PRJ_PATH, 'VPC_CV_3_model.pth', 'VPC')
R_3_T_CV4, R_3_1_CV4, R_3_2_CV4, pred_3_CV4, ans_3_CV4 = TEST_CPU(PRJ_PATH, 'CV_4', my_config, PRJ_PATH, 'VPC_CV_4_model.pth', 'VPC')



R_4_T_CV0, R_4_1_CV0, R_4_2_CV0, pred_4_CV0, ans_4_CV0 = TEST_CPU(PRJ_PATH, 'CV_0', my_config, PRJ_PATH, 'VSC_CV_0_model.pth', 'VLS')
R_4_T_CV1, R_4_1_CV1, R_4_2_CV1, pred_4_CV1, ans_4_CV1 = TEST_CPU(PRJ_PATH, 'CV_1', my_config, PRJ_PATH, 'VSC_CV_1_model.pth', 'VLS')
R_4_T_CV2, R_4_1_CV2, R_4_2_CV2, pred_4_CV2, ans_4_CV2 = TEST_CPU(PRJ_PATH, 'CV_2', my_config, PRJ_PATH, 'VSC_CV_2_model.pth', 'VLS')
R_4_T_CV3, R_4_1_CV3, R_4_2_CV3, pred_4_CV3, ans_4_CV3 = TEST_CPU(PRJ_PATH, 'CV_3', my_config, PRJ_PATH, 'VSC_CV_3_model.pth', 'VLS')
R_4_T_CV4, R_4_1_CV4, R_4_2_CV4, pred_4_CV4, ans_4_CV4 = TEST_CPU(PRJ_PATH, 'CV_4', my_config, PRJ_PATH, 'VSC_CV_4_model.pth', 'VLS')



# 이거 끝나면 저장해서 plot 그려야함 

아. 이제 이해함. 
PRED_1 은 그냥 특정 모델번호인거임. 
바보냐 진자 

ABCS_test_0 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV0_test'])]
ABCS_test_0['ANS'] = ans_1_CV0 ; ABCS_test_0['PRED_1'] = pred_1_CV0; ABCS_test_0['PRED_2'] = pred_2_CV0 ; ABCS_test_0['PRED_3'] = pred_3_CV0 ;  ABCS_test_0['PRED_4'] = pred_4_CV0

ABCS_test_1 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV1_test'])]
ABCS_test_1['ANS'] = ans_1_CV1 ;ABCS_test_1['PRED_1'] = pred_1_CV1 ;ABCS_test_1['PRED_2'] = pred_2_CV1 ;ABCS_test_1['PRED_3'] = pred_3_CV1 ;ABCS_test_1['PRED_4'] = pred_4_CV1

ABCS_test_2 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV2_test'])]
ABCS_test_2['ANS'] = ans_1_CV2 ; ABCS_test_2['PRED_1'] = pred_1_CV2; ABCS_test_2['PRED_2'] = pred_2_CV2; ABCS_test_2['PRED_3'] = pred_3_CV2; ABCS_test_2['PRED_4'] = pred_4_CV2

ABCS_test_3 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV3_test'])]
ABCS_test_3['ANS'] = ans_1_CV3 ;ABCS_test_3['PRED_1'] = pred_1_CV3 ; ABCS_test_3['PRED_2'] = pred_2_CV3 ; ABCS_test_3['PRED_3'] = pred_3_CV3 ;ABCS_test_3['PRED_4'] = pred_4_CV3

ABCS_test_4 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV4_test'])]
ABCS_test_4['ANS'] = ans_1_CV4 ; ABCS_test_4['PRED_1'] = pred_1_CV4 ;ABCS_test_4['PRED_2'] = pred_2_CV4 ;ABCS_test_4['PRED_3'] = pred_3_CV4 ;ABCS_test_4['PRED_4'] = pred_4_CV4


ABCS_test_result = pd.concat([ABCS_test_0, ABCS_test_1, ABCS_test_2, ABCS_test_3, ABCS_test_4])


ABCS_test_result.to_csv(PRJ_PATH+'ABCS_test_result.csv', index = False)


round(stats.pearsonr(ABCS_test_result['ANS'] , ABCS_test_result['PRED_1'])[0] , 4) # 0.7056
round(stats.pearsonr(ABCS_test_result['ANS'] , ABCS_test_result['PRED_2'])[0] , 4) # 0.7092
round(stats.pearsonr(ABCS_test_result['ANS'] , ABCS_test_result['PRED_3'])[0] , 4) # 0.7098
round(stats.pearsonr(ABCS_test_result['ANS'] , ABCS_test_result['PRED_4'])[0] , 4) # 0.7088





########################################
이제 bar plot 이랑 뭐든 좀 그려보자 


WORK_NAME = 'WORK_203' # 349
W_NAME = 'W203'
PRJ_NAME = 'M3V6'
MJ_NAME = 'M3V6'
MISS_NAME = 'MIS2'
PPI_NAME = '349'

PRJ_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/{}_{}_{}_{}/'.format(PRJ_NAME, W_NAME, PPI_NAME, MISS_NAME)


ABCS_test_result = pd.read_csv(PRJ_PATH+'ABCS_test_result.csv')



ABCS_test_result['tissue'] = [ '_'.join(a.split('_')[1:]) if type(a) == str else 'NA' for a in list(ABCS_test_result['CELL'])]
DC_CELL_info_filt['tissue'] = [ '_'.join(a.split('_')[1:]) if type(a) == str else 'NA' for a in list(DC_CELL_info_filt['CELL'])]

test_cell_df = pd.DataFrame({'DC_cellname' : list(set(ABCS_test_result.DC_cellname))})

cell_P = []
cell_S = []
cell_num = []

for cell in list(test_cell_df.DC_cellname) :
	tmp_test_re = ABCS_test_result[ABCS_test_result.DC_cellname == cell]
	cell_P_corr, _ = stats.pearsonr(tmp_test_re.ANS, tmp_test_re.PRED_3)
	cell_S_corr, _ = stats.spearmanr(tmp_test_re.ANS, tmp_test_re.PRED_3)
	cell_nums = tmp_test_re.shape[0]
	cell_P.append(cell_P_corr)
	cell_S.append(cell_S_corr)
	cell_num.append(cell_nums)


test_cell_df['P_COR'] = cell_P
test_cell_df['S_COR'] = cell_S
test_cell_df['cell_num'] = cell_num

test_cell_df = pd.merge(test_cell_df, DC_CELL_info_filt[['DC_cellname','tissue']], on = 'DC_cellname', how = 'left'  )

tissue_set = ['CENTRAL_NERVOUS_SYSTEM', 'LUNG', 'BREAST', 'BONE', 'OVARY', 'PROSTATE', 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'LARGE_INTESTINE', 'SKIN', 'PLEURA', 'KIDNEY' ] # list(set(test_cell_df['tissue']))
color_set = ['#CF3476','#FF7514','#025669','#308446','#84C3BE','#D53032','#4ddcfd','#ffcd36','#ac8cff',"#0000ffff","#7bff68ff"] # "#1E1E1E"
color_dict = {a : color_set[tissue_set.index(a)] for a in tissue_set}

test_cell_df['tissue_oh'] = [color_dict[a] for a in list(test_cell_df['tissue'])]



# 이쁜 그림을 위한 func 

# Pearson corr
test_cell_df = test_cell_df.sort_values('P_COR')

fig = plt.figure(figsize=(30,8))
x_pos = [a*2 for a in range(test_cell_df.shape[0])]

plt.bar(x_pos, test_cell_df['P_COR'], color=test_cell_df['tissue_oh']) # 
plt.xticks(x_pos,list(test_cell_df['DC_cellname']), rotation=90, fontsize= 18)
plt.yticks(np.arange(0, 1, step=0.2),np.round(np.arange(0, 1, step=0.2),2), fontsize= 18)
for i in range(test_cell_df.shape[0]):
	#plt.annotate(str(list(test_cell_df['cell_num'])[i]), xy=(x_pos[i],list(test_cell_df['P_COR'])[i]), ha='center', va='bottom', fontsize= 18)
	plt.annotate(str(list(np.round(test_cell_df['P_COR'],1))[i]), xy=(x_pos[i],list(test_cell_df['P_COR'])[i]), ha='center', va='bottom', fontsize= 15)

plt.legend(loc = 'upper left')
plt.tight_layout()
fig.savefig('{}/{}.png'.format(PRJ_PATH, 'new_plot_pearson'), bbox_inches = 'tight')

plt.close()




# 이쁜 그림을 위한 func  # 보고서용 다시 

# Pearson corr
test_cell_df = test_cell_df.sort_values('P_COR')

fig = plt.figure(figsize=(30,8))
x_pos = [a*2 for a in range(test_cell_df.shape[0])]

plt.bar(x_pos, test_cell_df['P_COR'], color=test_cell_df['tissue_oh']) # 
plt.xticks(x_pos,list(test_cell_df['DC_cellname']), rotation=90, fontsize= 18)
plt.yticks(np.arange(0, 1, step=0.2),np.round(np.arange(0, 1, step=0.2),2), fontsize= 18)
#plt.grid(True)
#plt.axhline(0.7)
plt.tight_layout()
fig.savefig('{}/{}.png'.format(PRJ_PATH, 'new_plot_pearson2'), bbox_inches = 'tight')
fig.savefig('{}/{}.pdf'.format(PRJ_PATH, 'new_plot_pearson2'), format="pdf", bbox_inches = 'tight')

plt.close()



















# 추가적으로 궁금해져서 
# AXBX 의 비율을 살펴보기로 함 
# total 10552
TEST_ALL_O = ABCS_test_result[ABCS_test_result.type.isin(['AOBO'])] # 1348 -> 659 
TEST_HALF_O = ABCS_test_result[ABCS_test_result.type.isin(['AXBO','AOBX'])] # 2179 -> 1074
TEST_NO_O = ABCS_test_result[ABCS_test_result.type.isin(['AXBX'])] # 7025 -> 16854




def give_test_result_corDF (ABCS_test_result) : 
	THIS_TEST = pd.merge(ABCS_test_result, DC_CELL_info_filt[['CELL', 'tissue']], on = 'CELL', how = 'left'  )
	test_cell_df = pd.DataFrame({'DC_cellname' : list(set(THIS_TEST.DC_cellname))})
	#
	cell_P = []
	cell_S = []
	cell_num = []
	#
	for cell in list(test_cell_df.DC_cellname) :
		tmp_test_re = THIS_TEST[THIS_TEST.DC_cellname == cell]
		if tmp_test_re.shape[0] > 1 :
			cell_P_corr, _ = stats.pearsonr(tmp_test_re.ANS, tmp_test_re.PRED_3)
			cell_S_corr, _ = stats.spearmanr(tmp_test_re.ANS, tmp_test_re.PRED_3)
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
	tissue_set = ['CENTRAL_NERVOUS_SYSTEM', 'LUNG', 'BREAST', 'BONE', 'OVARY', 'PROSTATE', 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'LARGE_INTESTINE', 'SKIN', 'PLEURA', 'KIDNEY' ] # list(set(test_cell_df['tissue']))
	color_set = ['#CF3476','#FF7514','#025669','#308446','#84C3BE','#D53032','#4ddcfd','#ffcd36','#ac8cff',"#0000ffff","#7bff68ff"] # "#1E1E1E"
	color_dict = {a : color_set[tissue_set.index(a)] for a in tissue_set}
	#
	test_cell_df['tissue_oh'] = [color_dict[a] for a in list(test_cell_df['tissue'])]
	test_cell_df = test_cell_df.sort_values('S_COR')
	return test_cell_df


TEST_ALL_O_RESULT = give_test_result_corDF(TEST_ALL_O)
TEST_HALF_O_RESULT = give_test_result_corDF(TEST_HALF_O)
TEST_NO_O_RESULT = give_test_result_corDF(TEST_NO_O)

TEST_ALL_O_RESULT

TEST_HALF_O_RESULT

TEST_NO_O_RESULT




max(ABCS_test_result.PRED_1)
max(ABCS_test_result.PRED_2)
max(ABCS_test_result.PRED_3)
max(ABCS_test_result.PRED_4)


min(ABCS_test_result.PRED_1)
min(ABCS_test_result.PRED_2)
min(ABCS_test_result.PRED_3)
min(ABCS_test_result.PRED_4)

max(ABCS_test_result.ANS)
min(ABCS_test_result.ANS)




# violin plot for tissue 
from matplotlib import colors as mcolors

tiss_list = tissue_set
my_order = test_cell_df.groupby(by=["tissue"])["P_COR"].mean().sort_values().iloc[::-1].index

fig, ax = plt.subplots(figsize=(25, 15))
sns.violinplot(ax = ax, data  = test_cell_df, x = 'tissue', y = 'P_COR', linewidth=1,  edgecolor="dimgrey", order=my_order,  inner = 'point') # width = 3,
violins = [c for i,c in enumerate(ax.collections) if i%2 == 0] # pollycollection 가져오기 위함 

				#'LARGE_INTESTINE', 'PROSTATE', 'OVARY', 'PLEURA', 'LUNG', 'SKIN','KIDNEY', 'CENTRAL_NERVOUS_SYSTEM', 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'BREAST', 'BONE'
				# [9, 2, 12, 1, 13, 22, 5, 6, 5, 16, 1]




# 하나밖에 없는애들은 violin 안그려져서 문제 
violins[0].set_facecolor(mcolors.to_rgba(color_dict['LARGE_INTESTINE'], 0.7))
violins[1].set_facecolor(mcolors.to_rgba(color_dict['PROSTATE'], 0.7))
violins[2].set_facecolor(mcolors.to_rgba(color_dict['OVARY'], 0.7))
violins[3].set_facecolor(mcolors.to_rgba(color_dict['LUNG'], 0.7))
violins[4].set_facecolor(mcolors.to_rgba(color_dict['SKIN'], 0.7))
violins[5].set_facecolor(mcolors.to_rgba(color_dict['KIDNEY'], 0.7))
violins[6].set_facecolor(mcolors.to_rgba(color_dict['CENTRAL_NERVOUS_SYSTEM'], 0.7))
violins[7].set_facecolor(mcolors.to_rgba(color_dict['HAEMATOPOIETIC_AND_LYMPHOID_TISSUE'], 0.7))
violins[8].set_facecolor(mcolors.to_rgba(color_dict['BREAST'], 0.7))




ax.set_xlabel('tissue names', fontsize=10)
ax.set_ylabel('Pearson Corr', fontsize=10)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
ax.tick_params(axis='both', which='major', labelsize=10 )
plt.tight_layout()
plt.grid(True)
plt.savefig(os.path.join(PRJ_PATH,'tissue_pearson.{}.png'.format(W_NAME)), dpi = 300)

plt.close()




# violin plot for tissue  22222 좀더 이쁜 버전 
from matplotlib import colors as mcolors

tiss_list = tissue_set
my_order = test_cell_df.groupby(by=["tissue"])["P_COR"].mean().sort_values().iloc[::-1].index

fig, ax = plt.subplots(figsize=(25, 10))
sns.violinplot(ax = ax, data  = test_cell_df, x = 'tissue', y = 'P_COR', linewidth=1,  edgecolor="dimgrey", order=my_order) # width = 3,,  inner = 'point'
violins = [c for i,c in enumerate(ax.collections) if i%2 == 0] # pollycollection 가져오기 위함 

				#'LARGE_INTESTINE', 'PROSTATE', 'OVARY', 'PLEURA', 'LUNG', 'SKIN','KIDNEY', 'CENTRAL_NERVOUS_SYSTEM', 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'BREAST', 'BONE'
				# [9, 2, 12, 1, 13, 22, 5, 6, 5, 16, 1]



# 하나밖에 없는애들은 violin 안그려져서 문제 
violins[0].set_facecolor(mcolors.to_rgba(color_dict['LARGE_INTESTINE'], 0.7))
violins[1].set_facecolor(mcolors.to_rgba(color_dict['PROSTATE'], 0.7))
violins[2].set_facecolor(mcolors.to_rgba(color_dict['OVARY'], 0.7))
violins[3].set_facecolor(mcolors.to_rgba(color_dict['LUNG'], 0.7))
violins[4].set_facecolor(mcolors.to_rgba(color_dict['SKIN'], 0.7))
violins[5].set_facecolor(mcolors.to_rgba(color_dict['KIDNEY'], 0.7))
violins[6].set_facecolor(mcolors.to_rgba(color_dict['CENTRAL_NERVOUS_SYSTEM'], 0.7))
violins[7].set_facecolor(mcolors.to_rgba(color_dict['HAEMATOPOIETIC_AND_LYMPHOID_TISSUE'], 0.7))
violins[8].set_facecolor(mcolors.to_rgba(color_dict['BREAST'], 0.7))

avail_cell_dict = {'PROSTATE': ['VCAP', 'PC3'], 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE': ['UHO1', 'K562', 'L1236', 'SR786', 'RPMI8226'], 'BREAST': ['BT549', 'BT474', 'MDAMB361', 'MDAMB231', 'UACC812', 'KPL1', 'MDAMB468', 'MDAMB175VII', 'HCC1500', 'HCC1419', 'MDAMB436', 'ZR751', 'MCF7', 'HS578T', 'T47D', 'CAMA1'], 'CENTRAL_NERVOUS_SYSTEM': ['T98G', 'SF539', 'U251MG', 'SNB75', 'SF268', 'SF295'], 'LUNG': ['A549', 'A427', 'HOP92', 'NCIH23', 'NCIH1650', 'NCIH520', 'HOP62', 'NCIH460', 'NCIH2122', 'NCIH226', 'EKVX', 'SKMES1', 'NCIH522'], 'LARGE_INTESTINE': ['SW620', 'LOVO', 'RKO', 'KM12', 'HCT15', 'HCT116', 'HT29', 'DLD1', 'SW837'], 'OVARY': ['NIHOVCAR3', 'OVCAR4', 'OVCAR5', 'IGROV1', 'OVCAR8', 'PA1', 'UWB1289', 'SKOV3', 'CAOV3', 'ES2', 'A2780', 'OV90'], 'SKIN': ['SKMEL5', 'HT144', 'RVH421', 'SKMEL28', 'A375', 'A101D', 'COLO792', 'UACC257', 'COLO800', 'MEWO', 'MELHO', 'A2058', 'RPMI7951', 'IPC298', 'MALME3M', 'SKMEL2', 'UACC62', 'LOXIMVI', 'WM115', 'G361', 'SKMEL30', 'COLO829'], 'BONE': ['A673'], 'KIDNEY': ['CAKI1', 'UO31', 'ACHN', 'A498', '786O'], 'PLEURA': ['MSTO211H']}
breast_check = ccle_info[ccle_info.stripped_cell_line_name.isin(avail_cell_dict['BREAST'])][['cell_line_name','lineage_molecular_subtype']]
breast_check.columns = ['DC_cellname','subclass']


test_cell_df2 = pd.merge(test_cell_df, breast_check, on = 'DC_cellname', how = 'left')
test_cell_df2['subclass'] = test_cell_df2.subclass.apply(lambda x : "NA" if type(x) != str else x)

test_cell_df2.at[test_cell_df2[test_cell_df2.DC_cellname=='ZR751'].index.item(),'subclass'] = 'luminal'
test_cell_df2.at[test_cell_df2[test_cell_df2.DC_cellname=='HS 578T'].index.item(),'subclass'] = 'basal_B'
test_cell_df2.at[test_cell_df2[test_cell_df2.DC_cellname=='KPL1'].index.item(),'subclass'] = 'luminal'
test_cell_df2.at[test_cell_df2[test_cell_df2.DC_cellname=='MDAMB436'].index.item(),'subclass'] = 'basal_B'

# test_cell_df2.subclass.factorize()
sns.swarmplot(ax = ax, data  = test_cell_df2, x = 'tissue', y = 'P_COR', hue='subclass', palette=sns.color_palette(['grey', 'yellow', 'pink', 'white','green']), order=my_order)



ax.set_xlabel('tissue names', fontsize=10)
ax.set_ylabel('Pearson Corr', fontsize=10)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, fontsize=25)
ax.tick_params(axis='both', which='major', labelsize=10 )
plt.tight_layout()
plt.grid(True)
plt.savefig(os.path.join(PRJ_PATH,'tissue_pearson2.{}.png'.format(W_NAME)), dpi = 300)

plt.close()









# violin plot for tissue  33333 좀더 이쁜 버전 
from matplotlib import colors as mcolors

tiss_list = tissue_set
my_order = test_cell_df.groupby(by=["tissue"])["P_COR"].mean().sort_values().iloc[::-1].index

fig, ax = plt.subplots(figsize=(23, 13))
sns.violinplot(ax = ax, data  = test_cell_df, x = 'tissue', y = 'P_COR', linewidth=2,  edgecolor="black", order=my_order, width = 1, inner = None) # width = 3,,  
violins = [c for i,c in enumerate(ax.collections) ] # pollycollection 가져오기 위함 # if i%2 == 0

				#'LARGE_INTESTINE', 'PROSTATE', 'OVARY', 'PLEURA', 'LUNG', 'SKIN','KIDNEY', 'CENTRAL_NERVOUS_SYSTEM', 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'BREAST', 'BONE'
				# [9, 2, 12, 1, 13, 22, 5, 6, 5, 16, 1]



# 하나밖에 없는애들은 violin 안그려져서 문제 
violins[0].set_facecolor(mcolors.to_rgba(color_dict['LARGE_INTESTINE'], 0.8))
violins[1].set_facecolor(mcolors.to_rgba(color_dict['PROSTATE'], 0.8))
violins[2].set_facecolor(mcolors.to_rgba(color_dict['OVARY'], 0.8))
violins[3].set_facecolor(mcolors.to_rgba(color_dict['LUNG'], 0.8))
violins[4].set_facecolor(mcolors.to_rgba(color_dict['SKIN'], 0.8))
violins[5].set_facecolor(mcolors.to_rgba(color_dict['KIDNEY'], 0.8))
violins[6].set_facecolor(mcolors.to_rgba(color_dict['CENTRAL_NERVOUS_SYSTEM'], 0.8))
violins[7].set_facecolor(mcolors.to_rgba(color_dict['HAEMATOPOIETIC_AND_LYMPHOID_TISSUE'], 0.8))
violins[8].set_facecolor(mcolors.to_rgba(color_dict['BREAST'], 0.8))

avail_cell_dict = {'PROSTATE': ['VCAP', 'PC3'], 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE': ['UHO1', 'K562', 'L1236', 'SR786', 'RPMI8226'], 'BREAST': ['BT549', 'BT474', 'MDAMB361', 'MDAMB231', 'UACC812', 'KPL1', 'MDAMB468', 'MDAMB175VII', 'HCC1500', 'HCC1419', 'MDAMB436', 'ZR751', 'MCF7', 'HS578T', 'T47D', 'CAMA1'], 'CENTRAL_NERVOUS_SYSTEM': ['T98G', 'SF539', 'U251MG', 'SNB75', 'SF268', 'SF295'], 'LUNG': ['A549', 'A427', 'HOP92', 'NCIH23', 'NCIH1650', 'NCIH520', 'HOP62', 'NCIH460', 'NCIH2122', 'NCIH226', 'EKVX', 'SKMES1', 'NCIH522'], 'LARGE_INTESTINE': ['SW620', 'LOVO', 'RKO', 'KM12', 'HCT15', 'HCT116', 'HT29', 'DLD1', 'SW837'], 'OVARY': ['NIHOVCAR3', 'OVCAR4', 'OVCAR5', 'IGROV1', 'OVCAR8', 'PA1', 'UWB1289', 'SKOV3', 'CAOV3', 'ES2', 'A2780', 'OV90'], 'SKIN': ['SKMEL5', 'HT144', 'RVH421', 'SKMEL28', 'A375', 'A101D', 'COLO792', 'UACC257', 'COLO800', 'MEWO', 'MELHO', 'A2058', 'RPMI7951', 'IPC298', 'MALME3M', 'SKMEL2', 'UACC62', 'LOXIMVI', 'WM115', 'G361', 'SKMEL30', 'COLO829'], 'BONE': ['A673'], 'KIDNEY': ['CAKI1', 'UO31', 'ACHN', 'A498', '786O'], 'PLEURA': ['MSTO211H']}
breast_check = ccle_info[ccle_info.stripped_cell_line_name.isin(avail_cell_dict['BREAST'])][['cell_line_name','lineage_molecular_subtype']]
breast_check.columns = ['DC_cellname','subclass']


test_cell_df2 = pd.merge(test_cell_df, breast_check, on = 'DC_cellname', how = 'left')
test_cell_df2['subclass'] = test_cell_df2.subclass.apply(lambda x : "NA" if type(x) != str else x)

test_cell_df2.at[test_cell_df2[test_cell_df2.DC_cellname=='ZR751'].index.item(),'subclass'] = 'luminal'
test_cell_df2.at[test_cell_df2[test_cell_df2.DC_cellname=='HS 578T'].index.item(),'subclass'] = 'basal_B'
test_cell_df2.at[test_cell_df2[test_cell_df2.DC_cellname=='KPL1'].index.item(),'subclass'] = 'luminal'
test_cell_df2.at[test_cell_df2[test_cell_df2.DC_cellname=='MDAMB436'].index.item(),'subclass'] = 'basal_B'

# test_cell_df2.subclass.factorize()
sns.swarmplot(ax = ax, data  = test_cell_df2, x = 'tissue', y = 'P_COR', order=my_order, 
hue='subclass', linewidth=0.1, edgecolor="white", palette=sns.color_palette(['black', 'black', 'black', 'black','black']))


ax.set_xlabel('tissue names', fontsize=15)
ax.set_ylabel('Pearson Corr', fontsize=15)
ax.set_xticks(ax.get_xticks())
ax.set_yticks(ax.get_yticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, fontsize=25)

ytick_names = np.array(['', 0.0 , 0.2, 0.4, 0.6, 0.8, 1.0, ''])
ax.set_yticklabels(ytick_names,  fontsize=25)
ax.tick_params(axis='both', which='major', labelsize=20 )
ax.get_legend().remove()
plt.tight_layout()
plt.grid(True, linewidth = 0.2, linestyle = '--')
plt.savefig(os.path.join(PRJ_PATH,'tissue_pearson3.{}.png'.format(W_NAME)), dpi = 300)
plt.savefig('{}/{}.pdf'.format(PRJ_PATH, 'tissue_pearson3.{}.pdf'.format(W_NAME)), format="pdf", bbox_inches = 'tight')

plt.close()





















# breast 따로 그리기 1 


fig, ax = plt.subplots(figsize=(6, 13))

test_cell_df_mini = test_cell_df[test_cell_df.tissue=='BREAST']
test_cell_df_mini = pd.merge(test_cell_df_mini, DC_CELL_DF2[['DC_cellname','DrugCombCCLE']], on = 'DC_cellname', how='left')


sns.violinplot(ax = ax, data  = test_cell_df_mini, x = 'tissue', y = 'P_COR', linewidth=1,  edgecolor="dimgrey") # width = 3,,  inner = 'point'
violins = [c for i,c in enumerate(ax.collections) if i%2 == 0] # pollycollection 가져오기 위함 
violins[0].set_facecolor(mcolors.to_rgba(color_dict['BREAST'], 0.7))


MY_B = ['BT474_BREAST' ,'MDAMB231_BREAST','MDAMB175VII_BREAST','MDAMB468_BREAST','HCC1419_BREAST','MDAMB361_BREAST','HCC1500_BREAST','T47D_BREAST','MCF7_BREAST','KPL1_BREAST','CAMA1_BREAST','MDAMB436_BREAST','ZR751_BREAST','BT549_BREAST','HS578T_BREAST','UACC812_BREAST']

breast_check = ccle_info[ccle_info.CCLE_Name.isin(MY_B)][['CCLE_Name','sample_collection_site','primary_or_metastasis','lineage_subtype','lineage_sub_subtype','lineage_molecular_subtype']]

test_cell_df_mini2 = pd.merge(test_cell_df_mini, breast_check, left_on = 'DrugCombCCLE', right_on = 'CCLE_Name', how = 'left')

# test_cell_df2.subclass.factorize()
sns.swarmplot(ax = ax, data  = test_cell_df_mini2, size = 10,
x = 'tissue', y = 'P_COR', hue='lineage_molecular_subtype', palette=sns.color_palette(['yellow', 'pink', 'lawngreen','cyan']))


ax.set_xlabel('tissue names', fontsize=15)
ax.set_ylabel('Pearson Corr', fontsize=15)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, fontsize=25)
ax.set_yticklabels(ax.get_yticklabels(),  fontsize=25)
ax.tick_params(axis='both', which='major', labelsize=20 )
plt.tight_layout()
plt.grid(True)
plt.savefig(os.path.join(PRJ_PATH,'tissue_pearson.BREAST1.{}.png'.format(W_NAME)), dpi = 300)

plt.close()


# breast 2 

fig, ax = plt.subplots(figsize=(6, 13))

test_cell_df_mini = test_cell_df[test_cell_df.tissue=='BREAST']
test_cell_df_mini = pd.merge(test_cell_df_mini, DC_CELL_DF2[['DC_cellname','DrugCombCCLE']], on = 'DC_cellname', how='left')

sns.violinplot(ax = ax, data  = test_cell_df_mini, x = 'tissue', y = 'P_COR', linewidth=1,  edgecolor="dimgrey") # width = 3,,  inner = 'point'
violins = [c for i,c in enumerate(ax.collections) if i%2 == 0] # pollycollection 가져오기 위함 
violins[0].set_facecolor(mcolors.to_rgba(color_dict['BREAST'], 0.7))

MY_B = ['BT474_BREAST' ,'MDAMB231_BREAST','MDAMB175VII_BREAST','MDAMB468_BREAST','HCC1419_BREAST','MDAMB361_BREAST','HCC1500_BREAST','T47D_BREAST','MCF7_BREAST','KPL1_BREAST','CAMA1_BREAST','MDAMB436_BREAST','ZR751_BREAST','BT549_BREAST','HS578T_BREAST','UACC812_BREAST']

breast_check = ccle_info[ccle_info.CCLE_Name.isin(MY_B)][['CCLE_Name','sample_collection_site','primary_or_metastasis','lineage_subtype','lineage_sub_subtype','lineage_molecular_subtype']]

test_cell_df_mini2 = pd.merge(test_cell_df_mini, breast_check, left_on = 'DrugCombCCLE', right_on = 'CCLE_Name', how = 'left')

# test_cell_df2.subclass.factorize()
sns.swarmplot(ax = ax, data  = test_cell_df_mini2, size = 10,
x = 'tissue', y = 'P_COR', hue='lineage_sub_subtype', palette=sns.color_palette(['yellow', 'pink','cyan']))


ax.set_xlabel('tissue names', fontsize=15)
ax.set_ylabel('Pearson Corr', fontsize=15)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, fontsize=25)
ax.set_yticklabels(ax.get_yticklabels(),  fontsize=25)
ax.tick_params(axis='both', which='major', labelsize=20 )
plt.tight_layout()
plt.grid(True)
plt.savefig(os.path.join(PRJ_PATH,'tissue_pearson.BREAST2.{}.png'.format(W_NAME)), dpi = 300)

plt.close()




# breast 3

fig, ax = plt.subplots(figsize=(6, 13))

test_cell_df_mini = test_cell_df[test_cell_df.tissue=='BREAST']
test_cell_df_mini = pd.merge(test_cell_df_mini, DC_CELL_DF2[['DC_cellname','DrugCombCCLE']], on = 'DC_cellname', how='left')

sns.violinplot(ax = ax, data  = test_cell_df_mini, x = 'tissue', y = 'P_COR', linewidth=1,  edgecolor="dimgrey") # width = 3,,  inner = 'point'
violins = [c for i,c in enumerate(ax.collections) if i%2 == 0] # pollycollection 가져오기 위함 
violins[0].set_facecolor(mcolors.to_rgba(color_dict['BREAST'], 0.7))

MY_B = ['BT474_BREAST' ,'MDAMB231_BREAST','MDAMB175VII_BREAST','MDAMB468_BREAST','HCC1419_BREAST','MDAMB361_BREAST','HCC1500_BREAST','T47D_BREAST','MCF7_BREAST','KPL1_BREAST','CAMA1_BREAST','MDAMB436_BREAST','ZR751_BREAST','BT549_BREAST','HS578T_BREAST','UACC812_BREAST']

breast_check = ccle_info[ccle_info.CCLE_Name.isin(MY_B)][['CCLE_Name','sample_collection_site','primary_or_metastasis','lineage_subtype','lineage_sub_subtype','lineage_molecular_subtype']]

test_cell_df_mini2 = pd.merge(test_cell_df_mini, breast_check, left_on = 'DrugCombCCLE', right_on = 'CCLE_Name', how = 'left')

# test_cell_df2.subclass.factorize()
sns.swarmplot(ax = ax, data  = test_cell_df_mini2, size = 10,
x = 'tissue', y = 'P_COR', hue='sample_collection_site', palette=sns.color_palette(['yellow', 'pink','cyan','lawngreen']))


ax.set_xlabel('tissue names', fontsize=15)
ax.set_ylabel('Pearson Corr', fontsize=15)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, fontsize=25)
ax.set_yticklabels(ax.get_yticklabels(),  fontsize=25)
ax.tick_params(axis='both', which='major', labelsize=20 )
plt.tight_layout()
plt.grid(True)
plt.savefig(os.path.join(PRJ_PATH,'tissue_pearson.BREAST3.{}.png'.format(W_NAME)), dpi = 300)

plt.close()




# breast 4

fig, ax = plt.subplots(figsize=(6, 13))

test_cell_df_mini = test_cell_df[test_cell_df.tissue=='BREAST']
test_cell_df_mini = pd.merge(test_cell_df_mini, DC_CELL_DF2[['DC_cellname','DrugCombCCLE']], on = 'DC_cellname', how='left')

sns.violinplot(ax = ax, data  = test_cell_df_mini, x = 'tissue', y = 'P_COR', linewidth=1,  edgecolor="dimgrey") # width = 3,,  inner = 'point'
violins = [c for i,c in enumerate(ax.collections) if i%2 == 0] # pollycollection 가져오기 위함 
violins[0].set_facecolor(mcolors.to_rgba(color_dict['BREAST'], 0.7))

MY_B = ['BT474_BREAST' ,'MDAMB231_BREAST','MDAMB175VII_BREAST','MDAMB468_BREAST','HCC1419_BREAST','MDAMB361_BREAST','HCC1500_BREAST','T47D_BREAST','MCF7_BREAST','KPL1_BREAST','CAMA1_BREAST','MDAMB436_BREAST','ZR751_BREAST','BT549_BREAST','HS578T_BREAST','UACC812_BREAST']

breast_check = ccle_info[ccle_info.CCLE_Name.isin(MY_B)][['CCLE_Name','sample_collection_site','primary_or_metastasis','lineage_subtype','lineage_sub_subtype','lineage_molecular_subtype']]

test_cell_df_mini2 = pd.merge(test_cell_df_mini, breast_check, left_on = 'DrugCombCCLE', right_on = 'CCLE_Name', how = 'left')

# test_cell_df2.subclass.factorize()
sns.swarmplot(ax = ax, data  = test_cell_df_mini2, size = 10,
x = 'tissue', y = 'P_COR', hue='lineage_subtype', palette=sns.color_palette(['yellow', 'pink']))


ax.set_xlabel('tissue names', fontsize=15)
ax.set_ylabel('Pearson Corr', fontsize=15)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, fontsize=25)
ax.set_yticklabels(ax.get_yticklabels(),  fontsize=25)
ax.tick_params(axis='both', which='major', labelsize=20 )
plt.tight_layout()
plt.grid(True)
plt.savefig(os.path.join(PRJ_PATH,'tissue_pearson.BREAST4.{}.png'.format(W_NAME)), dpi = 300)

plt.close()










# breast 따로 그리기 5
# 세가지로만 만들어서 보여주기 


fig, ax = plt.subplots(figsize=(6, 13))

test_cell_df_mini = test_cell_df[test_cell_df.tissue=='BREAST']
test_cell_df_mini = pd.merge(test_cell_df_mini, DC_CELL_DF2[['DC_cellname','DrugCombCCLE']], on = 'DC_cellname', how='left')


sns.violinplot(ax = ax, data  = test_cell_df_mini, x = 'tissue', y = 'P_COR', linewidth=2,  edgecolor="black") # width = 3,,  inner = 'point'
violins = [c for i,c in enumerate(ax.collections) if i%2 == 0] # pollycollection 가져오기 위함 
violins[0].set_facecolor(mcolors.to_rgba(color_dict['BREAST'], 0.8))


MY_B = ['BT474_BREAST' ,'MDAMB231_BREAST','MDAMB175VII_BREAST','MDAMB468_BREAST','HCC1419_BREAST','MDAMB361_BREAST','HCC1500_BREAST','T47D_BREAST','MCF7_BREAST','KPL1_BREAST','CAMA1_BREAST','MDAMB436_BREAST','ZR751_BREAST','BT549_BREAST','HS578T_BREAST','UACC812_BREAST']

breast_check = ccle_info[ccle_info.CCLE_Name.isin(MY_B)][['CCLE_Name','sample_collection_site','primary_or_metastasis','lineage_subtype','lineage_sub_subtype','lineage_molecular_subtype']]
breast_check['Subtypes'] = breast_check.lineage_molecular_subtype.apply(lambda x : 'Basal' if 'basal' in x else x)
breast_check['Subtypes'] = breast_check.Subtypes.apply(lambda x : 'Luminal' if 'luminal' in x else x)

test_cell_df_mini2 = pd.merge(test_cell_df_mini, breast_check, left_on = 'DrugCombCCLE', right_on = 'CCLE_Name', how = 'left')

# test_cell_df2.subclass.factorize()
sns.swarmplot(ax = ax, data  = test_cell_df_mini2, linewidth=1, edgecolor="white", size = 15,
x = 'tissue', y = 'P_COR', hue='Subtypes', 
palette=sns.color_palette(['yellow', 'pink','cyan']))


ax.set_xlabel('', fontsize=15)
ax.set_ylabel('Avg Pearson Corr', fontsize=15)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels([''], rotation = 90, fontsize=25) # ax.get_xticklabels()

ytick_names = np.array(['', 0.0 , 0.2, 0.4, 0.6, 0.8, 1.0, ''])
ax.set_yticklabels(ytick_names,  fontsize=25)
ax.tick_params(axis='both', which='major', labelsize=20 )
plt.grid(True)
plt.legend(bbox_to_anchor=(1.1, 1.1), loc="upper right", fontsize=15)
plt.tight_layout()

plt.savefig(os.path.join(PRJ_PATH,'tissue_pearson.BREAST5.{}.png'.format(W_NAME)), dpi = 300)

plt.close()






































# CENTRAL_NERVOUS_SYSTEM 1

NN_filt = ccle_info[ccle_info.CCLE_Name.isin(N_list)][['CCLE_Name','Subtype','lineage_subtype','lineage_sub_subtype','lineage_molecular_subtype']]


fig, ax = plt.subplots(figsize=(6, 13))

test_cell_df_mini = test_cell_df[test_cell_df.tissue=='CENTRAL_NERVOUS_SYSTEM']
test_cell_df_mini = pd.merge(test_cell_df_mini, DC_CELL_DF2[['DC_cellname','DrugCombCCLE']], on = 'DC_cellname', how='left')

sns.violinplot(ax = ax, data  = test_cell_df_mini, x = 'tissue', y = 'P_COR', linewidth=1,  edgecolor="dimgrey") # width = 3,,  inner = 'point'
violins = [c for i,c in enumerate(ax.collections) if i%2 == 0] # pollycollection 가져오기 위함 
violins[0].set_facecolor(mcolors.to_rgba(color_dict['CENTRAL_NERVOUS_SYSTEM'], 0.7))

MY_N = ['SF295_CENTRAL_NERVOUS_SYSTEM','U251MG_CENTRAL_NERVOUS_SYSTEM','T98G_CENTRAL_NERVOUS_SYSTEM','SF268_CENTRAL_NERVOUS_SYSTEM','SNB75_CENTRAL_NERVOUS_SYSTEM','SF539_CENTRAL_NERVOUS_SYSTEM']

nn_check = ccle_info[ccle_info.CCLE_Name.isin(MY_N)][['CCLE_Name','Subtype','lineage_subtype','lineage_sub_subtype','lineage_molecular_subtype']]

test_cell_df_mini2 = pd.merge(test_cell_df_mini, nn_check, left_on = 'DrugCombCCLE', right_on = 'CCLE_Name', how = 'left')

# test_cell_df2.subclass.factorize()
sns.swarmplot(ax = ax, data  = test_cell_df_mini2, size = 10,
x = 'tissue', y = 'P_COR', hue='Subtype', palette=sns.color_palette(['yellow', 'lawngreen','cyan']))


ax.set_xlabel('tissue names', fontsize=15)
ax.set_ylabel('Pearson Corr', fontsize=15)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, fontsize=25)
ax.set_yticklabels(ax.get_yticklabels(),  fontsize=25)
ax.tick_params(axis='both', which='major', labelsize=20 )
plt.tight_layout()
plt.grid(True)
plt.savefig(os.path.join(PRJ_PATH,'tissue_pearson.NN1.{}.png'.format(W_NAME)), dpi = 300)

plt.close()






# CENTRAL_NERVOUS_SYSTEM 2

NN_filt = ccle_info[ccle_info.CCLE_Name.isin(N_list)][['CCLE_Name','Subtype','lineage_subtype','lineage_sub_subtype','lineage_molecular_subtype']]


fig, ax = plt.subplots(figsize=(6, 13))

test_cell_df_mini = test_cell_df[test_cell_df.tissue=='CENTRAL_NERVOUS_SYSTEM']
test_cell_df_mini = pd.merge(test_cell_df_mini, DC_CELL_DF2[['DC_cellname','DrugCombCCLE']], on = 'DC_cellname', how='left')

sns.violinplot(ax = ax, data  = test_cell_df_mini, x = 'tissue', y = 'P_COR', linewidth=1,  edgecolor="dimgrey") # width = 3,,  inner = 'point'
violins = [c for i,c in enumerate(ax.collections) if i%2 == 0] # pollycollection 가져오기 위함 
violins[0].set_facecolor(mcolors.to_rgba(color_dict['CENTRAL_NERVOUS_SYSTEM'], 0.7))

MY_N = ['SF295_CENTRAL_NERVOUS_SYSTEM','U251MG_CENTRAL_NERVOUS_SYSTEM','T98G_CENTRAL_NERVOUS_SYSTEM','SF268_CENTRAL_NERVOUS_SYSTEM','SNB75_CENTRAL_NERVOUS_SYSTEM','SF539_CENTRAL_NERVOUS_SYSTEM']

nn_check = ccle_info[ccle_info.CCLE_Name.isin(MY_N)][['CCLE_Name','Subtype','lineage_subtype','lineage_sub_subtype','lineage_molecular_subtype','culture_type']]

test_cell_df_mini2 = pd.merge(test_cell_df_mini, nn_check, left_on = 'DrugCombCCLE', right_on = 'CCLE_Name', how = 'left')
test_cell_df_mini2['culture_type'] = test_cell_df_mini2.culture_type.apply(lambda x : 'NA' if type(x)!= str else x)

# test_cell_df2.subclass.factorize()
sns.swarmplot(ax = ax, data  = test_cell_df_mini2, size = 10,
x = 'tissue', y = 'P_COR', hue='culture_type', palette=sns.color_palette(['yellow', 'lawngreen','cyan']))


ax.set_xlabel('tissue names', fontsize=15)
ax.set_ylabel('Pearson Corr', fontsize=15)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, fontsize=25)
ax.set_yticklabels(ax.get_yticklabels(),  fontsize=25)
ax.tick_params(axis='both', which='major', labelsize=20 )
plt.tight_layout()
plt.grid(True)
plt.savefig(os.path.join(PRJ_PATH,'tissue_pearson.NN2.{}.png'.format(W_NAME)), dpi = 300)

plt.close()





# CENTRAL_NERVOUS_SYSTEM 3

NN_filt = ccle_info[ccle_info.CCLE_Name.isin(N_list)][['CCLE_Name','Subtype','lineage_subtype','lineage_sub_subtype','lineage_molecular_subtype']]


fig, ax = plt.subplots(figsize=(6, 13))

test_cell_df_mini = test_cell_df[test_cell_df.tissue=='CENTRAL_NERVOUS_SYSTEM']
test_cell_df_mini = pd.merge(test_cell_df_mini, DC_CELL_DF2[['DC_cellname','DrugCombCCLE']], on = 'DC_cellname', how='left')

sns.violinplot(ax = ax, data  = test_cell_df_mini, x = 'tissue', y = 'P_COR', linewidth=1,  edgecolor="dimgrey") # width = 3,,  inner = 'point'
violins = [c for i,c in enumerate(ax.collections) if i%2 == 0] # pollycollection 가져오기 위함 
violins[0].set_facecolor(mcolors.to_rgba(color_dict['CENTRAL_NERVOUS_SYSTEM'], 0.7))

MY_N = ['SF295_CENTRAL_NERVOUS_SYSTEM','U251MG_CENTRAL_NERVOUS_SYSTEM','T98G_CENTRAL_NERVOUS_SYSTEM','SF268_CENTRAL_NERVOUS_SYSTEM','SNB75_CENTRAL_NERVOUS_SYSTEM','SF539_CENTRAL_NERVOUS_SYSTEM']

nn_check = ccle_info[ccle_info.CCLE_Name.isin(MY_N)][['CCLE_Name','Subtype','lineage_subtype','lineage_sub_subtype','lineage_molecular_subtype','culture_type','source']]

test_cell_df_mini2 = pd.merge(test_cell_df_mini, nn_check, left_on = 'DrugCombCCLE', right_on = 'CCLE_Name', how = 'left')
test_cell_df_mini2['culture_type'] = test_cell_df_mini2.culture_type.apply(lambda x : 'NA' if type(x)!= str else x)

# test_cell_df2.subclass.factorize()
sns.swarmplot(ax = ax, data  = test_cell_df_mini2, size = 10,
x = 'tissue', y = 'P_COR', hue='source', palette=sns.color_palette(['yellow', 'lawngreen','cyan', 'black']))


ax.set_xlabel('tissue names', fontsize=15)
ax.set_ylabel('Pearson Corr', fontsize=15)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, fontsize=25)
ax.set_yticklabels(ax.get_yticklabels(),  fontsize=25)
ax.tick_params(axis='both', which='major', labelsize=20 )
plt.tight_layout()
plt.grid(True)
plt.savefig(os.path.join(PRJ_PATH,'tissue_pearson.NN3.{}.png'.format(W_NAME)), dpi = 300)

plt.close()





















# haemato 


fig, ax = plt.subplots(figsize=(6, 13))

test_cell_df_mini = test_cell_df[test_cell_df.tissue=='HAEMATOPOIETIC_AND_LYMPHOID_TISSUE']
test_cell_df_mini = pd.merge(test_cell_df_mini, DC_CELL_DF2[['DC_cellname','DrugCombCCLE']], on = 'DC_cellname', how='left')

sns.violinplot(ax = ax, data  = test_cell_df_mini, x = 'tissue', y = 'P_COR', linewidth=1,  edgecolor="dimgrey") # width = 3,,  inner = 'point'
violins = [c for i,c in enumerate(ax.collections) if i%2 == 0] # pollycollection 가져오기 위함 
violins[0].set_facecolor(mcolors.to_rgba(color_dict['HAEMATOPOIETIC_AND_LYMPHOID_TISSUE'], 0.7))

MY_M = ['UHO1_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE','RPMI8226_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE','K562_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE','L1236_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE','SR786_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE']

nn_check = ccle_info[ccle_info.CCLE_Name.isin(MY_M)][['CCLE_Name','sample_collection_site','primary_or_metastasis','lineage_subtype','lineage_sub_subtype']]

test_cell_df_mini2 = pd.merge(test_cell_df_mini, nn_check, left_on = 'DrugCombCCLE', right_on = 'CCLE_Name', how = 'left')
# test_cell_df_mini2['culture_type'] = test_cell_df_mini2.culture_type.apply(lambda x : 'NA' if type(x)!= str else x)

# test_cell_df2.subclass.factorize()
sns.swarmplot(ax = ax, data  = test_cell_df_mini2, size = 10,
x = 'tissue', y = 'P_COR', hue='sample_collection_site', palette=sns.color_palette(['yellow', 'pink']))


ax.set_xlabel('tissue names', fontsize=15)
ax.set_ylabel('Pearson Corr', fontsize=15)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, fontsize=25)
ax.set_yticklabels(ax.get_yticklabels(),  fontsize=25)
ax.tick_params(axis='both', which='major', labelsize=20 )
plt.tight_layout()
plt.grid(True)
plt.savefig(os.path.join(PRJ_PATH,'tissue_pearson.HM1.{}.png'.format(W_NAME)), dpi = 300)

plt.close()



22222 


fig, ax = plt.subplots(figsize=(6, 13))

test_cell_df_mini = test_cell_df[test_cell_df.tissue=='HAEMATOPOIETIC_AND_LYMPHOID_TISSUE']
test_cell_df_mini = pd.merge(test_cell_df_mini, DC_CELL_DF2[['DC_cellname','DrugCombCCLE']], on = 'DC_cellname', how='left')

sns.violinplot(ax = ax, data  = test_cell_df_mini, x = 'tissue', y = 'P_COR', linewidth=1,  edgecolor="dimgrey") # width = 3,,  inner = 'point'
violins = [c for i,c in enumerate(ax.collections) if i%2 == 0] # pollycollection 가져오기 위함 
violins[0].set_facecolor(mcolors.to_rgba(color_dict['HAEMATOPOIETIC_AND_LYMPHOID_TISSUE'], 0.7))

MY_M = ['UHO1_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE','RPMI8226_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE','K562_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE','L1236_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE','SR786_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE']

nn_check = ccle_info[ccle_info.CCLE_Name.isin(MY_M)][['CCLE_Name','sample_collection_site','primary_or_metastasis','lineage_subtype','lineage_sub_subtype']]

test_cell_df_mini2 = pd.merge(test_cell_df_mini, nn_check, left_on = 'DrugCombCCLE', right_on = 'CCLE_Name', how = 'left')
test_cell_df_mini2['primary_or_metastasis'] = test_cell_df_mini2.primary_or_metastasis.apply(lambda x : 'NA' if type(x)!= str else x)

# test_cell_df2.subclass.factorize()
sns.swarmplot(ax = ax, data  = test_cell_df_mini2, size = 10,
x = 'tissue', y = 'P_COR', hue='primary_or_metastasis', palette=sns.color_palette(['yellow', 'pink', 'black']))


ax.set_xlabel('tissue names', fontsize=15)
ax.set_ylabel('Pearson Corr', fontsize=15)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, fontsize=25)
ax.set_yticklabels(ax.get_yticklabels(),  fontsize=25)
ax.tick_params(axis='both', which='major', labelsize=20 )
plt.tight_layout()
plt.grid(True)
plt.savefig(os.path.join(PRJ_PATH,'tissue_pearson.HM2.{}.png'.format(W_NAME)), dpi = 300)

plt.close()







fig, ax = plt.subplots(figsize=(6, 13))

test_cell_df_mini = test_cell_df[test_cell_df.tissue=='HAEMATOPOIETIC_AND_LYMPHOID_TISSUE']
test_cell_df_mini = pd.merge(test_cell_df_mini, DC_CELL_DF2[['DC_cellname','DrugCombCCLE']], on = 'DC_cellname', how='left')

sns.violinplot(ax = ax, data  = test_cell_df_mini, x = 'tissue', y = 'P_COR', linewidth=1,  edgecolor="dimgrey") # width = 3,,  inner = 'point'
violins = [c for i,c in enumerate(ax.collections) if i%2 == 0] # pollycollection 가져오기 위함 
violins[0].set_facecolor(mcolors.to_rgba(color_dict['HAEMATOPOIETIC_AND_LYMPHOID_TISSUE'], 0.7))

MY_M = ['UHO1_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE','RPMI8226_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE','K562_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE','L1236_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE','SR786_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE']

nn_check = ccle_info[ccle_info.CCLE_Name.isin(MY_M)][['CCLE_Name','sample_collection_site','primary_or_metastasis','lineage_subtype','lineage_sub_subtype','source', 'culture_type']]

test_cell_df_mini2 = pd.merge(test_cell_df_mini, nn_check, left_on = 'DrugCombCCLE', right_on = 'CCLE_Name', how = 'left')
# test_cell_df_mini2['primary_or_metastasis'] = test_cell_df_mini2.primary_or_metastasis.apply(lambda x : 'NA' if type(x)!= str else x)

# test_cell_df2.subclass.factorize()
sns.swarmplot(ax = ax, data  = test_cell_df_mini2, size = 10,
x = 'tissue', y = 'P_COR', hue='lineage_subtype', palette=sns.color_palette(['yellow', 'pink', 'black', 'red']))


ax.set_xlabel('tissue names', fontsize=15)
ax.set_ylabel('Pearson Corr', fontsize=15)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, fontsize=25)
ax.set_yticklabels(ax.get_yticklabels(),  fontsize=25)
ax.tick_params(axis='both', which='major', labelsize=20 )
plt.tight_layout()
plt.grid(True)
plt.savefig(os.path.join(PRJ_PATH,'tissue_pearson.HM3.{}.png'.format(W_NAME)), dpi = 300)

plt.close()








# haemato 다시!!!

fig, ax = plt.subplots(figsize=(6, 13))

test_cell_df_mini = test_cell_df[test_cell_df.tissue=='HAEMATOPOIETIC_AND_LYMPHOID_TISSUE']
test_cell_df_mini = pd.merge(test_cell_df_mini, DC_CELL_DF2[['DC_cellname','DrugCombCCLE']], on = 'DC_cellname', how='left')

sns.violinplot(ax = ax, data  = test_cell_df_mini, x = 'tissue', y = 'P_COR', linewidth=2,  edgecolor="black") # width = 3,,  inner = 'point'
violins = [c for i,c in enumerate(ax.collections) if i%2 == 0] # pollycollection 가져오기 위함 
violins[0].set_facecolor(mcolors.to_rgba(color_dict['HAEMATOPOIETIC_AND_LYMPHOID_TISSUE'], 0.8))

MY_M = ['UHO1_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE','RPMI8226_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE','K562_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE','L1236_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE','SR786_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE']

nn_check = ccle_info[ccle_info.CCLE_Name.isin(MY_M)][['CCLE_Name','sample_collection_site','primary_or_metastasis','lineage_subtype','lineage_sub_subtype']]
nn_check['Subtype'] = ['Non Hodgkin Lymphoma', 'Chronic Myelogenous Leukemia', 'Hodgkin Lymphoma', 'Multiple Myeloma', 'Hodgkin Lymphoma']

test_cell_df_mini2 = pd.merge(test_cell_df_mini, nn_check, left_on = 'DrugCombCCLE', right_on = 'CCLE_Name', how = 'left')
# test_cell_df_mini2['culture_type'] = test_cell_df_mini2.culture_type.apply(lambda x : 'NA' if type(x)!= str else x)

# test_cell_df2.subclass.factorize()
sns.swarmplot(ax = ax, data  = test_cell_df_mini2, linewidth=1, edgecolor="white", size = 15,
x = 'tissue', y = 'P_COR', hue='Subtype', palette=sns.color_palette(['purple', 'red', 'darkgreen', 'yellow']))


ax.set_xlabel('tissue names', fontsize=15)
ax.set_ylabel('Pearson Corr', fontsize=15)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels([''], rotation = 90, fontsize=25) # ax.get_xticklabels()

ytick_names = np.array(['', 0.0 , 0.2, 0.4, 0.6, 0.8, 1.0, ''])
ax.set_yticklabels(ytick_names,  fontsize=25)
ax.tick_params(axis='both', which='major', labelsize=20 )
plt.grid(True)
plt.legend(bbox_to_anchor=(1.2, 1.2), loc="upper right", fontsize=15)
plt.tight_layout()

plt.savefig(os.path.join(PRJ_PATH,'tissue_pearson.HM4.{}.png'.format(W_NAME)), dpi = 300)

plt.close()






고통스러우므로 그냥 subplot 으로 해결해보자 


fig, ax = plt.subplots(1,2 ,figsize=(12, 4))
ax1 = ax[0]
ax2 = ax[1]

test_cell_df_mini = test_cell_df[test_cell_df.tissue=='BREAST']
test_cell_df_mini = pd.merge(test_cell_df_mini, DC_CELL_DF2[['DC_cellname','DrugCombCCLE']], on = 'DC_cellname', how='left')
sns.violinplot(ax = ax1, data  = test_cell_df_mini, x = 'tissue', y = 'P_COR', linewidth=2,  edgecolor="black") # width = 3,,  inner = 'point'
violins = [c for i,c in enumerate(ax1.collections) if i%2 == 0] # pollycollection 가져오기 위함 
violins[0].set_facecolor(mcolors.to_rgba(color_dict['BREAST'], 0.8))

MY_B = ['BT474_BREAST' ,'MDAMB231_BREAST','MDAMB175VII_BREAST','MDAMB468_BREAST','HCC1419_BREAST','MDAMB361_BREAST','HCC1500_BREAST','T47D_BREAST','MCF7_BREAST','KPL1_BREAST','CAMA1_BREAST','MDAMB436_BREAST','ZR751_BREAST','BT549_BREAST','HS578T_BREAST','UACC812_BREAST']

breast_check = ccle_info[ccle_info.CCLE_Name.isin(MY_B)][['CCLE_Name','sample_collection_site','primary_or_metastasis','lineage_subtype','lineage_sub_subtype','lineage_molecular_subtype']]
breast_check['Subtypes'] = breast_check.lineage_molecular_subtype.apply(lambda x : 'Basal' if 'basal' in x else x)
breast_check['Subtypes'] = breast_check.Subtypes.apply(lambda x : 'Luminal' if 'luminal' in x else x)

test_cell_df_mini2 = pd.merge(test_cell_df_mini, breast_check, left_on = 'DrugCombCCLE', right_on = 'CCLE_Name', how = 'left')

sns.swarmplot(ax = ax1, data  = test_cell_df_mini2, linewidth=1, edgecolor="white", size = 15,
x = 'tissue', y = 'P_COR', hue='Subtypes', 
palette=sns.color_palette(['yellow', 'pink','cyan']))

ax1.set_xlabel('', fontsize=15)
ax1.set_ylabel('Avg Pearson Corr', fontsize=15)
ax1.set_xticks(ax1.get_xticks())
ax1.set_xticklabels([''], rotation = 90, fontsize=25) # ax.get_xticklabels()

ytick_names = np.array(['', 0.0 , 0.2, 0.4, 0.6, 0.8, 1.0, ''])
ax1.set_yticklabels(ytick_names,  fontsize=25)
ax1.tick_params(axis='both', which='major', labelsize=20 )
ax1.legend(bbox_to_anchor=(1, 1.2), loc="upper right", fontsize = 15)



test_cell_df_mini = test_cell_df[test_cell_df.tissue=='HAEMATOPOIETIC_AND_LYMPHOID_TISSUE']
test_cell_df_mini = pd.merge(test_cell_df_mini, DC_CELL_DF2[['DC_cellname','DrugCombCCLE']], on = 'DC_cellname', how='left')

sns.violinplot(ax = ax2, data  = test_cell_df_mini, x = 'tissue', y = 'P_COR', linewidth=2,  edgecolor="black") # width = 3,,  inner = 'point'
violins = [c for i,c in enumerate(ax2.collections) if i%2 == 0] # pollycollection 가져오기 위함 
violins[0].set_facecolor(mcolors.to_rgba(color_dict['HAEMATOPOIETIC_AND_LYMPHOID_TISSUE'], 0.8))

MY_M = ['UHO1_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE','RPMI8226_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE','K562_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE','L1236_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE','SR786_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE']

nn_check = ccle_info[ccle_info.CCLE_Name.isin(MY_M)][['CCLE_Name','sample_collection_site','primary_or_metastasis','lineage_subtype','lineage_sub_subtype']]
nn_check['Subtype'] = ['Non Hodgkin Lymphoma', 'Chronic Myelogenous Leukemia', 'Hodgkin Lymphoma', 'Multiple Myeloma', 'Hodgkin Lymphoma']

test_cell_df_mini2 = pd.merge(test_cell_df_mini, nn_check, left_on = 'DrugCombCCLE', right_on = 'CCLE_Name', how = 'left')

sns.swarmplot(ax = ax2, data  = test_cell_df_mini2, linewidth=1, edgecolor="white", size = 15,
x = 'tissue', y = 'P_COR', hue='Subtype', palette=sns.color_palette(['purple', 'red', 'darkgreen', 'yellow']))

ax2.set_xlabel('', fontsize=15)
ax2.set_ylabel('Avg Pearson Corr', fontsize=15)
ax2.set_xticks(ax2.get_xticks())
ax2.set_xticklabels([''], rotation = 90, fontsize=25) # ax.get_xticklabels()

ytick_names = np.array(['', 0.0 , 0.2, 0.4, 0.6, 0.8, 1.0, ''])
ax2.set_yticklabels(ytick_names,  fontsize=25)
ax2.tick_params(axis='both', which='major', labelsize=20 )
ax2.legend(bbox_to_anchor=(1, 1.2), loc="upper right", fontsize = 15)



plt.tight_layout()

plt.savefig(os.path.join(PRJ_PATH,'tissue_pearson.2tissue.{}.png'.format(W_NAME)), dpi = 300)
plt.savefig(os.path.join(PRJ_PATH,'tissue_pearson.2tissue.{}.pdf'.format(W_NAME)), format="pdf", bbox_inches = 'tight')

plt.close()




















"hue="label", palette={
    'Label 1': '#d7191c',
    'Label 2': '#2b83ba'
}

ABCS_test_result['tissue'] = [ '_'.join(a.split('_')[1:]) if type(a) == str else 'NA' for a in list(ABCS_test_result['CELL'])]
ABCS_test_result['color'] = [color_dict[a] for a in ABCS_test_result.tissue]

jplot = sns.jointplot(data = ABCS_test_result , x='PRED_3', y='ANS', ci=68, kind='reg', scatter = False)
pr,pp = stats.pearsonr(ABCS_test_result['PRED_3'], ABCS_test_result['ANS'])
sr,sp = stats.spearmanr(ABCS_test_result['PRED_3'], ABCS_test_result['ANS'])
jplot.ax_joint.annotate(f'$pearson = {pr:.3f}, spearman = {sr:.3f}$',xy=(min(ABCS_test_result.PRED_3)+ 0.01, max(ABCS_test_result.ANS)- 0.01 ), ha='left', va='center',)
#jplot.ax_joint.scatter(data = ABCS_test_result , x='PRED_3', y='ANS', c =  )
sns.scatterplot(data = ABCS_test_result , x='PRED_3', y='ANS', hue='tissue', palette=color_dict)
jplot.set_axis_labels(xlabel='Predicted', ylabel='Answer', size=15)
jplot.figure.savefig('{}/{}.png'.format(PRJ_PATH, 'total_scatter'), bbox_inches = 'tight')
jplot.figure.savefig('{}/{}.pdf'.format(PRJ_PATH, 'total_scatter'), format="pdf", bbox_inches = 'tight')


g = sns.jointplot(data = ABCS_test_result , x='PRED_3', y='ANS', ci=68, kind='reg', color='black', scatter = False )
for i, subdata in ABCS_test_result.groupby("tissue"):
    sns.kdeplot(subdata.loc[:,'PRED_3'], ax=g.ax_marg_x, legend=False)
    sns.kdeplot(subdata.loc[:,'ANS'], ax=g.ax_marg_y, vertical=True, legend=False)
    g.ax_joint.plot(subdata.loc[:,'PRED_3'], subdata.loc[:,'ANS'], "o", ms = 1, alpha=.5, mfc=list(subdata.color)[0] )

pr,pp = stats.pearsonr(ABCS_test_result['PRED_3'], ABCS_test_result['ANS'])
sr,sp = stats.spearmanr(ABCS_test_result['PRED_3'], ABCS_test_result['ANS'])
g.ax_joint.annotate(f'$pearson = {pr:.3f}, spearman = {sr:.3f}$',xy=(min(ABCS_test_result.PRED_3)+ 0.01, max(ABCS_test_result.ANS)- 0.01 ), ha='left', va='center',)
g.set_axis_labels(xlabel='Predicted', ylabel='Answer', size=15)
g.figure.savefig('{}/{}.png'.format(PRJ_PATH, 'total_scatter'), bbox_inches = 'tight')
g.figure.savefig('{}/{}.pdf'.format(PRJ_PATH, 'total_scatter'), format="pdf", bbox_inches = 'tight')



tissue_set = ['CENTRAL_NERVOUS_SYSTEM', 'LUNG', 'BREAST', 'BONE', 'OVARY', 'PROSTATE', 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'LARGE_INTESTINE', 'SKIN', 'PLEURA', 'KIDNEY' ] # list(set(test_cell_df['tissue']))
color_set = ['#CF3476','#FF7514','#025669','#308446','#84C3BE','#D53032','#4ddcfd','#ffcd36','#ac8cff',"#0000ffff","#7bff68ff"] # "#1E1E1E"
color_dict = {a : color_set[tissue_set.index(a)] for a in tissue_set}















def jy_corrplot(PRED_list, Y_list, path, plotname ):

PRED_list = list(ABCS_test_result.PRED_3)
Y_list = list(ABCS_test_result.ANS)

jplot = sns.jointplot(x=PRED_list, y=Y_list, ci=68, kind='reg', color='black', scatter = False  )
pr,pp = stats.pearsonr(PRED_list, Y_list)
print("Pearson correlation is {} and related p_value is {}".format(pr, pp), flush=True)
sr,sp = stats.spearmanr(PRED_list, Y_list)
print("Spearman correlation is {} and related p_value is {}".format(sr, sp), flush=True)
jplot.ax_joint.annotate(f'$pearson = {pr:.3f}, spearman = {sr:.3f}$',xy=(min(PRED_list)+ 0.01, max(Y_list)- 0.01 ), ha='left', va='center',)
jplot.ax_joint.scatter(PRED_list, Y_list, s = 1, alpha=.8)
jplot.set_axis_labels(xlabel='Predicted', ylabel='Answer', size=15)
jplot.figure.savefig('/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V6_W203_349_MIS2/new.corrplot.png', bbox_inches = 'tight')
jplot.figure.savefig('/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V6_W203_349_MIS2/new.corrplot.pdf', format="pdf", bbox_inches = 'tight')

plt.close()













# pred * ans violin 확인 
##### 

order = ['A-673', 'MDA-MB-361', 'CAMA-1', 'MDA-MB-175-VII', 'L-1236', 'UACC-812', 'HCC1500', 'HCC1419', 'U-HO1', 'T98G', 'BT-474', 'LOX IMVI', 'ZR751', 'NCIH2122', 'MDAMB436', 'COLO 829', 'KPL1', 'EKVX', 'A498', 'MDA-MB-231', 'HT144', 'T-47D', 'A375', 'RPMI7951', 'A2058', 'G-361', 'KM12', 'MCF7', 'CAOV3', 'COLO 800', 'Mel Ho', 'NCI-H226', 'UACC-257', 'OVCAR3', 'SF-539', 'PC-3', 'CAKI-1', 'UACC62', 'SR', 'UO-31', 'NCI-H522', 'SK-MEL-5', 'HOP-92', 'ACHN', '786O', 'HS 578T', 'RVH-421', 'K-562', 'HOP-62', 'OVCAR-5', 'MSTO', 'U251', 'COLO 792', 'NCIH23', 'OV90', 'LOVO', 'RPMI-8226', 'A101D', 'NCIH520', 'SK-MEL-2', 'MALME-3M', 'IPC-298', 'HT29', 'SF-295', 'UWB1289', 'SW-620', 'OVCAR-4', 'SKMES1', 'SW837', 'MDA-MB-468', 'HCT-15', 'OVCAR-8', 'SNB-75', 'SF-268', 'HCT116', 'BT-549', 'A549', 'IGROV1', 'A427', 'SK-MEL-28', 'SK-OV-3', 'NCIH1650', 'WM115', 'NCI-H460', 'MeWo', 'ES2', 'SKMEL30', 'A2780', 'RKO', 'VCAP', 'PA1', 'DLD1']


ABCS_test_result_filt1 = ABCS_test_result[['DC_cellname','CELL','tissue','ANS']]
ABCS_test_result_filt1.columns = ['DC_cellname','CELL','tissue','value']
ABCS_test_result_filt1['label'] = 'y_data'


ABCS_test_result_filt2 = ABCS_test_result[['DC_cellname','CELL','tissue','PRED_3']]
ABCS_test_result_filt2.columns = ['DC_cellname','CELL','tissue','value']
ABCS_test_result_filt2['label'] = 'pred_result'


for_split_vio = pd.concat([ABCS_test_result_filt1, ABCS_test_result_filt2])

#my_order = test_cell_df.groupby(by=["tissue"])["P_COR"].mean().sort_values().iloc[::-1].index


fig, ax = plt.subplots(figsize=(20, 8))
sns.violinplot(
	ax =ax, data  = for_split_vio, order=order,
	x = 'DC_cellname', y = 'value', 
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
plt.savefig(os.path.join(PRJ_PATH,'cell_compare.pdf'),format="pdf", bbox_inches = 'tight')

plt.close()









for cell_name in list(set(result_merge_df.cell_name)) :
	cell_name
	min(result_merge_df[result_merge_df.cell_name==cell_name]['y_data'])
	max(result_merge_df[result_merge_df.cell_name==cell_name]['y_data'])
	np.std(result_merge_df[result_merge_df.cell_name==cell_name]['y_data'])












































###############################################################
###############################################################
###############################################################
###############################################################


# 혹시 TSNE 그려보기 

from sklearn.datasets import load_digits
from sklearn.manifold import TSNE

ABCS_test_result_tsne = ABCS_test_result[['CID_CID','CELL','tissue', 'ANS']]
cell_list = list(set(ABCS_test_result_tsne.CELL))
cell_list.sort()

test_1 = ABCS_test_result_tsne[ABCS_test_result_tsne.CELL == 'A375_SKIN']
test_2 = ABCS_test_result_tsne[ABCS_test_result_tsne.CELL == 'RVH421_SKIN']
common_n = len(set(test_1.CID_CID) & set(test_2.CID_CID))


'148177___135398510'
test_1[test_1.CID_CID=='148177___135398510']['ANS'].item()
test_2[test_2.CID_CID=='148177___135398510']['ANS'].item()


cell_array = pd.DataFrame(columns = cell_list, index = cell_list)

for cell_1 in cell_list : 
	cell_1
	for cell_2 in cell_list : 
		ABCS_res_cell_1 = ABCS_test_result_tsne[ABCS_test_result_tsne.CELL == cell_1]
		ABCS_res_cell_2 = ABCS_test_result_tsne[ABCS_test_result_tsne.CELL == cell_2]
		common_n = list(set(ABCS_res_cell_1.CID_CID) & set(ABCS_res_cell_2.CID_CID))
		common_n.sort()
		std_list = []
		for cc in common_n : 
			test_1 = ABCS_res_cell_1[ABCS_res_cell_1.CID_CID==cc]['ANS'].item()
			test_2 = ABCS_res_cell_2[ABCS_res_cell_2.CID_CID==cc]['ANS'].item()
			std_res = np.std([test_1, test_2])
			std_list.append(std_res)
		cell_cell_res = np.mean(std_list)
		cell_array.at[cell_1, cell_2] = cell_cell_res
	



cell_array.to_csv('/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V6_W203_349_MIS2/TSNE_array.csv', sep = '\t')

cell_array2 = np.array(cell_array)
cell_array2[np.isnan(cell_array2)] = 0
# 안머깋는디 

# 이거 반복해줘야할듯 
new_df = []
for ind in range(92) : 
	cell_row = cell_array2[ind]
	new_row_ind = [a for a in range(len(cell_row)) if np.isnan(cell_row[a])]
	cell_row[new_row_ind] = 0.0
	new_df.append(cell_row)


new_df2 = np.array(new_df)

n_components = 2
# TSNE 모델의 인스턴스를 만듭니다.
model = TSNE(n_components=n_components)
# data를 가지고 TSNE 모델을 훈련(적용) 합니다.
X_embedded = model.fit_transform(np.array(new_df2))

TSNE_DF = pd.DataFrame(X_embedded)
TSNE_DF.columns = ['comp1','comp2']

cell_check = list(cell_array.index)
TSNE_DF['cell_lines'] = cell_check

cell_tiss = ['_'.join(a.split('_')[1:]) for a in cell_check]
TSNE_DF['tissue'] = cell_tiss

cell_strip = [a.split('_')[0] for a in cell_check]
TSNE_DF['strip'] = cell_strip


tissue_set = ['CENTRAL_NERVOUS_SYSTEM', 'LUNG', 'BREAST', 'BONE', 'OVARY', 'PROSTATE', 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'LARGE_INTESTINE', 'SKIN', 'PLEURA', 'KIDNEY' ] # list(set(test_cell_df['tissue']))
# 여기서만 central 색깔 좀 여리게 바꿔줌 
color_set = ['#d95e92','#FF7514','#025669','#308446','#84C3BE','#D53032','#4ddcfd','#ffcd36','#ac8cff',"#0000ffff","#7bff68ff"] # "#1E1E1E"
color_dict = {a : color_set[tissue_set.index(a)] for a in tissue_set}


# 밥먹고 오면 그림 그려보기 ! 
# palette = sns.color_palette("bright", 10)
fig = plt.figure(figsize=(10,10))
sns.scatterplot(data = TSNE_DF, x = 'comp1', y = 'comp2', hue = 'tissue', legend='full', palette=color_dict, size = 'tissue', sizes=[105]*11) 
for i in range(TSNE_DF.shape[0]):
    plt.text(TSNE_DF.comp1[i]-0.8, TSNE_DF.comp2[i]+0.6, TSNE_DF.strip[i], size = 8)

plt.tight_layout()
path = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V6_W203_349_MIS2'
plotname = 'tsne_trial'
fig.savefig('{}/{}.png'.format(path, plotname), bbox_inches = 'tight')



MY_L = ['BT474_BREAST', 'MDAMB231_BREAST', 'MDAMB175VII_BREAST', 'MDAMB468_BREAST', 'HCC1419_BREAST', 'MDAMB361_BREAST', 'HCC1500_BREAST', 'T47D_BREAST', 'MCF7_BREAST', 'KPL1_BREAST', 'CAMA1_BREAST', 'MDAMB436_BREAST', 'ZR751_BREAST', 'BT549_BREAST', 'HS578T_BREAST', 'UACC812_BREAST']
MY_L = ['IPC298_SKIN', 'SKMEL2_SKIN', 'G361_SKIN', 'UACC62_SKIN', 'A375_SKIN', 'MELHO_SKIN', 'MEWO_SKIN', 'UACC257_SKIN', 'A101D_SKIN', 'RVH421_SKIN', 'COLO829_SKIN', 'SKMEL5_SKIN', 'LOXIMVI_SKIN', 'WM115_SKIN', 'A2058_SKIN', 'COLO800_SKIN', 'SKMEL28_SKIN', 'SKMEL30_SKIN', 'RPMI7951_SKIN', 'HT144_SKIN', 'MALME3M_SKIN', 'COLO792_SKIN']
MY_L = ['KM12_LARGE_INTESTINE', 'HT29_LARGE_INTESTINE', 'RKO_LARGE_INTESTINE', 'SW620_LARGE_INTESTINE', 'LOVO_LARGE_INTESTINE', 'SW837_LARGE_INTESTINE', 'HCT15_LARGE_INTESTINE', 'HCT116_LARGE_INTESTINE', 'DLD1_LARGE_INTESTINE']
MY_L = ['NCIH2122_LUNG', 'NCIH226_LUNG', 'NCIH460_LUNG', 'A427_LUNG', 'SKMES1_LUNG', 'NCIH522_LUNG', 'NCIH23_LUNG', 'HOP92_LUNG', 'EKVX_LUNG', 'HOP62_LUNG', 'NCIH1650_LUNG', 'NCIH520_LUNG', 'A549_LUNG']
MY_L = ['UHO1_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'RPMI8226_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'K562_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'L1236_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'SR786_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE']
MY_L = ['SF295_CENTRAL_NERVOUS_SYSTEM', 'U251MG_CENTRAL_NERVOUS_SYSTEM', 'T98G_CENTRAL_NERVOUS_SYSTEM', 'SF268_CENTRAL_NERVOUS_SYSTEM', 'SNB75_CENTRAL_NERVOUS_SYSTEM', 'SF539_CENTRAL_NERVOUS_SYSTEM']
MY_L = ['786O_KIDNEY', 'A498_KIDNEY', 'ACHN_KIDNEY', 'UO31_KIDNEY', 'CAKI1_KIDNEY']
MY_L = ['VCAP_PROSTATE', 'PC3_PROSTATE']
MY_L = ['MSTO211H_PLEURA']
MY_L = ['OVCAR4_OVARY', 'OVCAR8_OVARY', 'PA1_OVARY', 'NIHOVCAR3_OVARY', 'A2780_OVARY', 'IGROV1_OVARY', 'ES2_OVARY', 'OV90_OVARY', 'UWB1289_OVARY', 'OVCAR5_OVARY', 'SKOV3_OVARY', 'CAOV3_OVARY']
MY_L = ['A673_BONE']


my_check = ccle_info[ccle_info.CCLE_Name.isin(MY_L)][['CCLE_Name','sample_collection_site','primary_or_metastasis','lineage_subtype','lineage_sub_subtype']]






# 혹시 TSNE 3D 가능?

n_components = 3
model = TSNE(n_components=n_components)
X_embedded = model.fit_transform(np.array(new_df2))

TSNE_DF = pd.DataFrame(X_embedded)
TSNE_DF.columns = ['comp1','comp2','comp3']

cell_check = list(cell_array.index)
TSNE_DF['cell_lines'] = cell_check

cell_tiss = ['_'.join(a.split('_')[1:]) for a in cell_check]
TSNE_DF['tissue'] = cell_tiss
TSNE_DF['tissue'] = TSNE_DF['tissue'].as

cell_strip = [a.split('_')[0] for a in cell_check]
TSNE_DF['strip'] = cell_strip


tissue_set = ['CENTRAL_NERVOUS_SYSTEM', 'LUNG', 'BREAST', 'BONE', 'OVARY', 'PROSTATE', 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'LARGE_INTESTINE', 'SKIN', 'PLEURA', 'KIDNEY' ] # list(set(test_cell_df['tissue']))
# 여기서만 central 색깔 좀 여리게 바꿔줌 
color_set = ['#d95e92','#FF7514','#025669','#308446','#84C3BE','#D53032','#4ddcfd','#ffcd36','#ac8cff',"#0000ffff","#7bff68ff"] # "#1E1E1E"
color_dict2 = {a : mcolors.to_rgba(color_set[tissue_set.index(a)]) for a in tissue_set}
color_set2 = ['rgb(217,94,146)', 'rgb(255,117,20)', 'rgb(2,86,105)', 'rgb(48,132,70)', 'rgb(132,195,190)', 'rgb(213,48,50)', 'rgb(77,220,253)', 'rgb(255,205,54)', 'rgb(172,140,255)', 'rgb(0,255,255)',  'rgb(255,104,255)']
color_dict2= {a : color_set2[tissue_set.index(a)] for a in tissue_set}



# 밥먹고 오면 그림 그려보기 ! 
import plotly.express as px

fig = px.scatter_3d(TSNE_DF, x='comp1', y='comp2', z='comp3',hover_name="cell_lines",
              color='tissue', color_discrete_map = color_dict2)


path = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V6_W203_349_MIS2'
plotname = 'tsne_trial'
fig.write_html('{}/{}.html'.format(path, plotname))

# 아... 3D 는 좀 별로임.... 










# 하위 빼고 그리면 어떨지? 

from sklearn.datasets import load_digits
from sklearn.manifold import TSNE


cells_bad = [
	'A673_BONE','MDAMB361_BREAST','CAMA1_BREAST','MDAMB175VII_BREAST','L1236_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE',
	'UACC812_BREAST','HCC1500_BREAST','HCC1419_BREAST', 'UHO1_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'T98G_CENTRAL_NERVOUS_SYSTEM', 'BT474_BREAST']

cells_good = [a for a in cell_array.index if a not in cells_bad]


cell_array33 = cell_array.loc[cells_good,cells_good]

cell_array33_2 = np.array(cell_array33)

# 이거 반복해줘야할듯 
new_df = []
for ind in range(81) : 
	cell_row = cell_array33_2[ind]
	new_row_ind = [a for a in range(len(cell_row)) if np.isnan(cell_row[a])]
	cell_row[new_row_ind] = 0.0
	new_df.append(cell_row)


new_df2 = np.array(new_df)

n_components = 2
# TSNE 모델의 인스턴스를 만듭니다.
model = TSNE(n_components=n_components)
# data를 가지고 TSNE 모델을 훈련(적용) 합니다.
X_embedded = model.fit_transform(np.array(new_df2))

TSNE_DF = pd.DataFrame(X_embedded)
TSNE_DF.columns = ['comp1','comp2']

cell_check = list(cell_array33.index)
TSNE_DF['cell_lines'] = cell_check

cell_tiss = ['_'.join(a.split('_')[1:]) for a in cell_check]
TSNE_DF['tissue'] = cell_tiss

cell_strip = [a.split('_')[0] for a in cell_check]
TSNE_DF['strip'] = cell_strip


tissue_set = ['CENTRAL_NERVOUS_SYSTEM', 'LUNG', 'BREAST', 'BONE', 'OVARY', 'PROSTATE', 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'LARGE_INTESTINE', 'SKIN', 'PLEURA', 'KIDNEY' ] # list(set(test_cell_df['tissue']))
# 여기서만 central 색깔 좀 여리게 바꿔줌 
color_set = ['#d95e92','#FF7514','#025669','#308446','#84C3BE','#D53032','#4ddcfd','#ffcd36','#ac8cff',"#0000ffff","#7bff68ff"] # "#1E1E1E"
color_dict = {a : color_set[tissue_set.index(a)] for a in tissue_set}


# 밥먹고 오면 그림 그려보기 ! 
# palette = sns.color_palette("bright", 10)
fig = plt.figure(figsize=(10,10))
sns.scatterplot(data = TSNE_DF, x = 'comp1', y = 'comp2', hue = 'tissue', legend='full', palette=color_dict, size = 'tissue', sizes=[105]*10) 
for i in range(TSNE_DF.shape[0]):
    plt.text(TSNE_DF.comp1[i]-0.2, TSNE_DF.comp2[i]+0.2, TSNE_DF.strip[i], size = 8)

plt.tight_layout()
path = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V6_W203_349_MIS2'
plotname = 'tsne_trial2'
fig.savefig('{}/{}.png'.format(path, plotname), bbox_inches = 'tight')




# 혹시 TSNE 3D 가능?

n_components = 3
model = TSNE(n_components=n_components)
X_embedded = model.fit_transform(np.array(new_df2))

TSNE_DF = pd.DataFrame(X_embedded)
TSNE_DF.columns = ['comp1','comp2','comp3']

cell_check = list(cell_array33.index)
TSNE_DF['cell_lines'] = cell_check

cell_tiss = ['_'.join(a.split('_')[1:]) for a in cell_check]
TSNE_DF['tissue'] = cell_tiss

cell_strip = [a.split('_')[0] for a in cell_check]
TSNE_DF['strip'] = cell_strip


tissue_set = ['CENTRAL_NERVOUS_SYSTEM', 'LUNG', 'BREAST', 'BONE', 'OVARY', 'PROSTATE', 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'LARGE_INTESTINE', 'SKIN', 'PLEURA', 'KIDNEY' ] # list(set(test_cell_df['tissue']))
# 여기서만 central 색깔 좀 여리게 바꿔줌 
color_set = ['#d95e92','#FF7514','#025669','#308446','#84C3BE','#D53032','#4ddcfd','#ffcd36','#ac8cff',"#0000ffff","#7bff68ff"] # "#1E1E1E"
color_dict2 = {a : mcolors.to_rgba(color_set[tissue_set.index(a)]) for a in tissue_set}
color_set2 = ['rgb(217,94,146)', 'rgb(255,117,20)', 'rgb(2,86,105)', 'rgb(48,132,70)', 'rgb(132,195,190)', 'rgb(213,48,50)', 'rgb(77,220,253)', 'rgb(255,205,54)', 'rgb(172,140,255)', 'rgb(0,255,255)',  'rgb(255,104,255)']
color_dict2= {a : color_set2[tissue_set.index(a)] for a in tissue_set}



# 밥먹고 오면 그림 그려보기 ! 
import plotly.express as px

fig = px.scatter_3d(TSNE_DF, x='comp1', y='comp2', z='comp3',hover_name="cell_lines",
              color='tissue', color_discrete_map = color_dict2)


path = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V6_W203_349_MIS2'
plotname = 'tsne_trial2'
fig.write_html('{}/{}.html'.format(path, plotname))


























# barplot 도 그리기  
# 그래서 가져와봅시다 결과 다시한번 
# 그러면 kisti 에서 아예 가져와야함 























