# 601 

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

file_name = 'M3V6_349_MISS1_PRE' # 0608
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
CELL_CUT = 10

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

A_B_C_S_SET_SM = copy.deepcopy(A_B_C_S_SET_COH2) # 182174

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
		self.G_Common_dim = min([G_hiddim_chem,G_hiddim_exp])
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
				input_drug1 = F.relu(input_drug1)
			else :
				input_drug1 = self.Convs_1[L1](input_drug1)
		#
		for L2 in range(len(self.Convs_1)):
			if L2 != len(self.Convs_1)-1 :
				input_drug2 = self.Convs_1[L2](input_drug2)
				input_drug2 = F.dropout(input_drug2, p=self.inDrop, training = self.training)
				input_drug2 = F.relu(input_drug2)
			else :
				input_drug2 = self.Convs_1[L2](input_drug2)
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
	use_cuda = True  #  #  #  #  #  #  # True
	#
	dsn1_layers = [config["feat_size_0"], config["feat_size_1"] , config["feat_size_2"] ]
	dsn2_layers = [config["feat_size_0"], config["feat_size_1"] , config["feat_size_2"] ]
	snp_layers = [config["feat_size_3"] ,config["feat_size_4"]]
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
			dsn1_layers, dsn2_layers, snp_layers,      # drug 1 layers, drug 2 layers, merged layer, 
			len(set(A_B_C_S_SET_SM.CELL)), 1,      # cell_dim ,out_dim,
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



# 이건 테스트 버전임. 생각하고 해 

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
		"feat_size_0" : tune.grid_search([ my_config['config/feat_size_0'].item() ]),
		"feat_size_1" : tune.grid_search([ my_config['config/feat_size_1'].item() ]),
		"feat_size_2" : tune.grid_search([ my_config['config/feat_size_2'].item() ]),
		"feat_size_3" : tune.grid_search([ my_config['config/feat_size_3'].item() ]),
		"feat_size_4" : tune.grid_search([ my_config['config/feat_size_4'].item() ]),
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










W_NAME = 'W100' # 고른 내용으로 5CV 다시 
MJ_NAME = 'M3V6'
WORK_DATE = '23.06.15' # 349
MISS_NAME = 'MIS0'
PPI_NAME = '349'
WORK_NAME = 'WORK_100' # 349###################################################################################################


W_NAME = 'W101' # 고른 내용으로 5CV 다시 
MJ_NAME = 'M3V6'
WORK_DATE = '23.06.15' # 349
MISS_NAME = 'MIS1'
PPI_NAME = '349'
WORK_NAME = 'WORK_101' # 349###################################################################################################



WORK_PATH = '/home01/k040a01/02.{}/{}_{}_{}_{}/'.format(MJ_NAME,MJ_NAME,W_NAME,PPI_NAME,MISS_NAME)


OLD_PATH = '/home01/k040a01/02.M3V5/M3V5_W322_349_MIS2'
ANA_DF_CSV = pd.read_csv(os.path.join(OLD_PATH,'RAY_ANA_DF.{}.csv'.format('M3V5_W322_349_MIS2')))

my_config = ANA_DF_CSV[ANA_DF_CSV.trial_id=='4efc26cc'] # 349 



#4gpu for 5cv 
MAIN('PRJ02.{}.{}.{}.{}.{}'.format(WORK_DATE, MJ_NAME, WORK_NAME, PPI_NAME, MISS_NAME), my_config, 1, 1000, 24, 0.5)


with open(file='{}/CV_SM_list.pickle'.format(WORK_PATH), mode='wb') as f:
	pickle.dump(CV_ND_INDS, f)











# W 100 : MIS0 

tail /home01/k040a01/02.M3V6/M3V6_W100_349_MIS0/RESULT.G4.CV5.txt -n 100 
tail ~/logs/M3V6W100_GPU4_12927.log 


# W 101 : MIS0 

tail /home01/k040a01/02.M3V6/M3V6_W101_349_MIS0/RESULT.G4.CV5.txt -n 100 
tail ~/logs/M3V6W102_GPU4_12927.log 

# W 102 : MIS0 

tail /home01/k040a01/02.M3V6/M3V6_W102_349_MIS0/RESULT.G4.CV5.txt -n 100 
tail ~/logs/M3V6W102_GPU4_12927.log 



sbatch gpu4.W100.CV5.any M3V6_WORK100.349.CV5.py
sbatch gpu4.W101.CV5.any M3V6_WORK101.349.CV5.py
sbatch gpu4.W102.CV5.any M3V6_WORK102.349.CV5.py
sbatch gpu4.W103.CV5.any M3V6_WORK103.349.CV5.py
sbatch gpu4.W104.CV5.any M3V6_WORK104.349.CV5.py
sbatch gpu4.W105.CV5.any M3V6_WORK105.349.CV5.py






# W 103 : MIS1

tail /home01/k040a01/02.M3V6/M3V6_W103_349_MIS1/RESULT.G4.CV5.txt -n 100 
tail ~/logs/M3V6W103_GPU4_12927.log 

# W 104 : MIS1

tail /home01/k040a01/02.M3V6/M3V6_W104_349_MIS1/RESULT.G4.CV5.txt -n 100 
tail ~/logs/M3V6W104_GPU4_12927.log 

# W 105 : MIS1

tail /home01/k040a01/02.M3V6/M3V6_W105_349_MIS1/RESULT.G4.CV5.txt -n 100 
tail ~/logs/M3V6W105_GPU4_12927.log 







# W 106 : MIS2

tail /home01/k040a01/02.M3V6/M3V6_W106_349_MIS2/RESULT.G4.CV5.txt -n 100 

# W 107 : MIS2

tail /home01/k040a01/02.M3V6/M3V6_W107_349_MIS2/RESULT.G4.CV5.txt -n 100 

# W 108 : MIS2

tail /home01/k040a01/02.M3V6/M3V6_W108_349_MIS2/RESULT.G4.CV5.txt -n 100 





# W 109 : MIS2

tail /home01/k040a01/02.M3V6/M3V6_W109_349_MIS2/RESULT.G4.CV5.txt -n 100 

# W 110 : MIS2

tail /home01/k040a01/02.M3V6/M3V6_W110_349_MIS2/RESULT.G4.CV5.txt -n 100 

# W 111 : MIS2

tail /home01/k040a01/02.M3V6/M3V6_W111_349_MIS2/RESULT.G4.CV5.txt -n 100 








W_NAME = 'W100' # 고른 내용으로 5CV 다시 
MJ_NAME = 'M3V6'
WORK_DATE = '23.06.19' # 349
MISS_NAME = 'MIS0'
PPI_NAME = '349'
WORK_NAME = 'WORK_100' # 349###################################################################################################

cp M3V6_WORK100.349.CV5.py M3V6_WORK118.349.CV5.py
cp M3V6_WORK101.349.CV5.py M3V6_WORK119.349.CV5.py
cp M3V6_WORK102.349.CV5.py M3V6_WORK120.349.CV5.py
cp M3V6_WORK103.349.CV5.py M3V6_WORK121.349.CV5.py
cp M3V6_WORK104.349.CV5.py M3V6_WORK122.349.CV5.py
cp M3V6_WORK105.349.CV5.py M3V6_WORK123.349.CV5.py


sed -i 's/M3V6_349_MISS1_PRE/M3V6_349_MISS2_FULL/g' M3V6_WORK118.349.CV5.py
sed -i 's/W100/W118/g' M3V6_WORK118.349.CV5.py
sed -i 's/WORK_100/WORK_118/g' M3V6_WORK118.349.CV5.py
sed -i 's/23.06.17/23.06.19/g' M3V6_WORK118.349.CV5.py

sed -i 's/M3V6_349_MISS1_PRE/M3V6_349_MISS2_FULL/g' M3V6_WORK119.349.CV5.py
sed -i 's/W101/W119/g' M3V6_WORK119.349.CV5.py
sed -i 's/WORK_101/WORK_119/g' M3V6_WORK119.349.CV5.py
sed -i 's/23.06.17/23.06.19/g' M3V6_WORK119.349.CV5.py

sed -i 's/M3V6_349_MISS1_PRE/M3V6_349_MISS2_FULL/g' M3V6_WORK120.349.CV5.py
sed -i 's/W102/W120/g' M3V6_WORK120.349.CV5.py
sed -i 's/WORK_102/WORK_120/g' M3V6_WORK120.349.CV5.py
sed -i 's/23.06.17/23.06.19/g' M3V6_WORK120.349.CV5.py

sed -i 's/M3V6_349_MISS1_PRE/M3V6_349_MISS2_FULL/g' M3V6_WORK121.349.CV5.py
sed -i 's/W103/W121/g' M3V6_WORK121.349.CV5.py
sed -i 's/WORK_103/WORK_121/g' M3V6_WORK121.349.CV5.py
sed -i 's/23.06.17/23.06.19/g' M3V6_WORK121.349.CV5.py

sed -i 's/M3V6_349_MISS1_PRE/M3V6_349_MISS2_FULL/g' M3V6_WORK122.349.CV5.py
sed -i 's/W104/W122/g' M3V6_WORK122.349.CV5.py
sed -i 's/WORK_104/WORK_122/g' M3V6_WORK122.349.CV5.py
sed -i 's/23.06.17/23.06.19/g' M3V6_WORK122.349.CV5.py

sed -i 's/M3V6_349_MISS1_PRE/M3V6_349_MISS2_FULL/g' M3V6_WORK123.349.CV5.py
sed -i 's/W105/W123/g' M3V6_WORK123.349.CV5.py
sed -i 's/WORK_105/WORK_123/g' M3V6_WORK123.349.CV5.py
sed -i 's/23.06.17/23.06.19/g' M3V6_WORK123.349.CV5.py

cp gpu1.W112.CV5.any gpu1.W118.CV5.any
cp gpu1.W112.CV5.any gpu1.W119.CV5.any
cp gpu1.W112.CV5.any gpu1.W120.CV5.any

cp gpu1.W117.CV5.any gpu1.W121.CV5.any
cp gpu1.W117.CV5.any gpu1.W122.CV5.any
cp gpu1.W117.CV5.any gpu1.W123.CV5.any

sed -i 's/W112/W118/g' gpu1.W118.CV5.any
sed -i 's/W112/W119/g' gpu1.W119.CV5.any
sed -i 's/W112/W120/g' gpu1.W120.CV5.any

sed -i 's/W117/W121/g' gpu1.W121.CV5.any
sed -i 's/W117/W122/g' gpu1.W122.CV5.any
sed -i 's/W117/W123/g' gpu1.W123.CV5.any



mkdir /home01/k040a01/02.M3V6/M3V6_W118_349_MIS0/
mkdir /home01/k040a01/02.M3V6/M3V6_W119_349_MIS0/
mkdir /home01/k040a01/02.M3V6/M3V6_W120_349_MIS0/


mkdir /home01/k040a01/02.M3V6/M3V6_W121_349_MIS1/
mkdir /home01/k040a01/02.M3V6/M3V6_W122_349_MIS1/
mkdir /home01/k040a01/02.M3V6/M3V6_W123_349_MIS1/


sbatch gpu1.W118.CV5.any M3V6_WORK118.349.CV5.py
sbatch gpu1.W119.CV5.any M3V6_WORK119.349.CV5.py
sbatch gpu1.W120.CV5.any M3V6_WORK120.349.CV5.py

sbatch gpu1.W121.CV5.any M3V6_WORK121.349.CV5.py
sbatch gpu1.W122.CV5.any M3V6_WORK122.349.CV5.py
sbatch gpu1.W123.CV5.any M3V6_WORK123.349.CV5.py





cp M3V6_WORK111.349.CV5.py M3V6_WORK126.349.CV5.py
sed -i 's/W111/W126/g' M3V6_WORK126.349.CV5.py
sed -i 's/WORK_111/WORK_126/g' M3V6_WORK126.349.CV5.py
sed -i 's/23.06.17/23.06.19/g' M3V6_WORK126.349.CV5.py


cp M3V6_WORK110.349.CV5.py M3V6_WORK125.349.CV5.py
sed -i 's/W110/W125/g' M3V6_WORK125.349.CV5.py
sed -i 's/WORK_110/WORK_125/g' M3V6_WORK125.349.CV5.py
sed -i 's/23.06.17/23.06.19/g' M3V6_WORK125.349.CV5.py


cp M3V6_WORK109.349.CV5.py M3V6_WORK124.349.CV5.py
sed -i 's/W109/W124/g' M3V6_WORK124.349.CV5.py
sed -i 's/WORK_109/WORK_124/g' M3V6_WORK124.349.CV5.py
sed -i 's/23.06.17/23.06.19/g' M3V6_WORK124.349.CV5.py

cp gpu1.W109.CV5.any gpu1.W126.CV5.any
cp gpu1.W109.CV5.any gpu1.W125.CV5.any
cp gpu1.W109.CV5.any gpu1.W124.CV5.any

sed -i 's/W109/W126/g' gpu1.W126.CV5.any
sed -i 's/W109/W125/g' gpu1.W125.CV5.any
sed -i 's/W109/W124/g' gpu1.W124.CV5.any

mkdir /home01/k040a01/02.M3V6/M3V6_W126_349_MIS2/
mkdir /home01/k040a01/02.M3V6/M3V6_W125_349_MIS2/
mkdir /home01/k040a01/02.M3V6/M3V6_W124_349_MIS2/

sbatch gpu1.W126.CV5.any M3V6_WORK126.349.CV5.py
sbatch gpu1.W125.CV5.any M3V6_WORK125.349.CV5.py
sbatch gpu1.W124.CV5.any M3V6_WORK124.349.CV5.py







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

WORK_NAME = 'WORK_124' # 349
W_NAME = 'W124'
WORK_DATE = '23.06.19' # 349


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
































######################################################

그래서 로컬에서 잘 예측한 cid-cell 과 그렇지 않은 cid-cell 을 확인해보기로 함 












