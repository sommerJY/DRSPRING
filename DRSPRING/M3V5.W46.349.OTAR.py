
# target 제한 없는 oneil 



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


# 내가보기엔, 데이터 늘릴 필요가 없어졌으니가 그냥 말 그대로의 ablation study 만 하면 될듯
# 그러면 그냥 원래 데이터에다가 lincs 없애서 돌리는거만 해보면 될듯 




ray.init()

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


SAVE_PATH = '/home01/k040a01/02.M3V5/M3V5_W32_349_DATA/'
# SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V5_W32_349_FULL/'

file_name = 'M3V5_349_MISS2_ONEIL' # 0608


A_B_C_S_SET_ADD = pd.read_csv(SAVE_PATH+'{}.A_B_C_S_SET_ADD.csv'.format(file_name), low_memory=False)
MY_chem_A_feat = torch.load(SAVE_PATH+'{}.MY_chem_A_feat.pt'.format(file_name))
MY_chem_B_feat = torch.load(SAVE_PATH+'{}.MY_chem_B_feat.pt'.format(file_name))
MY_chem_A_adj = torch.load(SAVE_PATH+'{}.MY_chem_A_adj.pt'.format(file_name))
MY_chem_B_adj = torch.load(SAVE_PATH+'{}.MY_chem_B_adj.pt'.format(file_name))
MY_g_EXP_A = torch.load(SAVE_PATH+'{}.MY_g_EXP_A.pt'.format(file_name))
MY_g_EXP_B = torch.load(SAVE_PATH+'{}.MY_g_EXP_B.pt'.format(file_name))
MY_Target_1_A = torch.load(SAVE_PATH+'{}.MY_Target_1_A.pt'.format(file_name))
MY_Target_1_B = torch.load(SAVE_PATH+'{}.MY_Target_1_B.pt'.format(file_name))
#MY_Target_2_A = torch.load(SAVE_PATH+'{}.MY_Target_2_A.pt'.format(file_name))
#MY_Target_2_B = torch.load(SAVE_PATH+'{}.MY_Target_2_B.pt'.format(file_name))
MY_CellBase = torch.load(SAVE_PATH+'{}.MY_CellBase.pt'.format(file_name))
MY_syn = torch.load(SAVE_PATH+'{}.MY_syn.pt'.format(file_name))



																									SAVE_PATH_786O = '/home01/k040a01/02.M3V5/M3V5_W32_349_DATA/'
																									# SAVE_PATH_786O = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V5_W32_349_FULL/'

																									file_name_786O = 'M3V5_349_MISS2_FULL_786O'

																									A_B_C_S_SET_ADD_786O = pd.read_csv(SAVE_PATH_786O+'{}.A_B_C_S_SET_ADD.csv'.format(file_name_786O), low_memory=False)
																									MY_chem_A_feat_786O = torch.load(SAVE_PATH_786O+'{}.MY_chem_A_feat.pt'.format(file_name_786O))
																									MY_chem_B_feat_786O = torch.load(SAVE_PATH_786O+'{}.MY_chem_B_feat.pt'.format(file_name_786O))
																									MY_chem_A_adj_786O = torch.load(SAVE_PATH_786O+'{}.MY_chem_A_adj.pt'.format(file_name_786O))
																									MY_chem_B_adj_786O = torch.load(SAVE_PATH_786O+'{}.MY_chem_B_adj.pt'.format(file_name_786O))
																									MY_g_EXP_A_786O = torch.load(SAVE_PATH_786O+'{}.MY_g_EXP_A.pt'.format(file_name_786O))
																									MY_g_EXP_B_786O = torch.load(SAVE_PATH_786O+'{}.MY_g_EXP_B.pt'.format(file_name_786O))
																									MY_Target_1_A_786O = torch.load(SAVE_PATH_786O+'{}.MY_Target_1_A.pt'.format(file_name_786O))
																									MY_Target_1_B_786O = torch.load(SAVE_PATH_786O+'{}.MY_Target_1_B.pt'.format(file_name_786O))
																									#MY_Target_2_A = torch.load(SAVE_PATH_786O+'{}.MY_Target_2_A.pt'.format(file_name_786O))
																									#MY_Target_2_B = torch.load(SAVE_PATH_786O+'{}.MY_Target_2_B.pt'.format(file_name_786O))
																									MY_CellBase_786O = torch.load(SAVE_PATH_786O+'{}.MY_CellBase.pt'.format(file_name_786O))
																									MY_syn_786O = torch.load(SAVE_PATH_786O+'{}.MY_syn.pt'.format(file_name_786O))



																									# 315936
																									A_B_C_S_SET_ADD = pd.concat(
																										[A_B_C_S_SET_ADD[['drug_row_CID', 'drug_col_CID', 'DrugCombCCLE','ROWCHECK', 'COLCHECK', 'ROW_CAN_SMILES', 'COL_CAN_SMILES','ROW_pert_id', 'ROW_BETA_sig_id', 'COL_pert_id', 'COL_BETA_sig_id','type', 'ROW_len', 'COL_len', 'Basal_Exp', 'SYN_OX', 'T1OX','tani01', 'tani_02', 'tani_Q', 'ONEIL']],
																										A_B_C_S_SET_ADD_786O[['drug_row_CID', 'drug_col_CID', 'DrugCombCCLE','ROWCHECK', 'COLCHECK', 'ROW_CAN_SMILES', 'COL_CAN_SMILES','ROW_pert_id', 'ROW_BETA_sig_id', 'COL_pert_id', 'COL_BETA_sig_id','type', 'ROW_len', 'COL_len', 'Basal_Exp', 'SYN_OX', 'T1OX','tani01', 'tani_02', 'tani_Q', 'ONEIL']]
																										]
																									)

																									MY_chem_A_feat = torch.concat([MY_chem_A_feat,MY_chem_A_feat_786O])
																									MY_chem_B_feat = torch.concat([MY_chem_B_feat,MY_chem_B_feat_786O])
																									MY_chem_A_adj = torch.concat([MY_chem_A_adj,MY_chem_A_adj_786O])
																									MY_chem_B_adj = torch.concat([MY_chem_B_adj,MY_chem_B_adj_786O])
																									MY_g_EXP_A = torch.concat([MY_g_EXP_A,MY_g_EXP_A_786O])
																									MY_g_EXP_B = torch.concat([MY_g_EXP_B,MY_g_EXP_B_786O])
																									MY_Target_1_A = torch.concat([MY_Target_1_A,MY_Target_1_A_786O])
																									MY_Target_1_B = torch.concat([MY_Target_1_B,MY_Target_1_B_786O])
																									MY_CellBase = torch.concat([MY_CellBase,MY_CellBase_786O])
																									MY_syn = torch.concat([MY_syn,MY_syn_786O])









# 초장부터 데이터 refine 필요 
# 0605 version
A_B_C_S_SET_ADD2 = copy.deepcopy(A_B_C_S_SET_ADD)
A_B_C_S_SET_ADD2 = A_B_C_S_SET_ADD2.reset_index(drop = True)
A_B_C_S_SET_ADD2['type'] = ['AXBO' if a=="AOBX" else a  for a in A_B_C_S_SET_ADD2.type ]


A_B_C_S_SET_ADD2['ori_index'] = list(A_B_C_S_SET_ADD2.index)
aaa = list(A_B_C_S_SET_ADD2['drug_row_CID'])
bbb = list(A_B_C_S_SET_ADD2['drug_col_CID'])
aa = list(A_B_C_S_SET_ADD2['ROW_CAN_SMILES'])
bb = list(A_B_C_S_SET_ADD2['COL_CAN_SMILES'])
cc = list(A_B_C_S_SET_ADD2['DrugCombCCLE'])
aaaa = [a if type(a) == str else 'NA' for a in list(A_B_C_S_SET_ADD2['ROW_BETA_sig_id'])]
bbbb = [a if type(a) == str else 'NA' for a in list(A_B_C_S_SET_ADD2['COL_BETA_sig_id'])]

A_B_C_S_SET_ADD2['CID_CID'] = [str(int(aaa[i])) + '___' + str(int(bbb[i])) if aaa[i] < bbb[i] else str(int(bbb[i])) + '___' + str(int(aaa[i])) for i in range(A_B_C_S_SET_ADD2.shape[0])]
A_B_C_S_SET_ADD2['CID_CID_CCLE'] = [str(int(aaa[i])) + '___' + str(int(bbb[i]))+ '___' + cc[i] if aaa[i] < bbb[i] else str(int(bbb[i])) + '___' + str(int(aaa[i]))+ '___' + cc[i] for i in range(A_B_C_S_SET_ADD2.shape[0])]
A_B_C_S_SET_ADD2['SM_C_CHECK'] = [aa[i] + '___' + bb[i]+ '___' + cc[i] if aa[i] < bb[i] else bb[i] + '___' + aa[i]+ '___' + cc[i] for i in range(A_B_C_S_SET_ADD2.shape[0])]
A_B_C_S_SET_ADD2['SIG_SIG'] = [aaaa[i] + '___' + bbbb[i] if aaaa[i] < bbbb[i] else bbbb[i] + '___' + aaaa[i] for i in range(A_B_C_S_SET_ADD2.shape[0])]
A_B_C_S_SET_ADD2['SIG_SIG_CCLE'] = [aaaa[i] + '___' + bbbb[i] + '___' + cc[i] if aaaa[i] < bbbb[i] else bbbb[i] + '___' + aaaa[i] + '___' + cc[i] for i in range(A_B_C_S_SET_ADD2.shape[0])]




len(set(A_B_C_S_SET_ADD2['CID_CID_CCLE'])) # 307644
len(set(A_B_C_S_SET_ADD2['SM_C_CHECK'])) # 307315
len(set(A_B_C_S_SET_ADD2['SIG_SIG_CCLE'])) # 8756

A_B_C_S_SET_ADD3 = A_B_C_S_SET_ADD2[['ori_index','CID_CID','CID_CID_CCLE','SM_C_CHECK','SIG_SIG','SIG_SIG_CCLE', 'DrugCombCCLE','Basal_Exp', 'SYN_OX', 'T1OX','type']]
dup_index = A_B_C_S_SET_ADD2[['CID_CID','CID_CID_CCLE','SM_C_CHECK','SIG_SIG','SIG_SIG_CCLE','Basal_Exp', 'SYN_OX', 'T1OX','type']].duplicated()== False
A_B_C_S_SET_ADD4 = A_B_C_S_SET_ADD3[dup_index] # 309004 -> 308456







																									# 우선 drugcomb 자체에서 CID 중복이 일어나는거 필터링을 제대로 못했음

																									DC_dup_check = pd.read_csv(DC_PATH+'DC_duplicates.csv', sep ='\t')
																									# DC_dup_check = pd.read_csv('/st06/jiyeonH/11.TOX/DR_SPRING/trials/DC_duplicates.csv', sep ='\t') 
																									DC_dup_list = list(DC_dup_check['id_id_cell'])

																									A_B_C_S_SET_ADD2_dup_check = A_B_C_S_SET_ADD2[A_B_C_S_SET_ADD2.CID_CID_CCLE.isin(DC_dup_list)]

																									check_dc_info = list(A_B_C_S_SET_ADD2_dup_check.CID_CID_CCLE)

																									# 내꺼에서는 딱히 dup 가 아닐수도 (이미 모종의 이유로 지워져서? ) if tmp.shape[0] != len(set(tmp.SIG_SIG)) :
																									rm_index = []
																									syn_prob = []
																									for a in check_dc_info :
																										tmp = A_B_C_S_SET_ADD2_dup_check[A_B_C_S_SET_ADD2_dup_check.CID_CID_CCLE == a]
																										sig_set = list(set(tmp.SIG_SIG))
																										#print(len(sig_set))
																										for i in sig_set :
																											tmp_index = list(tmp[tmp.SIG_SIG == i].index)
																											if MY_syn[tmp_index[0]].item() * MY_syn[tmp_index[1]].item() <0 :
																												syn_prob.append(a)
																												rm_index = rm_index + tmp_index
																										tmp2 = tmp[['CID_CID_CCLE','SIG_SIG']].drop_duplicates(keep='last')
																										rm_index = rm_index + list(tmp2.index)


																									A_B_C_S_SET_rmdup = A_B_C_S_SET_ADD2.drop(rm_index) # 302039

																									A_B_C_S_SET_rmdup[['CID_CID_CCLE','SIG_SIG']].drop_duplicates() # 302039 # CID-SIG 중복인것만 남음
																									A_B_C_S_SET_rmdup[['CID_CID_CCLE','SM_C_CHECK','SIG_SIG']].drop_duplicates() # 302039
																									A_B_C_S_SET_rmdup[['drug_row_CID','drug_col_CID','DrugCombCCLE']].drop_duplicates() # 301232
																									A_B_C_S_SET_rmdup[['ROW_CAN_SMILES','COL_CAN_SMILES','DrugCombCCLE']].drop_duplicates() # 300997
																									len(set(A_B_C_S_SET_rmdup['CID_CID_CCLE'])) # 51212
																									len(set(A_B_C_S_SET_rmdup['SM_C_CHECK'])) # 51160



A_B_C_S_SET_rmdup = copy.deepcopy(A_B_C_S_SET_ADD4)

A_B_C_S_SET_rmdup[['CID_CID_CCLE','SIG_SIG']].drop_duplicates() # 308456 # CID-SIG 중복인것만 남음
A_B_C_S_SET_rmdup[['CID_CID_CCLE','SM_C_CHECK','SIG_SIG']].drop_duplicates() # 308456
A_B_C_S_SET_rmdup[['CID_CID_CCLE']].drop_duplicates() # 307644
A_B_C_S_SET_rmdup[['SM_C_CHECK']].drop_duplicates() # 307315
len(set(A_B_C_S_SET_rmdup['CID_CID_CCLE'])) # 
len(set(A_B_C_S_SET_rmdup['SM_C_CHECK'])) # 








MISS_filter = ['AOBO','AXBO','AOBX','AXBX']

A_B_C_S_SET = A_B_C_S_SET_rmdup[A_B_C_S_SET_rmdup.Basal_Exp == 'O']

A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.SYN_OX == 'O']

## A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.T2OX == 'O'] ###################### old targets 
#A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.T1OX == 'O'] ####################### new targets 

A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.type.isin(MISS_filter)]

# 근데 그럼 synergy 정답값 바꿔야해... 언제바꿔 또 하아아아ㄴ아ㅏ아






# basal node exclude -> CCLE match 만 사용
CCLE_PATH = '/home01/k040a01/01.Data/CCLE/'
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
DC_CELL_DF2 = pd.concat([
	DC_CELL_DF2, 
	pd.DataFrame({'cell_line_id' : [1],'DC_cellname' : ['786O'],'DrugCombCello' : ['CVCL_1051'],'DrugCombCCLE':['786O_KIDNEY']})])

DC_CELL_info_filt = DC_CELL_DF2[DC_CELL_DF2.DrugCombCCLE.isin(A_B_C_S_SET.DrugCombCCLE)] # 38

A_B_C_S_SET_COH = pd.merge(A_B_C_S_SET, DC_CELL_info_filt[['DrugCombCCLE','DC_cellname']], on = 'DrugCombCCLE', how = 'left'  )






# sample number filter 

# 빈도 확인 

C_names = list(set(A_B_C_S_SET_COH.DC_cellname))

C_freq = [list(A_B_C_S_SET_COH.DC_cellname).count(a) for a in C_names]
C_cclename = [list(A_B_C_S_SET_COH[A_B_C_S_SET_COH.DC_cellname==a]['DrugCombCCLE'])[0] for a in C_names]

C_df = pd.DataFrame({'cell' : C_names, 'freq' : C_freq, 'ccle' : C_cclename})
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
					path = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V5_W32_349_FULL/'
					fig.savefig('{}/{}.png'.format(path, plotname), bbox_inches = 'tight')
					plt.close()

						# C_MJ = pd.merge(C_df, DC_CELL_info_filt[['DC_cellname','DrugCombCello','DrugCombCCLE']], left_on = 'cell', right_on = 'DC_cellname', how = 'left')
						# C_MJ.to_csv('/st06/jiyeonH/13.DD_SESS/01.PRJ2/cell_cut_example.csv')





CELL_CUT = 200 ##################################################################################

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




print('CID_CID', flush = True)
tmp = list(set(A_B_C_S_SET_COH2.CID_CID))
tmp2 = sum([a.split('___') for a in tmp],[])

len(set(tmp2))


print('CID_CID', flush = True)
len(set(A_B_C_S_SET_COH2.CID_CID))



print('CID_CID_CCLE', flush = True)
len(set(A_B_C_S_SET_COH2.CID_CID_CCLE))

print('DrugCombCCLE', flush = True)
len(set(A_B_C_S_SET_COH2.DrugCombCCLE))










###########################################################################################
###########################################################################################
###########################################################################################

# 이번엔 그냥 생 5CV 임 
# test 따로 안뺐음 
# 후.. 근데 이거 다들 leave drug-drug out 이네.. 
# 돌려봐야 알것 같다.... 
# check ing... 

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
SM_SM_list.sort() ##################### 
sm_sm_list_1 = sklearn.utils.shuffle(SM_SM_list, random_state=42)

bins = [a for a in range(0, len(sm_sm_list_1), round(len(sm_sm_list_1)*0.2) )]
bins = bins[1:]
res = np.split(sm_sm_list_1, bins)

CV_1_smsm = list(res[0])
CV_2_smsm = list(res[1])
CV_3_smsm = list(res[2])
CV_4_smsm = list(res[3])
CV_5_smsm = list(res[4])

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
		FEAT_A = torch.Tensor(np.array([ self.gcn_gene_A[index].squeeze().tolist() , self.cell_basal[index].tolist()]).T)
		FEAT_B = torch.Tensor(np.array([ self.gcn_gene_B[index].squeeze().tolist() , self.cell_basal[index].tolist()]).T)
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
		expA = expA.view(-1,2)#### 다른점 
		expB = expB.view(-1,2)#### 다른점 
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
			expA = expA.view(-1,2)#### 다른점 
			expB = expB.view(-1,2)#### 다른점 
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
			config["G_exp_layer"], 2 , config["G_exp_hdim"],      # G_layer_exp, G_indim_exp, G_hiddim_exp, 
			dsn1_layers, dsn2_layers, snp_layers,      # drug 1 layers, drug 2 layers, merged layer, 
			len(set(A_B_C_S_SET_SM.DrugCombCCLE)), 1,      # cell_dim ,out_dim,
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



W_NAME = 'W44' # 고른 내용으로 5CV 다시 
MJ_NAME = 'M3V5'
WORK_DATE = '23.06.13' # 349
MISS_NAME = 'MIS2'
PPI_NAME = '349'
WORK_NAME = 'WORK_44' # 349###################################################################################################

WORK_PATH = '/home01/k040a01/02.M3V5/M3V5_W44_349_MIS2/'


OLD_PATH = '/home01/k040a01/02.M3V5/M3V5_W322_349_MIS2'
ANA_DF_CSV = pd.read_csv(os.path.join(OLD_PATH,'RAY_ANA_DF.{}.csv'.format('M3V5_W322_349_MIS2')))

my_config = ANA_DF_CSV[ANA_DF_CSV.trial_id=='4efc26cc'] # 349 
# 그 전까지 쓰던거 : 3fd8fd68
# 6월 9일부터 쓰는거 : '094ca18a' # 8/ 4/ 32/ 3/ 0.1, 0.2/ 256-128-32-64-128 / 0.001
# 6월 11일부터 쓰는거 : '4efc26cc' # 32/ 4/ 32/ 3


#4gpu
MAIN('PRJ02.{}.{}.{}.{}.{}'.format(WORK_DATE, MJ_NAME, WORK_NAME, PPI_NAME, MISS_NAME), my_config, 1, 1000, 32, 1)


#1gpu
MAIN('PRJ02.{}.{}.{}.{}.{}'.format(WORK_DATE, MJ_NAME, WORK_NAME, PPI_NAME, MISS_NAME), my_config, 1, 1000, 128, 1)




sbatch gpu1.W46.CV5.any M3V5_WORK46.349.CV5.py


tail ~/logs/M3V5W46_GPU1_12867.log


tail /home01/k040a01/02.M3V5/M3V5_W46_349_MIS2/RESULT.G1.CV5.txt -n 100 

























import pandas as pd 
import matplotlib.pyplot as plt
from ray.tune import ExperimentAnalysis
from ray.tune import Analysis
import pickle
import math
import torch
import os 
import copy




WORK_NAME = 'WORK_46' # 349
W_NAME = 'W46'
MJ_NAME = 'M3V5'
MISS_NAME = 'MIS2'
PPI_NAME = '349'

WORK_PATH = '/home01/k040a01/02.M3V5/M3V5_W46_349_MIS2/'

WORK_DATE = '23.06.13' # 349
anal_dir = "/home01/k040a01/ray_results/PRJ02.{}.{}.{}.{}.{}".format(WORK_DATE, MJ_NAME,  WORK_NAME, PPI_NAME, MISS_NAME)

list_dir = os.listdir(anal_dir)
exp_json = [a for a in list_dir if 'experiment_state' in a]
exp_json
anal_df = ExperimentAnalysis(os.path.join(anal_dir, exp_json[0]))

ANA_DF_1 = anal_df.dataframe()
ANA_ALL_DF_1 = anal_df.trial_dataframes



ANA_DF = ANA_DF_1
ANA_ALL_DF= ANA_ALL_DF_1


ANA_DF = ANA_DF.sort_values('config/CV')
ANA_DF.index = [0,1,2,3,4]


ANA_DF.to_csv('/home01/k040a01/02.M3V5/{}_{}_{}_{}/RAY_ANA_DF.{}_{}_{}_{}.csv'.format(MJ_NAME, W_NAME, PPI_NAME, MISS_NAME,  MJ_NAME, W_NAME, PPI_NAME, MISS_NAME))
import pickle
with open("/home01/k040a01/02.M3V5/{}_{}_{}_{}/RAY_ANA_DF.{}_{}_{}_{}.pickle".format(MJ_NAME, W_NAME, PPI_NAME, MISS_NAME, MJ_NAME, W_NAME, PPI_NAME, MISS_NAME), "wb") as fp:
	pickle.dump(ANA_ALL_DF,fp) 

'/home01/k040a01/02.M3V5/{}_{}_{}_{}/RAY_ANA_DF.{}_{}_{}_{}.csv'.format(MJ_NAME, W_NAME, PPI_NAME, MISS_NAME,  MJ_NAME, W_NAME, PPI_NAME, MISS_NAME)
"/home01/k040a01/02.M3V5/{}_{}_{}_{}/RAY_ANA_DF.{}_{}_{}_{}.pickle".format(MJ_NAME, W_NAME, PPI_NAME, MISS_NAME, MJ_NAME, W_NAME, PPI_NAME, MISS_NAME)



cv0_key = ANA_DF['logdir'][0] ;	cv1_key = ANA_DF['logdir'][1]; 	cv2_key = ANA_DF['logdir'][2] ;	cv3_key = ANA_DF['logdir'][3];	cv4_key = ANA_DF['logdir'][4]

epc_T_LS_mean = np.mean([ANA_ALL_DF[cv0_key]['T_LS'], ANA_ALL_DF[cv1_key]['T_LS'],ANA_ALL_DF[cv2_key]['T_LS'], ANA_ALL_DF[cv3_key]['T_LS'], ANA_ALL_DF[cv4_key]['T_LS']], axis = 0)
epc_T_LS_std = np.std([ANA_ALL_DF[cv0_key]['T_LS'], ANA_ALL_DF[cv1_key]['T_LS'],ANA_ALL_DF[cv2_key]['T_LS'], ANA_ALL_DF[cv3_key]['T_LS'], ANA_ALL_DF[cv4_key]['T_LS']], axis = 0)

epc_T_PC_mean = np.mean([ANA_ALL_DF[cv0_key]['T_PC'], ANA_ALL_DF[cv1_key]['T_PC'],ANA_ALL_DF[cv2_key]['T_PC'], ANA_ALL_DF[cv3_key]['T_PC'], ANA_ALL_DF[cv4_key]['T_PC']], axis = 0)
epc_T_PC_std = np.std([ANA_ALL_DF[cv0_key]['T_PC'], ANA_ALL_DF[cv1_key]['T_PC'],ANA_ALL_DF[cv2_key]['T_PC'], ANA_ALL_DF[cv3_key]['T_PC'], ANA_ALL_DF[cv4_key]['T_PC']], axis = 0)

epc_T_SC_mean = np.mean([ANA_ALL_DF[cv0_key]['T_SC'], ANA_ALL_DF[cv1_key]['T_SC'],ANA_ALL_DF[cv2_key]['T_SC'], ANA_ALL_DF[cv3_key]['T_SC'], ANA_ALL_DF[cv4_key]['T_SC']], axis = 0)
epc_T_SC_std = np.std([ANA_ALL_DF[cv0_key]['T_SC'], ANA_ALL_DF[cv1_key]['T_SC'],ANA_ALL_DF[cv2_key]['T_SC'], ANA_ALL_DF[cv3_key]['T_SC'], ANA_ALL_DF[cv4_key]['T_SC']], axis = 0)

epc_V_LS_mean = np.mean([ANA_ALL_DF[cv0_key]['V_LS'], ANA_ALL_DF[cv1_key]['V_LS'],ANA_ALL_DF[cv2_key]['V_LS'], ANA_ALL_DF[cv3_key]['V_LS'], ANA_ALL_DF[cv4_key]['V_LS']], axis = 0)
epc_V_LS_std = np.std([ANA_ALL_DF[cv0_key]['V_LS'], ANA_ALL_DF[cv1_key]['V_LS'],ANA_ALL_DF[cv2_key]['V_LS'], ANA_ALL_DF[cv3_key]['V_LS'], ANA_ALL_DF[cv4_key]['V_LS']], axis = 0)

epc_V_PC_mean = np.mean([ANA_ALL_DF[cv0_key]['V_PC'], ANA_ALL_DF[cv1_key]['V_PC'],ANA_ALL_DF[cv2_key]['V_PC'], ANA_ALL_DF[cv3_key]['V_PC'], ANA_ALL_DF[cv4_key]['V_PC']], axis = 0)
epc_V_PC_std = np.std([ANA_ALL_DF[cv0_key]['V_PC'], ANA_ALL_DF[cv1_key]['V_PC'],ANA_ALL_DF[cv2_key]['V_PC'], ANA_ALL_DF[cv3_key]['V_PC'], ANA_ALL_DF[cv4_key]['V_PC']], axis = 0)

epc_V_SC_mean = np.mean([ANA_ALL_DF[cv0_key]['V_SC'], ANA_ALL_DF[cv1_key]['V_SC'],ANA_ALL_DF[cv2_key]['V_SC'], ANA_ALL_DF[cv3_key]['V_SC'], ANA_ALL_DF[cv4_key]['V_SC']], axis = 0)
epc_V_SC_std = np.std([ANA_ALL_DF[cv0_key]['V_SC'], ANA_ALL_DF[cv1_key]['V_SC'],ANA_ALL_DF[cv2_key]['V_SC'], ANA_ALL_DF[cv3_key]['V_SC'], ANA_ALL_DF[cv4_key]['V_SC']], axis = 0)



epc_result = pd.DataFrame({
	'T_LS_mean' : epc_T_LS_mean, 'T_PC_mean' : epc_T_PC_mean, 'T_SC_mean' : epc_T_SC_mean, 
	'T_LS_std' : epc_T_LS_std, 'T_PC_std' : epc_T_PC_std, 'T_SC_std' : epc_T_SC_std, 
	'V_LS_mean' : epc_V_LS_mean, 'V_PC_mean' : epc_V_PC_mean, 'V_SC_mean' : epc_V_SC_mean, 
	'V_LS_std' : epc_V_LS_std, 'V_PC_std' : epc_V_PC_std, 'V_SC_std' : epc_V_SC_std,
})

epc_result[['T_LS_mean', 'T_LS_std', 'T_PC_mean', 'T_PC_std','T_SC_mean','T_SC_std', 'V_LS_mean', 'V_LS_std', 'V_PC_mean', 'V_PC_std','V_SC_mean','V_SC_std']].to_csv("/home01/k040a01/02.M3V5/{}_{}_{}_{}/RAY_ANA_DF.{}_{}_{}_{}.resDF".format(MJ_NAME, W_NAME, PPI_NAME, MISS_NAME, MJ_NAME, W_NAME, PPI_NAME, MISS_NAME))

"/home01/k040a01/02.M3V5/{}_{}_{}_{}/RAY_ANA_DF.{}_{}_{}_{}.resDF".format(MJ_NAME, W_NAME, PPI_NAME, MISS_NAME, MJ_NAME, W_NAME, PPI_NAME, MISS_NAME)
      


1) min loss # 401

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
round(epc_result.loc[KEY_EPC]['V_LS_mean'],3)
round(epc_result.loc[KEY_EPC]['V_LS_std'],3)



2) PC best  # 734

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
round(epc_result.loc[KEY_EPC]['V_PC_mean'],3)
round(epc_result.loc[KEY_EPC]['V_PC_std'],3)



3) SC best # 790

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
round(epc_result.loc[KEY_EPC]['V_SC_mean'],3)
round(epc_result.loc[KEY_EPC]['V_SC_std'],3)










###########################################
########### LOCAL ##################
###########################################


from ray.tune import ExperimentAnalysis






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
			expA = expA.view(-1,2)#### 다른점 
			expB = expB.view(-1,2)#### 다른점 
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
	dsn1_layers = [my_config['config/feat_size_0'].item(), my_config['config/feat_size_1'].item(), my_config['config/feat_size_2'].item()]
	dsn2_layers = [my_config['config/feat_size_0'].item(), my_config['config/feat_size_1'].item(), my_config['config/feat_size_2'].item()] 
	snp_layers = [my_config['config/feat_size_3'].item() , my_config['config/feat_size_4'].item()]
	inDrop = my_config['config/dropout_1'].item()
	Drop = my_config['config/dropout_2'].item()
	#
	best_model = MY_expGCN_parallel_model(
				G_chem_layer, T_test.gcn_drug1_F.shape[-1] , G_chem_hdim,
				G_exp_layer, 2, G_exp_hdim,
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






WORK_NAME = 'WORK_46' # 349
W_NAME = 'W46'
PRJ_NAME = 'M3V5'
MJ_NAME = 'M3V5'
MISS_NAME = 'MIS2'
PPI_NAME = '349'

WORK_PATH = '/home01/k040a01/02.M3V5/M3V5_W46_349_MIS2/'



PRJ_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/{}_{}_{}_{}/'.format(PRJ_NAME, W_NAME, PPI_NAME, MISS_NAME)
ANA_DF = pd.read_csv(PRJ_PATH+'RAY_ANA_DF.{}_{}_{}_{}.csv'.format(PRJ_NAME, W_NAME, PPI_NAME, MISS_NAME))
with open(PRJ_PATH+'RAY_ANA_DF.{}_{}_{}_{}.pickle'.format(PRJ_NAME, W_NAME, PPI_NAME, MISS_NAME), 'rb') as f:
	ANA_ALL_DF = pickle.load(f)


epc_result = pd.read_csv(PRJ_PATH+"RAY_ANA_DF.{}_{}_{}_{}.resDF".format(MJ_NAME, W_NAME, PPI_NAME, MISS_NAME, MJ_NAME, W_NAME, PPI_NAME, MISS_NAME))



my_config = ANA_DF.loc[0]


plot_three(
	"5CV",
	list(epc_result.T_LS_mean), list(epc_result.V_LS_mean), 
	list(epc_result.V_PC_mean), list(epc_result.V_PC_mean), 
	list(epc_result.V_SC_mean), list(epc_result.V_SC_mean), 
	PRJ_PATH, '{}_{}_{}'.format(MJ_NAME, MISS_NAME, WORK_NAME) )



1) min loss

min(epc_result.sort_values('V_LS_mean')['V_LS_mean']) ; min_VLS = min(epc_result.sort_values('V_LS_mean')['V_LS_mean'])
KEY_EPC = epc_result[epc_result.V_LS_mean == min_VLS].index.item()
checkpoint = "/checkpoint_"+str(KEY_EPC).zfill(6)

R_1_T_CV0, R_1_1_CV0, R_1_2_CV0, pred_1_CV0, ans_1_CV0 = TEST_CPU(PRJ_PATH, 'CV_0', my_config, PRJ_PATH, 'VLS_CV_0_model.pth', 'VLS')
R_1_T_CV1, R_1_1_CV1, R_1_2_CV1, pred_1_CV1, ans_1_CV1 = TEST_CPU(PRJ_PATH, 'CV_1', my_config, PRJ_PATH, 'VLS_CV_1_model.pth', 'VLS')
R_1_T_CV2, R_1_1_CV2, R_1_2_CV2, pred_1_CV2, ans_1_CV2 = TEST_CPU(PRJ_PATH, 'CV_2', my_config, PRJ_PATH, 'VLS_CV_2_model.pth', 'VLS')
R_1_T_CV3, R_1_1_CV3, R_1_2_CV3, pred_1_CV3, ans_1_CV3 = TEST_CPU(PRJ_PATH, 'CV_3', my_config, PRJ_PATH, 'VLS_CV_3_model.pth', 'VLS')
R_1_T_CV4, R_1_1_CV4, R_1_2_CV4, pred_1_CV4, ans_1_CV4 = TEST_CPU(PRJ_PATH, 'CV_4', my_config, PRJ_PATH, 'VLS_CV_4_model.pth', 'VLS')


list(ABCS_test_0.index) + list(ABCS_test_1.index) + list(ABCS_test_2.index) + list(ABCS_test_3.index) + list(ABCS_test_4.index) + 

ABCS_test_0 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV0_test'])]
ABCS_test_1 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV1_test'])]
ABCS_test_2 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV2_test'])]
ABCS_test_3 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV3_test'])]
ABCS_test_4 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV4_test'])]

test_data = pd.concat([ABCS_test_0,ABCS_test_1,ABCS_test_2,ABCS_test_3,ABCS_test_4])
pred_data = pred_1_CV0 + pred_1_CV1 + pred_1_CV2 + pred_1_CV3 + pred_1_CV4
ans_data = ans_1_CV0 + ans_1_CV1 + ans_1_CV2 + ans_1_CV3 + ans_1_CV4

test_data['tissue'] = [a.split('_')[1] for a in test_data.DrugCombCCLE]
test_data['ANS'] = ans_data
test_data['PRED'] = pred_data
test_data['DIFF'] = abs(test_data.ANS - test_data.PRED)
test_data['D_level'] = ABCS_test_result.DIFF.apply(lambda x:'under5' if x<5 else 'under10' if x < 10 else 'over10' if x < 100 else 'NA' )




tissue_set = ['CENTRAL_NERVOUS_SYSTEM', 'LUNG', 'BREAST', 'BONE', 'OVARY', 'PROSTATE', 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'LARGE_INTESTINE', 'SKIN', 'PLEURA', 'KIDNEY' ] # list(set(test_cell_df['tissue']))
color_set = ['#CF3476','#FF7514','#025669','#308446','#84C3BE','#D53032','#4ddcfd','#ffcd36','#ac8cff',"#0000ffff","#7bff68ff"] # "#1E1E1E"
color_dict = {a : color_set[tissue_set.index(a)] for a in tissue_set}


# make heatmap table

cells_info = pd.DataFrame(
    data = np.inf,
    columns=['tissue'],
    index=list(set(test_data.DrugCombCCLE))
)

cells_info['tissue'] = ['_'.join(a.split('_')[1:]) for a in list(set(test_data.DrugCombCCLE))]





# complexheatmap 포기하고 그냥 matplotlib 써보기 
from matplotlib import colors as mcolors
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib






my_heatmap_dot = pd.DataFrame(
    data = "NA", #np.inf,
    columns=list(set(test_data.DrugCombCCLE)),
    index=list(set(test_data.CID_CID))
)

c_c_c_list = list(test_data.CID_CID_CCLE)
c_c_c_list_set = list(set(test_data.CID_CID_CCLE))

for c_ind in range(len(c_c_c_list_set)) : # 대충 18만줄 대상으로 30분 소요 
    if c_ind%1000 == 0 : 
        print(str(c_ind)+'/'+str(len(c_c_c_list_set)) )
        datetime.now() 
    c_c_c = c_c_c_list_set[c_ind]
    tmp_res = test_data[test_data.CID_CID_CCLE==c_c_c]
    c_c = c_c_c.split('___')[0] + '___'+ c_c_c.split('___')[1]
    c = c_c_c.split('___')[2]
    if tmp_res.shape[0] ==1 :
        tmp_result = tmp_res['DIFF'].item()
    else  :
        tmp_result = np.mean(tmp_res['DIFF'])
    # 
    if tmp_result < 5 : 
        my_heatmap_dot.at[c_c, c] = "under5"
    elif  tmp_result < 10 : 
        my_heatmap_dot.at[c_c, c] = "under10"
    else :
        my_heatmap_dot.at[c_c, c] = "over10"
    


tissue_order = ['SKIN', 'BREAST', 'LUNG', 'OVARY', 'LARGE_INTESTINE', 'CENTRAL_NERVOUS_SYSTEM', 'KIDNEY', 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE','PROSTATE', 'BONE', 'PLEURA']
tiss_cell_dict ={a : list(cells_info[cells_info.tissue==a].index) for a in tissue_order }


my_map_skin = my_heatmap_dot[tiss_cell_dict['SKIN']]
my_map_breast = my_heatmap_dot[tiss_cell_dict['BREAST']]
my_map_lung = my_heatmap_dot[tiss_cell_dict['LUNG']]
my_map_ovary = my_heatmap_dot[tiss_cell_dict['OVARY']]
my_map_Lins = my_heatmap_dot[tiss_cell_dict['LARGE_INTESTINE']]
my_map_nerv = my_heatmap_dot[tiss_cell_dict['CENTRAL_NERVOUS_SYSTEM']]
my_map_kidn = my_heatmap_dot[tiss_cell_dict['KIDNEY']]
my_map_hema = my_heatmap_dot[tiss_cell_dict['HAEMATOPOIETIC_AND_LYMPHOID_TISSUE']]
my_map_prot = my_heatmap_dot[tiss_cell_dict['PROSTATE']]
my_map_bone = my_heatmap_dot[tiss_cell_dict['BONE']]
my_map_pleu = my_heatmap_dot[tiss_cell_dict['PLEURA']]


tissue_set = ['CENTRAL_NERVOUS_SYSTEM', 'LUNG', 'BREAST', 'BONE', 'OVARY', 'PROSTATE', 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'LARGE_INTESTINE', 'SKIN', 'PLEURA', 'KIDNEY' ] # list(set(test_cell_df['tissue']))
color_set = ['#CF3476','#FF7514','#025669','#308446','#84C3BE','#D53032','#4ddcfd','#ffcd36','#ac8cff',"#0000ffff","#7bff68ff"] # "#1E1E1E"
color_dict = {a : color_set[tissue_set.index(a)] for a in tissue_set}


tissue_map = pd.DataFrame(my_heatmap_dot.columns, columns= ['cell'])
tissue_map['tissue'] = ['_'.join(a.split('_')[1:]) for a in tissue_map.cell]
tissue_map['col'] = tissue_map['tissue'].map(color_dict)

col_colors = list(tissue_map['col'])



value_to_int = {j:i for i,j in enumerate(pd.unique(my_heatmap_dot.values.ravel()))} # like you did
n = len(value_to_int)     
# cmap = sns.color_palette("Pastel2", n) 
cmap = [mcolors.to_rgba('#f5f8fb'),mcolors.to_rgba('#ffe28a'),mcolors.to_rgba("#ffc1cb"),mcolors.to_rgba("#ff647e")]

gg = sns.clustermap(
    my_heatmap_dot.replace(value_to_int),  cmap=cmap, 
    figsize=(20,20),
    row_cluster=True, col_cluster = True, 
    col_colors = col_colors, dendrogram_ratio=0.2, yticklabels=False) 


plt.tight_layout()
plt.savefig("/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/example1.W46.pdf", bbox_inches='tight')
plt.close()


