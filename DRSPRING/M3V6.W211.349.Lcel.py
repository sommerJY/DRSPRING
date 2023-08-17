

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




# leave cell out 하기 위해서 일단 이걸로 확인하기 

data_nodup_df = pd.DataFrame({'setset' : data_no_dup.tolist(), 'cell' : data_no_dup_cells })
data_nodup_df2 = data_nodup_df.sort_values('cell')
data_nodup_df2 = data_nodup_df2.reset_index(drop =True)

data_nodup_df2['CHEM_A'] = [setset.split('___')[0] for setset in list(data_nodup_df2['setset'])]
data_nodup_df2['CHEM_B'] = [setset.split('___')[1] for setset in list(data_nodup_df2['setset'])]

# 25 CV  -> 92 CV

all_setset = list(data_nodup_df2.setset)

# 7 prob 

211_1
CV_0_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='BT474_BREAST']['setset']) ; CV_0_cell_train = [a for a in all_setset if a not in CV_0_cell_test]
CV_1_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='IPC298_SKIN']['setset']) ; CV_1_cell_train = [a for a in all_setset if a not in CV_1_cell_test]
CV_2_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='SKMEL2_SKIN']['setset']) ; CV_2_cell_train = [a for a in all_setset if a not in CV_2_cell_test]
CV_3_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='G361_SKIN']['setset']) ; CV_3_cell_train = [a for a in all_setset if a not in CV_3_cell_test]
CV_4_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='OVCAR4_OVARY']['setset']) ; CV_4_cell_train = [a for a in all_setset if a not in CV_4_cell_test]
CV_5_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='KM12_LARGE_INTESTINE']['setset']) ; CV_5_cell_train = [a for a in all_setset if a not in CV_5_cell_test]
CV_6_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='NCIH2122_LUNG']['setset']) ; CV_6_cell_train = [a for a in all_setset if a not in CV_6_cell_test]
CV_7_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='UACC62_SKIN']['setset']) ; CV_7_cell_train = [a for a in all_setset if a not in CV_7_cell_test]

211_2
CV_8_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='SF295_CENTRAL_NERVOUS_SYSTEM']['setset']) ; CV_8_cell_train = [a for a in all_setset if a not in CV_8_cell_test]
CV_9_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='OVCAR8_OVARY']['setset']) ; CV_9_cell_train = [a for a in all_setset if a not in CV_9_cell_test]
CV_10_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='NCIH226_LUNG']['setset']) ; CV_10_cell_train = [a for a in all_setset if a not in CV_10_cell_test]
CV_11_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='HT29_LARGE_INTESTINE']['setset']) ; CV_11_cell_train = [a for a in all_setset if a not in CV_11_cell_test]
CV_12_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='786O_KIDNEY']['setset']) ; CV_12_cell_train = [a for a in all_setset if a not in CV_12_cell_test]
CV_13_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='UHO1_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE']['setset']) ; CV_13_cell_train = [a for a in all_setset if a not in CV_13_cell_test]
CV_14_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='RPMI8226_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE']['setset']) ; CV_14_cell_train = [a for a in all_setset if a not in CV_14_cell_test]
CV_15_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='A375_SKIN']['setset']) ; CV_15_cell_train = [a for a in all_setset if a not in CV_15_cell_test]

211_3
CV_16_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='K562_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE']['setset']) ; CV_16_cell_train = [a for a in all_setset if a not in CV_16_cell_test]
CV_17_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='NCIH460_LUNG']['setset']) ; CV_17_cell_train = [a for a in all_setset if a not in CV_17_cell_test]
CV_18_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='U251MG_CENTRAL_NERVOUS_SYSTEM']['setset']) ; CV_18_cell_train = [a for a in all_setset if a not in CV_18_cell_test]
CV_19_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='PA1_OVARY']['setset']) ; CV_19_cell_train = [a for a in all_setset if a not in CV_19_cell_test]
CV_20_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='RKO_LARGE_INTESTINE']['setset']) ; CV_20_cell_train = [a for a in all_setset if a not in CV_20_cell_test]
CV_21_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='A427_LUNG']['setset']) ; CV_21_cell_train = [a for a in all_setset if a not in CV_21_cell_test]
CV_22_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='A498_KIDNEY']['setset']) ; CV_22_cell_train = [a for a in all_setset if a not in CV_22_cell_test]
CV_23_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='SW620_LARGE_INTESTINE']['setset']) ; CV_23_cell_train = [a for a in all_setset if a not in CV_23_cell_test]
CV_24_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='SKMES1_LUNG']['setset']) ; CV_24_cell_train = [a for a in all_setset if a not in CV_24_cell_test]
CV_25_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='T98G_CENTRAL_NERVOUS_SYSTEM']['setset']) ; CV_25_cell_train = [a for a in all_setset if a not in CV_25_cell_test]
CV_26_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='NCIH522_LUNG']['setset']) ; CV_26_cell_train = [a for a in all_setset if a not in CV_26_cell_test]
CV_27_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='LOVO_LARGE_INTESTINE']['setset']) ; CV_27_cell_train = [a for a in all_setset if a not in CV_27_cell_test]
CV_28_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='MELHO_SKIN']['setset']) ; CV_28_cell_train = [a for a in all_setset if a not in CV_28_cell_test]
CV_29_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='L1236_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE']['setset']) ; CV_29_cell_train = [a for a in all_setset if a not in CV_29_cell_test]
CV_30_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='ACHN_KIDNEY']['setset']) ; CV_30_cell_train = [a for a in all_setset if a not in CV_30_cell_test]
CV_31_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='MDAMB231_BREAST']['setset']) ; CV_31_cell_train = [a for a in all_setset if a not in CV_31_cell_test]

211_4
CV_32_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='UO31_KIDNEY']['setset']) ; CV_32_cell_train = [a for a in all_setset if a not in CV_32_cell_test]
CV_33_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='MEWO_SKIN']['setset']) ; CV_33_cell_train = [a for a in all_setset if a not in CV_33_cell_test]
CV_34_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='UACC257_SKIN']['setset']) ; CV_34_cell_train = [a for a in all_setset if a not in CV_34_cell_test]
CV_35_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='SR786_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE']['setset']) ; CV_35_cell_train = [a for a in all_setset if a not in CV_35_cell_test]
CV_36_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='A101D_SKIN']['setset']) ; CV_36_cell_train = [a for a in all_setset if a not in CV_36_cell_test]
CV_37_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='NCIH23_LUNG']['setset']) ; CV_37_cell_train = [a for a in all_setset if a not in CV_37_cell_test]
CV_38_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='RVH421_SKIN']['setset']) ; CV_38_cell_train = [a for a in all_setset if a not in CV_38_cell_test]
CV_39_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='MDAMB175VII_BREAST']['setset']) ; CV_39_cell_train = [a for a in all_setset if a not in CV_39_cell_test]

211_5
CV_40_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='COLO829_SKIN']['setset']) ; CV_40_cell_train = [a for a in all_setset if a not in CV_40_cell_test]
CV_41_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='NIHOVCAR3_OVARY']['setset']) ; CV_41_cell_train = [a for a in all_setset if a not in CV_41_cell_test]
CV_42_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='SKMEL5_SKIN']['setset']) ; CV_42_cell_train = [a for a in all_setset if a not in CV_42_cell_test]
CV_43_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='LOXIMVI_SKIN']['setset']) ; CV_43_cell_train = [a for a in all_setset if a not in CV_43_cell_test]
CV_44_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='A673_BONE']['setset']) ; CV_44_cell_train = [a for a in all_setset if a not in CV_44_cell_test]
CV_45_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='MDAMB468_BREAST']['setset']) ; CV_45_cell_train = [a for a in all_setset if a not in CV_45_cell_test]
CV_46_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='SW837_LARGE_INTESTINE']['setset']) ; CV_46_cell_train = [a for a in all_setset if a not in CV_46_cell_test]
CV_47_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='A2780_OVARY']['setset']) ; CV_47_cell_train = [a for a in all_setset if a not in CV_47_cell_test]

211_6
CV_48_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='HCC1419_BREAST']['setset']) ; CV_48_cell_train = [a for a in all_setset if a not in CV_48_cell_test]
CV_49_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='HOP92_LUNG']['setset']) ; CV_49_cell_train = [a for a in all_setset if a not in CV_49_cell_test]
CV_50_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='WM115_SKIN']['setset']) ; CV_50_cell_train = [a for a in all_setset if a not in CV_50_cell_test]
CV_51_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='CAKI1_KIDNEY']['setset']) ; CV_51_cell_train = [a for a in all_setset if a not in CV_51_cell_test]
CV_52_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='MDAMB361_BREAST']['setset']) ; CV_52_cell_train = [a for a in all_setset if a not in CV_52_cell_test]
CV_53_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='SF268_CENTRAL_NERVOUS_SYSTEM']['setset']) ; CV_53_cell_train = [a for a in all_setset if a not in CV_53_cell_test]
CV_54_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='A2058_SKIN']['setset']) ; CV_54_cell_train = [a for a in all_setset if a not in CV_54_cell_test]
CV_55_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='COLO800_SKIN']['setset']) ; CV_55_cell_train = [a for a in all_setset if a not in CV_55_cell_test]

211_7
CV_56_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='IGROV1_OVARY']['setset']) ; CV_56_cell_train = [a for a in all_setset if a not in CV_56_cell_test]
CV_57_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='MSTO211H_PLEURA']['setset']) ; CV_57_cell_train = [a for a in all_setset if a not in CV_57_cell_test]
CV_58_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='HCT15_LARGE_INTESTINE']['setset']) ; CV_58_cell_train = [a for a in all_setset if a not in CV_58_cell_test]
CV_59_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='HCT116_LARGE_INTESTINE']['setset']) ; CV_59_cell_train = [a for a in all_setset if a not in CV_59_cell_test]
CV_60_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='VCAP_PROSTATE']['setset']) ; CV_60_cell_train = [a for a in all_setset if a not in CV_60_cell_test]
CV_61_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='EKVX_LUNG']['setset']) ; CV_61_cell_train = [a for a in all_setset if a not in CV_61_cell_test]
CV_62_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='ES2_OVARY']['setset']) ; CV_62_cell_train = [a for a in all_setset if a not in CV_62_cell_test]
CV_63_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='OV90_OVARY']['setset']) ; CV_63_cell_train = [a for a in all_setset if a not in CV_63_cell_test]

211_8
CV_64_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='HCC1500_BREAST']['setset']) ; CV_64_cell_train = [a for a in all_setset if a not in CV_64_cell_test]
CV_65_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='T47D_BREAST']['setset']) ; CV_65_cell_train = [a for a in all_setset if a not in CV_65_cell_test]
CV_66_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='UWB1289_OVARY']['setset']) ; CV_66_cell_train = [a for a in all_setset if a not in CV_66_cell_test]
CV_67_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='SKMEL28_SKIN']['setset']) ; CV_67_cell_train = [a for a in all_setset if a not in CV_67_cell_test]
CV_68_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='MCF7_BREAST']['setset']) ; CV_68_cell_train = [a for a in all_setset if a not in CV_68_cell_test]
CV_69_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='SNB75_CENTRAL_NERVOUS_SYSTEM']['setset']) ; CV_69_cell_train = [a for a in all_setset if a not in CV_69_cell_test]
CV_70_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='HOP62_LUNG']['setset']) ; CV_70_cell_train = [a for a in all_setset if a not in CV_70_cell_test]
CV_71_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='NCIH1650_LUNG']['setset']) ; CV_71_cell_train = [a for a in all_setset if a not in CV_71_cell_test]

211_9
CV_72_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='KPL1_BREAST']['setset']) ; CV_72_cell_train = [a for a in all_setset if a not in CV_72_cell_test]
CV_73_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='SKMEL30_SKIN']['setset']) ; CV_73_cell_train = [a for a in all_setset if a not in CV_73_cell_test]
CV_74_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='PC3_PROSTATE']['setset']) ; CV_74_cell_train = [a for a in all_setset if a not in CV_74_cell_test]
CV_75_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='CAMA1_BREAST']['setset']) ; CV_75_cell_train = [a for a in all_setset if a not in CV_75_cell_test]
CV_76_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='OVCAR5_OVARY']['setset']) ; CV_76_cell_train = [a for a in all_setset if a not in CV_76_cell_test]
CV_77_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='MDAMB436_BREAST']['setset']) ; CV_77_cell_train = [a for a in all_setset if a not in CV_77_cell_test]
CV_78_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='RPMI7951_SKIN']['setset']) ; CV_78_cell_train = [a for a in all_setset if a not in CV_78_cell_test]
CV_79_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='NCIH520_LUNG']['setset']) ; CV_79_cell_train = [a for a in all_setset if a not in CV_79_cell_test]

211_10
CV_80_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='SKOV3_OVARY']['setset']) ; CV_80_cell_train = [a for a in all_setset if a not in CV_80_cell_test]
CV_81_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='ZR751_BREAST']['setset']) ; CV_81_cell_train = [a for a in all_setset if a not in CV_81_cell_test]
CV_82_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='BT549_BREAST']['setset']) ; CV_82_cell_train = [a for a in all_setset if a not in CV_82_cell_test]
CV_83_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='HS578T_BREAST']['setset']) ; CV_83_cell_train = [a for a in all_setset if a not in CV_83_cell_test]
CV_84_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='HT144_SKIN']['setset']) ; CV_84_cell_train = [a for a in all_setset if a not in CV_84_cell_test]
CV_85_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='MALME3M_SKIN']['setset']) ; CV_85_cell_train = [a for a in all_setset if a not in CV_85_cell_test]
CV_86_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='SF539_CENTRAL_NERVOUS_SYSTEM']['setset']) ; CV_86_cell_train = [a for a in all_setset if a not in CV_86_cell_test]
CV_87_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='CAOV3_OVARY']['setset']) ; CV_87_cell_train = [a for a in all_setset if a not in CV_87_cell_test]

211_11
CV_88_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='DLD1_LARGE_INTESTINE']['setset']) ; CV_88_cell_train = [a for a in all_setset if a not in CV_88_cell_test]
CV_89_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='A549_LUNG']['setset']) ; CV_89_cell_train = [a for a in all_setset if a not in CV_89_cell_test]
CV_90_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='COLO792_SKIN']['setset']) ; CV_90_cell_train = [a for a in all_setset if a not in CV_90_cell_test]
CV_91_cell_test = list(data_nodup_df2[data_nodup_df2.cell=='UACC812_BREAST']['setset']) ; CV_91_cell_train = [a for a in all_setset if a not in CV_91_cell_test]





# 주어진 환경 맞춰서 5CV만 돌릴거라서 수정버전 
def prepare_data_GCN(CV_num, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, 
MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, 
cell_one_hot, MY_syn_RE2, norm ) : 
	# 
	# CV_num = 0
	train_key = 'CV_{}_cell_train'.format(CV_num)
	test_key = 'CV_{}_cell_test'.format(CV_num)
	train_cell = globals()[train_key] 
	test_cell = globals()[test_key] 
	# 
	#
	ABCS_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(train_cell)]
	ABCS_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(test_cell)]
	ABCS_train_ind = list(ABCS_train.index)
	ABCS_train = ABCS_train.loc[ABCS_train_ind]
	#
	train_ind = list(ABCS_train.index)
	random.shuffle(train_ind)
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
	return train_data, test_data






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

211_1
train_data_0, test_data_0 = prepare_data_GCN(0, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_1, test_data_1 = prepare_data_GCN(1, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_2, test_data_2 = prepare_data_GCN(2, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_3, test_data_3 = prepare_data_GCN(3, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_4, test_data_4 = prepare_data_GCN(4, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_5, test_data_5 = prepare_data_GCN(5, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_6, test_data_6 = prepare_data_GCN(6, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_7, test_data_7 = prepare_data_GCN(7, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)

211_2
train_data_8, test_data_8 = prepare_data_GCN(8, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_9, test_data_9 = prepare_data_GCN(9, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_10, test_data_10 = prepare_data_GCN(10, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_11, test_data_11 = prepare_data_GCN(11, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_12, test_data_12 = prepare_data_GCN(12, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_13, test_data_13 = prepare_data_GCN(13, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_14, test_data_14 = prepare_data_GCN(14, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_15, test_data_15 = prepare_data_GCN(15, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)

211_3
train_data_16, test_data_16 = prepare_data_GCN(16, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_17, test_data_17 = prepare_data_GCN(17, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_18, test_data_18 = prepare_data_GCN(18, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_19, test_data_19 = prepare_data_GCN(19, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_20, test_data_20 = prepare_data_GCN(20, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_21, test_data_21 = prepare_data_GCN(21, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_22, test_data_22 = prepare_data_GCN(22, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_23, test_data_23 = prepare_data_GCN(23, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_24, test_data_24 = prepare_data_GCN(24, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_25, test_data_25 = prepare_data_GCN(25, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_26, test_data_26 = prepare_data_GCN(26, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_27, test_data_27 = prepare_data_GCN(27, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_28, test_data_28 = prepare_data_GCN(28, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_29, test_data_29 = prepare_data_GCN(29, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_30, test_data_30 = prepare_data_GCN(30, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_31, test_data_31 = prepare_data_GCN(31, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)

211_4
train_data_32, test_data_32 = prepare_data_GCN(32, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_33, test_data_33 = prepare_data_GCN(33, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_34, test_data_34 = prepare_data_GCN(34, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_35, test_data_35 = prepare_data_GCN(35, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_36, test_data_36 = prepare_data_GCN(36, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_37, test_data_37 = prepare_data_GCN(37, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_38, test_data_38 = prepare_data_GCN(38, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_39, test_data_39 = prepare_data_GCN(39, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)

211_5
train_data_40, test_data_40 = prepare_data_GCN(40, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_41, test_data_41 = prepare_data_GCN(41, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_42, test_data_42 = prepare_data_GCN(42, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_43, test_data_43 = prepare_data_GCN(43, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_44, test_data_44 = prepare_data_GCN(44, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_45, test_data_45 = prepare_data_GCN(45, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_46, test_data_46 = prepare_data_GCN(46, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_47, test_data_47 = prepare_data_GCN(47, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)

211_6
train_data_48, test_data_48 = prepare_data_GCN(48, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_49, test_data_49 = prepare_data_GCN(49, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_50, test_data_50 = prepare_data_GCN(50, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_51, test_data_51 = prepare_data_GCN(51, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_52, test_data_52 = prepare_data_GCN(52, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_53, test_data_53 = prepare_data_GCN(53, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_54, test_data_54 = prepare_data_GCN(54, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_55, test_data_55 = prepare_data_GCN(55, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)

211_7
train_data_56, test_data_56 = prepare_data_GCN(56, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_57, test_data_57 = prepare_data_GCN(57, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_58, test_data_58 = prepare_data_GCN(58, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_59, test_data_59 = prepare_data_GCN(59, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_60, test_data_60 = prepare_data_GCN(60, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_61, test_data_61 = prepare_data_GCN(61, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_62, test_data_62 = prepare_data_GCN(62, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_63, test_data_63 = prepare_data_GCN(63, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)

211_8
train_data_64, test_data_64 = prepare_data_GCN(64, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_65, test_data_65 = prepare_data_GCN(65, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_66, test_data_66 = prepare_data_GCN(66, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_67, test_data_67 = prepare_data_GCN(67, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_68, test_data_68 = prepare_data_GCN(68, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_69, test_data_69 = prepare_data_GCN(69, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_70, test_data_70 = prepare_data_GCN(70, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_71, test_data_71 = prepare_data_GCN(71, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)

211_9
train_data_72, test_data_72 = prepare_data_GCN(72, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_73, test_data_73 = prepare_data_GCN(73, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_74, test_data_74 = prepare_data_GCN(74, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_75, test_data_75 = prepare_data_GCN(75, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_76, test_data_76 = prepare_data_GCN(76, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_77, test_data_77 = prepare_data_GCN(77, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_78, test_data_78 = prepare_data_GCN(78, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_79, test_data_79 = prepare_data_GCN(79, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)

211_10
train_data_80, test_data_80 = prepare_data_GCN(80, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_81, test_data_81 = prepare_data_GCN(81, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_82, test_data_82 = prepare_data_GCN(82, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_83, test_data_83 = prepare_data_GCN(83, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_84, test_data_84 = prepare_data_GCN(84, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_85, test_data_85 = prepare_data_GCN(85, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_86, test_data_86 = prepare_data_GCN(86, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_87, test_data_87 = prepare_data_GCN(87, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)

211_11
train_data_88, test_data_88 = prepare_data_GCN(88, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_89, test_data_89 = prepare_data_GCN(89, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_90, test_data_90 = prepare_data_GCN(90, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)
train_data_91, test_data_91 = prepare_data_GCN(91, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, cell_one_hot, MY_syn_RE2, norm)



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
LOSS_WEIGHT_5 = get_loss_weight(5)
LOSS_WEIGHT_6 = get_loss_weight(6)
LOSS_WEIGHT_7 = get_loss_weight(7)

LOSS_WEIGHT_8 = get_loss_weight(8)
LOSS_WEIGHT_9 = get_loss_weight(9)
LOSS_WEIGHT_10 = get_loss_weight(10)
LOSS_WEIGHT_11 = get_loss_weight(11)
LOSS_WEIGHT_12 = get_loss_weight(12)
LOSS_WEIGHT_13 = get_loss_weight(13)
LOSS_WEIGHT_14 = get_loss_weight(14)
LOSS_WEIGHT_15 = get_loss_weight(15)

LOSS_WEIGHT_16 = get_loss_weight(16)
LOSS_WEIGHT_17 = get_loss_weight(17)
LOSS_WEIGHT_18 = get_loss_weight(18)
LOSS_WEIGHT_19 = get_loss_weight(19)
LOSS_WEIGHT_20 = get_loss_weight(20)
LOSS_WEIGHT_21 = get_loss_weight(21)
LOSS_WEIGHT_22 = get_loss_weight(22)
LOSS_WEIGHT_23 = get_loss_weight(23)

LOSS_WEIGHT_24 = get_loss_weight(24)
LOSS_WEIGHT_25 = get_loss_weight(25)
LOSS_WEIGHT_26 = get_loss_weight(26)
LOSS_WEIGHT_27 = get_loss_weight(27)
LOSS_WEIGHT_28 = get_loss_weight(28)
LOSS_WEIGHT_29 = get_loss_weight(29)
LOSS_WEIGHT_30 = get_loss_weight(30)
LOSS_WEIGHT_31 = get_loss_weight(31)

LOSS_WEIGHT_32 = get_loss_weight(32)
LOSS_WEIGHT_33 = get_loss_weight(33)
LOSS_WEIGHT_34 = get_loss_weight(34)
LOSS_WEIGHT_35 = get_loss_weight(35)
LOSS_WEIGHT_36 = get_loss_weight(36)
LOSS_WEIGHT_37 = get_loss_weight(37)
LOSS_WEIGHT_38 = get_loss_weight(38)
LOSS_WEIGHT_39 = get_loss_weight(39)

LOSS_WEIGHT_40 = get_loss_weight(40)
LOSS_WEIGHT_41 = get_loss_weight(41)
LOSS_WEIGHT_42 = get_loss_weight(42)
LOSS_WEIGHT_43 = get_loss_weight(43)
LOSS_WEIGHT_44 = get_loss_weight(44)
LOSS_WEIGHT_45 = get_loss_weight(45)
LOSS_WEIGHT_46 = get_loss_weight(46)
LOSS_WEIGHT_47 = get_loss_weight(47)

LOSS_WEIGHT_48 = get_loss_weight(48)
LOSS_WEIGHT_49 = get_loss_weight(49)
LOSS_WEIGHT_50 = get_loss_weight(50)
LOSS_WEIGHT_51 = get_loss_weight(51)
LOSS_WEIGHT_52 = get_loss_weight(52)
LOSS_WEIGHT_53 = get_loss_weight(53)
LOSS_WEIGHT_54 = get_loss_weight(54)
LOSS_WEIGHT_55 = get_loss_weight(55)

LOSS_WEIGHT_56 = get_loss_weight(56)
LOSS_WEIGHT_57 = get_loss_weight(57)
LOSS_WEIGHT_58 = get_loss_weight(58)
LOSS_WEIGHT_59 = get_loss_weight(59)
LOSS_WEIGHT_60 = get_loss_weight(60)
LOSS_WEIGHT_61 = get_loss_weight(61)
LOSS_WEIGHT_62 = get_loss_weight(62)
LOSS_WEIGHT_63 = get_loss_weight(63)

LOSS_WEIGHT_64 = get_loss_weight(64)
LOSS_WEIGHT_65 = get_loss_weight(65)
LOSS_WEIGHT_66 = get_loss_weight(66)
LOSS_WEIGHT_67 = get_loss_weight(67)
LOSS_WEIGHT_68 = get_loss_weight(68)
LOSS_WEIGHT_69 = get_loss_weight(69)
LOSS_WEIGHT_70 = get_loss_weight(70)
LOSS_WEIGHT_71 = get_loss_weight(71)

LOSS_WEIGHT_72 = get_loss_weight(72)
LOSS_WEIGHT_73 = get_loss_weight(73)
LOSS_WEIGHT_74 = get_loss_weight(74)
LOSS_WEIGHT_75 = get_loss_weight(75)
LOSS_WEIGHT_76 = get_loss_weight(76)
LOSS_WEIGHT_77 = get_loss_weight(77)
LOSS_WEIGHT_78 = get_loss_weight(78)
LOSS_WEIGHT_79 = get_loss_weight(79)

LOSS_WEIGHT_80 = get_loss_weight(80)
LOSS_WEIGHT_81 = get_loss_weight(81)
LOSS_WEIGHT_82 = get_loss_weight(82)
LOSS_WEIGHT_83 = get_loss_weight(83)
LOSS_WEIGHT_84 = get_loss_weight(84)
LOSS_WEIGHT_85 = get_loss_weight(85)
LOSS_WEIGHT_86 = get_loss_weight(86)
LOSS_WEIGHT_87 = get_loss_weight(87)

LOSS_WEIGHT_88 = get_loss_weight(88)
LOSS_WEIGHT_89 = get_loss_weight(89)
LOSS_WEIGHT_90 = get_loss_weight(90)
LOSS_WEIGHT_91 = get_loss_weight(91)







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





too much, have to split 

211_1
T_train_0, T_test_0 = make_merged_data(0) ; RAY_loss_weight_0 = ray.put(LOSS_WEIGHT_0) ; RAY_train_0 = ray.put(T_train_0) ; RAY_test_0 = ray.put(T_test_0)
T_train_1, T_test_1 = make_merged_data(1) ; RAY_loss_weight_1 = ray.put(LOSS_WEIGHT_1) ; RAY_train_1 = ray.put(T_train_1) ; RAY_test_1 = ray.put(T_test_1)
T_train_2, T_test_2 = make_merged_data(2) ; RAY_loss_weight_2 = ray.put(LOSS_WEIGHT_2) ; RAY_train_2 = ray.put(T_train_2) ; RAY_test_2 = ray.put(T_test_2)
T_train_3, T_test_3 = make_merged_data(3) ; RAY_loss_weight_3 = ray.put(LOSS_WEIGHT_3) ; RAY_train_3 = ray.put(T_train_3) ; RAY_test_3 = ray.put(T_test_3)
T_train_4, T_test_4 = make_merged_data(4) ; RAY_loss_weight_4 = ray.put(LOSS_WEIGHT_4) ; RAY_train_4 = ray.put(T_train_4) ; RAY_test_4 = ray.put(T_test_4)
T_train_5, T_test_5 = make_merged_data(5) ; RAY_loss_weight_5 = ray.put(LOSS_WEIGHT_5) ; RAY_train_5 = ray.put(T_train_5) ; RAY_test_5 = ray.put(T_test_5)
T_train_6, T_test_6 = make_merged_data(6) ; RAY_loss_weight_6 = ray.put(LOSS_WEIGHT_6) ; RAY_train_6 = ray.put(T_train_6) ; RAY_test_6 = ray.put(T_test_6)
T_train_7, T_test_7 = make_merged_data(7) ; RAY_loss_weight_7 = ray.put(LOSS_WEIGHT_7) ; RAY_train_7 = ray.put(T_train_7) ; RAY_test_7 = ray.put(T_test_7)

211_2
T_train_8, T_test_8 = make_merged_data(8) ; RAY_loss_weight_8 = ray.put(LOSS_WEIGHT_8) ; RAY_train_8 = ray.put(T_train_8) ; RAY_test_8 = ray.put(T_test_8)
T_train_9, T_test_9 = make_merged_data(9) ; RAY_loss_weight_9 = ray.put(LOSS_WEIGHT_9) ; RAY_train_9 = ray.put(T_train_9) ; RAY_test_9 = ray.put(T_test_9)
T_train_10, T_test_10 = make_merged_data(10) ; RAY_loss_weight_10 = ray.put(LOSS_WEIGHT_10) ; RAY_train_10 = ray.put(T_train_10) ; RAY_test_10 = ray.put(T_test_10)
T_train_11, T_test_11 = make_merged_data(11) ; RAY_loss_weight_11 = ray.put(LOSS_WEIGHT_11) ; RAY_train_11 = ray.put(T_train_11) ; RAY_test_11 = ray.put(T_test_11)
T_train_12, T_test_12 = make_merged_data(12) ; RAY_loss_weight_12 = ray.put(LOSS_WEIGHT_12) ; RAY_train_12 = ray.put(T_train_12) ; RAY_test_12 = ray.put(T_test_12)
T_train_13, T_test_13 = make_merged_data(13) ; RAY_loss_weight_13 = ray.put(LOSS_WEIGHT_13) ; RAY_train_13 = ray.put(T_train_13) ; RAY_test_13 = ray.put(T_test_13)
T_train_14, T_test_14 = make_merged_data(14) ; RAY_loss_weight_14 = ray.put(LOSS_WEIGHT_14) ; RAY_train_14 = ray.put(T_train_14) ; RAY_test_14 = ray.put(T_test_14)
T_train_15, T_test_15 = make_merged_data(15) ; RAY_loss_weight_15 = ray.put(LOSS_WEIGHT_15) ; RAY_train_15 = ray.put(T_train_15) ; RAY_test_15 = ray.put(T_test_15)

211_3
T_train_16, T_test_16 = make_merged_data(16) ; RAY_loss_weight_16 = ray.put(LOSS_WEIGHT_16) ; RAY_train_16 = ray.put(T_train_16) ; RAY_test_16 = ray.put(T_test_16)
T_train_17, T_test_17 = make_merged_data(17) ; RAY_loss_weight_17 = ray.put(LOSS_WEIGHT_17) ; RAY_train_17 = ray.put(T_train_17) ; RAY_test_17 = ray.put(T_test_17)
T_train_18, T_test_18 = make_merged_data(18) ; RAY_loss_weight_18 = ray.put(LOSS_WEIGHT_18) ; RAY_train_18 = ray.put(T_train_18) ; RAY_test_18 = ray.put(T_test_18)
T_train_19, T_test_19 = make_merged_data(19) ; RAY_loss_weight_19 = ray.put(LOSS_WEIGHT_19) ; RAY_train_19 = ray.put(T_train_19) ; RAY_test_19 = ray.put(T_test_19)
T_train_20, T_test_20 = make_merged_data(20) ; RAY_loss_weight_20 = ray.put(LOSS_WEIGHT_20) ; RAY_train_20 = ray.put(T_train_20) ; RAY_test_20 = ray.put(T_test_20)
T_train_21, T_test_21 = make_merged_data(21) ; RAY_loss_weight_21 = ray.put(LOSS_WEIGHT_21) ; RAY_train_21 = ray.put(T_train_21) ; RAY_test_21 = ray.put(T_test_21)
T_train_22, T_test_22 = make_merged_data(22) ; RAY_loss_weight_22 = ray.put(LOSS_WEIGHT_22) ; RAY_train_22 = ray.put(T_train_22) ; RAY_test_22 = ray.put(T_test_22)
T_train_23, T_test_23 = make_merged_data(23) ; RAY_loss_weight_23 = ray.put(LOSS_WEIGHT_23) ; RAY_train_23 = ray.put(T_train_23) ; RAY_test_23 = ray.put(T_test_23)
T_train_24, T_test_24 = make_merged_data(24) ; RAY_loss_weight_24 = ray.put(LOSS_WEIGHT_24) ; RAY_train_24 = ray.put(T_train_24) ; RAY_test_24 = ray.put(T_test_24)
T_train_25, T_test_25 = make_merged_data(25) ; RAY_loss_weight_25 = ray.put(LOSS_WEIGHT_25) ; RAY_train_25 = ray.put(T_train_25) ; RAY_test_25 = ray.put(T_test_25)
T_train_26, T_test_26 = make_merged_data(26) ; RAY_loss_weight_26 = ray.put(LOSS_WEIGHT_26) ; RAY_train_26 = ray.put(T_train_26) ; RAY_test_26 = ray.put(T_test_26)
T_train_27, T_test_27 = make_merged_data(27) ; RAY_loss_weight_27 = ray.put(LOSS_WEIGHT_27) ; RAY_train_27 = ray.put(T_train_27) ; RAY_test_27 = ray.put(T_test_27)
T_train_28, T_test_28 = make_merged_data(28) ; RAY_loss_weight_28 = ray.put(LOSS_WEIGHT_28) ; RAY_train_28 = ray.put(T_train_28) ; RAY_test_28 = ray.put(T_test_28)
T_train_29, T_test_29 = make_merged_data(29) ; RAY_loss_weight_29 = ray.put(LOSS_WEIGHT_29) ; RAY_train_29 = ray.put(T_train_29) ; RAY_test_29 = ray.put(T_test_29)
T_train_30, T_test_30 = make_merged_data(30) ; RAY_loss_weight_30 = ray.put(LOSS_WEIGHT_30) ; RAY_train_30 = ray.put(T_train_30) ; RAY_test_30 = ray.put(T_test_30)
T_train_31, T_test_31 = make_merged_data(31) ; RAY_loss_weight_31 = ray.put(LOSS_WEIGHT_31) ; RAY_train_31 = ray.put(T_train_31) ; RAY_test_31 = ray.put(T_test_31)

211_4
T_train_32, T_test_32 = make_merged_data(32) ; RAY_loss_weight_32 = ray.put(LOSS_WEIGHT_32) ; RAY_train_32 = ray.put(T_train_32) ; RAY_test_32 = ray.put(T_test_32)
T_train_33, T_test_33 = make_merged_data(33) ; RAY_loss_weight_33 = ray.put(LOSS_WEIGHT_33) ; RAY_train_33 = ray.put(T_train_33) ; RAY_test_33 = ray.put(T_test_33)
T_train_34, T_test_34 = make_merged_data(34) ; RAY_loss_weight_34 = ray.put(LOSS_WEIGHT_34) ; RAY_train_34 = ray.put(T_train_34) ; RAY_test_34 = ray.put(T_test_34)
T_train_35, T_test_35 = make_merged_data(35) ; RAY_loss_weight_35 = ray.put(LOSS_WEIGHT_35) ; RAY_train_35 = ray.put(T_train_35) ; RAY_test_35 = ray.put(T_test_35)
T_train_36, T_test_36 = make_merged_data(36) ; RAY_loss_weight_36 = ray.put(LOSS_WEIGHT_36) ; RAY_train_36 = ray.put(T_train_36) ; RAY_test_36 = ray.put(T_test_36)
T_train_37, T_test_37 = make_merged_data(37) ; RAY_loss_weight_37 = ray.put(LOSS_WEIGHT_37) ; RAY_train_37 = ray.put(T_train_37) ; RAY_test_37 = ray.put(T_test_37)
T_train_38, T_test_38 = make_merged_data(38) ; RAY_loss_weight_38 = ray.put(LOSS_WEIGHT_38) ; RAY_train_38 = ray.put(T_train_38) ; RAY_test_38 = ray.put(T_test_38)
T_train_39, T_test_39 = make_merged_data(39) ; RAY_loss_weight_39 = ray.put(LOSS_WEIGHT_39) ; RAY_train_39 = ray.put(T_train_39) ; RAY_test_39 = ray.put(T_test_39)

211_5
T_train_40, T_test_40 = make_merged_data(40) ; RAY_loss_weight_40 = ray.put(LOSS_WEIGHT_40) ; RAY_train_40 = ray.put(T_train_40) ; RAY_test_40 = ray.put(T_test_40)
T_train_41, T_test_41 = make_merged_data(41) ; RAY_loss_weight_41 = ray.put(LOSS_WEIGHT_41) ; RAY_train_41 = ray.put(T_train_41) ; RAY_test_41 = ray.put(T_test_41)
T_train_42, T_test_42 = make_merged_data(42) ; RAY_loss_weight_42 = ray.put(LOSS_WEIGHT_42) ; RAY_train_42 = ray.put(T_train_42) ; RAY_test_42 = ray.put(T_test_42)
T_train_43, T_test_43 = make_merged_data(43) ; RAY_loss_weight_43 = ray.put(LOSS_WEIGHT_43) ; RAY_train_43 = ray.put(T_train_43) ; RAY_test_43 = ray.put(T_test_43)
T_train_44, T_test_44 = make_merged_data(44) ; RAY_loss_weight_44 = ray.put(LOSS_WEIGHT_44) ; RAY_train_44 = ray.put(T_train_44) ; RAY_test_44 = ray.put(T_test_44)
T_train_45, T_test_45 = make_merged_data(45) ; RAY_loss_weight_45 = ray.put(LOSS_WEIGHT_45) ; RAY_train_45 = ray.put(T_train_45) ; RAY_test_45 = ray.put(T_test_45)
T_train_46, T_test_46 = make_merged_data(46) ; RAY_loss_weight_46 = ray.put(LOSS_WEIGHT_46) ; RAY_train_46 = ray.put(T_train_46) ; RAY_test_46 = ray.put(T_test_46)
T_train_47, T_test_47 = make_merged_data(47) ; RAY_loss_weight_47 = ray.put(LOSS_WEIGHT_47) ; RAY_train_47 = ray.put(T_train_47) ; RAY_test_47 = ray.put(T_test_47)

211_6
T_train_48, T_test_48 = make_merged_data(48) ; RAY_loss_weight_48 = ray.put(LOSS_WEIGHT_48) ; RAY_train_48 = ray.put(T_train_48) ; RAY_test_48 = ray.put(T_test_48)
T_train_49, T_test_49 = make_merged_data(49) ; RAY_loss_weight_49 = ray.put(LOSS_WEIGHT_49) ; RAY_train_49 = ray.put(T_train_49) ; RAY_test_49 = ray.put(T_test_49)
T_train_50, T_test_50 = make_merged_data(50) ; RAY_loss_weight_50 = ray.put(LOSS_WEIGHT_50) ; RAY_train_50 = ray.put(T_train_50) ; RAY_test_50 = ray.put(T_test_50)
T_train_51, T_test_51 = make_merged_data(51) ; RAY_loss_weight_51 = ray.put(LOSS_WEIGHT_51) ; RAY_train_51 = ray.put(T_train_51) ; RAY_test_51 = ray.put(T_test_51)
T_train_52, T_test_52 = make_merged_data(52) ; RAY_loss_weight_52 = ray.put(LOSS_WEIGHT_52) ; RAY_train_52 = ray.put(T_train_52) ; RAY_test_52 = ray.put(T_test_52)
T_train_53, T_test_53 = make_merged_data(53) ; RAY_loss_weight_53 = ray.put(LOSS_WEIGHT_53) ; RAY_train_53 = ray.put(T_train_53) ; RAY_test_53 = ray.put(T_test_53)
T_train_54, T_test_54 = make_merged_data(54) ; RAY_loss_weight_54 = ray.put(LOSS_WEIGHT_54) ; RAY_train_54 = ray.put(T_train_54) ; RAY_test_54 = ray.put(T_test_54)
T_train_55, T_test_55 = make_merged_data(55) ; RAY_loss_weight_55 = ray.put(LOSS_WEIGHT_55) ; RAY_train_55 = ray.put(T_train_55) ; RAY_test_55 = ray.put(T_test_55)

211_7
T_train_56, T_test_56 = make_merged_data(56) ; RAY_loss_weight_56 = ray.put(LOSS_WEIGHT_56) ; RAY_train_56 = ray.put(T_train_56) ; RAY_test_56 = ray.put(T_test_56)
T_train_57, T_test_57 = make_merged_data(57) ; RAY_loss_weight_57 = ray.put(LOSS_WEIGHT_57) ; RAY_train_57 = ray.put(T_train_57) ; RAY_test_57 = ray.put(T_test_57)
T_train_58, T_test_58 = make_merged_data(58) ; RAY_loss_weight_58 = ray.put(LOSS_WEIGHT_58) ; RAY_train_58 = ray.put(T_train_58) ; RAY_test_58 = ray.put(T_test_58)
T_train_59, T_test_59 = make_merged_data(59) ; RAY_loss_weight_59 = ray.put(LOSS_WEIGHT_59) ; RAY_train_59 = ray.put(T_train_59) ; RAY_test_59 = ray.put(T_test_59)
T_train_60, T_test_60 = make_merged_data(60) ; RAY_loss_weight_60 = ray.put(LOSS_WEIGHT_60) ; RAY_train_60 = ray.put(T_train_60) ; RAY_test_60 = ray.put(T_test_60)
T_train_61, T_test_61 = make_merged_data(61) ; RAY_loss_weight_61 = ray.put(LOSS_WEIGHT_61) ; RAY_train_61 = ray.put(T_train_61) ; RAY_test_61 = ray.put(T_test_61)
T_train_62, T_test_62 = make_merged_data(62) ; RAY_loss_weight_62 = ray.put(LOSS_WEIGHT_62) ; RAY_train_62 = ray.put(T_train_62) ; RAY_test_62 = ray.put(T_test_62)
T_train_63, T_test_63 = make_merged_data(63) ; RAY_loss_weight_63 = ray.put(LOSS_WEIGHT_63) ; RAY_train_63 = ray.put(T_train_63) ; RAY_test_63 = ray.put(T_test_63)

211_8
T_train_64, T_test_64 = make_merged_data(64) ; RAY_loss_weight_64 = ray.put(LOSS_WEIGHT_64) ; RAY_train_64 = ray.put(T_train_64) ; RAY_test_64 = ray.put(T_test_64)
T_train_65, T_test_65 = make_merged_data(65) ; RAY_loss_weight_65 = ray.put(LOSS_WEIGHT_65) ; RAY_train_65 = ray.put(T_train_65) ; RAY_test_65 = ray.put(T_test_65)
T_train_66, T_test_66 = make_merged_data(66) ; RAY_loss_weight_66 = ray.put(LOSS_WEIGHT_66) ; RAY_train_66 = ray.put(T_train_66) ; RAY_test_66 = ray.put(T_test_66)
T_train_67, T_test_67 = make_merged_data(67) ; RAY_loss_weight_67 = ray.put(LOSS_WEIGHT_67) ; RAY_train_67 = ray.put(T_train_67) ; RAY_test_67 = ray.put(T_test_67)
T_train_68, T_test_68 = make_merged_data(68) ; RAY_loss_weight_68 = ray.put(LOSS_WEIGHT_68) ; RAY_train_68 = ray.put(T_train_68) ; RAY_test_68 = ray.put(T_test_68)
T_train_69, T_test_69 = make_merged_data(69) ; RAY_loss_weight_69 = ray.put(LOSS_WEIGHT_69) ; RAY_train_69 = ray.put(T_train_69) ; RAY_test_69 = ray.put(T_test_69)
T_train_70, T_test_70 = make_merged_data(70) ; RAY_loss_weight_70 = ray.put(LOSS_WEIGHT_70) ; RAY_train_70 = ray.put(T_train_70) ; RAY_test_70 = ray.put(T_test_70)
T_train_71, T_test_71 = make_merged_data(71) ; RAY_loss_weight_71 = ray.put(LOSS_WEIGHT_71) ; RAY_train_71 = ray.put(T_train_71) ; RAY_test_71 = ray.put(T_test_71)

211_9
T_train_72, T_test_72 = make_merged_data(72) ; RAY_loss_weight_72 = ray.put(LOSS_WEIGHT_72) ; RAY_train_72 = ray.put(T_train_72) ; RAY_test_72 = ray.put(T_test_72)
T_train_73, T_test_73 = make_merged_data(73) ; RAY_loss_weight_73 = ray.put(LOSS_WEIGHT_73) ; RAY_train_73 = ray.put(T_train_73) ; RAY_test_73 = ray.put(T_test_73)
T_train_74, T_test_74 = make_merged_data(74) ; RAY_loss_weight_74 = ray.put(LOSS_WEIGHT_74) ; RAY_train_74 = ray.put(T_train_74) ; RAY_test_74 = ray.put(T_test_74)
T_train_75, T_test_75 = make_merged_data(75) ; RAY_loss_weight_75 = ray.put(LOSS_WEIGHT_75) ; RAY_train_75 = ray.put(T_train_75) ; RAY_test_75 = ray.put(T_test_75)
T_train_76, T_test_76 = make_merged_data(76) ; RAY_loss_weight_76 = ray.put(LOSS_WEIGHT_76) ; RAY_train_76 = ray.put(T_train_76) ; RAY_test_76 = ray.put(T_test_76)
T_train_77, T_test_77 = make_merged_data(77) ; RAY_loss_weight_77 = ray.put(LOSS_WEIGHT_77) ; RAY_train_77 = ray.put(T_train_77) ; RAY_test_77 = ray.put(T_test_77)
T_train_78, T_test_78 = make_merged_data(78) ; RAY_loss_weight_78 = ray.put(LOSS_WEIGHT_78) ; RAY_train_78 = ray.put(T_train_78) ; RAY_test_78 = ray.put(T_test_78)
T_train_79, T_test_79 = make_merged_data(79) ; RAY_loss_weight_79 = ray.put(LOSS_WEIGHT_79) ; RAY_train_79 = ray.put(T_train_79) ; RAY_test_79 = ray.put(T_test_79)

211_10
T_train_80, T_test_80 = make_merged_data(80) ; RAY_loss_weight_80 = ray.put(LOSS_WEIGHT_80) ; RAY_train_80 = ray.put(T_train_80) ; RAY_test_80 = ray.put(T_test_80)
T_train_81, T_test_81 = make_merged_data(81) ; RAY_loss_weight_81 = ray.put(LOSS_WEIGHT_81) ; RAY_train_81 = ray.put(T_train_81) ; RAY_test_81 = ray.put(T_test_81)
T_train_82, T_test_82 = make_merged_data(82) ; RAY_loss_weight_82 = ray.put(LOSS_WEIGHT_82) ; RAY_train_82 = ray.put(T_train_82) ; RAY_test_82 = ray.put(T_test_82)
T_train_83, T_test_83 = make_merged_data(83) ; RAY_loss_weight_83 = ray.put(LOSS_WEIGHT_83) ; RAY_train_83 = ray.put(T_train_83) ; RAY_test_83 = ray.put(T_test_83)
T_train_84, T_test_84 = make_merged_data(84) ; RAY_loss_weight_84 = ray.put(LOSS_WEIGHT_84) ; RAY_train_84 = ray.put(T_train_84) ; RAY_test_84 = ray.put(T_test_84)
T_train_85, T_test_85 = make_merged_data(85) ; RAY_loss_weight_85 = ray.put(LOSS_WEIGHT_85) ; RAY_train_85 = ray.put(T_train_85) ; RAY_test_85 = ray.put(T_test_85)
T_train_86, T_test_86 = make_merged_data(86) ; RAY_loss_weight_86 = ray.put(LOSS_WEIGHT_86) ; RAY_train_86 = ray.put(T_train_86) ; RAY_test_86 = ray.put(T_test_86)
T_train_87, T_test_87 = make_merged_data(87) ; RAY_loss_weight_87 = ray.put(LOSS_WEIGHT_87) ; RAY_train_87 = ray.put(T_train_87) ; RAY_test_87 = ray.put(T_test_87)

211_11
T_train_88, T_test_88 = make_merged_data(88) ; RAY_loss_weight_88 = ray.put(LOSS_WEIGHT_88) ; RAY_train_88 = ray.put(T_train_88) ; RAY_test_88 = ray.put(T_test_88)
T_train_89, T_test_89 = make_merged_data(89) ; RAY_loss_weight_89 = ray.put(LOSS_WEIGHT_89) ; RAY_train_89 = ray.put(T_train_89) ; RAY_test_89 = ray.put(T_test_89)
T_train_90, T_test_90 = make_merged_data(90) ; RAY_loss_weight_90 = ray.put(LOSS_WEIGHT_90) ; RAY_train_90 = ray.put(T_train_90) ; RAY_test_90 = ray.put(T_test_90)
T_train_91, T_test_91 = make_merged_data(91) ; RAY_loss_weight_91 = ray.put(LOSS_WEIGHT_91) ; RAY_train_91 = ray.put(T_train_91) ; RAY_test_91 = ray.put(T_test_91)




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






	RAY_train_list = [RAY_train_32 ,RAY_train_33 ,RAY_train_34,RAY_train_35RAY_train_36,RAY_train_37,RAY_train_38,RAY_train_39 ,] #########################
	RAY_test_list = [RAY_test_32 ,RAY_test_33 ,RAY_test_34,RAY_test_35,RAY_test_36,RAY_test_37,RAY_test_38, RAY_test_39 ]#########################
	RAY_loss_weight_list = [RAY_loss_weight_32 ,RAY_loss_weight_33 ,RAY_loss_weight_34,RAY_loss_weight_35,RAY_loss_weight_36,RAY_loss_weight_37,RAY_loss_weight_38, RAY_loss_weight_39 ] #########################
	#
	RAY_train_list = [RAY_train_40 ,RAY_train_41,RAY_train_42,RAY_train_43,RAY_train_44,RAY_train_45,RAY_train_46,RAY_train_47]
	RAY_test_list = [RAY_test_40 ,RAY_test_41,RAY_test_42,RAY_test_43,RAY_test_44,RAY_test_45,RAY_test_46, RAY_test_47]
	RAY_loss_weight_list = [RAY_loss_weight_40 ,RAY_loss_weight_41,RAY_loss_weight_42,RAY_loss_weight_43,RAY_loss_weight_44,RAY_loss_weight_45,RAY_loss_weight_46, RAY_loss_weight_47]


	RAY_train_list = [RAY_train_48 ,RAY_train_49 ,RAY_train_50,RAY_train_51, RAY_train_52,RAY_train_53,RAY_train_54,RAY_train_55  ] #########################
	RAY_test_list = [RAY_test_48 ,RAY_test_49 ,RAY_test_50,RAY_test_51,RAY_test_52,RAY_test_53,RAY_test_54, RAY_test_55 ]#########################
	RAY_loss_weight_list = [RAY_loss_weight_48 ,RAY_loss_weight_49 ,RAY_loss_weight_50,RAY_loss_weight_51,RAY_loss_weight_52,RAY_loss_weight_53,RAY_loss_weight_54, RAY_loss_weight_55 ] #########################
	
,8,9,10,11,12,13,14,15

	RAY_train_list = [RAY_train_56, RAY_train_57,RAY_train_58,RAY_train_59,RAY_train_60,RAY_train_61,RAY_train_62,RAY_train_63]
	RAY_test_list = [RAY_test_56 ,RAY_test_57,RAY_test_58,RAY_test_59,RAY_test_60,RAY_test_61,RAY_test_62, RAY_test_63]
	RAY_loss_weight_list = [RAY_loss_weight_56 ,RAY_loss_weight_57,RAY_loss_weight_58,RAY_loss_weight_59,RAY_loss_weight_60,RAY_loss_weight_61,RAY_loss_weight_62, RAY_loss_weight_63]


	RAY_train_list = [RAY_train_64 ,RAY_train_65 ,RAY_train_66,RAY_train_67, RAY_train_68,RAY_train_69,RAY_train_70,RAY_train_71 ] 
	RAY_test_list = [RAY_test_64 ,RAY_test_65 ,RAY_test_66,RAY_test_67,RAY_test_68,RAY_test_69,RAY_test_70, RAY_test_71 ]
	RAY_loss_weight_list = [RAY_loss_weight_64 ,RAY_loss_weight_65 ,RAY_loss_weight_66,RAY_loss_weight_67,RAY_loss_weight_68,RAY_loss_weight_69,RAY_loss_weight_70, RAY_loss_weight_71 ] 
	
	RAY_train_list = [RAY_train_72 ,RAY_train_73,RAY_train_74,RAY_train_75,RAY_train_76,RAY_train_77,RAY_train_78,RAY_train_79]
	RAY_test_list = [RAY_test_72 ,RAY_test_73,RAY_test_74,RAY_test_75,RAY_test_76,RAY_test_77,RAY_test_78, RAY_test_79]
	RAY_loss_weight_list = [RAY_loss_weight_72 ,RAY_loss_weight_73,RAY_loss_weight_74,RAY_loss_weight_75,RAY_loss_weight_76,RAY_loss_weight_77,RAY_loss_weight_78,RAY_loss_weight_79]

	RAY_train_list = [RAY_train_80 ,RAY_train_81 ,RAY_train_82,RAY_train_83, RAY_train_84,RAY_train_85,RAY_train_86,RAY_train_87 ] 
	RAY_test_list = [RAY_test_80 ,RAY_test_81 ,RAY_test_82,RAY_test_83,RAY_test_84,RAY_test_85,RAY_test_86, RAY_test_87 ]
	RAY_loss_weight_list = [RAY_loss_weight_80 ,RAY_loss_weight_81 ,RAY_loss_weight_82,RAY_loss_weight_83,RAY_loss_weight_84,RAY_loss_weight_85,RAY_loss_weight_86, RAY_loss_weight_87] 
	
	RAY_train_list = [RAY_train_89 ,RAY_train_90 ,RAY_train_91,RAY_train_92 ] 
	RAY_test_list = [RAY_test_89 ,RAY_test_90 ,RAY_test_91,RAY_test_92 ]
	RAY_loss_weight_list = [RAY_loss_weight_89 ,RAY_loss_weight_90 ,RAY_loss_weight_91,RAY_loss_weight_92] 
	




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
	#
	RAY_train_list = [RAY_train_89 ,RAY_train_90 ,RAY_train_91,RAY_train_92 ] 
	RAY_test_list = [RAY_test_89 ,RAY_test_90 ,RAY_test_91,RAY_test_92 ]
	RAY_loss_weight_list = [RAY_loss_weight_89 ,RAY_loss_weight_90 ,RAY_loss_weight_91,RAY_loss_weight_92] 
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
		'CV' : tune.grid_search([0,1,2,3,4,5,6,7]),
		'n_workers' : tune.grid_search([cpus_per_trial]),
		"epoch" : tune.grid_search([max_num_epochs]),
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




W_NAME = 'W211' # 고른 내용으로 5CV 다시 
MJ_NAME = 'M3V6'
WORK_DATE = '23.06.24' # 349
MISS_NAME = 'MIS2'
PPI_NAME = '349'
WORK_NAME = 'WORK_211_9' # 349###################################################################################################


WORK_PATH = '/home01/k040a01/02.{}/{}_{}_{}_{}/'.format(MJ_NAME,MJ_NAME,W_NAME,PPI_NAME,MISS_NAME)


OLD_PATH = '/home01/k040a01/02.M3V6/M3V6_W202_349_MIS2'
ANA_DF_CSV = pd.read_csv(os.path.join(OLD_PATH,'RAY_ANA_DF.{}.csv'.format('M3V6_W202_349_MIS2')))

my_config = ANA_DF_CSV[ANA_DF_CSV.trial_id=='1cf5052a'] # 349 


#4gpu for 5cv 
MAIN('PRJ02.{}.{}.{}.{}.{}'.format(WORK_DATE, MJ_NAME, WORK_NAME, PPI_NAME, MISS_NAME), my_config, 1, 1000, 16, 0.5)

# 8gpu
MAIN('PRJ02.{}.{}.{}.{}.{}'.format(WORK_DATE, MJ_NAME, WORK_NAME, PPI_NAME, MISS_NAME), my_config, 1, 1000, 8, 0.5)


with open(file='{}/CV_SM_list.pickle'.format(WORK_PATH), mode='wb') as f:
	pickle.dump(CV_ND_INDS, f)






sbatch gpu4.W203.CV5.any M3V6_WORK203.349.py
tail -n 100 /home01/k040a01/02.M3V6/M3V6_W203_349_MIS2/RESULT.G4.CV5.txt
tail ~/logs/M3V6W203_GPU4_13062.log





########################################### GPU
########################################### GPU
########################################### GPU
########################################### GPU
########################################### GPU




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

W_NAME = 'W211'
WORK_NAME = 'WORK_211' # 349
WORK_DATE = '23.06.24' # 349


anal_dir = '/home01/k040a01/ray_results/PRJ02.23.06.24.M3V6.WORK_211_10.349.MIS2'
anal_dir = '/home01/k040a01/ray_results/PRJ02.23.06.24.M3V6.WORK_211_2.349.MIS2'
anal_dir = '/home01/k040a01/ray_results/PRJ02.23.06.24.M3V6.WORK_211_3.349.MIS2'
anal_dir = '/home01/k040a01/ray_results/PRJ02.23.06.24.M3V6.WORK_211.349.MIS2'
anal_dir = '/home01/k040a01/ray_results/PRJ02.23.06.24.M3V6.WORK_211_4.349.MIS2'
anal_dir = '/home01/k040a01/ray_results/PRJ02.23.06.24.M3V6.WORK_211_5.349.MIS2'
anal_dir = '/home01/k040a01/ray_results/PRJ02.23.06.24.M3V6.WORK_211_6.349.MIS2'
anal_dir = '/home01/k040a01/ray_results/PRJ02.23.06.24.M3V6.WORK_211_7.349.MIS2'
anal_dir = '/home01/k040a01/ray_results/PRJ02.23.06.24.M3V6.WORK_211_8.349.MIS2'
anal_dir = '/home01/k040a01/ray_results/PRJ02.23.06.24.M3V6.WORK_211_9.349.MIS2'
anal_dir = '/home01/k040a01/ray_results/PRJ02.23.06.24.M3V6.WORK_212.349.MIS2'

list_dir = os.listdir(anal_dir)
exp_json = [a for a in list_dir if 'experiment_state' in a]
anal_df = ExperimentAnalysis(os.path.join(anal_dir, exp_json[0]))


'BT474_BREAST' : 
'IPC298_SKIN'
'SKMEL2_SKIN'
'G361_SKIN'
'OVCAR4_OVARY'
'KM12_LARGE_INTESTINE'
'NCIH2122_LUNG'
'UACC62_SKIN'
'SF295_CENTRAL_NERVOUS_SYSTEM'
'OVCAR8_OVARY'
'NCIH226_LUNG'
'HT29_LARGE_INTESTINE'
'786O_KIDNEY'
'UHO1_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE'
'RPMI8226_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE'
'A375_SKIN'
'K562_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE'
'NCIH460_LUNG'
'U251MG_CENTRAL_NERVOUS_SYSTEM'
'PA1_OVARY'
'RKO_LARGE_INTESTINE'
'A427_LUNG'
'A498_KIDNEY'
'SW620_LARGE_INTESTINE'
'SKMES1_LUNG'
'T98G_CENTRAL_NERVOUS_SYSTEM'
'NCIH522_LUNG'
'LOVO_LARGE_INTESTINE'
'MELHO_SKIN'
'L1236_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE'
'ACHN_KIDNEY'
'MDAMB231_BREAST'
'UO31_KIDNEY'
'MEWO_SKIN'
'UACC257_SKIN'
'SR786_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE'
'A101D_SKIN'
'NCIH23_LUNG'
'RVH421_SKIN'
'MDAMB175VII_BREAST'
'COLO829_SKIN'
'NIHOVCAR3_OVARY'
'SKMEL5_SKIN'
'LOXIMVI_SKIN'
'A673_BONE'
'MDAMB468_BREAST'
'SW837_LARGE_INTESTINE'
'A2780_OVARY'
'HCC1419_BREAST'
'HOP92_LUNG'
'WM115_SKIN'
'CAKI1_KIDNEY'
'MDAMB361_BREAST'
'SF268_CENTRAL_NERVOUS_SYSTEM'
'A2058_SKIN'
'COLO800_SKIN'
'IGROV1_OVARY'
'MSTO211H_PLEURA'
'HCT15_LARGE_INTESTINE'
'HCT116_LARGE_INTESTINE'
'VCAP_PROSTATE'
'EKVX_LUNG'
'ES2_OVARY'
'OV90_OVARY'
'HCC1500_BREAST'
'T47D_BREAST'
'UWB1289_OVARY'
'SKMEL28_SKIN'
'MCF7_BREAST'
'SNB75_CENTRAL_NERVOUS_SYSTEM'
'HOP62_LUNG'
'NCIH1650_LUNG'
'KPL1_BREAST'
'SKMEL30_SKIN'
'PC3_PROSTATE'
'CAMA1_BREAST'
'OVCAR5_OVARY'
'MDAMB436_BREAST'
'RPMI7951_SKIN'
'NCIH520_LUNG'
'SKOV3_OVARY'
'ZR751_BREAST'
'BT549_BREAST'
'HS578T_BREAST'
'HT144_SKIN'
'MALME3M_SKIN'
'SF539_CENTRAL_NERVOUS_SYSTEM'
'CAOV3_OVARY'
'DLD1_LARGE_INTESTINE'
'A549_LUNG'
'COLO792_SKIN'
'UACC812_BREAST'











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


