

# 편의상 로컬에서 파일을 다 만들어다가 바치기로 했어요 
# random 섞은것도 마찬가지임 
# 그래서 아예 GPU 에서는 learning 만 진행하도록 


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


WORK_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_6/'
DC_PATH = '/st06/jiyeonH/13.DD_SESS/DrugComb.1.5/' 
IDK_PATH = '/st06/jiyeonH/13.DD_SESS/ideker/' 
LINCS_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/' 
TARGET_PATH = '/st06/jiyeonH/13.DD_SESS/merged_target/'
cid_PATH = '/st06/jiyeonH/11.TOX/Pubchem_Classi_176/'



# Pubchem Whole Data
PUBCHEM_ALL = pd.read_csv('/st06/jiyeonH/12.HTP_DB/08.PUBCHEM/PUBCHEM_MJ_031022.csv',  low_memory = False)


# LINCS data filter 
# LINCS data filter 
# LINCS data filter 

BETA_BIND = pd.read_csv('/st06/jiyeonH/11.TOX/MY_TRIAL_5/'+"BETA_DATA_for_20220705_978.csv")
BETA_CP_info = pd.read_table('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+'compoundinfo_beta.txt')
BETA_CEL_info = pd.read_table('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+'cellinfo_beta.txt')
BETA_SIG_info = pd.read_table('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+'siginfo_beta.txt', low_memory = False)

filter1 = BETA_SIG_info[BETA_SIG_info.pert_type.isin(['ctl_vehicle', 'ctl_untrt' ,'trt_cp' ])]
# 764996
filter2 = filter1[filter1.is_exemplar_sig==1]
# 136460
BETA_MJ = pd.read_csv('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+"Lincs_pubchem_mj.csv") # SID 필터까지 그렇게 완성한 무엇 

smiles_df = BETA_MJ[['SMILES_cid','canonical_smiles_re']].drop_duplicates()
only_smiles = list(smiles_df['canonical_smiles_re'])
leng = []

for i in range(len(only_smiles)) :
	smiles = only_smiles[i]
	iMol = Chem.MolFromSmiles(smiles.strip())
	try:
		NUM = iMol.GetNumAtoms()
		leng.append(NUM)
	except:
		leng.append("error")
		print("error",z,i)

smiles_df['leng'] = leng # max : 115

BETA_MJ_edit = pd.merge(BETA_MJ, smiles_df[['SMILES_cid','leng']], on = 'SMILES_cid', how = 'left')
# BETA_MJ_edit.to_csv(WORK_PATH+'BETA_MJ_edit.csv', sep = '\t')
BETA_MJ_edit = pd.read_csv(WORK_PATH+'BETA_MJ_edit.csv', sep = '\t')



BETA_MJ_RE = BETA_MJ_edit[['pert_id','SMILES_cid','canonical_smiles',
	   'pubchem_cid', 'h_bond_acceptor_count', 'h_bond_donor_count',
	   'rotatable_bond_count', 'MolLogP', 'molecular_weight',
	   'canonical_smiles_re', 'tpsa', 'leng']].drop_duplicates() # 25903

BETA_EXM = pd.merge(filter2, BETA_MJ_RE, on='pert_id', how = 'left')
BETA_EXM2 = BETA_EXM[BETA_EXM.SMILES_cid > 0] # 127595

BETA_CEL_info2 = BETA_CEL_info[['cell_iname','cellosaurus_id','ccle_name']] # 240 
BETA_SELEC_SIG_wCell = pd.merge(BETA_EXM2, BETA_CEL_info2, on = 'cell_iname', how = 'left') # 127595
BETA_SELEC_SIG_wCell2 = BETA_SELEC_SIG_wCell[['pert_id','pubchem_cid','cellosaurus_id','ccle_name','sig_id']]
BETA_SELEC_SIG_wCell2 = BETA_SELEC_SIG_wCell2.drop_duplicates() # 127595

cello_tt=[True if type(a)==str else False for a in list(BETA_SELEC_SIG_wCell2.cellosaurus_id)]
BETA_CID_CELLO_SIG = BETA_SELEC_SIG_wCell2[cello_tt][['pert_id','pubchem_cid','cellosaurus_id','sig_id']].drop_duplicates() # 110714
beta_cid_cello_sig_tf = [ True if a>0 else False for a in list(BETA_CID_CELLO_SIG.pubchem_cid)]
BETA_CID_CELLO_SIG = BETA_CID_CELLO_SIG[beta_cid_cello_sig_tf]
BETA_CID_CELLO_SIG['pubchem_cid'] = [int(a) for a in list(BETA_CID_CELLO_SIG['pubchem_cid']) ]

# -> CCLE 필터까지 완료




# Drug Comb 데이터 가져오기 
# Drug Comb 데이터 가져오기 
# Drug Comb 데이터 가져오기 

DC_tmp_DF1 = pd.read_csv(DC_PATH+'TOTAL_BLOCK.csv', low_memory=False) # 1469182
DC_tmp_DF2 = DC_tmp_DF1[DC_tmp_DF1['quality'] != 'bad'] # 1457561

# Drug info 
with open(DC_PATH+'drugs.json') as json_file :
	DC_DRUG =json.load(json_file)

DC_DRUG_K = list(DC_DRUG[0].keys())
DC_DRUG_DF = pd.DataFrame(columns=DC_DRUG_K)

for DD in range(1,len(DC_DRUG)):
	tmpdf = pd.DataFrame({k:[DC_DRUG[DD][k]] for k in DC_DRUG_K})
	DC_DRUG_DF = pd.concat([DC_DRUG_DF, tmpdf], axis = 0)

DC_DRUG_DF['id_re'] = [float(a) for a in list(DC_DRUG_DF['id'])]




# drug info merge to pubchem data (특히 smiles 아 여기 smiles 없는데..? )
DC_DRUG_DF_FULL = pd.merge(DC_DRUG_DF, PUBCHEM_ALL , left_on = 'cid', right_on ='CID', how = 'left')

smiles_df = DC_DRUG_DF_FULL[['cid','CAN_SMILES']].drop_duplicates()
only_smiles = list(smiles_df['CAN_SMILES'])
leng = []

for i in range(len(only_smiles)) :
	smiles = only_smiles[i]
	z = 0
	try:
		iMol = Chem.MolFromSmiles(smiles.strip())
		NUM = iMol.GetNumAtoms()
		leng.append(NUM)
	except:
		leng.append("error")
		print("error",z,i)

smiles_df['leng'] = leng # max : 115

DC_DRUG_DF_FULL = pd.merge(DC_DRUG_DF_FULL, smiles_df[['cid','leng']], on = 'cid', how = 'left')
# DC_DRUG_DF_FULL.to_csv('/st06/jiyeonH/11.TOX/MY_TRIAL_6/DC_DRUG_DF_PC.csv', sep ='\t')
DC_DRUG_DF_FULL = pd.read_csv('/st06/jiyeonH/11.TOX/MY_TRIAL_6/DC_DRUG_DF_PC.csv', sep ='\t')



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

DC_DATA_filter = DC_tmp_DF2[['drug_row_id','drug_col_id','cell_line_id','synergy_loewe']] # 1457561
DC_DATA_filter2 = DC_DATA_filter.drop_duplicates() # 1374958
DC_DATA_filter2['drug_row_id_re'] = [float(a) for a in list(DC_DATA_filter2['drug_row_id'])]
DC_DATA_filter2['drug_col_id_re'] = [float(a) for a in list(DC_DATA_filter2['drug_col_id'])]

DC_DATA_filter3 = DC_DATA_filter2[DC_DATA_filter2.drug_row_id_re>0] # 1374958 -> 1363698
DC_DATA_filter4 = DC_DATA_filter3[DC_DATA_filter3.drug_col_id_re>0] # 751450 -> 740884
DC_DATA_filter4.cell_line_id # unique 295


# cid renaming
DC_DRUG_DF2 = DC_DRUG_DF[['id_re','dname','cid']]
																	# dnames = list(DC_DRUG_DF2['dname']) # 그래 이부분때문에 다시 다운로드 받은거였음. 이게 문제가 drug 마다 CID 가 여러개 붙는 애들이 있어서
DC_DRUG_DF2.columns = ['drug_row_id_re','drug_row','drug_row_cid']
DC_DATA5 = pd.merge(DC_DATA_filter4, DC_DRUG_DF2, on ='drug_row_id_re', how='left' ) # 751450 -> 740884

DC_DRUG_DF2.columns = ['drug_col_id_re','drug_col','drug_col_cid']
DC_DATA6 = pd.merge(DC_DATA5, DC_DRUG_DF2, on ='drug_col_id_re', how='left') # 751450 -> 740884

#  Add cell data and cid filter
DC_DATA7_1 = pd.merge(DC_DATA6, DC_CELL_DF2 , on= 'cell_line_id', how = 'left') # 751450 -> 740884
DC_DATA7_1['drug_row_cid_re'] = [a if type(a) == int else 0 for a in list(DC_DATA7_1['drug_row_cid'])]
DC_DATA7_1['drug_col_cid_re'] = [a if type(a) == int else 0 for a in list(DC_DATA7_1['drug_col_cid'])]
DC_DATA7_2 = DC_DATA7_1[DC_DATA7_1.drug_row_cid_re>0] # 747621 -> 737106
DC_DATA7_3 = DC_DATA7_2[DC_DATA7_2.drug_col_cid_re>0] # 735595 -> 725496
cello_t=[True if type(a)==str else False for a in list(DC_DATA7_3.DrugCombCello)]

DC_DATA7_4_cello = DC_DATA7_3[cello_t] # 730348 -> 720249
DC_cello_final = DC_DATA7_4_cello[['drug_row_cid_re','drug_col_cid_re','DrugCombCello']].drop_duplicates() # 563367 -> 554431
DC_cello_final_dup = DC_DATA7_4_cello[['drug_row_cid_re','drug_col_cid_re','DrugCombCello', 'synergy_loewe']].drop_duplicates() # 730348 -> 720249

DC_cello_final_dup["drug_row_cid2"] = [float(a) for a in list(DC_cello_final_dup.drug_row_cid_re)]
DC_cello_final_dup["drug_col_cid2"] = [float(a) for a in list(DC_cello_final_dup.drug_col_cid_re)]

DC_cello_final_dup = DC_cello_final_dup[['drug_row_cid2','drug_col_cid2','DrugCombCello','synergy_loewe']]
DC_cello_final_dup.columns = ['drug_row_cid','drug_col_cid','DrugCombCello','synergy_loewe'] # 730348 -> 720249
TF_check = [True if np.isnan(a)==False else False for a in DC_cello_final_dup.synergy_loewe] 
DC_cello_final_dup = DC_cello_final_dup[TF_check] # 719946



# MATCH DC & LINCS 
print('DC and LINCS')
# 무조건 pubchem 공식 파일 사용하기 (mj ver)

BETA_CID_CELLO_SIG.columns=['pert_id', 'drug_row_cid', 'DrugCombCello', 'BETA_sig_id']
CELLO_DC_BETA_1 = pd.merge(DC_cello_final_dup, BETA_CID_CELLO_SIG, on = ['drug_row_cid','DrugCombCello'], how = 'left') # 731051 -> 720619 -> 719999

BETA_CID_CELLO_SIG.columns=['pert_id', 'drug_col_cid', 'DrugCombCello', 'BETA_sig_id']
CELLO_DC_BETA_2 = pd.merge(CELLO_DC_BETA_1, BETA_CID_CELLO_SIG, on = ['drug_col_cid','DrugCombCello'], how = 'left') # 731644 -> 721206

BETA_CID_CELLO_SIG.columns=['pert_id', 'pubchem_cid', 'cellosaurus_id', 'sig_id']
# CELLO_DC_BETA_2.to_csv('/st06/jiyeonH/13.DD_SESS/00.PRJ/DC_EXP_Full.csv', '\t')



# (1) AO BO  
FILTER_AO_BO = [a for a in range(CELLO_DC_BETA_2.shape[0]) if (type(CELLO_DC_BETA_2.BETA_sig_id_x[a]) == str) & (type(CELLO_DC_BETA_2.BETA_sig_id_y[a]) == str)]
DATA_AO_BO = CELLO_DC_BETA_2.loc[FILTER_AO_BO] # 11742 -> 11379 
DATA_AO_BO[['BETA_sig_id_x','BETA_sig_id_y','DrugCombCello']].drop_duplicates() # 9230 -> 8914
DATA_AO_BO_cids = list(set(list(DATA_AO_BO.drug_row_cid) + list(DATA_AO_BO.drug_col_cid))) # 172 


# (2) AX BO 
FILTER_AX_BO = [a for a in range(CELLO_DC_BETA_2.shape[0]) if (type(CELLO_DC_BETA_2.BETA_sig_id_x[a]) != str) & (type(CELLO_DC_BETA_2.BETA_sig_id_y[a]) == str)]
DATA_AX_BO = CELLO_DC_BETA_2.loc[FILTER_AX_BO] # 11967
DATA_AX_BO[['BETA_sig_id_x','BETA_sig_id_y','DrugCombCello']].drop_duplicates() # 452
DATA_AX_BO_cids = list(set(list(DATA_AX_BO.drug_row_cid) + list(DATA_AX_BO.drug_col_cid))) # 635 

# DATA_AX_BO.to_csv('/st06/jiyeonH/13.DD_SESS/00.PRJ/DC_EXP_JY1.csv', sep= '\t')

tmp = DATA_AX_BO[['drug_row_cid','DrugCombCello']].drop_duplicates()
tmp1 = [str(a) for a in tmp['drug_row_cid']]
tmp2 = [tmp1[a] + "__" + list(tmp['DrugCombCello'])[a] for a in range(tmp.shape[0])] # 1274

# (3) AO BX 
FILTER_AO_BX = [a for a in range(CELLO_DC_BETA_2.shape[0]) if (type(CELLO_DC_BETA_2.BETA_sig_id_x[a]) == str) & (type(CELLO_DC_BETA_2.BETA_sig_id_y[a]) != str)]
DATA_AO_BX = CELLO_DC_BETA_2.loc[FILTER_AO_BX] # 14998
DATA_AO_BX[['BETA_sig_id_x','BETA_sig_id_y','DrugCombCello']].drop_duplicates() # 449
DATA_AO_BX_cids = list(set(list(DATA_AO_BX.drug_row_cid) + list(DATA_AO_BX.drug_col_cid))) # 274 

# DATA_AO_BX.to_csv('/st06/jiyeonH/13.DD_SESS/00.PRJ/DC_EXP_JY2.csv', sep= '\t')

tmp = DATA_AO_BX[['drug_col_cid','DrugCombCello']].drop_duplicates()
tmp1 = [str(a) for a in tmp['drug_col_cid']]
tmp2 = [tmp1[a] + "__" + list(tmp['DrugCombCello'])[a] for a in range(len(tmp1))] # 900


# (4) AX BX 
FILTER_AX_BX = [a for a in range(CELLO_DC_BETA_2.shape[0]) if (type(CELLO_DC_BETA_2.BETA_sig_id_x[a]) != str) & (type(CELLO_DC_BETA_2.BETA_sig_id_y[a]) != str)]
DATA_AX_BX = CELLO_DC_BETA_2.loc[FILTER_AX_BX] # 682862
DATA_AX_BX = DATA_AX_BX[DATA_AX_BX.DrugCombCello!='NA']
DATA_AX_BX[['BETA_sig_id_x','BETA_sig_id_y','DrugCombCello']].drop_duplicates() # 의미가 없음 
DATA_AX_BX_cids = list(set(list(DATA_AX_BX.drug_row_cid) + list(DATA_AX_BX.drug_col_cid))) # 4280 

# DATA_AX_BX.to_csv('/st06/jiyeonH/13.DD_SESS/00.PRJ/DC_EXP_JY3.csv', sep= '\t')


tmp_r = DATA_AX_BX[['drug_row_cid','DrugCombCello']].drop_duplicates()
tmp_c = DATA_AX_BX[['drug_col_cid','DrugCombCello']].drop_duplicates()
tmp1 = [str(a) for a in tmp_r['drug_row_cid']]
tmp2 = [str(a) for a in tmp_c['drug_col_cid']]
tmp3 = [tmp1[a] + "__" + list(tmp_r['DrugCombCello'])[a] for a in range(len(tmp1))] # 
tmp4 = [tmp2[a] + "__" + list(tmp_c['DrugCombCello'])[a] for a in range(len(tmp2))] # 
len(set(tmp3+tmp4)) # 22,051


DATA_AO_BO['type'] = 'AOBO'
DATA_AX_BO['type'] = 'AXBO'
DATA_AO_BX['type'] = 'AOBX'
DATA_AX_BX['type'] = 'AXBX'

#################################################################################################
##################################################################################################



print('NETWORK')

hunet_dir = '/home01/k006a01/01.DATA/HumanNet/'
# hunet_dir = '/st06/jiyeonH/13.DD_SESS/HumanNetV3/'


hunet_gsp = pd.read_csv(hunet_dir+'HumanNet-GSP.tsv', sep = '\t', header = None)
hunet_gsp.columns = ['G_A','G_B']

L_matching_list = pd.read_csv(IDK_PATH+'L_12_string.csv', sep = '\t')

hnet_IAS_L1 = hunet_gsp[hunet_gsp['G_A'].isin(L_matching_list.entrez)]
hnet_IAS_L2 = hnet_IAS_L1[hnet_IAS_L1['G_B'].isin(L_matching_list.entrez)] # 20232

len(set(list(hnet_IAS_L2['G_A']) + list(hnet_IAS_L2['G_B']))) # 611
ID_G = nx.from_pandas_edgelist(hnet_IAS_L2, 'G_A', 'G_B')

ESSEN_NAMES = list(set(L_matching_list['entrez']))

MSSNG = [a for  a in ESSEN_NAMES if a not in list(ID_G.nodes)]
# MSSNG = L_matching_list[L_matching_list['PPI_name'].isin(list(ID_G.nodes))==False]['PPI_name']
for nn in list(MSSNG):
	ID_G.add_node(nn)

# edge 3871
ID_GENE_ORDER_mini = list(ID_G.nodes()) # 978
ID_ADJ = nx.adjacency_matrix(ID_G)
ID_ADJ_tmp = torch.LongTensor(ID_ADJ.toarray())
ID_ADJ_IDX = ID_ADJ_tmp.to_sparse().indices()  # [2, 7742]
ID_WEIGHT = [] # len : 3871 -> 7742

ID_WEIGHT_SCORE = [1 for a in range(ID_ADJ_IDX.shape[1])]


new_node_name = []
for a in ID_G.nodes():
	tmp_name = list(set(L_matching_list[L_matching_list.entrez == a ]['L_gene_symbol'])) # 6118
	new_node_name = new_node_name + tmp_name

mapping = {list(ID_G.nodes())[a]:new_node_name[a] for a in range(len(new_node_name))}

ID_G_RE = nx.relabel_nodes(ID_G, mapping)

MY_G = ID_G_RE 
MY_WEIGHT_SCORE = ID_WEIGHT_SCORE # SCORE



# TARGET # 민지가 다시 올려준다고 함 

TARGET_DB = pd.read_csv(TARGET_PATH+'combined_target.csv', low_memory=False)



#################################################################################################
##################################################################################################

# 데이터 만들기 위한 단계 

# Graph 확인 

JY_GRAPH = MY_G
JY_GRAPH_ORDER = MY_G.nodes()
JY_ADJ = nx.adjacency_matrix(JY_GRAPH)

JY_ADJ_tmp = torch.LongTensor(JY_ADJ.toarray())
JY_ADJ_IDX = JY_ADJ_tmp.to_sparse().indices()
JY_IDX_WEIGHT = MY_WEIGHT_SCORE


# 써야하는 DC set 확인 

(1)
A_B_C_S = DATA_AO_BO.reset_index(drop = True) # 11379
(2) -> 5_2_3
A_B_C_S = pd.concat([DATA_AO_BO, DATA_AX_BO, DATA_AO_BX])
A_B_C_S = A_B_C_S.reset_index(drop = True) # 38344 -> 31652
(3) -> 5_3_3
A_B_C_S = pd.concat([DATA_AO_BO, DATA_AX_BO, DATA_AO_BX, DATA_AX_BX])
A_B_C_S = A_B_C_S.reset_index(drop = True) # 622809


# 5_2_3 : 31652
# 5_3_3 : 538362
A_B_C_S_SET = A_B_C_S[['drug_row_cid','drug_col_cid','BETA_sig_id_x','BETA_sig_id_y','DrugCombCello','type']].drop_duplicates()
ori_cids = list(set(list(set(A_B_C_S_SET.drug_row_cid)) + list(set(A_B_C_S_SET.drug_col_cid)))) # 695



# LINCS 확인 
BETA_ORDER_pre =[list(L_matching_list.L_gene_symbol).index(a) for a in JY_GRAPH_ORDER]
BETA_ORDER_DF = L_matching_list.iloc[BETA_ORDER_pre] # 어차피 ref 다르고 같은 애들이라 괜춘 
BETA_ORDER = list(BETA_ORDER_DF.entrez)




# max atom check 
DC_smiles = DC_DRUG_DF_FULL[['cid','CAN_SMILES','leng']].drop_duplicates()
DC_smiles.columns = ['cid','canonical_smiles_re','leng']
DC_smiles = DC_smiles[DC_smiles.cid>0]
DC_smiles = DC_smiles.reset_index(drop=True)
L_smiles = BETA_MJ_edit[['SMILES_cid','canonical_smiles_re','leng']].drop_duplicates()
L_smiles.columns = ['cid','canonical_smiles_re','leng']

max_len = 100
len_sm = pd.concat( [ L_smiles, DC_smiles ] )
len_sm['leng'] = [int(a) if a !='error' else 0 for a in list(len_sm['leng']) ]
len_sm = len_sm.drop_duplicates() # 31054

len_sm_100 = len_sm[len_sm.leng<=100] # 31014


# ABCS with atom len 
len_sm_100.columns = ['cid', 'canonical_smiles_re', 'leng']
len_sm_100_cut = len_sm_100[['cid','leng']].drop_duplicates()
A_B_C_S_SET_len = pd.merge(A_B_C_S_SET, len_sm_100_cut, left_on='drug_row_cid', right_on='cid', how = 'left')
A_B_C_S_SET_len = pd.merge(A_B_C_S_SET_len, len_sm_100_cut, left_on='drug_col_cid', right_on='cid', how = 'left')

A_B_C_S_SET_len2 = A_B_C_S_SET_len[A_B_C_S_SET_len.leng_x>0]
A_B_C_S_SET_len2 = A_B_C_S_SET_len2[A_B_C_S_SET_len2.leng_y>0] # 5_2_3 : 31650 | 5_3_3 : 538346
A_B_C_S_SET = copy.deepcopy(A_B_C_S_SET_len2)



# Tanimoto filter 
len_cut_cids = list(set(list(set(A_B_C_S_SET.drug_row_cid)) + list(set(A_B_C_S_SET.drug_col_cid))))# 694

DATA_Filter_ori = len_sm[len_sm.cid.isin(ori_cids)]
DATA_Filter_cut = len_sm[len_sm.cid.isin(len_cut_cids)]

def calculate_internal_pairwise_similarities(smiles_list) :
	mols = [Chem.MolFromSmiles(x.strip()) for x in smiles_list]
	fps = [Chem.RDKFingerprint(x) for x in mols]
	nfps = len(fps)
	#
	similarities = np.zeros((nfps, nfps))
	#
	for i in range(1, nfps):
		sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
		similarities[i, :i] = sims
		similarities[:i, i] = sims
	return similarities 


sim_matrix_order = list(DATA_Filter_ori['canonical_smiles_re'])
sim_matrix = calculate_internal_pairwise_similarities(sim_matrix_order)



row_means = []
for i in range(sim_matrix.shape[0]):
	indexes = [a for a in range(sim_matrix.shape[0])]
	indexes.pop(i)
	row_tmp = sim_matrix[i][indexes]
	row_mean = np.mean(row_tmp)
	row_means = row_means + [row_mean]


DATA_Filter_ori['MEAN'] = row_means
means_df = DATA_Filter_ori.sort_values('MEAN')
means_df['cat']=['MEAN' for a in range(len(ori_cids))]
# means_df['filter']=['stay' if (a in cids_all) else 'nope' for a in list(means_df.CIDs)] 
means_df['dot_col']= ['IN' if (a in len_cut_cids) else 'OUT' for a in list(means_df['cid'])] 

# means_df.MEAN.describe()

means_df['over0.1'] = ['IN' if a > 0.1  else 'OUT' for a in list(means_df.MEAN)] 
means_df['over0.2'] = ['IN' if a > 0.2  else 'OUT' for a in list(means_df.MEAN)] 
means_df['overQ'] = ['IN' if a > 0.28  else 'OUT' for a in list(means_df.MEAN)] 


#  over 0.1 only -> 7720
tmp_list = list(means_df[(means_df['dot_col'] == 'IN') & (means_df['over0.1'] == 'IN') ]['cid'])
tmp_row = A_B_C_S_SET[A_B_C_S_SET.drug_row_cid.isin(tmp_list)]
tmp_col = tmp_row[tmp_row.drug_col_cid.isin(tmp_list)] # 5_2_3 : 26303 | 5_3_3 :  441281

A_B_C_S_SET = copy.deepcopy(tmp_col) 
# 5_2_3 :  cid-cid-cell AOBO 7238, other 18245
# 5_3_3 :  cid-cid-cell AOBO 7238, other 433223


# Cell line vector
ABCS_cells= list(set(A_B_C_S_SET.DrugCombCello))
DC_CELL_DF3 = copy.deepcopy(DC_CELL_DF2)
DC_CELL_DF3 = DC_CELL_DF3[DC_CELL_DF3.DrugCombCello!='NA']
DC_CELL_DF4 = DC_CELL_DF3[DC_CELL_DF3.DrugCombCello.isin(ABCS_cells)]
DC_CELL_DF4.iloc[list(DC_CELL_DF4.DrugCombCello).index('CVCL_0395')]['tissue'] = 'PROSTATE'
DC_CELL_DF4.iloc[list(DC_CELL_DF4.DrugCombCello).index('CVCL_A442')]['tissue'] = 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE'
DC_CELL_DF4['cell_onehot'] = [a for a in range( DC_CELL_DF4.shape[0])]

A_B_C_S_SET_COH = pd.merge(A_B_C_S_SET, DC_CELL_DF4[['DrugCombCello','DC_cellname','cell_onehot']], on = 'DrugCombCello', how = 'left'  )

cell_one_hot = torch.nn.functional.one_hot(torch.Tensor(A_B_C_S_SET_COH['cell_onehot']).long())

A_B_C_S_SET = copy.deepcopy(A_B_C_S_SET_COH)




# another smiles merge 
len_sm_100.columns = ['drug_row_cid','ROW_CAN_SMILES','ROW_SM_leng']
ABCS_SM1 = pd.merge(A_B_C_S_SET, len_sm_100, on='drug_row_cid', how = 'left' )
len_sm_100.columns = ['drug_col_cid','COL_CAN_SMILES','COL_SM_leng']
ABCS_SM2 = pd.merge(ABCS_SM1, len_sm_100, on='drug_col_cid', how = 'left' )

len_sm_100.columns = ['cid','canonical_smiles_re','leng']

aa = list(ABCS_SM2['ROW_CAN_SMILES'])
bb = list(ABCS_SM2['COL_CAN_SMILES'])
cc = list(ABCS_SM2['DrugCombCello'])
ABCS_SM2['SM_SM'] = [aa[i] + '___' + bb[i]+ '___' + cc[i] for i in range(ABCS_SM2.shape[0])]



																				DC_CELL_DF3['tissue'] = [ '_'.join(a.split('_')[1:]) for a in list(DC_CELL_DF3['DrugCombCCLE'])]
																				missing_tissue = {a:'' for a in list(DC_CELL_DF3[DC_CELL_DF3.DrugCombCCLE =="NA"]['DrugCombCello'])}
																				TISSUE_SET = list(set(DC_CELL_DF3['tissue']))
																				DC_CELL_DF3['tissue_onehot'] = [TISSUE_SET.index(a) for a in list(DC_CELL_DF3['tissue'])]

																				A_B_C_S_SET_COH = pd.merge(A_B_C_S_SET, DC_CELL_DF3[['DrugCombCello','DC_cellname','cell_onehot']], on = 'DrugCombCello', how = 'left'  )
																				cell_one_hot = torch.nn.functional.one_hot(torch.Tensor(A_B_C_S_SET_COH['cell_onehot']).long())


																				{'', 'VULVA', 'LARGE_INTESTINE', 'PANCREAS', 'TESTIS', 'LUNG', 'CERVIX', 'AUTONOMIC_GANGLIA', 'OVARY', 
																				'CENTRAL_NERVOUS_SYSTEM', 'THYROID', 'LIVER', 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'KIDNEY', 'STOMACH', 'SMALL_INTESTINE', 'SOFT_TISSUE', 
																				'SOFT_TISSUE; SJRH30_SOFT_TISSUE', 'MATCHED_NORMAL_TISSUE', 'PROSTATE', 'ENDOMETRIUM', 'URINARY_TRACT', 'FIBROBLAST', 'SALIVARY_GLAND', 
																				'SKIN', 'PLEURA', 'UPPER_AERODIGESTIVE_TRACT', 'BONE', 'PLACENTA', 'BREAST', 'ADRENAL_CORTEX', 'BILIARY_TRACT', 'OESOPHAGUS'}




def get_CHEM(cid, k):
	maxNumAtoms = max_len
	smiles = len_sm[len_sm.cid == cid]['canonical_smiles_re'].item()
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



def get_LINCS_data(DRUG_SIG):
	Drug_EXP = BETA_BIND[['id',DRUG_SIG]]
	BIND_ORDER =[list(BETA_BIND.id).index(a) for a in BETA_ORDER_DF.entrez]
	Drug_EXP_ORD = Drug_EXP.iloc[BIND_ORDER]
	#
	return list(Drug_EXP_ORD[DRUG_SIG])



# read MJ data
g = globals() # global 변수들의 리스트 
DATA_LIST = ['JY1_MJ_1_NF','JY1_MJ_1_MJB','JY1_MJ_2_NF','JY1_MJ_2_MJB','JY2_MJ_1_NF','JY2_MJ_1_MJB','JY2_MJ_2_NF','JY2_MJ_2_MJB','JY3_MJ_1_NF','JY3_MJ_1_MJB','JY3_MJ_2_NF','JY3_MJ_2_MJB']
for i in DATA_LIST:
	try:
		g[i] = pd.read_csv('/st06/jiyeonH/13.DD_SESS/00.PRJ/'+i+'.csv', index_col = 0)
	except:
		print(i)



# 1) MJ_1 series
MJ_entrez = list(JY1_MJ_1_MJB.index)
JY2_MJ_1_MJB.index = MJ_entrez
JY3_MJ_1_MJB.index = MJ_entrez

MJ_1_MJB_DF = pd.concat([JY1_MJ_1_MJB,JY2_MJ_1_MJB,JY3_MJ_1_MJB], axis = 1)#########
MJ_1_MJB_DF = MJ_1_MJB_DF.T.drop_duplicates().T

def get_MJ_data(CID, CELL, dat_type): 
	if dat_type == 'AXBO' :
		i = 'JY1_MJ_1_MJB'################### PRJ change ! 
	elif dat_type == 'AOBX' :
		i = 'JY2_MJ_1_MJB'###################
	elif dat_type == 'AXBX' :
		i = 'JY3_MJ_1_MJB' ###################
	else : 
		print('nodb')
	#
	MJ_DATA = g[i]
	MJ_DATA_re = MJ_DATA.loc[BETA_ORDER]
	RES = MJ_DATA_re[str(int(CID))+'__'+str(CELL)]
	return list(RES)


# 2) MJ_2 series
MJ_entrez = list(JY1_MJ_2_MJB.index)
JY2_MJ_2_MJB.index = MJ_entrez
JY3_MJ_2_MJB.index = MJ_entrez

MJ_2_MJB_DF = pd.concat([JY1_MJ_2_MJB, JY2_MJ_2_MJB, JY3_MJ_2_MJB], axis = 1)#########
MJ_2_MJB_DF = MJ_2_MJB_DF.T.drop_duplicates().T


def get_MJ_data(CID, CELL, dat_type): 
	if dat_type == 'AXBO' :
		i = 'JY1_MJ_2_MJB'################### PRJ change ! 
	elif dat_type == 'AOBX' :
		i = 'JY2_MJ_2_MJB'###################
	elif dat_type == 'AXBX' :
		i = 'JY3_MJ_2_MJB' ###################
	else : 
		print('nodb')
	#
	MJ_DATA = g[i]
	MJ_DATA_re = MJ_DATA.loc[BETA_ORDER]
	RES = MJ_DATA_re[str(int(CID))+'__'+str(CELL)]
	return list(RES)




# 3) MJ_1 NF series
MJ_entrez = list(JY1_MJ_1_MJB.index) # 민지가 얘만 안붙여줌.... 다른 애들이랑 같은것 같긴 함 
JY1_MJ_1_NF.index = MJ_entrez
JY2_MJ_1_NF.index = MJ_entrez
JY3_MJ_1_NF.index = MJ_entrez

MJ_1_NF_DF = pd.concat([JY1_MJ_1_NF, JY2_MJ_1_NF, JY3_MJ_1_NF], axis = 1)#########
MJ_1_NF_DF = MJ_1_NF_DF.T.drop_duplicates().T

def get_MJ_data(CID, CELL, dat_type): 
	if dat_type == 'AXBO' :
		i = 'JY1_MJ_1_NF'################### PRJ change ! 
	elif dat_type == 'AOBX' :
		i = 'JY2_MJ_1_NF'###################
	elif dat_type == 'AXBX' :
		i = 'JY3_MJ_1_NF' ###################
	else : 
		print('nodb')
	#
	MJ_DATA = g[i]
	MJ_DATA_re = MJ_DATA.loc[BETA_ORDER]
	RES = MJ_DATA_re[str(int(CID))+'__'+str(CELL)]
	return list(RES)

'5746__CVCL_0132'
CID, CELL, dat_type = DrugA_CID, Cell, dat_type





# 4) MJ_2 NF series
MJ_entrez = list(JY1_MJ_2_NF.index)
JY2_MJ_2_NF.index = MJ_entrez
JY3_MJ_2_NF.index = MJ_entrez

MJ_2_NF_DF = pd.concat([JY1_MJ_2_NF,JY2_MJ_2_NF,JY3_MJ_2_NF], axis = 1)#########
MJ_2_NF_DF = MJ_2_NF_DF.T.drop_duplicates().T

def get_MJ_data(CID, CELL, dat_type): 
	if dat_type == 'AXBO' :
		i = 'JY1_MJ_2_NF'################### PRJ change ! 
	elif dat_type == 'AOBX' :
		i = 'JY2_MJ_2_NF'###################
	elif dat_type == 'AXBX' :
		i = 'JY3_MJ_2_NF' ###################
	else : 
		print('nodb')
	#
	MJ_DATA = g[i]
	MJ_DATA_re = MJ_DATA.loc[BETA_ORDER]
	RES = MJ_DATA_re[str(int(CID))+'__'+str(CELL)]
	return list(RES)




def get_targets(CID): # 이걸 수정해야함? 아닌가 굳이 해야하나 아니지 해야지. CID 가 없는 경우를 나타내야지  
	target_cids = list(set(TARGET_DB.cid))
	if CID in target_cids:
		tmp_df = TARGET_DB[TARGET_DB.cid == CID]
		targets = list(set(tmp_df.target))
		gene_symbols = list(BETA_ORDER_DF.L_gene_symbol)
		vec = [1 if a in targets else 0 for a in gene_symbols ]
	else :
		gene_symbols = list(BETA_ORDER_DF.L_gene_symbol)
		vec = [0 for a in gene_symbols ]
	return vec


def get_cell(IND) : 
	cell_res = cell_one_hot[IND]
	return(cell_res)



def get_synergy_data(DrugA_CID, DrugB_CID, Cell):
	ABCS1 = A_B_C_S[A_B_C_S.drug_row_cid == DrugA_CID]
	ABCS2 = ABCS1[ABCS1.drug_col_cid == DrugB_CID]
	ABCS3 = ABCS2[ABCS2.DrugCombCello == Cell]
	synergy_score = np.median(ABCS3.synergy_loewe)
	return synergy_score






A_B_C_S_SET = A_B_C_S_SET.reset_index(drop = True)
# AOBO cid-cid-cell : 7238
# AXBO cid-cid-cell : 8116
# AOBX cid-cid-cell : 10129
# AXBX cid-cid-cell : 414978

# 5_2_3 : cid-cid-cell : 25483
# 5_3_3 : cid-cid-cell : 440461

A_B_C_S_SET_AOBO = A_B_C_S_SET[A_B_C_S_SET.type=='AOBO']
A_B_C_S_SET_AXBO = A_B_C_S_SET[A_B_C_S_SET.type=='AXBO']
A_B_C_S_SET_AOBX = A_B_C_S_SET[A_B_C_S_SET.type=='AOBX']
A_B_C_S_SET_AXBX = A_B_C_S_SET[A_B_C_S_SET.type=='AXBX']

A_B_C_S_SET_AXBO['MJ_x'] = [str(int(A_B_C_S_SET_AXBO.loc[a]['drug_row_cid']))+"__"+A_B_C_S_SET_AXBO.loc[a]['DrugCombCello'] for a in list( A_B_C_S_SET_AXBO.index)]
A_B_C_S_SET_AOBX['MJ_y'] = [str(int(A_B_C_S_SET_AOBX.loc[a]['drug_col_cid']))+"__"+A_B_C_S_SET_AOBX.loc[a]['DrugCombCello'] for a in list( A_B_C_S_SET_AOBX.index)]
A_B_C_S_SET_AXBX['MJ_x'] = [str(int(A_B_C_S_SET_AXBX.loc[a]['drug_row_cid']))+"__"+A_B_C_S_SET_AXBX.loc[a]['DrugCombCello'] for a in list( A_B_C_S_SET_AXBX.index)]
A_B_C_S_SET_AXBX['MJ_y'] = [str(int(A_B_C_S_SET_AXBX.loc[a]['drug_col_cid']))+"__"+A_B_C_S_SET_AXBX.loc[a]['DrugCombCello'] for a in list( A_B_C_S_SET_AXBX.index)]



# 1) MJ_1_MJB
A_B_C_S_SET_AXBO = A_B_C_S_SET_AXBO[A_B_C_S_SET_AXBO.MJ_x.isin(list(MJ_1_MJB_DF.columns))]
A_B_C_S_SET_AOBX = A_B_C_S_SET_AOBX[A_B_C_S_SET_AOBX.MJ_y.isin(list(MJ_1_MJB_DF.columns))]
A_B_C_S_SET_AXBX = A_B_C_S_SET_AXBX[(A_B_C_S_SET_AXBX.MJ_x.isin(list(MJ_1_MJB_DF.columns))) & (A_B_C_S_SET_AXBX.MJ_y.isin(list(MJ_1_MJB_DF.columns)))]

# 2) MJ_2_MJB
A_B_C_S_SET_AXBO = A_B_C_S_SET_AXBO[A_B_C_S_SET_AXBO.MJ_x.isin(list(MJ_2_MJB_DF.columns))]
A_B_C_S_SET_AOBX = A_B_C_S_SET_AOBX[A_B_C_S_SET_AOBX.MJ_y.isin(list(MJ_2_MJB_DF.columns))]
A_B_C_S_SET_AXBX = A_B_C_S_SET_AXBX[(A_B_C_S_SET_AXBX.MJ_x.isin(list(MJ_2_MJB_DF.columns))) & (A_B_C_S_SET_AXBX.MJ_y.isin(list(MJ_2_MJB_DF.columns)))]

# 3) MJ_1_NF
A_B_C_S_SET_AXBO = A_B_C_S_SET_AXBO[A_B_C_S_SET_AXBO.MJ_x.isin(list(MJ_1_NF_DF.columns))]
A_B_C_S_SET_AOBX = A_B_C_S_SET_AOBX[A_B_C_S_SET_AOBX.MJ_y.isin(list(MJ_1_NF_DF.columns))]
A_B_C_S_SET_AXBX = A_B_C_S_SET_AXBX[(A_B_C_S_SET_AXBX.MJ_x.isin(list(MJ_1_NF_DF.columns))) & (A_B_C_S_SET_AXBX.MJ_y.isin(list(MJ_1_NF_DF.columns)))]

# 4) MJ_2_NF
A_B_C_S_SET_AXBO = A_B_C_S_SET_AXBO[A_B_C_S_SET_AXBO.MJ_x.isin(list(MJ_2_NF_DF.columns))]
A_B_C_S_SET_AOBX = A_B_C_S_SET_AOBX[A_B_C_S_SET_AOBX.MJ_y.isin(list(MJ_2_NF_DF.columns))]
A_B_C_S_SET_AXBX = A_B_C_S_SET_AXBX[(A_B_C_S_SET_AXBX.MJ_x.isin(list(MJ_2_NF_DF.columns))) & (A_B_C_S_SET_AXBX.MJ_y.isin(list(MJ_2_NF_DF.columns)))]


# 40926__CVCL_0320
# ABC = A_B_C_S_SET_RE[['drug_row_cid','drug_col_cid','DrugCombCello']]
# A_B_C_S_SET_RE[(A_B_C_S_SET_RE.drug_row_cid == 24856436) & (A_B_C_S_SET_RE.drug_col_cid == 208908) & (A_B_C_S_SET_RE.DrugCombCello == 'CVCL_0132')]


A_B_C_S_SET_RE = pd.concat([A_B_C_S_SET_AOBO, A_B_C_S_SET_AXBO, A_B_C_S_SET_AOBX, A_B_C_S_SET_AXBX]) 
# 5_2_3 & MJ_1_MJB : 12584
# A_B_C_S_SET_RE[['drug_row_cid','drug_col_cid','DrugCombCello']].drop_duplicates() # 11929 (cid : 232)
# 
# 5_3_3 & MJ_1_MJB : 13318
# A_B_C_S_SET_RE[['drug_row_cid','drug_col_cid','DrugCombCello']].drop_duplicates() # 12667 (cid : 219) # diff num because of filtering
#
# 5_2_3 & MJ_2_MJB : 16279
# A_B_C_S_SET_RE[['drug_row_cid','drug_col_cid','DrugCombCello']].drop_duplicates() # 15522 (cid : )
#
# 5_3_3 & MJ_2_MJB : 19921
# A_B_C_S_SET_RE[['drug_row_cid','drug_col_cid','DrugCombCello']].drop_duplicates() # 19164 (cid : )
#
# 5_2_3 & MJ_1_NF : 17316
# A_B_C_S_SET_RE[['drug_row_cid','drug_col_cid','DrugCombCello']].drop_duplicates() # 16675 (cid : )
#
# 5_3_3 & MJ_1_NF : 46613
# A_B_C_S_SET_RE[['drug_row_cid','drug_col_cid','DrugCombCello']].drop_duplicates() # 45972  (cid : )
#
# 5_2_3 & MJ_2_NF : 13786
# A_B_C_S_SET_RE[['drug_row_cid','drug_col_cid','DrugCombCello']].drop_duplicates() # 13089 (cid : )
#
# 5_3_3 & MJ_2_NF : 15672
# A_B_C_S_SET_RE[['drug_row_cid','drug_col_cid','DrugCombCello']].drop_duplicates() # 14975 (cid : )
#


# don't change the index 


MY_chem_A_feat = torch.empty(size=(A_B_C_S_SET_RE.shape[0], max_len, 64))
MY_chem_B_feat= torch.empty(size=(A_B_C_S_SET_RE.shape[0], max_len, 64))
MY_chem_A_adj = torch.empty(size=(A_B_C_S_SET_RE.shape[0], max_len, max_len))
MY_chem_B_adj= torch.empty(size=(A_B_C_S_SET_RE.shape[0], max_len, max_len))
MY_g_feat_A = torch.empty(size=(A_B_C_S_SET_RE.shape[0], 978, 2))
MY_g_feat_B = torch.empty(size=(A_B_C_S_SET_RE.shape[0], 978, 2))
MY_Cell = torch.empty(size=(A_B_C_S_SET_RE.shape[0], cell_one_hot.shape[1]))
MY_syn =  torch.empty(size=(A_B_C_S_SET_RE.shape[0],1))


MY_chem_A_feat = torch.empty(size=(128, max_len, 64))
MY_chem_B_feat= torch.empty(size=(128, max_len, 64))
MY_chem_A_adj = torch.empty(size=(128, max_len, max_len))
MY_chem_B_adj= torch.empty(size=(128, max_len, max_len))
MY_g_feat_A = torch.empty(size=(128, 978, 2))
MY_g_feat_B = torch.empty(size=(128, 978, 2))
MY_Cell = torch.empty(size=(128, cell_one_hot.shape[1]))
MY_syn =  torch.empty(size=(128,1))



MY_chem_A_feat_long = torch.empty(size=(A_B_C_S_SET_RE.shape[0], max_len, 64))
MY_chem_B_feat_long= torch.empty(size=(A_B_C_S_SET_RE.shape[0], max_len, 64))
MY_chem_A_adj_long = torch.empty(size=(A_B_C_S_SET_RE.shape[0], max_len, max_len))
MY_chem_B_adj_long= torch.empty(size=(A_B_C_S_SET_RE.shape[0], max_len, max_len))
MY_g_feat_A_long = torch.empty(size=(A_B_C_S_SET_RE.shape[0], 978, 2))
MY_g_feat_B_long = torch.empty(size=(A_B_C_S_SET_RE.shape[0], 978, 2))
MY_Cell_long = torch.empty(size=(A_B_C_S_SET_RE.shape[0], cell_one_hot.shape[1]))
MY_syn_long =  torch.empty(size=(A_B_C_S_SET_RE.shape[0],1))

MY_chem_A_feat_long[0:39371] = MY_chem_A_feat
MY_chem_B_feat_long[0:39371] = MY_chem_B_feat
MY_chem_A_adj_long[0:39371] = MY_chem_A_adj
MY_chem_B_adj_long[0:39371] = MY_chem_B_adj
MY_g_feat_A_long[0:39371] = MY_g_feat_A
MY_g_feat_B_long[0:39371] = MY_g_feat_B
MY_Cell_long[0:39371] = MY_Cell
MY_syn_long[0:39371] = MY_syn

MY_chem_A_feat = MY_chem_A_feat_long
MY_chem_B_feat = MY_chem_B_feat_long
MY_chem_A_adj = MY_chem_A_adj_long
MY_chem_B_adj = MY_chem_B_adj_long
MY_g_feat_A = MY_g_feat_A_long
MY_g_feat_B = MY_g_feat_B_long
MY_Cell = MY_Cell_long
MY_syn = MY_syn_long



Fail_ind = []
from datetime import datetime


for IND in range(10000, MY_chem_A_feat.shape[0]) : #  MY_chem_A_feat.shape[0] # a = random.sample(range(0,31650), 128) # 18698
	if IND%100 == 0 :
		print(IND)
		Fail_ind
		datetime.now()
	index_num = list(A_B_C_S_SET_RE.index)[IND]
	dat_type = A_B_C_S_SET_RE.iloc[IND,]['type']
	DrugA_SIG = A_B_C_S_SET_RE.iloc[IND,]['BETA_sig_id_x']
	DrugB_SIG = A_B_C_S_SET_RE.iloc[IND,]['BETA_sig_id_y']
	DrugA_CID = A_B_C_S_SET_RE.iloc[IND,]['cid_x']
	DrugB_CID = A_B_C_S_SET_RE.iloc[IND,]['cid_y']
	Cell = A_B_C_S_SET_RE.iloc[IND,]['DrugCombCello']
	#
	k=1
	DrugA_Feat, DrugA_ADJ = get_CHEM(DrugA_CID, k)
	DrugB_Feat, DrugB_ADJ = get_CHEM(DrugB_CID, k)
	#
	if dat_type == "AOBO" : 
		EXP_A = get_LINCS_data(DrugA_SIG)
		EXP_B = get_LINCS_data(DrugB_SIG)
	elif dat_type == "AXBO" :
		try:
			EXP_A = get_MJ_data(DrugA_CID, Cell, dat_type)
			EXP_B = get_LINCS_data(DrugB_SIG)
		except :
			EXP_A = [0]*978
			EXP_B = [0]*978
			Fail_ind.append(IND)
	elif dat_type == "AOBX" :
		try:
			EXP_A = get_LINCS_data(DrugA_SIG)
			EXP_B = get_MJ_data(DrugB_CID, Cell, dat_type)
		except :
			EXP_A = [0]*978
			EXP_B = [0]*978
			Fail_ind.append(IND)
	elif dat_type == "AXBX" :
		try:
			EXP_A = get_MJ_data(DrugA_CID, Cell, dat_type)
			EXP_B = get_MJ_data(DrugB_CID, Cell, dat_type)
		except :
			EXP_A = [0]*978
			EXP_B = [0]*978
			Fail_ind.append(IND)       
	else : 
		EXP_A = [0]*978
		EXP_B = [0]*978
		Fail_ind.append(IND)   
	#
	TGT_A = get_targets(DrugA_CID)
	TGT_B = get_targets(DrugB_CID)
	#
	ARR_A = np.array([EXP_A, TGT_A]).T
	ARR_B = np.array([EXP_B, TGT_B]).T
	#
	Cell_Vec = get_cell(index_num)
	#
	AB_SYN = get_synergy_data(DrugA_CID, DrugB_CID, Cell)
	#
	#
	MY_chem_A_feat[IND] = torch.Tensor(DrugA_Feat)
	MY_chem_B_feat[IND] = torch.Tensor(DrugB_Feat)
	MY_chem_A_adj[IND] = torch.Tensor(DrugA_ADJ)
	MY_chem_B_adj[IND] = torch.Tensor(DrugB_ADJ)	
	MY_g_feat_A[IND] = torch.Tensor(ARR_A)	
	MY_g_feat_B[IND] = torch.Tensor(ARR_B)	
	MY_Cell[IND] = Cell_Vec
	MY_syn[IND] = torch.Tensor([AB_SYN])



True_ind = [a for a in range(MY_chem_A_feat.shape[0]) if a not in Fail_ind]
MY_chem_A_feat_re = MY_chem_A_feat[True_ind]
MY_chem_B_feat_re = MY_chem_B_feat[True_ind]
MY_chem_A_adj_re = MY_chem_A_adj[True_ind]
MY_chem_B_adj_re = MY_chem_B_adj[True_ind]
MY_g_feat_A_re = MY_g_feat_A[True_ind]
MY_g_feat_B_re = MY_g_feat_B[True_ind]
MY_Cell_re = MY_Cell[True_ind]
MY_syn_re = MY_syn[True_ind]


G_NAME = '5_2_3.MJ_1_MJB'
SAVE_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_6/DATA.5_2_3/'
torch.save(MY_chem_A_feat_re, SAVE_PATH+'1001.{}.MY_chem_A_feat.pt'.format(G_NAME))
torch.save(MY_chem_B_feat_re, SAVE_PATH+'1001.{}.MY_chem_B_feat.pt'.format(G_NAME))
torch.save(MY_chem_A_adj_re, SAVE_PATH+'1001.{}.MY_chem_A_adj.pt'.format(G_NAME))
torch.save(MY_chem_B_adj_re, SAVE_PATH+'1001.{}.MY_chem_B_adj.pt'.format(G_NAME))
torch.save(MY_g_feat_A_re, SAVE_PATH+'1001.{}.MY_g_feat_A.pt'.format(G_NAME))
torch.save(MY_g_feat_B_re, SAVE_PATH+'1001.{}.MY_g_feat_B.pt'.format(G_NAME))
torch.save(MY_Cell_re, SAVE_PATH+'1001.{}.MY_Cell.pt'.format(G_NAME))
torch.save(MY_syn_re, SAVE_PATH+'1001.{}.MY_syn.pt'.format(G_NAME))

A_B_C_S_SET_RE.to_csv(SAVE_PATH+'5_2_3.MJ_1_MJB.A_B_C_S.csv')





G_NAME = '5_3_3.MJ_1_MJB'
SAVE_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_6/DATA.5_3_3/'
torch.save(MY_chem_A_feat_re, SAVE_PATH+'1002.{}.MY_chem_A_feat.pt'.format(G_NAME))
torch.save(MY_chem_B_feat_re, SAVE_PATH+'1002.{}.MY_chem_B_feat.pt'.format(G_NAME))
torch.save(MY_chem_A_adj_re, SAVE_PATH+'1002.{}.MY_chem_A_adj.pt'.format(G_NAME))
torch.save(MY_chem_B_adj_re, SAVE_PATH+'1002.{}.MY_chem_B_adj.pt'.format(G_NAME))
torch.save(MY_g_feat_A_re, SAVE_PATH+'1002.{}.MY_g_feat_A.pt'.format(G_NAME))
torch.save(MY_g_feat_B_re, SAVE_PATH+'1002.{}.MY_g_feat_B.pt'.format(G_NAME))
torch.save(MY_Cell_re, SAVE_PATH+'1002.{}.MY_Cell.pt'.format(G_NAME))
torch.save(MY_syn_re, SAVE_PATH+'1002.{}.MY_syn.pt'.format(G_NAME))

A_B_C_S_SET_RE.to_csv(SAVE_PATH+'5_3_3.MJ_1_MJB.A_B_C_S.csv')








G_NAME = '5_2_3.MJ_2_MJB'
SAVE_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_6/DATA.5_2_3/'
torch.save(MY_chem_A_feat_re, SAVE_PATH+'1002.{}.MY_chem_A_feat.pt'.format(G_NAME))
torch.save(MY_chem_B_feat_re, SAVE_PATH+'1002.{}.MY_chem_B_feat.pt'.format(G_NAME))
torch.save(MY_chem_A_adj_re, SAVE_PATH+'1002.{}.MY_chem_A_adj.pt'.format(G_NAME))
torch.save(MY_chem_B_adj_re, SAVE_PATH+'1002.{}.MY_chem_B_adj.pt'.format(G_NAME))
torch.save(MY_g_feat_A_re, SAVE_PATH+'1002.{}.MY_g_feat_A.pt'.format(G_NAME))
torch.save(MY_g_feat_B_re, SAVE_PATH+'1002.{}.MY_g_feat_B.pt'.format(G_NAME))
torch.save(MY_Cell_re, SAVE_PATH+'1002.{}.MY_Cell.pt'.format(G_NAME))
torch.save(MY_syn_re, SAVE_PATH+'1002.{}.MY_syn.pt'.format(G_NAME))

A_B_C_S_SET_RE.to_csv(SAVE_PATH+'5_2_3.MJ_2_MJB.A_B_C_S.csv')



G_NAME = '5_3_3.MJ_2_MJB'
SAVE_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_6/DATA.5_3_3/'
torch.save(MY_chem_A_feat_re, SAVE_PATH+'1002.{}.MY_chem_A_feat.pt'.format(G_NAME))
torch.save(MY_chem_B_feat_re, SAVE_PATH+'1002.{}.MY_chem_B_feat.pt'.format(G_NAME))
torch.save(MY_chem_A_adj_re, SAVE_PATH+'1002.{}.MY_chem_A_adj.pt'.format(G_NAME))
torch.save(MY_chem_B_adj_re, SAVE_PATH+'1002.{}.MY_chem_B_adj.pt'.format(G_NAME))
torch.save(MY_g_feat_A_re, SAVE_PATH+'1002.{}.MY_g_feat_A.pt'.format(G_NAME))
torch.save(MY_g_feat_B_re, SAVE_PATH+'1002.{}.MY_g_feat_B.pt'.format(G_NAME))
torch.save(MY_Cell_re, SAVE_PATH+'1002.{}.MY_Cell.pt'.format(G_NAME))
torch.save(MY_syn_re, SAVE_PATH+'1002.{}.MY_syn.pt'.format(G_NAME))

A_B_C_S_SET_RE.to_csv(SAVE_PATH+'5_3_3.MJ_2_MJB.A_B_C_S.csv')




G_NAME = '5_2_3.MJ_1_NF'
SAVE_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_GPU/PJ01.TRIAL.5_2_3.MJ_1_NF/'
torch.save(MY_chem_A_feat_re, SAVE_PATH+'1006.{}.MY_chem_A_feat.pt'.format(G_NAME))
torch.save(MY_chem_B_feat_re, SAVE_PATH+'1006.{}.MY_chem_B_feat.pt'.format(G_NAME))
torch.save(MY_chem_A_adj_re, SAVE_PATH+'1006.{}.MY_chem_A_adj.pt'.format(G_NAME))
torch.save(MY_chem_B_adj_re, SAVE_PATH+'1006.{}.MY_chem_B_adj.pt'.format(G_NAME))
torch.save(MY_g_feat_A_re, SAVE_PATH+'1006.{}.MY_g_feat_A.pt'.format(G_NAME))
torch.save(MY_g_feat_B_re, SAVE_PATH+'1006.{}.MY_g_feat_B.pt'.format(G_NAME))
torch.save(MY_Cell_re, SAVE_PATH+'1006.{}.MY_Cell.pt'.format(G_NAME))
torch.save(MY_syn_re, SAVE_PATH+'1006.{}.MY_syn.pt'.format(G_NAME))

A_B_C_S_SET_RE.to_csv(SAVE_PATH+'5_2_3.MJ_1_NF.A_B_C_S.csv')




G_NAME = '5_3_3.MJ_1_NF'
SAVE_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_GPU/PJ01.TRIAL.5_3_3.MJ_1_NF/'
torch.save(MY_chem_A_feat_re, SAVE_PATH+'1006.{}.MY_chem_A_feat.pt'.format(G_NAME))
torch.save(MY_chem_B_feat_re, SAVE_PATH+'1006.{}.MY_chem_B_feat.pt'.format(G_NAME))
torch.save(MY_chem_A_adj_re, SAVE_PATH+'1006.{}.MY_chem_A_adj.pt'.format(G_NAME))
torch.save(MY_chem_B_adj_re, SAVE_PATH+'1006.{}.MY_chem_B_adj.pt'.format(G_NAME))
torch.save(MY_g_feat_A_re, SAVE_PATH+'1006.{}.MY_g_feat_A.pt'.format(G_NAME))
torch.save(MY_g_feat_B_re, SAVE_PATH+'1006.{}.MY_g_feat_B.pt'.format(G_NAME))
torch.save(MY_Cell_re, SAVE_PATH+'1006.{}.MY_Cell.pt'.format(G_NAME))
torch.save(MY_syn_re, SAVE_PATH+'1006.{}.MY_syn.pt'.format(G_NAME))

A_B_C_S_SET_RE.to_csv(SAVE_PATH+'5_3_3.MJ_1_NF.A_B_C_S.csv')









G_NAME = '5_2_3.MJ_2_NF'
SAVE_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_6/DATA.5_2_3/'
torch.save(MY_chem_A_feat_re, SAVE_PATH+'1003.{}.MY_chem_A_feat.pt'.format(G_NAME))
torch.save(MY_chem_B_feat_re, SAVE_PATH+'1003.{}.MY_chem_B_feat.pt'.format(G_NAME))
torch.save(MY_chem_A_adj_re, SAVE_PATH+'1003.{}.MY_chem_A_adj.pt'.format(G_NAME))
torch.save(MY_chem_B_adj_re, SAVE_PATH+'1003.{}.MY_chem_B_adj.pt'.format(G_NAME))
torch.save(MY_g_feat_A_re, SAVE_PATH+'1003.{}.MY_g_feat_A.pt'.format(G_NAME))
torch.save(MY_g_feat_B_re, SAVE_PATH+'1003.{}.MY_g_feat_B.pt'.format(G_NAME))
torch.save(MY_Cell_re, SAVE_PATH+'1003.{}.MY_Cell.pt'.format(G_NAME))
torch.save(MY_syn_re, SAVE_PATH+'1003.{}.MY_syn.pt'.format(G_NAME))

A_B_C_S_SET_RE.to_csv(SAVE_PATH+'5_2_3.MJ_2_NF.A_B_C_S.csv')





G_NAME = '5_3_3.MJ_2_NF'
SAVE_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_6/DATA.5_3_3/'
torch.save(MY_chem_A_feat_re, SAVE_PATH+'1003.{}.MY_chem_A_feat.pt'.format(G_NAME))
torch.save(MY_chem_B_feat_re, SAVE_PATH+'1003.{}.MY_chem_B_feat.pt'.format(G_NAME))
torch.save(MY_chem_A_adj_re, SAVE_PATH+'1003.{}.MY_chem_A_adj.pt'.format(G_NAME))
torch.save(MY_chem_B_adj_re, SAVE_PATH+'1003.{}.MY_chem_B_adj.pt'.format(G_NAME))
torch.save(MY_g_feat_A_re, SAVE_PATH+'1003.{}.MY_g_feat_A.pt'.format(G_NAME))
torch.save(MY_g_feat_B_re, SAVE_PATH+'1003.{}.MY_g_feat_B.pt'.format(G_NAME))
torch.save(MY_Cell_re, SAVE_PATH+'1003.{}.MY_Cell.pt'.format(G_NAME))
torch.save(MY_syn_re, SAVE_PATH+'1003.{}.MY_syn.pt'.format(G_NAME))

A_B_C_S_SET_RE.to_csv(SAVE_PATH+'5_3_3.MJ_2_NF.A_B_C_S.csv')


# save and move to kisti 










아니 그리고 왜 smiles 가 일치를 안하지
DC 랑 pubchem 전체를 merge 했는데
중복되서 나오는 애들은 왜그런거야
Lincs 를 만들때 민지가 pubchem py 쓴게 문제가 있나?
아니면 그냥 다 pubchem 전체 필터로 진행해보자 
아 짜증나 진짜 

tmp_sm = pd.concat( [L_smiles, DC_smiles] )
tmp_sm2 = tmp_sm.drop_duplicates()
tmp_sm2 = tmp_sm2.sort_values('cid')
tmp_sm2['leng'] = [int(a) if a !='error' else 0 for a in list(tmp_sm2['leng']) ]
tmp_sm2 = tmp_sm2.drop_duplicates()

smsm = list(set(tmp_sm2.cid))
smt = list(tmp_sm2['cid'])
[a for a in smsm if smt.count(a) >1]

[a for a in range(8159) if type(list(DC_smiles.canonical_smiles_re)[a])==float]

25791047

9830392

그래 이제 그러면 맞춰서 가져오는걸로 합시다 


plot_dir = '/st06/jiyeonH/11.TOX/MY_TRIAL_6/figs/'
import seaborn 
sns.set_context('talk', font_scale=1.1)
plt.figure(figsize=(8,6))
sns.violinplot(y = 'leng', x= 'nana', data =len_sm, color = '#cde7ff', inner= 'quartile')
sns.stripplot(y = 'leng', x = 'nana', data = len_sm, edgecolor= 'gray', jitter =True, alpha = 0.5)
plt.savefig(plot_dir+"Violin_atlen.png", format = 'png', dpi=300)


그래서 max_len 은 100 으로 정하기로 했어요
gogo









# just length check 
smiles_df_L = BETA_MJ[['SMILES_cid','canonical_smiles_re']].drop_duplicates() # 원래 Lincs 에서의 smiles list 
smiles_df_DC = DC_DRUG_DF[['cid','smiles']].drop_duplicates() # drugcomb 에서 주는 모든 smiles list

# 혹시 lincs 랑 DC 랑 겹치는 cid 들 중에서 smiles 가 다른 경우에는 어떡하지 라는 노파심 
comm = smiles_df_L[smiles_df_L.SMILES_cid.isin(smiles_df_DC.cid)]['SMILES_cid'] # Lincs 랑 DC 겹치는 cid 들 
comm_L = smiles_df_L[smiles_df_L.SMILES_cid.isin(comm)]
comm_DC = smiles_df_DC[smiles_df_DC.cid.isin(comm)]

L_sm = list(comm_L.sort_values('SMILES_cid')['canonical_smiles_re'])
dc_sm = list(comm_DC.sort_values('cid')['smiles'])
# 응 무려 2747 중에서 1768 만 같고, 천여개는 다르다고 한다 
# 지금까지는 그냥 cid가 pubchem 중심이라서 달라도 상관이 없었는데, 이렇게 되면 DC 에 있는 cid 들에 대해서 smiles 한번 다 바꿔야할듯 


PUBCHEM_ALL = pd.read_csv('/st06/jiyeonH/12.HTP_DB/08.PUBCHEM/PUBCHEM_MJ_031022.csv',  low_memory = False)
# 그럼 그냥 DC 도 맞춰버리자
# DC 맞춰서 저장






# lincs 




BETA_CID_CELLO_SIG.columns=['pert_id', 'drug_row_cid', 'DrugCombCello', 'BETA_sig_id']
CELLO_DC_BETA_1 = pd.merge(DC_cello_final_dup, BETA_CID_CELLO_SIG, on = ['drug_row_cid','DrugCombCello'], how = 'left') # 731051 -> 720619

BETA_CID_CELLO_SIG.columns=['pert_id', 'drug_col_cid', 'DrugCombCello', 'BETA_sig_id']
CELLO_DC_BETA_2 = pd.merge(CELLO_DC_BETA_1, BETA_CID_CELLO_SIG, on = ['drug_col_cid','DrugCombCello'], how = 'left') # 731644 -> 721206

BETA_CID_CELLO_SIG.columns=['pert_id', 'pubchem_cid', 'cellosaurus_id', 'sig_id']

CELLO_DC_BETA_2.to_csv('/st06/jiyeonH/13.DD_SESS/00.PRJ/DC_EXP_Full.csv', '\t')


# (1) AO BO  
FILTER_AO_BO = [a for a in range(CELLO_DC_BETA_2.shape[0]) if (type(CELLO_DC_BETA_2.BETA_sig_id_x[a]) == str) & (type(CELLO_DC_BETA_2.BETA_sig_id_y[a]) == str)]
DATA_AO_BO = CELLO_DC_BETA_2.loc[FILTER_AO_BO] # 11742 -> 11379
DATA_AO_BO[['BETA_sig_id_x','BETA_sig_id_y','DrugCombCello']].drop_duplicates() # 9230 -> 8914
DATA_AO_BO_cids = list(set(list(DATA_AO_BO.drug_row_cid) + list(DATA_AO_BO.drug_col_cid))) # 172 


# (2) AX BO 
FILTER_AX_BO = [a for a in range(CELLO_DC_BETA_2.shape[0]) if (type(CELLO_DC_BETA_2.BETA_sig_id_x[a]) != str) & (type(CELLO_DC_BETA_2.BETA_sig_id_y[a]) == str)]
DATA_AX_BO = CELLO_DC_BETA_2.loc[FILTER_AX_BO] # 11967
DATA_AX_BO[['BETA_sig_id_x','BETA_sig_id_y','DrugCombCello']].drop_duplicates() # 452
DATA_AX_BO_cids = list(set(list(DATA_AX_BO.drug_row_cid) + list(DATA_AX_BO.drug_col_cid))) # 635 

DATA_AX_BO.to_csv('/st06/jiyeonH/13.DD_SESS/00.PRJ/DC_EXP_JY1.csv', sep= '\t')

tmp = DATA_AX_BO[['drug_row_cid','DrugCombCello']].drop_duplicates()
tmp1 = [str(a) for a in tmp['drug_row_cid']]
tmp2 = [tmp1[a] + "__" + list(tmp['DrugCombCello'])[a] for a in range(tmp.shape[0])] # 1274

# (3) AO BX 
FILTER_AO_BX = [a for a in range(CELLO_DC_BETA_2.shape[0]) if (type(CELLO_DC_BETA_2.BETA_sig_id_x[a]) == str) & (type(CELLO_DC_BETA_2.BETA_sig_id_y[a]) != str)]
DATA_AO_BX = CELLO_DC_BETA_2.loc[FILTER_AO_BX] # 14998
DATA_AO_BX[['BETA_sig_id_x','BETA_sig_id_y','DrugCombCello']].drop_duplicates() # 449
DATA_AO_BX_cids = list(set(list(DATA_AO_BX.drug_row_cid) + list(DATA_AO_BX.drug_col_cid))) # 274 

DATA_AO_BX.to_csv('/st06/jiyeonH/13.DD_SESS/00.PRJ/DC_EXP_JY2.csv', sep= '\t')

tmp = DATA_AO_BX[['drug_col_cid','DrugCombCello']].drop_duplicates()
tmp1 = [str(a) for a in tmp['drug_col_cid']]
tmp2 = [tmp1[a] + "__" + list(tmp['DrugCombCello'])[a] for a in range(len(tmp1))] # 900


# (4) AX BX 
FILTER_AX_BX = [a for a in range(CELLO_DC_BETA_2.shape[0]) if (type(CELLO_DC_BETA_2.BETA_sig_id_x[a]) != str) & (type(CELLO_DC_BETA_2.BETA_sig_id_y[a]) != str)]
DATA_AX_BX = CELLO_DC_BETA_2.loc[FILTER_AX_BX] # 682862
DATA_AX_BX[['BETA_sig_id_x','BETA_sig_id_y','DrugCombCello']].drop_duplicates() # 의미가 없음 
DATA_AX_BX_cids = list(set(list(DATA_AX_BX.drug_row_cid) + list(DATA_AX_BX.drug_col_cid))) # 4280 

DATA_AX_BX.to_csv('/st06/jiyeonH/13.DD_SESS/00.PRJ/DC_EXP_JY3.csv', sep= '\t')


tmp_r = DATA_AX_BX[['drug_row_cid','DrugCombCello']].drop_duplicates()
tmp_c = DATA_AX_BX[['drug_col_cid','DrugCombCello']].drop_duplicates()
tmp1 = [str(a) for a in tmp_r['drug_row_cid']]
tmp2 = [str(a) for a in tmp_c['drug_col_cid']]
tmp3 = [tmp1[a] + "__" + list(tmp_r['DrugCombCello'])[a] for a in range(len(tmp1))] # 
tmp4 = [tmp2[a] + "__" + list(tmp_c['DrugCombCello'])[a] for a in range(len(tmp2))] # 
len(set(tmp3+tmp4)) # 22,051

DATA_AO_BO['type'] = 'AOBO'
DATA_AX_BO['type'] = 'AXBO'
DATA_AO_BX['type'] = 'AOBX'
DATA_AX_BX['type'] = 'AXBX'


######################################################
