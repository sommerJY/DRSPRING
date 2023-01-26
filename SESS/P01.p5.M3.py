아예5_1_3 가져다가 M1 만 바꿔가지고 진행해보기 
대체 왜 결과가 이렇게 제대로 안나오는건지 이해가 안되므로 


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

max_len = 50
len_sm = pd.concat( [ L_smiles, DC_smiles ] )
len_sm['leng'] = [int(a) if a !='error' else 0 for a in list(len_sm['leng']) ]
len_sm = len_sm.drop_duplicates() # 31054

len_sm_100 = len_sm[len_sm.leng<=50] # 31014


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




A_B_C_S_SET_ROW_CHECK = [int(a) for a in list(A_B_C_S_SET.drug_row_cid)]
A_B_C_S_SET_COL_CHECK = [int(a) for a in list(A_B_C_S_SET.drug_col_cid)]
A_B_C_S_SET_CELL_CHECK = list(A_B_C_S_SET.DrugCombCello)

A_B_C_S_SET['ROWCHECK'] = [str(int(A_B_C_S_SET_ROW_CHECK[i]))+'__'+A_B_C_S_SET_CELL_CHECK[i] for i in range(A_B_C_S_SET.shape[0])]
A_B_C_S_SET['COLCHECK'] = [str(int(A_B_C_S_SET_COL_CHECK[i]))+'__'+A_B_C_S_SET_CELL_CHECK[i] for i in range(A_B_C_S_SET.shape[0])]



# read MJ data -> full 말고 deside 로 다시 가보자 

MJ_DIR = '/st06/jiyeonH/13.DD_SESS/01.PRJ2/'
####MJ_request_ANS = pd.read_csv(MJ_DIR+'PRJ2_EXP_fugcn_vers1.csv')
MJ_request_ANS = pd.read_csv(MJ_DIR+'PRJ2_EXP_deside_NF.csv')

MJ_request_ANS2 = copy.deepcopy(MJ_request_ANS)
MJ_request_ANS_COL = list(MJ_request_ANS2.columns)

MJ_request_ANS_re_col = MJ_request_ANS_COL[0:3]+[int(a.split('__')[0])  for a in MJ_request_ANS_COL[3:]]
MJ_request_ANS2.columns =MJ_request_ANS_re_col 
MJ_request_ANS3 = MJ_request_ANS2.T.drop_duplicates().T


# deside & mu
A_B_C_S_SET_MJ = A_B_C_S_SET[A_B_C_S_SET.drug_row_cid.isin(MJ_request_ANS3.columns)]
A_B_C_S_SET_MJ = A_B_C_S_SET_MJ[A_B_C_S_SET_MJ.drug_col_cid.isin(MJ_request_ANS3.columns)]
A_B_C_S_SET_MJ = A_B_C_S_SET_MJ.reset_index(drop = True)



# Cell line vector
ABCS_cells= list(set(A_B_C_S_SET_MJ.DrugCombCello))
DC_CELL_DF3 = copy.deepcopy(DC_CELL_DF2)
DC_CELL_DF3 = DC_CELL_DF3[DC_CELL_DF3.DrugCombCello!='NA']
DC_CELL_DF4 = DC_CELL_DF3[DC_CELL_DF3.DrugCombCello.isin(ABCS_cells)]
DC_CELL_DF4.iloc[list(DC_CELL_DF4.DrugCombCello).index('CVCL_0395')]['tissue'] = 'PROSTATE'
DC_CELL_DF4.iloc[list(DC_CELL_DF4.DrugCombCello).index('CVCL_A442')]['tissue'] = 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE'
DC_CELL_DF4['cell_onehot'] = [a for a in range( DC_CELL_DF4.shape[0])]

A_B_C_S_SET_COH = pd.merge(A_B_C_S_SET_MJ, DC_CELL_DF4[['DrugCombCello','DC_cellname','cell_onehot']], on = 'DrugCombCello', how = 'left'  )

cell_one_hot = torch.nn.functional.one_hot(torch.Tensor(A_B_C_S_SET_COH['cell_onehot']).long())

A_B_C_S_SET_RE = copy.deepcopy(A_B_C_S_SET_COH)




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



# LINCS exp order 따지기 
BETA_ORDER_pre = [list(LINCS_978.new_node).index(a) for a in JY_GRAPH_ORDER]
BETA_ORDER_DF = LINCS_978.iloc[BETA_ORDER_pre] # 어차피 ref 다르고 같은 애들이라 괜춘 
BETA_ENTREZ_ORDER = list(BETA_ORDER_DF.gene_id)
BETA_SYMBOL_ORDER = list(BETA_ORDER_DF.gene_symbol)
BETA_NEWNOD_ORDER = list(BETA_ORDER_DF.new_node)


# DESIDE(M1) / mu1 (M2)
def get_MJ_data( CID ): 
	if CID in list(MJ_request_ANS3.columns) :
		MJ_DATA = MJ_request_ANS3[['entrez_id', CID]]
		ord = [list(MJ_DATA.entrez_id).index(a) for a in BETA_ORDER]
		MJ_DATA_re = MJ_DATA.loc[ord] 
		RES = MJ_DATA_re[CID]
		OX = 'O'
	else : 
		RES = [0]*978
		OX = 'X'
	return list(RES), OX







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


for IND in range(0, MY_chem_A_feat.shape[0]) : #  MY_chem_A_feat.shape[0] # a = random.sample(range(0,31650), 128) # 18698
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
		EXP_A, OX = get_MJ_data(DrugA_CID)
		EXP_B = get_LINCS_data(DrugB_SIG)
		if OX == 'X' :
			Fail_ind.append(IND)
	elif dat_type == "AOBX" :
		EXP_A = get_LINCS_data(DrugA_SIG)
		EXP_B, OX = get_MJ_data(DrugB_CID)
		if OX == 'X' :
			Fail_ind.append(IND)
	elif dat_type == "AXBX" :
		EXP_A, OX1 = get_MJ_data(DrugA_CID)
		EXP_B, OX2 = get_MJ_data(DrugB_CID)
		if 'X' in [OX1, OX2]:
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





G_NAME = 'oldver.MJ_3.MISS_2'
SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3_OLDJ/'
torch.save(MY_chem_A_feat_re, SAVE_PATH+'1222.{}.MY_chem_A_feat.pt'.format(G_NAME))
torch.save(MY_chem_B_feat_re, SAVE_PATH+'1222.{}.MY_chem_B_feat.pt'.format(G_NAME))
torch.save(MY_chem_A_adj_re, SAVE_PATH+'1222.{}.MY_chem_A_adj.pt'.format(G_NAME))
torch.save(MY_chem_B_adj_re, SAVE_PATH+'1222.{}.MY_chem_B_adj.pt'.format(G_NAME))
torch.save(MY_g_feat_A_re, SAVE_PATH+'1222.{}.MY_g_feat_A.pt'.format(G_NAME))
torch.save(MY_g_feat_B_re, SAVE_PATH+'1222.{}.MY_g_feat_B.pt'.format(G_NAME))
torch.save(MY_Cell_re, SAVE_PATH+'1222.{}.MY_Cell.pt'.format(G_NAME))
torch.save(MY_syn_re, SAVE_PATH+'1222.{}.MY_syn.pt'.format(G_NAME))

A_B_C_S_SET_RE.to_csv(SAVE_PATH+'1222.OLD.MJ_3.MISS_2.A_B_C_S.csv')


# save and move to kisti 







###############################################
###############################################

###############################################
###############################################

###############################################
###############################################

###############################################
###############################################

###############################################
###############################################

###############################################
###############################################

###############################################
###############################################

###############################################
###############################################



G_NAME = '5_3_3'

WORK_PATH = '/home01/k020a01/03.old_trials/DATA/'
# WORK_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_GPU/PJ01.TRIAL.5_3_3.MJ_1_NF/'

MY_chem_A_feat = torch.load(WORK_PATH+'1006.{}.MJ_1_NF.MY_chem_A_feat.pt'.format(G_NAME))
MY_chem_B_feat = torch.load(WORK_PATH+'1006.{}.MJ_1_NF.MY_chem_B_feat.pt'.format(G_NAME))
MY_chem_A_adj = torch.load(WORK_PATH+'1006.{}.MJ_1_NF.MY_chem_A_adj.pt'.format(G_NAME))
MY_chem_B_adj = torch.load(WORK_PATH+'1006.{}.MJ_1_NF.MY_chem_B_adj.pt'.format(G_NAME))
MY_g_feat_A = torch.load(WORK_PATH+'1006.{}.MJ_1_NF.MY_g_feat_A.pt'.format(G_NAME))
MY_g_feat_B = torch.load(WORK_PATH+'1006.{}.MJ_1_NF.MY_g_feat_B.pt'.format(G_NAME))
MY_Cell = torch.load(WORK_PATH+'1006.{}.MJ_1_NF.MY_Cell.pt'.format(G_NAME))
MY_syn = torch.load(WORK_PATH+'1006.{}.MJ_1_NF.MY_syn.pt'.format(G_NAME))

A_B_C_S_SET = pd.read_csv(WORK_PATH+'5_3_3.MJ_1_NF.A_B_C_S.csv')



print('NETWORK')

hunet_dir = '/home01/k020a01/01.Data/HumanNet/'
# hunet_dir = '/st06/jiyeonH/13.DD_SESS/HumanNetV3/'
IDK_PATH = '/home01/k020a01/01.Data/IDK/'
# IDK_PATH = '/st06/jiyeonH/13.DD_SESS/ideker/'


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


TARGET_PATH = '/home01/k020a01/01.Data/TARGET/'
# TARGET_PATH = '/st06/jiyeonH/13.DD_SESS/merged_target/'
TARGET_DB = pd.read_csv(TARGET_PATH+'combined_target.csv', low_memory=False)





# Graph 확인 

JY_GRAPH = MY_G
JY_GRAPH_ORDER = MY_G.nodes()
JY_ADJ = nx.adjacency_matrix(JY_GRAPH)

JY_ADJ_tmp = torch.LongTensor(JY_ADJ.toarray())
JY_ADJ_IDX = JY_ADJ_tmp.to_sparse().indices()
JY_IDX_WEIGHT = MY_WEIGHT_SCORE





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




# 아 smiles 읽어서 붙여줘야하는걸 안해줌. 
# 
PC_PATH = '/home01/k020a01/01.Data/Pubchem/'
# PC_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/'
for_CAN_smiles = pd.read_csv(PC_PATH+'CID_SMILES.csv',index_col=0)


for_CAN_smiles.columns = ['drug_row_cid','ROW_CAN_SMILES']
A_B_C_S_SET_SM_1 = pd.merge(A_B_C_S_SET, for_CAN_smiles, on = 'drug_row_cid', how = 'left')
for_CAN_smiles.columns = ['drug_col_cid','COL_CAN_SMILES']
A_B_C_S_SET_SM_2 = pd.merge(A_B_C_S_SET_SM_1, for_CAN_smiles, on = 'drug_col_cid', how = 'left')

A_B_C_S_SET_SM = copy.deepcopy(A_B_C_S_SET_SM_2)


A_B_C_S_SET_SM['ori_index'] = list(A_B_C_S_SET_SM.index)
aa = list(A_B_C_S_SET_SM['ROW_CAN_SMILES'])
bb = list(A_B_C_S_SET_SM['COL_CAN_SMILES'])
cc = list(A_B_C_S_SET_SM['DrugCombCello'])
A_B_C_S_SET_SM['SM_C_CHECK'] = [aa[i] + '___' + bb[i]+ '___' + cc[i] for i in range(A_B_C_S_SET_SM.shape[0])]


# get unique values, remove duplicates, but keep original counts
data_no_dup, counts = np.unique(list(A_B_C_S_SET_SM['SM_C_CHECK']), return_counts=True)
data_no_dup_cells = [setset.split('___')[2] for setset in data_no_dup]
data_nodup_df = pd.DataFrame({'setset' : data_no_dup.tolist(), 'cell' : data_no_dup_cells })



grouped_df = data_nodup_df.groupby('cell')

CV_1_list = []; CV_2_list = []; CV_3_list = []; CV_4_list = []; CV_5_list = []
CV_6_list = []; CV_7_list = []; CV_8_list = []; CV_9_list = []; CV_10_list = []

for i, g in grouped_df:
	if len(g) > 10 :
		nums = int(.1 * len(g))
		bins = []
		for ii in list(range(0, len(g), nums)):
			if len(bins)<= 9 :
				bins.append(ii)
		#
		bins = bins[1:]
		res = np.split(g, bins)
		CV_1_list = CV_1_list + res[0].index.tolist()
		CV_2_list = CV_2_list + res[1].index.tolist()
		CV_3_list = CV_3_list + res[2].index.tolist()
		CV_4_list = CV_4_list + res[3].index.tolist()
		CV_5_list = CV_5_list + res[4].index.tolist()
		CV_6_list = CV_6_list + res[5].index.tolist()
		CV_7_list = CV_7_list + res[6].index.tolist()
		CV_8_list = CV_8_list + res[7].index.tolist()
		CV_9_list = CV_9_list + res[8].index.tolist()
		CV_10_list = CV_10_list + res[9].index.tolist()
	else :
		CV_1_list = CV_1_list + g.index.tolist()



CV_ND_INDS = {'CV0_train' : CV_1_list+ CV_2_list+CV_3_list+CV_4_list+ CV_5_list+CV_6_list+CV_7_list+CV_8_list, 'CV0_val' : CV_9_list,'CV0_test' : CV_10_list,
			'CV1_train' : CV_3_list+CV_4_list+ CV_5_list+CV_6_list+CV_7_list+CV_8_list+CV_9_list+CV_10_list, 'CV1_val' : CV_1_list,'CV1_test' : CV_2_list,
			'CV2_train' : CV_5_list+CV_6_list+CV_7_list+CV_8_list+CV_9_list+CV_10_list+CV_1_list+ CV_2_list, 'CV2_val' : CV_3_list,'CV2_test' : CV_4_list,
			'CV3_train' : CV_7_list+CV_8_list+CV_9_list+CV_10_list+CV_1_list+ CV_2_list+CV_3_list+CV_4_list, 'CV3_val' : CV_5_list,'CV3_test' : CV_6_list,
			'CV4_train' : CV_9_list+CV_10_list+CV_1_list+ CV_2_list+CV_3_list+CV_4_list+CV_5_list+CV_6_list, 'CV4_val' : CV_7_list,'CV4_test' : CV_8_list }





# use just index 
# normalizing 과정에 대한 고민 필요 ... GCN 이니까 상관 없지 않을까
def prepare_data_GCN(A_B_C_S_SET_SM, MY_chem_A_feat, MY_chem_B_feat, MY_chem_A_adj, MY_chem_B_adj, MY_g_feat_A, MY_g_feat_B, MY_syn, MY_cell, norm ) : 
	#        
	CV_num = 0
	train_key = 'CV{}_train'.format(CV_num)
	val_key = 'CV{}_val'.format(CV_num)
	test_key = 'CV{}_test'.format(CV_num)
	#
	train_no_dup = data_nodup_df.loc[CV_ND_INDS[train_key]]
	val_no_dup = data_nodup_df.loc[CV_ND_INDS[val_key]]
	test_no_dup = data_nodup_df.loc[CV_ND_INDS[test_key]]
	#
	ABCS_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(train_no_dup.setset)]
	ABCS_val = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(val_no_dup.setset)]
	ABCS_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(test_no_dup.setset)]
	#
	train_ind = list(ABCS_train.index)
	val_ind = list(ABCS_val.index)
	test_ind = list(ABCS_test.index)
	# 
	chem_feat_A_train = MY_chem_A_feat[train_ind]; chem_feat_A_val = MY_chem_A_feat[val_ind]; chem_feat_A_test = MY_chem_A_feat[test_ind]
	chem_feat_B_train = MY_chem_B_feat[train_ind]; chem_feat_B_val = MY_chem_B_feat[val_ind]; chem_feat_B_test = MY_chem_B_feat[test_ind]
	chem_adj_A_train = MY_chem_A_adj[train_ind]; chem_adj_A_val = MY_chem_A_adj[val_ind]; chem_adj_A_test = MY_chem_A_adj[test_ind]
	chem_adj_B_train = MY_chem_B_adj[train_ind]; chem_adj_B_val = MY_chem_B_adj[val_ind]; chem_adj_B_test = MY_chem_B_adj[test_ind]
	gene_A_train = MY_g_feat_A[train_ind]; gene_A_val = MY_g_feat_A[val_ind]; gene_A_test = MY_g_feat_A[test_ind]
	gene_B_train = MY_g_feat_B[train_ind]; gene_B_val = MY_g_feat_B[val_ind]; gene_B_test = MY_g_feat_B[test_ind]
	syn_train = MY_syn[train_ind]; syn_val = MY_syn[val_ind]; syn_test = MY_syn[test_ind]
	cell_train = MY_cell[train_ind]; cell_val = MY_cell[val_ind]; cell_test = MY_cell[test_ind]
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
	train_data['EXP_A'] = torch.concat([gene_A_train, gene_B_train], axis = 0)
	val_data['EXP_A'] = gene_A_val
	test_data['EXP_A'] = gene_A_test
	#
	train_data['EXP_B'] = torch.concat([gene_B_train, gene_A_train], axis = 0)
	val_data['EXP_B'] = gene_B_val
	test_data['EXP_B'] = gene_B_test
	#
	train_data['cell'] = np.concatenate((cell_train, cell_train), axis=0)
	val_data['cell'] = cell_val
	test_data['cell'] = cell_test
	#               
	train_data['y'] = np.concatenate((syn_train, syn_train), axis=0)
	val_data['y'] = syn_val
	test_data['y'] = syn_test
	#
	print(train_data['drug1_feat'].shape, flush=True)
	print(val_data['drug1_feat'].shape, flush=True)
	print(test_data['drug1_feat'].shape, flush=True)
	return train_data, val_data, test_data






class DATASET_GCN_W_FT(Dataset):
	def __init__(self, gcn_drug1_F, gcn_drug2_F, gcn_drug1_ADJ, gcn_drug2_ADJ, gcn_exp_A, gcn_exp_B, gcn_adj, gcn_adj_weight, cell_info, syn_ans):
		self.gcn_drug1_F = gcn_drug1_F
		self.gcn_drug2_F = gcn_drug2_F
		self.gcn_drug1_ADJ = gcn_drug1_ADJ
		self.gcn_drug2_ADJ = gcn_drug2_ADJ
		self.gcn_exp_A = gcn_exp_A
		self.gcn_exp_B = gcn_exp_B
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
		self.adj_re_A = self.gcn_drug1_ADJ[index].long().to_sparse().indices()
		self.adj_re_B = self.gcn_drug2_ADJ[index].long().to_sparse().indices()
		return self.gcn_drug1_F[index], self.gcn_drug2_F[index], self.adj_re_A, self.adj_re_B, self.gcn_exp_A[index], self.gcn_exp_B[index], self.gcn_adj, self.gcn_adj_weight, self.cell_info[index], self.syn_ans[index] 






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
	for drug1_f, drug2_f, drug1_adj, drug2_adj, expA, expB, exp_adj, exp_adj_w, cell, y in batch :
		drug1_f_list.append(drug1_f)
		drug2_f_list.append(drug2_f)
		drug1_adj_list.append(drug1_adj+DRUG_1_num_nodes_seen)
		drug2_adj_list.append(drug2_adj+DRUG_2_num_nodes_seen)
		expA_list.append(expA)
		expB_list.append(expB)
		exp_adj_list.append(exp_adj+EXP_num_nodes_seen)
		exp_adj_w_list.append(exp_adj_w)
		y_list.append(y)
		cell_list.append(cell)
		EXP_num_nodes_seen += expA.shape[0]
		DRUG_1_num_nodes_seen += drug1_f.shape[0]
		DRUG_2_num_nodes_seen += drug2_f.shape[0]
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
train_data, val_data, test_data = prepare_data_GCN(A_B_C_S_SET_SM, MY_chem_A_feat, MY_chem_B_feat, MY_chem_A_adj, MY_chem_B_adj, MY_g_feat_A, MY_g_feat_B, MY_syn,MY_Cell, norm)


# WEIGHT 
ys = train_data['y'].squeeze().tolist()
min_s = np.amin(ys)
loss_weight = np.log(train_data['y'] - min_s + np.e)
JY_IDX_WEIGHT_T = torch.Tensor(JY_IDX_WEIGHT).view(1,-1)





# DATA check 
T_train = DATASET_GCN_W_FT(
	torch.Tensor(train_data['drug1_feat']), torch.Tensor(train_data['drug2_feat']), 
	torch.Tensor(train_data['drug1_adj']), torch.Tensor(train_data['drug2_adj']),
	torch.Tensor(train_data['EXP_A']), torch.Tensor(train_data['EXP_B']), 
	JY_ADJ_IDX, JY_IDX_WEIGHT_T,
	torch.Tensor(train_data['cell']), 
	torch.Tensor(train_data['y']))




T_val = DATASET_GCN_W_FT(
	torch.Tensor(val_data['drug1_feat']), torch.Tensor(val_data['drug2_feat']), 
	torch.Tensor(val_data['drug1_adj']), torch.Tensor(val_data['drug2_adj']),
	val_data['EXP_A'], val_data['EXP_B'], 
	JY_ADJ_IDX, JY_IDX_WEIGHT_T,
	torch.Tensor(val_data['cell']), 
	torch.Tensor(val_data['y']))
	
T_test = DATASET_GCN_W_FT(
	torch.Tensor(test_data['drug1_feat']), torch.Tensor(test_data['drug2_feat']), 
	torch.Tensor(test_data['drug1_adj']), torch.Tensor(test_data['drug2_adj']),
	test_data['EXP_A'], test_data['EXP_B'], 
	JY_ADJ_IDX, JY_IDX_WEIGHT_T,
	torch.Tensor(test_data['cell']), 
	torch.Tensor(test_data['y']))

RAY_train = ray.put(T_train)
RAY_val = ray.put(T_val)
RAY_test = ray.put(T_test)
RAY_loss_weight = ray.put(loss_weight)




# DATA check 
T_train = DATASET_GCN_W_FT(
	torch.Tensor(train_data['drug1_feat'][0:64]), torch.Tensor(train_data['drug2_feat'][0:64]), 
	torch.Tensor(train_data['drug1_adj'][0:64]), torch.Tensor(train_data['drug2_adj'][0:64]),
	torch.Tensor(train_data['EXP_A'][0:64]), torch.Tensor(train_data['EXP_B'][0:64]), 
	JY_ADJ_IDX, JY_IDX_WEIGHT_T,
	torch.Tensor(train_data['cell'][0:64]), 
	torch.Tensor(train_data['y'][0:64]))


T_val = DATASET_GCN_W_FT(
	torch.Tensor(val_data['drug1_feat'][0:64]), torch.Tensor(val_data['drug2_feat'][0:64]), 
	torch.Tensor(val_data['drug1_adj'][0:64]), torch.Tensor(val_data['drug2_adj'][0:64]),
	val_data['EXP_A'][0:64], val_data['EXP_B'][0:64], 
	JY_ADJ_IDX, JY_IDX_WEIGHT_T,
	torch.Tensor(val_data['cell'][0:64]), 
	torch.Tensor(val_data['y'][0:64]))
	
T_test = DATASET_GCN_W_FT(
	torch.Tensor(test_data['drug1_feat'][0:64]), torch.Tensor(test_data['drug2_feat'][0:64]), 
	torch.Tensor(test_data['drug1_adj'][0:64]), torch.Tensor(test_data['drug2_adj'][0:64]),
	test_data['EXP_A'][0:64], test_data['EXP_B'][0:64], 
	JY_ADJ_IDX, JY_IDX_WEIGHT_T,
	torch.Tensor(test_data['cell'][0:64]), 
	torch.Tensor(test_data['y'][0:64]))

RAY_train = ray.put(T_train)
RAY_val = ray.put(T_val)
RAY_test = ray.put(T_test)
RAY_loss_weight = ray.put(loss_weight)






############################ MAIN
print('MAIN')


class MY_expGCN_parallel_model(torch.nn.Module):
	def __init__(self, G_layer_chem, G_indim_chem, G_hiddim_chem, G_layer_exp, G_indim_exp, G_hiddim_exp, layers_1, layers_2, layers_3, cell_dim ,out_dim, inDrop, drop):
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
		for conv in self.G_convs_2_chem :
			conv.reset_parameters()
		for bns in self.G_bns_2_chem :
			bns.reset_parameters()
		for conv in self.G_convs_1_exp :
			conv.reset_parameters()
		for bns in self.G_bns_1_exp :
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
	n_epochs = config["epoch"]
	criterion = weighted_mse_loss
	use_cuda =  False# True# True #  
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
			config["G_layer"], T_train.gcn_drug1_F.shape[-1] , config["G_hiddim"],
			config["G_layer"], 2 , config["G_hiddim"],
			dsn1_layers, dsn2_layers, snp_layers, MY_Cell.shape[1], 1,
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
		for batch_idx_t, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y) in enumerate(loaders['train']):
			expA = expA.view(-1,2)#### 다른점 
			expB = expB.view(-1,2)#### 다른점 
			adj_w = adj_w.squeeze()
			# move to GPU
			if use_cuda:
				drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell = drug1_f.cuda(), drug2_f.cuda(), drug1_a.cuda(), drug2_a.cuda(), expA.cuda(), expB.cuda(), adj.cuda(), adj_w.cuda(), y.cuda(), cell.cuda() 
			## find the loss and update the model parameters accordingly
			# clear the gradients of all optimized variables
			optimizer.zero_grad()
			output = MM_MODEL(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y)
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
		for batch_idx_v, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y) in enumerate(loaders['eval']):
			expA = expA.view(-1,2)#### 다른점 
			expB = expB.view(-1,2)#### 다른점 
			adj_w = adj_w.squeeze()
			# move to GPU
			if use_cuda:
				drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell = drug1_f.cuda(), drug2_f.cuda(), drug1_a.cuda(), drug2_a.cuda(), expA.cuda(), expB.cuda(), adj.cuda(), adj_w.cuda(), y.cuda(), cell.cuda()
			## update the average validation loss
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
		#
		done = datetime.now()
		time_spent = done-now
		#
		with tune.checkpoint_dir(step = epoch) as checkpoint_dir:
			trial_name = '_'.join(checkpoint_dir.split('/')[-2].split('_')[0:4])
			print('trial : {}, epoch : {}, TrainLoss : {}, ValLoss : {}'.format(trial_name, epoch,TRAIN_LOSS,VAL_LOSS), flush=True)
			path = os.path.join(checkpoint_dir, "checkpoint")
			torch.save((MM_MODEL.state_dict(), optimizer.state_dict()), path)
			torch.save(MM_MODEL.state_dict(), './model.pth')
		tune.report(TrainLoss= TRAIN_LOSS,  ValLoss=VAL_LOSS )
	#
	print("Finished Training")
 





def RAY_TEST_MODEL(my_config, model_path, model_name, PRJ_PATH, Trial_name, G_NAME, number): 
	use_cuda =  True# True # False 
	T_test = ray.get(RAY_test)
	Test_loader = torch.utils.data.DataLoader(T_test, batch_size = my_config['config/batch_size'].item(), collate_fn = graph_collate_fn, shuffle =False, num_workers=my_config['config/n_workers'].item())
	#
	G_layer = my_config['config/G_layer'].item()
	G_hiddim = my_config['config/G_hiddim'].item()
	dsn1_layers = [my_config['config/feat_size_0'].item(), my_config['config/feat_size_1'].item(), my_config['config/feat_size_2'].item()]
	dsn2_layers = [my_config['config/feat_size_0'].item(), my_config['config/feat_size_1'].item(), my_config['config/feat_size_2'].item()] 
	snp_layers = [my_config['config/feat_size_3'].item() , my_config['config/feat_size_4'].item()]
	inDrop = my_config['config/dropout_1'].item()
	Drop = my_config['config/dropout_2'].item()
	#       
	best_model = MY_expGCN_parallel_model(
				G_layer, T_test.gcn_drug1_F.shape[-1] , G_hiddim,
				G_layer, 2, G_hiddim,
				dsn1_layers, dsn2_layers, snp_layers, MY_Cell.shape[1], 1,
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
	if torch.cuda.is_available():
		state_dict = torch.load(os.path.join(model_path, model_name))
	else:
		state_dict = torch.load(os.path.join(model_path, model_name), map_location=torch.device('cpu'))
	#
	print("state_dict_done", flush = True)
	print(type(state_dict))
	print(len(state_dict))
	if type(state_dict) == tuple:
		best_model.load_state_dict(state_dict[0])
	else : 
		best_model.load_state_dict(state_dict)	#
	print("state_load_done", flush = True)
	#
	#
	best_model.eval()
	test_loss = 0.0
	PRED_list = []
	Y_list = test_data['y'].squeeze().tolist()
	with torch.no_grad():
		best_model.eval()
		for batch_idx_t, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y) in enumerate(Test_loader):
			expA = expA.view(-1,2)
			expB = expB.view(-1,2)
			adj_w = adj_w.squeeze()
			if use_cuda:
				drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell = drug1_f.cuda(), drug2_f.cuda(), drug1_a.cuda(), drug2_a.cuda(), expA.cuda(), expB.cuda(), adj.cuda(), adj_w.cuda(), y.cuda(), cell.cuda()
			output= best_model(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y) 
			MSE = torch.nn.MSELoss()
			loss = MSE(output, y)
			test_loss = test_loss + loss.item()
			outputs = output.squeeze().tolist()
			PRED_list = PRED_list+outputs
	TEST_LOSS = test_loss/(batch_idx_t+1)
	R__T = TEST_LOSS
	R__1 , R__2 = jy_corrplot(PRED_list, Y_list, PRJ_PATH,'P.{}.{}.{}_model'.format(Trial_name, G_NAME, number) )
	return  R__T, R__1, R__2







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






def final_result(R_1_V, R_1_T, R_1_1, R_1_2, R_2_V, R_2_T, R_2_1, R_2_2, R_3_V, R_3_T, R_3_1, R_3_2) :
	print('---1---')
	print('- Val MSE : {:.2f}'.format(R_1_V))
	print('- Test MSE : {:.2f}'.format(R_1_T))
	print('- Test Pearson : {:.2f}'.format(R_1_1))
	print('- Test Spearman : {:.2f}'.format(R_1_2))
	print('---2---')
	print('- Val MSE : {:.2f}'.format(R_2_V))
	print('- Test MSE : {:.2f}'.format(R_2_T))
	print('- Test Pearson : {:.2f}'.format(R_2_1))
	print('- Test Spearman : {:.2f}'.format(R_2_2))
	print('---3---')
	print('- Val MSE : {:.2f}'.format(R_3_V))
	print('- Test MSE : {:.2f}'.format(R_3_T))
	print('- Test Pearson : {:.2f}'.format(R_3_1))
	print('- Test Spearman : {:.2f}'.format(R_3_2))




from ray.tune import ExperimentAnalysis



def MAIN(ANAL_name, WORK_PATH, PRJ_PATH, Trial_name, G_NAME, num_samples= 10, max_num_epochs=1000, grace_period = 150, cpus_per_trial = 16, gpus_per_trial = 1):
	# OPTUNA CONFIGURE 
	CONFIG={
		'n_workers' : tune.choice([cpus_per_trial]),
		"epoch" : tune.choice([max_num_epochs]),
		"G_layer" : tune.choice([2, 3, 4]), # 
		"G_hiddim" : tune.choice([512, 256, 128, 64, 32]), # 
		"batch_size" : tune.choice([ 128, 64, 32, 16]), # CPU 니까 
		"feat_size_0" : tune.choice([4096, 2048, 1024, 512, 256, 128, 64, 32]), # 4096, 2048, 1024, 512, 256, 128, 64, 32
		"feat_size_1" : tune.choice([4096, 2048, 1024, 512, 256, 128, 64, 32]),# 4096, 2048, 1024, 512, 256, 128, 64, 32
		"feat_size_2" : tune.choice([4096, 2048, 1024, 512, 256, 128, 64, 32]),# 4096, 2048, 1024, 512, 256, 128, 64, 32
		"feat_size_3" : tune.choice([4096, 2048, 1024, 512, 256, 128, 64, 32]),# 4096, 2048, 1024, 512, 256, 128, 64, 32
		"feat_size_4" : tune.choice([4096, 2048, 1024, 512, 256, 128, 64, 32]),# 4096, 2048, 1024, 512, 256, 128, 64, 32
		"dropout_1" : tune.choice([0.01, 0.2, 0.5, 0.8]), # 0.01, 0.2, 0.5, 0.8
		"dropout_2" : tune.choice([0.01, 0.2, 0.5, 0.8]), # 0.01, 0.2, 0.5, 0.8
		"lr" : tune.choice([0.00001, 0.0001, 0.001]),# 0.00001, 0.0001, 0.001
	}
	#
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
		name = ANAL_name,
		num_samples=num_samples,
		config=CONFIG,
		resources_per_trial={'cpu': cpus_per_trial }, # , 'gpu' : gpus_per_trial
		progress_reporter = reporter,
		search_alg = optuna_search,
		scheduler = ASHA_scheduler
	)
	best_trial = ANALYSIS.get_best_trial("ValLoss", "min", "last")
	print("Best trial config: {}".format(best_trial.config), flush=True)
	print("Best trial final validation loss: {}".format(
	best_trial.last_result["ValLoss"]), flush=True)
	#
	#
	anal_df = ExperimentAnalysis("~/ray_results/{}".format(ANAL_name))
	#
	# 1) best final
	#
	ANA_DF = anal_df.dataframe()
	ANA_ALL_DF = anal_df.trial_dataframes
	#
	DF_KEY = list(ANA_DF.sort_values('ValLoss')['logdir'])[0]
	print('best final', flush=True)
	print(DF_KEY, flush=True)
	TOPVAL_PATH = DF_KEY
	mini_df = ANA_ALL_DF[DF_KEY]
	my_config = ANA_DF[ANA_DF.logdir==DF_KEY]
	R_1_V = list(ANA_DF.sort_values('ValLoss')['ValLoss'])[0]
	R_1_T, R_1_1, R_1_2 = RAY_TEST_MODEL(my_config, TOPVAL_PATH, 'model.pth', PRJ_PATH, Trial_name, G_NAME, 'M1')
	#
	# 2) best final's checkpoint
	# 
	cck_num =mini_df[mini_df.ValLoss==min(mini_df.ValLoss)].index.item()
	checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
	TOPVAL_PATH = DF_KEY + checkpoint
	print('best final check', flush=True)
	print(TOPVAL_PATH, flush=True)
	R_2_V = min(mini_df.ValLoss)
	R_2_T, R_2_1, R_2_2 = RAY_TEST_MODEL(my_config, TOPVAL_PATH, 'checkpoint', PRJ_PATH, Trial_name, G_NAME, 'M2')
	#
	# 3) total checkpoint best 
	#	
	import numpy as np
	TOT_min = np.Inf
	TOT_key = ""
	for key in ANA_ALL_DF.keys():
		trial_min = min(ANA_ALL_DF[key]['ValLoss'])
		if trial_min < TOT_min :
			TOT_min = trial_min
			TOT_key = key
	print('best val', flush=True)
	print(TOT_key, flush=True)
	mini_df = ANA_ALL_DF[TOT_key]
	TOPVAL_PATH = TOT_key
	my_config = ANA_DF[ANA_DF.logdir==TOT_key]
	cck_num =mini_df[mini_df.ValLoss==min(mini_df.ValLoss)].index.item()
	checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
	TOPVAL_PATH = TOT_key + checkpoint
	print('best val check', flush=True)
	print(TOPVAL_PATH, flush=True)
	R_3_V = min(mini_df.ValLoss)
	R_3_T, R_3_1, R_3_2 = RAY_TEST_MODEL(my_config, TOPVAL_PATH, 'checkpoint', PRJ_PATH, Trial_name, G_NAME, 'M4')
	#
	final_result(R_1_V, R_1_T, R_1_1, R_1_2, R_2_V, R_2_T, R_2_1, R_2_2, R_3_V, R_3_T, R_3_1, R_3_2)
	return ANALYSIS


G_NAME = 'oldver.MJ_3.MISS_2'


# ANAL_name, WORK_PATH, PRJ_PATH, Trial_name, G_NAME, num_samples= 10, max_num_epochs=1000, grace_period = 150, cpus_per_trial = 16, gpus_per_trial = 1

# cpu
MAIN('PRJ01.TRIAL.5_3_3', WORK_PATH, WORK_PATH, 'OLD', 'oldver.MJ_3.MISS_2', 2, 3, 1, 6, 1)


# 8gpu
MAIN('PRJ01.TRIAL.5_3_3', WORK_PATH, WORK_PATH, 'OLD', 'oldver.MJ_3.MISS_2', 100, 1000, 150, 8, 0.5)











#######################################################
#######################################################
#######################################################
#######################################################


import pandas as pd 
import matplotlib.pyplot as plt
from ray.tune import ExperimentAnalysis
from ray.tune import Analysis
import pickle
import math
import torch
import os 


(1) original 에다가 CV change 넣은거 


anal_dir = "/home01/k020a01/ray_results/PRJ01.TRIAL.5_3_3.mix"
list_dir = os.listdir(anal_dir)
exp_json = [a for a in list_dir if 'experiment_state' in a]
exp_json
# anal_df = ExperimentAnalysis(anal_dir+exp_json[2])
anal_df = Analysis(anal_dir)


ANA_DF = anal_df.dataframe()
ANA_ALL_DF = anal_df.trial_dataframes



ANA_DF.to_csv('/home01/k020a01/02.Trial/M3_OLD_MISS_2/RAY_ANA_DF.NEWCV.csv')
import pickle
with open("/home01/k020a01/02.Trial/M3_OLD_MISS_2/RAY_ANA_DF.NEWCV.pickle", "wb") as fp:
	pickle.dump(ANA_ALL_DF,fp) 




min(ANA_DF.sort_values('ValLoss')['ValLoss'])
DF_KEY = list(ANA_DF.sort_values('ValLoss')['logdir'])[0]
DF_KEY






TOPVAL_PATH = DF_KEY

mini_df = ANA_ALL_DF[DF_KEY]
min(mini_df.ValLoss)
cck_num =mini_df[mini_df.ValLoss==min(mini_df.ValLoss)].index.item()
checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
TOPVAL_PATH = DF_KEY + checkpoint
TOPVAL_PATH




import numpy as np
TOT_min = np.Inf
TOT_key = ""
for key in ANA_ALL_DF.keys():
	trial_min = min(ANA_ALL_DF[key]['ValLoss'])
	if trial_min < TOT_min :
		TOT_min = trial_min
		TOT_key = key

TOT_min
TOT_key

mini_df = ANA_ALL_DF[TOT_key]
TOPVAL_PATH = TOT_key
min(mini_df.ValLoss)
cck_num =mini_df[mini_df.ValLoss==min(mini_df.ValLoss)].index.item()
checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
TOPVAL_PATH = TOT_key + checkpoint
TOPVAL_PATH






####################################################
####################################################
####################################################
####################################################





PRJ_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/OLD_811/'
ANA_DF = pd.read_csv(PRJ_PATH+'RAY_ANA_DF.NEWCV.csv')
with open(PRJ_PATH+'RAY_ANA_DF.NEWCV.pickle', 'rb') as f:
	ANA_ALL_DF = pickle.load(f)





TOPVAL_PATH = PRJ_PATH







def RAY_TEST_MODEL(my_config, model_path, model_name, PRJ_PATH, Trial_name, G_NAME, number): 
	use_cuda =  False# True #  
	T_test = ray.get(RAY_test)
	Test_loader = torch.utils.data.DataLoader(T_test, batch_size = my_config['config/batch_size'].item(), collate_fn = graph_collate_fn, shuffle =False, num_workers=my_config['config/n_workers'].item())
	#
	G_layer = my_config['config/G_layer'].item()
	G_hiddim = my_config['config/G_hiddim'].item()
	dsn1_layers = [my_config['config/feat_size_0'].item(), my_config['config/feat_size_1'].item(), my_config['config/feat_size_2'].item()]
	dsn2_layers = [my_config['config/feat_size_0'].item(), my_config['config/feat_size_1'].item(), my_config['config/feat_size_2'].item()] 
	snp_layers = [my_config['config/feat_size_3'].item() , my_config['config/feat_size_4'].item()]
	inDrop = my_config['config/dropout_1'].item()
	Drop = my_config['config/dropout_2'].item()
	#       
	best_model = MY_expGCN_parallel_model(
				G_layer, T_test.gcn_drug1_F.shape[-1] , G_hiddim,
				G_layer, 2, G_hiddim,
				dsn1_layers, dsn2_layers, snp_layers, MY_Cell.shape[1], 1,
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
	if torch.cuda.is_available():
		state_dict = torch.load(os.path.join(model_path, model_name))
	else:
		state_dict = torch.load(os.path.join(model_path, model_name), map_location=torch.device('cpu'))
	#
	print("state_dict_done", flush = True)
	print(type(state_dict))
	print(len(state_dict))
	if type(state_dict) == tuple:
		best_model.load_state_dict(state_dict[0])
	else : 
		best_model.load_state_dict(state_dict)	#
	print("state_load_done", flush = True)
	#
	#
	best_model.eval()
	test_loss = 0.0
	PRED_list = []
	Y_list = test_data['y'].squeeze().tolist()
	with torch.no_grad():
		best_model.eval()
		for batch_idx_t, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y) in enumerate(Test_loader):
			expA = expA.view(-1,2)
			expB = expB.view(-1,2)
			adj_w = adj_w.squeeze()
			if use_cuda:
				drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell = drug1_f.cuda(), drug2_f.cuda(), drug1_a.cuda(), drug2_a.cuda(), expA.cuda(), expB.cuda(), adj.cuda(), adj_w.cuda(), y.cuda(), cell.cuda()
			output= best_model(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y) 
			MSE = torch.nn.MSELoss()
			loss = MSE(output, y)
			test_loss = test_loss + loss.item()
			outputs = output.squeeze().tolist()
			PRED_list = PRED_list+outputs
	TEST_LOSS = test_loss/(batch_idx_t+1)
	R__T = TEST_LOSS
	R__1 , R__2 = jy_corrplot(PRED_list, Y_list, PRJ_PATH,'P.{}.{}.{}_model'.format(Trial_name, G_NAME, number) )
	return  R__T, R__1, R__2







PRJ_NAME = 'OLD811'
MISS_NAME = 'M2'



# 1) best final
# 
# 
DF_KEY = list(ANA_DF.sort_values('ValLoss')['logdir'])[0]
print('best final', flush=True)
print(DF_KEY, flush=True)
#TOPVAL_PATH = DF_KEY
mini_df = ANA_ALL_DF[DF_KEY]
my_config = ANA_DF[ANA_DF.logdir==DF_KEY]
R_1_V = list(ANA_DF.sort_values('ValLoss')['ValLoss'])[0]
R_1_V
R_1_T, R_1_1, R_1_2 = RAY_TEST_MODEL(my_config, TOPVAL_PATH, 'M1_model.pth', PRJ_PATH, PRJ_NAME, MISS_NAME, 'M1')
plot_loss(list(mini_df.TrainLoss), list(mini_df.ValLoss), PRJ_PATH, 'OLD811_BestLast'  )


# plot_loss(list(mini_df.TrainLoss), list(mini_df.ValLoss), PRJ_PATH, 'M1J3M0_BestLast'  )
# plot_loss(list(mini_df.TrainLoss), list(mini_df.ValLoss), PRJ_PATH, 'M2J3M0_BestLast'  )
# plot_loss(list(mini_df.TrainLoss), list(mini_df.ValLoss), PRJ_PATH, 'M2J3M1_BestLast'  )


#
# 2) best final's checkpoint
# 
cck_num =mini_df[mini_df.ValLoss==min(mini_df.ValLoss)].index.item()
checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
#TOPVAL_PATH = DF_KEY + checkpoint
print('best final check', flush=True)
print(TOPVAL_PATH, flush=True)
R_2_V = min(mini_df.ValLoss)
R_2_V
R_2_T, R_2_1, R_2_2 = RAY_TEST_MODEL(my_config, TOPVAL_PATH, 'M2_checkpoint', PRJ_PATH, PRJ_NAME, MISS_NAME, 'M2')
#
# 3) total checkpoint best 
#	
import numpy as np
TOT_min = np.Inf
TOT_key = ""
for key in ANA_ALL_DF.keys():
	trial_min = min(ANA_ALL_DF[key]['ValLoss'])
	if trial_min < TOT_min :
		TOT_min = trial_min
		TOT_key = key

print('best val', flush=True)
print(TOT_key, flush=True)
mini_df = ANA_ALL_DF[TOT_key]
#TOPVAL_PATH = TOT_key
my_config = ANA_DF[ANA_DF.logdir==TOT_key]
cck_num =mini_df[mini_df.ValLoss==min(mini_df.ValLoss)].index.item()
checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
#TOPVAL_PATH = TOT_key + checkpoint
print('best val check', flush=True)
print(TOPVAL_PATH, flush=True)
R_3_V = min(mini_df.ValLoss)
R_3_V
R_3_T, R_3_1, R_3_2 = RAY_TEST_MODEL(my_config, TOPVAL_PATH, 'M4_checkpoint', PRJ_PATH, PRJ_NAME, MISS_NAME, 'M4')

plot_loss(list(mini_df.TrainLoss), list(mini_df.ValLoss), PRJ_PATH, 'OLD811_BestVal'  )






def final_result(R_1_V, R_1_T, R_1_1, R_1_2, R_2_V, R_2_T, R_2_1, R_2_2, R_3_V, R_3_T, R_3_1, R_3_2) :
	print('---1---')
	print('- Val MSE : {:.2f}'.format(R_1_V))
	print('- Test MSE : {:.2f}'.format(R_1_T))
	print('- Test Pearson : {:.2f}'.format(R_1_1))
	print('- Test Spearman : {:.2f}'.format(R_1_2))
	print('---2---')
	print('- Val MSE : {:.2f}'.format(R_2_V))
	print('- Test MSE : {:.2f}'.format(R_2_T))
	print('- Test Pearson : {:.2f}'.format(R_2_1))
	print('- Test Spearman : {:.2f}'.format(R_2_2))
	print('---3---')
	print('- Val MSE : {:.2f}'.format(R_3_V))
	print('- Test MSE : {:.2f}'.format(R_3_T))
	print('- Test Pearson : {:.2f}'.format(R_3_1))
	print('- Test Spearman : {:.2f}'.format(R_3_2))

final_result(R_1_V, R_1_T, R_1_1, R_1_2, R_2_V, R_2_T, R_2_1, R_2_2, R_3_V, R_3_T, R_3_1, R_3_2)













