





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


G_NAME = '1_OTS'

WORK_PATH = '/home01/k006a01/PRJ.01/TRIAL_5.1.1/'
DC_PATH = '/home01/k006a01/01.DATA/DrugComb/'
IDK_PATH = '/home01/k006a01/01.DATA/IDK/'
LINCS_PATH = '/home01/k006a01/01.DATA/LINCS/' #
TARGET_PATH = '/home01/k006a01/01.DATA/TARGET/'
cid_PATH = '/home01/k006a01/01.DATA/PC_ATC_LIST/'



# LINCS DATA
#print("LINCS")
#BETA_BIND = pd.read_csv(LINCS_PATH+"BETA_DATA_for_20220705_978.csv")
#BETA_CP_info = pd.read_table(LINCS_PATH+'compoundinfo_beta.txt')
#BETA_CEL_info = pd.read_table(LINCS_PATH+'cellinfo_beta.txt')
#BETA_SIG_info = pd.read_table(LINCS_PATH+'siginfo_beta.txt', low_memory = False)
#filter1 = BETA_SIG_info[BETA_SIG_info.pert_type.isin(['ctl_vehicle', 'ctl_untrt' ,'trt_cp' ])]
# 764996
#filter2 = filter1[filter1.is_exemplar_sig==1]
# 136460
#BETA_MJ = pd.read_csv(LINCS_PATH+"Lincs_pubchem_mj.csv") # SID 필터까지 그렇게 완성한 무엇 


WORK_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_6/'
DC_PATH = '/st06/jiyeonH/13.DD_SESS/DrugComb.1.5/' 
IDK_PATH = '/st06/jiyeonH/13.DD_SESS/ideker/' 
LINCS_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/' 
TARGET_PATH = '/st06/jiyeonH/13.DD_SESS/merged_target/'
cid_PATH = '/st06/jiyeonH/11.TOX/Pubchem_Classi_176/'

BETA_BIND = pd.read_csv('/st06/jiyeonH/11.TOX/MY_TRIAL_5/'+"BETA_DATA_for_20220705_978.csv")
BETA_CP_info = pd.read_table('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+'compoundinfo_beta.txt')
BETA_CEL_info = pd.read_table('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+'cellinfo_beta.txt')
BETA_SIG_info = pd.read_table('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+'siginfo_beta.txt', low_memory = False)

filter1 = BETA_SIG_info[BETA_SIG_info.pert_type.isin(['ctl_vehicle', 'ctl_untrt' ,'trt_cp' ])]
# 764996
filter2 = filter1[filter1.is_exemplar_sig==1]
# 136460
BETA_MJ = pd.read_csv('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+"Lincs_pubchem_mj.csv") # SID 필터까지 그렇게 완성한 무엇 




BETA_MJ_RE = BETA_MJ[['pert_id','SMILES_cid','canonical_smiles',
	   'pubchem_cid', 'h_bond_acceptor_count', 'h_bond_donor_count',
	   'rotatable_bond_count', 'MolLogP', 'molecular_weight',
	   'canonical_smiles_re', 'tpsa']].drop_duplicates()

BETA_EXM = pd.merge(filter2, BETA_MJ_RE, on='pert_id', how = 'left')
BETA_EXM2 = BETA_EXM[BETA_EXM.SMILES_cid > 0]

BETA_CEL_info2 = BETA_CEL_info[['cell_iname','cellosaurus_id','ccle_name']] # 240 
BETA_SELEC_SIG_wCell = pd.merge(BETA_EXM2, BETA_CEL_info2, on = 'cell_iname', how = 'left') # 129116
BETA_SELEC_SIG_wCell2 = BETA_SELEC_SIG_wCell[['pert_id','pubchem_cid','cellosaurus_id','ccle_name','sig_id']]
BETA_SELEC_SIG_wCell2 = BETA_SELEC_SIG_wCell2.drop_duplicates() # 129116

cello_tt=[True if type(a)==str else False for a in list(BETA_SELEC_SIG_wCell2.cellosaurus_id)]
BETA_CID_CELLO_SIG = BETA_SELEC_SIG_wCell2[cello_tt][['pert_id','pubchem_cid','cellosaurus_id','sig_id']].drop_duplicates() # 111916
beta_cid_cello_sig_tf = [ True if a>0 else False for a in list(BETA_CID_CELLO_SIG.pubchem_cid)]
BETA_CID_CELLO_SIG = BETA_CID_CELLO_SIG[beta_cid_cello_sig_tf]


print("DC filtering")
DC_tmp_DF1 = pd.read_csv(DC_PATH+'TOTAL_BLOCK.csv', low_memory=False) # 1469182
DC_tmp_DF2 = DC_tmp_DF1[DC_tmp_DF1['quality'] != 'bad'] # 1457561

DC_DATA_filter = DC_tmp_DF2[['drug_row_id','drug_col_id','cell_line_id','synergy_loewe']] # 1457561
DC_DATA_filter2 = DC_DATA_filter.drop_duplicates() # 1374958
DC_DATA_filter2['drug_row_id_re'] = [float(a) for a in list(DC_DATA_filter2['drug_row_id'])]
DC_DATA_filter2['drug_col_id_re'] = [float(a) for a in list(DC_DATA_filter2['drug_col_id'])]

DC_DATA_filter3 = DC_DATA_filter2[DC_DATA_filter2.drug_row_id_re>0] # 1374958 -> 1363698
DC_DATA_filter4 = DC_DATA_filter3[DC_DATA_filter3.drug_col_id_re>0] # 751450 -> 740884
DC_DATA_filter4.cell_line_id # unique 295






# Drug DATA
with open(DC_PATH+'drugs.json') as json_file :
	DC_DRUG =json.load(json_file)

DC_DRUG_K = list(DC_DRUG[0].keys())
DC_DRUG_DF = pd.DataFrame(columns=DC_DRUG_K)

for DD in range(1,len(DC_DRUG)):
	tmpdf = pd.DataFrame({k:[DC_DRUG[DD][k]] for k in DC_DRUG_K})
	DC_DRUG_DF = pd.concat([DC_DRUG_DF, tmpdf], axis = 0)

DC_DRUG_DF['id_re'] = [float(a) for a in list(DC_DRUG_DF['id'])]




# cid filter 
DC_DRUG_DF2 = DC_DRUG_DF[['id_re','dname','cid']]
# dnames = list(DC_DRUG_DF2['dname']) # 그래 이부분때문에 다시 다운로드 받은거였음. 이게 문제가 drug 마다 CID 가 여러개 붙는 애들이 있어서
DC_DRUG_DF2.columns = ['drug_row_id_re','drug_row','drug_row_cid']
DC_DATA5 = pd.merge(DC_DATA_filter4, DC_DRUG_DF2, on ='drug_row_id_re', how='left' ) # 751450 -> 740884

DC_DRUG_DF2.columns = ['drug_col_id_re','drug_col','drug_col_cid']
DC_DATA6 = pd.merge(DC_DATA5, DC_DRUG_DF2, on ='drug_col_id_re', how='left') # 751450 -> 740884



# Cell DATA
with open(DC_PATH+'cell_lines.json') as json_file :
	DC_CELL =json.load(json_file)


DC_CELL_K = list(DC_CELL[0].keys())
DC_CELL_DF = pd.DataFrame(columns=DC_CELL_K)

for DD in range(1,len(DC_CELL)):
	tmpdf = pd.DataFrame({k:[DC_CELL[DD][k]] for k in DC_CELL_K})
	DC_CELL_DF = pd.concat([DC_CELL_DF, tmpdf], axis = 0)

DC_CELL_DF2 = DC_CELL_DF[['id','name','cellosaurus_accession', 'ccle_name']] # 2319
DC_CELL_DF2.columns = ['cell_line_id', 'DC_cellname','DrugCombCello', 'DrugCombCCLE']


#  DC cid check 

DC_DATA7_1 = pd.merge(DC_DATA6, DC_CELL_DF2 , on= 'cell_line_id', how = 'left') # 751450 -> 740884
DC_DATA7_1['drug_row_cid_re'] = [a if type(a) == int else 0 for a in list(DC_DATA7_1['drug_row_cid'])]
DC_DATA7_1['drug_col_cid_re'] = [a if type(a) == int else 0 for a in list(DC_DATA7_1['drug_col_cid'])]
DC_DATA7_2 = DC_DATA7_1[DC_DATA7_1.drug_row_cid_re>0] # 747621 -> 737106
DC_DATA7_3 = DC_DATA7_2[DC_DATA7_2.drug_col_cid_re>0] # 735595 -> 725496
cello_t=[True if type(a)==str else False for a in list(DC_DATA7_3.DrugCombCello)]


DC_DATA7_4_cello = DC_DATA7_3[cello_t] # 730348 -> 720249
DC_cello_final = DC_DATA7_4_cello[['drug_row_id_re','drug_col_cid_re','DrugCombCello']].drop_duplicates() # 563367 -> 554431
DC_cello_final_dup = DC_DATA7_4_cello[['drug_row_id_re','drug_col_cid_re','DrugCombCello', 'synergy_loewe']].drop_duplicates() # 730348 -> 720249

DC_cello_final_dup["drug_row_cid2"] = [float(a) for a in list(DC_cello_final_dup.drug_row_cid)]
DC_cello_final_dup["drug_col_cid2"] = [float(a) for a in list(DC_cello_final_dup.drug_col_cid)]

DC_cello_final_dup = DC_cello_final_dup[['drug_row_cid2','drug_col_cid2','DrugCombCello','synergy_loewe']]
DC_cello_final_dup.columns = ['drug_row_cid','drug_col_cid','DrugCombCello','synergy_loewe'] # 730348 -> 720249
TF_check = [True if np.isnan(a)==False else False for a in DC_cello_final_dup.synergy_loewe] 
DC_cello_final_dup = DC_cello_final_dup[TF_check] # 719946

print('DC and LINCS')

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


######################################################



















