normalization 을 해야하는지 말아야하는지 알기 위해 일단 분포를 좀 봅시다 

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







BETA_BIND = pd.read_csv('/st06/jiyeonH/11.TOX/MY_TRIAL_5/'+"BETA_DATA_for_20220705_978.csv")
BETA_CP_info = pd.read_table('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+'compoundinfo_beta.txt') # pert 34419 , cansmiles : 
BETA_CEL_info = pd.read_table('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+'cellinfo_beta.txt')
BETA_SIG_info = pd.read_table('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+'siginfo_beta.txt', low_memory = False)

# 범위를 좀 봐야겠다 

check_range = BETA_BIND.iloc[:,2:]
check_r_1 = np.array(check_range)
check_r_2 = check_r_1.tolist()
check_r_3 = sum(check_r_2, []) # 176444892

aa = sns.displot(check_r_3, kind="kde") # 978 기준 

plt.tight_layout()
plt.savefig("/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/LINCS.pdf", bbox_inches='tight')
plt.close()


# 349 기준은? 

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


BETA_BIND_349 = BETA_BIND[BETA_BIND.id.isin(ID_G.nodes())]


check_range = BETA_BIND_349.iloc[:,2:]
check_r_1 = np.array(check_range)
check_r_2 = check_r_1.tolist()
check_r_3 = sum(check_r_2, []) # 62964486

aa = sns.displot(check_r_3, kind="kde") # 349 기준 

plt.tight_layout()
plt.savefig("/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/LINCS2.pdf", bbox_inches='tight')
plt.close()



# 만약 dose 랑 시간 정해서 쓰는 경우?
추가적인 애들 필요 

filter1 = BETA_SIG_info[BETA_SIG_info.pert_type.isin([ 'ctl_vehicle', 'ctl_untrt' ,'trt_cp' ])]

dose = filter1['pert_idose']
dose = [a if type(a) == str else 'NA' for a in dose]
dose_set = list(set(dose))
dose_set.sort()

time = filter1['pert_itime']
time = [a if type(a) == str else 'NA' for a in time]
time_set = list(set(time))
time_set.sort()


dose_time = [dose[a]+'__'+time[a] for a in range(len(dose))]
dose_time_set = list(set(dose_time))
dose_time_set.sort()

filter1['dose_time'] = dose_time


freq = [dose_time.count(a) for a in dose_time_set]
freq_df = pd.DataFrame({ 'dose_time': dose_time_set, 'freq': freq })

freq_df.sort_values('freq')

filter1_trtcp = filter1[filter1.pert_type=='trt_cp']
dose_time_id = list(filter1_trtcp[filter1_trtcp.dose_time=='10 uM__24 h']['sig_id'])




beta_cols = BETA_BIND.columns
need_cols = list(set(dose_time_id) - set(beta_cols))

BETA_ADD = pd.read_csv("/st06/jiyeonH/11.TOX/LINCS/L_2020/level5_beta_trt_cp_n720216x12328.gct", usecols = need_cols, skiprows = 2, low_memory = False, sep ='\t')



/st06/jiyeonH/11.TOX/LINCS/L_2020
level5_beta_ctl_n58022x12328.gct
level5_beta_trt_cp_n720216x12328.gct
skiprows=2

BETA_test.gct











check_range = BETA_BIND[dose_time_id]
check_r_1 = np.array(check_range)
check_r_2 = check_r_1.tolist()
check_r_3 = sum(check_r_2, []) # 176444892

aa = sns.displot(check_r_3, kind="kde") # 978 기준 

plt.tight_layout()
plt.savefig("/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/LINCS.pdf", bbox_inches='tight')
plt.close()








































# basal node exclude -> CCLE match 만 사용
CCLE_PATH = '/st06/jiyeonH/13.DD_SESS/CCLE.22Q1/'
ccle_exp = pd.read_csv(CCLE_PATH+'CCLE_expression.csv', low_memory=False)
ccle_info= pd.read_csv(CCLE_PATH+'sample_info.csv', low_memory=False)


check_range = ccle_exp.iloc[:,1:]
check_r_1 = np.array(check_range)
check_r_2 = check_r_1.tolist()
check_r_3 = sum(check_r_2, []) # 26713561


aa = sns.displot(check_r_3, kind="kde") # 978 기준 

plt.tight_layout()
plt.savefig("/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/BASE1.pdf", bbox_inches='tight')
plt.close()





# CCLE ver

ccle_ori_col = list(ccle_exp.columns)
ccle_new_col =['DepMap_ID'] + [int(a.split(')')[0].split('(')[1]) for a in ccle_ori_col[1:]]

ccle_exp.columns = ccle_new_col

ccle_cell_info = ccle_info[['DepMap_ID','stripped_cell_line_name','CCLE_Name']]
ccle_cell_info.columns = ['DepMap_ID','STR_ID','DrugCombCCLE']
ccle_exp2 = pd.merge(ccle_exp, ccle_cell_info, on = 'DepMap_ID' , how='left')

g_names = list(ID_G.nodes())
check_range = ccle_exp[g_names]
check_r_1 = np.array(check_range)
check_r_2 = check_r_1.tolist()
check_r_3 = sum(check_r_2, []) # 26713561


aa = sns.displot(check_r_3, kind="kde") # 978 기준 

plt.tight_layout()
plt.savefig("/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/BASE2.pdf", bbox_inches='tight')
plt.close()




MJ 만들어준 데이터는? 


MJ_DIR = '/st06/jiyeonH/13.DD_SESS/01.PRJ2/'

MJ_request_ANS_PRE = pd.read_csv(MJ_DIR+'/PRJ2_EXP_ccle_all_fugcn_hhh3.csv')
MJ_request_ANS_786O = pd.read_csv(MJ_DIR+'/PRJ2_EXP_ccle_cell786O_fugcn_hhh3.csv')

MJ_request_ANS = pd.concat([MJ_request_ANS_PRE, MJ_request_ANS_786O], axis =1)
entrez_id = list(MJ_request_ANS.entrez_id.iloc[:,1])
MJ_request_ANS = MJ_request_ANS.drop(['entrez_id','Unnamed: 0', 'CID__CELL',], axis =1)



check_range = MJ_request_ANS
check_r_1 = np.array(check_range)
check_r_2 = check_r_1.tolist()
check_r_3 = sum(check_r_2, []) # 26713561

aa = sns.displot(check_r_3, kind="kde") # 349 기준 

plt.tight_layout()
plt.savefig("/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/MJ1.pdf", bbox_inches='tight')
plt.close()











# AOBO 에 해당되는 sig 의 결과 분포? 

A_B_C_S_SET_COH2


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


BETA_BIND_349 = BETA_BIND[BETA_BIND.id.isin(ID_G.nodes())]


print('SIG_SIG', flush = True)
tmp = list(set(A_B_C_S_SET_COH2.SIG_SIG))
tmp2 = sum([a.split('___') for a in tmp],[])

len(set(tmp2))



1) AOBO
check_range = BETA_BIND_349[tmp2]
check_r_1 = np.array(check_range)
check_r_2 = check_r_1.tolist()
check_r_3 = sum(check_r_2, []) # 5143562
len(check_r_3)

aa = sns.displot(check_r_3, kind="kde") # 349 기준 

plt.tight_layout()
plt.savefig("/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/LINCS_AOBO.pdf", bbox_inches='tight')
plt.close()



2) AOBX -- 459

tmp2 = [a for a in tmp2 if a !='NA']
check_range = BETA_BIND_349[tmp2]
check_r_1 = np.array(check_range)
check_r_2 = check_r_1.tolist()
check_r_3 = sum(check_r_2, []) # 5626229
len(check_r_3)

aa = sns.displot(check_r_3, kind="kde") # 349 기준 

plt.tight_layout()
plt.savefig("/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/LINCS_AOBX.pdf", bbox_inches='tight')
plt.close()


3) AXBX

tmp2 = [a for a in tmp2 if a !='NA']
check_range = BETA_BIND_349[tmp2]
check_r_1 = np.array(check_range)
check_r_2 = check_r_1.tolist()
check_r_3 = sum(check_r_2, []) # 5626229
len(check_r_3)

aa = sns.displot(check_r_3, kind="kde") # 349 기준 

plt.tight_layout()
plt.savefig("/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/LINCS_AXBX.pdf", bbox_inches='tight')
plt.close()









