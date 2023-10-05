

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
#from rdkit import Chem
#from rdkit.Chem import AllChem
#from rdkit import DataStructs
#import rdkit
#from rdkit import Chem
#from rdkit.Chem.QED import qed

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

#import ray
#from ray import tune
#from functools import partial
#from ray.tune.schedulers import ASHAScheduler
#from ray.tune import CLIReporter
#from ray.tune.suggest.optuna import OptunaSearch
#from ray.tune import ExperimentAnalysis

import numpy as np

import sys
import os
import pandas as pd





NETWORK_PATH = '/st06/jiyeonH/13.DD_SESS/HumanNetV3/'
LINCS_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/' 
DC_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/'







# 유전자 순서는 필요 

NETWORK_PATH = '/st06/jiyeonH/13.DD_SESS/HumanNetV3/'
LINCS_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/' 
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


















## GDSC 시작! 







gdsc_c_path = '/st06/jiyeonH/11.TOX/DR_SPRING/val_data/'


gdsc_breast = pd.read_csv(gdsc_c_path + 'breast_anchor_combo.csv') # 163470
gdsc_colon = pd.read_csv(gdsc_c_path + 'colon_anchor_combo.csv') # 75400
gdsc_pancreas = pd.read_csv(gdsc_c_path + 'pancreas_anchor_combo.csv') # 66300


일단 cid - cid - cell 에 대해서 synergy 결과가 갈리지는 않는지? -> 안갈리는것 같음..!!!! 예스 
근데 앞뒤는 있는듯 

# breast 부터 확인해보기 
g_breast = gdsc_breast[['Cell Line name','Anchor Name','Library Name','Synergy?']].drop_duplicates() # 63800
g_breast[['Cell Line name','Anchor Name','Library Name']].drop_duplicates() # 63800
g_breast_names = list(set(list(g_breast['Anchor Name']) + list(g_breast['Library Name']))) # 52 
g_breast_cell = list(set(g_breast['Cell Line name']))

# colon
g_colon = gdsc_colon[['Cell Line name','Anchor Name','Library Name','Synergy?']].drop_duplicates() # 29200
g_colon[['Cell Line name','Anchor Name','Library Name']].drop_duplicates() # 29200
g_colon_names = list(set(list(g_colon['Anchor Name']) + list(g_colon['Library Name']))) # 52 
g_colon_cell = list(set(g_colon['Cell Line name']))


# pancreas
g_pancreas = gdsc_pancreas[['Cell Line name','Anchor Name','Library Name','Synergy?']].drop_duplicates() # 18650
g_pancreas[['Cell Line name','Anchor Name','Library Name']].drop_duplicates() # 18650
g_pancreas_names = list(set(list(g_pancreas['Anchor Name']) + list(g_pancreas['Library Name']))) # 26
g_pancreas_cell = list(set(g_pancreas['Cell Line name']))



gdsc_id_path = '/st06/jiyeonH/13.DD_SESS/GDSC/'
# Drug_listWed_Jul_6_11_12_48_2022.csv → GDSC_LIST_0706.tsv
gdsc_chem_list = pd.read_csv(gdsc_id_path + 'GDSC_LIST_0706.tsv', sep = '\t')
gdsc_chem_list2 = gdsc_chem_list[['Name','PubCHEM']].drop_duplicates() # 472
len(set(gdsc_chem_list2.Name)) # 449

gdsc_dups = gdsc_chem_list2[gdsc_chem_list2.Name.duplicated()]

gdsc_chem_list_dups = gdsc_chem_list2[gdsc_chem_list2.Name.isin(gdsc_dups.Name)]
gdsc_chem_list_nodups = gdsc_chem_list2[gdsc_chem_list2.Name.isin(gdsc_dups.Name)==False]

dup_names = list(set(gdsc_chem_list_dups.Name))

str_inds = []
for nana in dup_names : 
	tmp = gdsc_chem_list_dups[gdsc_chem_list_dups.Name==nana]
	tmp2 = tmp[tmp.PubCHEM.apply(lambda x : type(x)==str)]
	str_inds = str_inds + list(tmp2.index)


gdsc_chem_list = pd.concat([gdsc_chem_list_nodups, gdsc_chem_list_dups.loc[str_inds]])






# 이름붙이기  
set(g_breast_names) - set(gdsc_chem_list.Name) # None 
set(g_colon_names) - set(gdsc_chem_list.Name) # {'Afatinib | Trametinib'} -> 진짜 이렇게 생겼음. 제외해도 될듯 
set(g_pancreas_names) - set(gdsc_chem_list.Name) # {'Galunisertib'} -> 10090485 나 131708758 면 될것 같음. canonical smiles 동일 


gdsc_chem_list = pd.concat([gdsc_chem_list, pd.DataFrame({'Name' : ['Galunisertib'], "PubCHEM" : ['10090485']})])

gdsc_chem_list_re = gdsc_chem_list[gdsc_chem_list.Name.isin(g_breast_names + g_colon_names + g_pancreas_names)]
gdsc_chem_list_re['CID'] = gdsc_chem_list_re.PubCHEM.apply(lambda x : int(x))

gdsc_chem_list_re = gdsc_chem_list_re[['Name','CID']]


# breast 
gdsc_chem_list_re.columns = ['Anchor Name','Anchor CID']
g_breast2 = pd.merge(g_breast, gdsc_chem_list_re, on ='Anchor Name', how ='left')

gdsc_chem_list_re.columns = ['Library Name','Library CID']
g_breast3 = pd.merge(g_breast2, gdsc_chem_list_re, on ='Library Name', how ='left')


# colon 
gdsc_chem_list_re.columns = ['Anchor Name','Anchor CID']
g_colon2 = pd.merge(g_colon, gdsc_chem_list_re, on ='Anchor Name', how ='left')

gdsc_chem_list_re.columns = ['Library Name','Library CID']
g_colon3 = pd.merge(g_colon2, gdsc_chem_list_re, on ='Library Name', how ='left')


# pancreas 
gdsc_chem_list_re.columns = ['Anchor Name','Anchor CID']
g_pancreas2 = pd.merge(g_pancreas, gdsc_chem_list_re, on ='Anchor Name', how ='left')

gdsc_chem_list_re.columns = ['Library Name','Library CID']
g_pancreas3 = pd.merge(g_pancreas2, gdsc_chem_list_re, on ='Library Name', how ='left')






# cell line 이름 붙이기 

CCLE_PATH = '/st06/jiyeonH/13.DD_SESS/CCLE.22Q1/'
ccle_info= pd.read_csv(CCLE_PATH+'sample_info.csv', low_memory=False)

gdsc_cell = pd.read_csv(gdsc_id_path +'GDSC_CELL_0707.tsv', sep = '\t')

set(g_breast_cell) - set(ccle_info.cell_line_name) # {'Hs-578-T', 'DU-4475', 'EVSA-T', 'T47D'}
set(g_colon_cell) - set(ccle_info.cell_line_name) # {'SW620', 'LS-411N', 'HCT-116', 'LS-513', 'LS-1034', 'LS-180', 'DiFi', 'SNU-C2B', 'HT-115', 'COLO-205', 'HCC2998', 'LS-123'}
set(g_pancreas_cell) - set(ccle_info.cell_line_name) # {'SW1990', 'MIA-PaCa-2', 'CAPAN-2', 'Hs-766T', 'KP-4', 'PANC-08-13', 'KP-1N', 'PANC-02-03', 'HuP-T3', 'SU8686', 'CAPAN-1', 'PANC-04-03', 'PANC-10-05', 'HuP-T4', 'PANC-03-27'}

set(g_breast_cell) - set(gdsc_cell.Name) # {'ZR-75-1'}
set(g_colon_cell) - set(gdsc_cell.Name) # 
set(g_pancreas_cell) - set(gdsc_cell.Name) # 

gdsc_cell_ccle = pd.merge(gdsc_cell, ccle_info, left_on = 'Passport', right_on = 'Sanger_Model_ID', how = 'left')
gdsc_cell_ccle2 = gdsc_cell_ccle[['Name','stripped_cell_line_name','CCLE_Name']]
gdsc_cell_ccle2 = pd.concat([gdsc_cell_ccle2 , pd.DataFrame({'Name' : ['ZR-75-1'],'stripped_cell_line_name':['ZR751'], 'CCLE_Name':['ZR751_BREAST']})])
gdsc_cell_ccle2 = gdsc_cell_ccle2.drop_duplicates()

gdsc_cell_ccle2.columns = ['Cell Line name','strip_name','ccle_name']

g_breast4 = pd.merge(g_breast3, gdsc_cell_ccle2, on = 'Cell Line name', how = 'left') # 63800
g_colon4 = pd.merge(g_colon3, gdsc_cell_ccle2, on = 'Cell Line name', how = 'left') # 29200
g_pancreas4 = pd.merge(g_pancreas3, gdsc_cell_ccle2, on = 'Cell Line name', how = 'left') # 18650


g_breast5 = g_breast4[['Anchor CID', 'Library CID', 'strip_name', 'ccle_name', 'Synergy?']]
# 63800

g_colon5 = g_colon4[['Anchor CID', 'Library CID', 'strip_name', 'ccle_name', 'Synergy?']]
# 얘 그 Afatinib | Trametinib 때문에 filter 해줘야함 
g_colon5 = g_colon5[g_colon5['Anchor CID']>0]
g_colon5 = g_colon5[g_colon5['Library CID']>0]
# 26952

g_pancreas5 = g_pancreas4[['Anchor CID', 'Library CID', 'strip_name', 'ccle_name', 'Synergy?']]
# 18650


# g_breast5 = g_breast4
# g_colon5 = g_colon4
# g_colon5 = g_colon5[g_colon5['Anchor CID']>0]
# g_colon5 = g_colon5[g_colon5['Library CID']>0]
# g_pancreas5 = g_pancreas4






# BREAST 
cid_a = list(g_breast5['Anchor CID'])
cid_b = list(g_breast5['Library CID'])
cell = list(g_breast5['ccle_name'])

g_breast5['cid_cid_cell'] = [str(int(cid_a[i])) + '___' + str(int(cid_b[i]))+ '___' + cell[i] if cid_a[i] < cid_b[i] else str(int(cid_b[i])) + '___' + str(int(cid_a[i]))+ '___' + cell[i] for i in range(g_breast5.shape[0])]
g_breast5['CID_A_CELL'] = g_breast5['Anchor CID'].apply(lambda a : str(a)) + '__' + g_breast5['ccle_name']
g_breast5['CID_B_CELL'] = g_breast5['Library CID'].apply(lambda b : str(b)) + '__' + g_breast5['ccle_name']
g_breast5['cid_cid'] = [str(int(cid_a[i])) + '___' + str(int(cid_b[i])) if cid_a[i] < cid_b[i] else str(int(cid_b[i])) + '___' + str(int(cid_a[i])) for i in range(g_breast5.shape[0])]

len(g_breast5.cid_cid_cell) # 63800
len(set(g_breast5.cid_cid_cell)) # 48704



# COLON
cid_a = list(g_colon5['Anchor CID'])
cid_b = list(g_colon5['Library CID'])
cell = list(g_colon5['ccle_name'])

g_colon5['cid_cid_cell'] = [str(int(cid_a[i])) + '___' + str(int(cid_b[i]))+ '___' + cell[i] if cid_a[i] < cid_b[i] else str(int(cid_b[i])) + '___' + str(int(cid_a[i]))+ '___' + cell[i] for i in range(g_colon5.shape[0])]
g_colon5['CID_A_CELL'] = g_colon5['Anchor CID'].apply(lambda a : str(int(a))) + '__' + g_colon5['ccle_name']
g_colon5['CID_B_CELL'] = g_colon5['Library CID'].apply(lambda b : str(int(b))) + '__' + g_colon5['ccle_name']
g_colon5['cid_cid'] = [str(int(cid_a[i])) + '___' + str(int(cid_b[i])) if cid_a[i] < cid_b[i] else str(int(cid_b[i])) + '___' + str(int(cid_a[i])) for i in range(g_colon5.shape[0])]

len(g_colon5.cid_cid_cell) # 26952
len(set(g_colon5.cid_cid_cell)) # 13499



# PANCREAS
cid_a = list(g_pancreas5['Anchor CID'])
cid_b = list(g_pancreas5['Library CID'])
cell = list(g_pancreas5['ccle_name'])

g_pancreas5['cid_cid_cell'] = [str(int(cid_a[i])) + '___' + str(int(cid_b[i]))+ '___' + cell[i] if cid_a[i] < cid_b[i] else str(int(cid_b[i])) + '___' + str(int(cid_a[i]))+ '___' + cell[i] for i in range(g_pancreas5.shape[0])]
g_pancreas5['CID_A_CELL'] = g_pancreas5['Anchor CID'].apply(lambda a : str(a)) + '__' + g_pancreas5['ccle_name']
g_pancreas5['CID_B_CELL'] = g_pancreas5['Library CID'].apply(lambda b : str(b)) + '__' + g_pancreas5['ccle_name']
g_pancreas5['cid_cid'] = [str(int(cid_a[i])) + '___' + str(int(cid_b[i])) if cid_a[i] < cid_b[i] else str(int(cid_b[i])) + '___' + str(int(cid_a[i])) for i in range(g_pancreas5.shape[0])]


len(g_pancreas5.cid_cid_cell) # 18650
len(set(g_pancreas5.cid_cid_cell)) # 9397











# lincs 10um 24h
filter2 = pd.read_csv('/st06/jiyeonH/11.TOX/MY_TRIAL_5/BETA_FILTER.20230614.csv')
LINCS_PERT_MATCH = pd.read_csv('/st06/jiyeonH/13.DD_SESS/01.PRJ2/LINCS_PERT_CID_MATCH.1207.csv', sep ='\t')
BETA_CEL_info = pd.read_table('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+'cellinfo_beta.txt')

filter3 = filter2[['pert_id','sig_id','cell_iname']]
filter4 = pd.merge(filter3, LINCS_PERT_MATCH[['pert_id','CID']] , on = 'pert_id', how = 'left')
filter5 = pd.merge(filter4, ccle_info[['stripped_cell_line_name','CCLE_Name']].drop_duplicates(), left_on = 'cell_iname', right_on = 'stripped_cell_line_name', how = 'left' )

#not_in_lincs = list(set(filter5.ccle_name) - set(filter5.CCLE_Name))
#[a for a in not_in_lincs if a in list(BETA_CEL_info.ccle_name)] # 0

#not_in_lincs = list(set(g_colon5.ccle_name) - set(filter5.CCLE_Name))
#[a for a in not_in_lincs if a in list(BETA_CEL_info.ccle_name)] # 8 개 있는데, 10_24 에 해당 없음. 

#not_in_lincs = list(set(g_pancreas5.ccle_name) - set(filter5.CCLE_Name))
#[a for a in not_in_lincs if a in list(BETA_CEL_info.ccle_name)] # 0 

filter6 = filter5[filter5.CID>0]
filter7 = filter6[filter6.CCLE_Name.apply(lambda x : type(x) == str)]
filter7['CID_CELL'] = filter7.CID.apply(lambda x : str(int(x))) + '__' +filter7.CCLE_Name
filter7['long_id'] = filter7.CID.apply(lambda x : str(int(x))) + '___' +filter7.cell_iname

filter8 = filter7[['long_id','CID_CELL']].drop_duplicates() # 물론 CID 여러개 붙어서 문제는 있음. 만약에 붙는거 보고 sig dup 일어나면 평균 취해줘야함 

#total_cids = list(set(list(g_breast5['Anchor CID']) + list(g_breast5['Library CID']) + list(g_colon5['Anchor CID']) + list(g_colon5['Library CID']) + list(g_pancreas5['Anchor CID']) + list(g_pancreas5['Library CID'])))
#pre_cids = all_chem_cids+pc_sm
#[a for a in total_cids if a not in pre_cids]
#44450571, 44224160, 10096043




그래서 민지가 새로 만들어서 주면 그거 맞게 input 가공 필요

# 민지 요청할거 
total_list = list(g_breast5.CID_A_CELL) + list(g_breast5.CID_B_CELL) + list(g_colon5.CID_A_CELL) + list(g_colon5.CID_B_CELL) + list(g_pancreas5.CID_A_CELL) + list(g_pancreas5.CID_B_CELL)
total_list = list(set(total_list))

with open(file='/st06/jiyeonH/13.DD_SESS/01.PRJ2/gdsc_add.pickle', mode='wb') as f:
	pickle.dump(total_list, f) # 4507

with open(file='/st06/jiyeonH/13.DD_SESS/01.PRJ2/gdsc_add.pickle', mode='rb') as f:
	requested = pickle.load(f) # 4507


# MJ data # 3604 만들어줌 
# MJ_gdsc = pd.read_csv('/st06/jiyeonH/13.DD_SESS/01.PRJ2/GDSC_EXP_ccle_cellall_fugcn_hhhdt3.csv')

# 0909 이후 
MJ_gdsc = pd.read_csv('/st06/jiyeonH/13.DD_SESS/01.PRJ2/GDSC_EXP_ccle_cellall_fugcn_hhhdttf3.csv')


entrez_id = list(MJ_gdsc['entrez_id'])
MJ_gdsc_re = MJ_gdsc.drop(['entrez_id','Unnamed: 0','CID__CELL'], axis =1)
MJ_gdsc_re['entrez_id'] = entrez_id

ord = [list(MJ_gdsc_re.entrez_id).index(a) for a in BETA_ENTREZ_ORDER]
MJ_gdsc_re = MJ_gdsc_re.loc[ord] 


# fu (M3 & M33)
def get_MJ_data( CHECK ): 
	if CHECK in list(MJ_gdsc_re.columns) :
		RES = MJ_gdsc_re[CHECK]
		OX = 'O'
	else : 
		RES = [0]*349        ##############
		OX = 'X'
	return list(RES), OX



LINCS_ALL_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/'

BETA_BIND_MEAN = torch.load( LINCS_ALL_PATH + "10_24_sig_cell_mean.0620.pt")
BETA_BIND_M_SIG_df_CID = pd.read_csv(LINCS_ALL_PATH + '10_24_sig_cell_mean.0620.csv')

BETA_BIND_M_SIG_df_CID['CID__CELL'] = BETA_BIND_M_SIG_df_CID.CID.apply(lambda x : str(x)) + "__" + BETA_BIND_M_SIG_df_CID.CCLE_Name

# 여기에 새로 줍줍한거 들어가야함 

def get_LINCS_data(CID__CELL):
	bb_ind = BETA_BIND_M_SIG_df_CID[BETA_BIND_M_SIG_df_CID.CID_CELL == CID__CELL ].index.item()
	sig_ts = BETA_BIND_MEAN[bb_ind]
	#
	return sig_ts




DC_ALL_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/'

all_chem_DF = pd.read_csv(DC_ALL_PATH+'DC_ALL_7555_ORDER.csv')
all_chem_feat_TS = torch.load(DC_ALL_PATH+'DC_ALL.MY_chem_feat.pt')
all_chem_feat_adj = torch.load(DC_ALL_PATH+'DC_ALL.MY_chem_adj.pt')
all_chem_cids = list(all_chem_DF.CID)

additional_cids = {
	44450571 : 'C1=CC2=C(C=CC(=C2)C=C3C(=O)N=C(S3)NCC4=CC=CS4)N=C1',
	44224160 : 'COC1=CC=CC2=CC(=C3C4=C(N=CNN4C(=N3)C5CCC(CC5)C(=O)O)N)N=C21',
	10096043 : 'CCNC(=O)C1=C(C(=C2C=C(C(=CC2=O)O)C(C)C)ON1)C3=CC=C(C=C3)CN4CCOCC4'
}


def check_drug_f_ts(CID) :
	if CID in all_chem_cids :  
		INDEX = all_chem_DF[all_chem_DF.CID == CID].index.item()
		adj_pre = all_chem_feat_adj[INDEX]
		feat = all_chem_feat_TS[INDEX]
	#
	elif CID in pc_sm :
		feat, adj_pre = get_CHEM(CID)
	else :
		feat, adj_pre = get_CHEM2(additional_cids[CID])
	return feat, adj_pre



PC_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/'
for_CAN_smiles = pd.read_csv(PC_PATH+'CID_SMILES.csv', low_memory = False)
pc_sm = list(for_CAN_smiles.CID)

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

def get_CHEM2(smiles, k=1):
	maxNumAtoms = max_len
	#smiles = for_CAN_smiles[for_CAN_smiles.CID == cid]['CAN_SMILES'].item()
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








# breast 대상 
# breast 대상 
# breast 대상 
# breast 대상 

gg_tmp = g_breast5[['cid_cid_cell','Synergy?']].drop_duplicates() # 49702 갈리는 애가 있네 


1) # synergy 갈리는 애들은 빼버리는 경우 -> 아예 없애버리는 경우 
dup_ccc = list(gg_tmp[gg_tmp.cid_cid_cell.duplicated()]['cid_cid_cell'])
g_breast5 = g_breast5[g_breast5.cid_cid_cell.isin(dup_ccc) == False] 
# 61804 , ccc: 47706


2) 아예 synergy 있는걸로 밀어버리는 경우 -> 더 살릴 수 있을지도 
dup_ccc = list(gg_tmp[gg_tmp.cid_cid_cell.duplicated()]['cid_cid_cell'])
tmp_change = g_breast5[g_breast5.cid_cid_cell.isin(dup_ccc)]
tmp_change['Synergy?'] = 1 
g_breast_ok = g_breast5[g_breast5.cid_cid_cell.isin(dup_ccc)==False] 
g_breast5 = pd.concat([g_breast_ok, tmp_change])
# 63800 , ccc: 47706



filter8.columns = ['long_id_A' , 'CID_A_CELL']
g_breast6 = pd.merge(g_breast5, filter8, on = 'CID_A_CELL', how = 'left') # row 61804, ccc : 47706

filter8.columns = ['long_id_B' , 'CID_B_CELL']
g_breast7 = pd.merge(g_breast6, filter8, on = 'CID_B_CELL', how = 'left') # 63800, ccc : 47706

g_long_A = list(g_breast7.long_id_A)
g_long_B = list(g_breast7.long_id_B)

ttype = [] 
for a in range(g_breast7.shape[0]) :
	type_a = type(g_long_A[a])
	type_b = type(g_long_B[a])
	if (type_a == str) & (type_b == str) : 
		ttype.append('AOBO')
	elif (type_a != str) & (type_b != str) : 
		ttype.append('AXBX')
	else : 
		ttype.append('AXBO')



g_breast7['type'] = ttype

g_breast7[g_breast7['type'] =='AOBO'] # 1908
g_breast7[g_breast7['type'] =='AXBO'] # 4348
g_breast7[g_breast7['type'] =='AXBX'] # 55548





g_breast8 = g_breast7[g_breast7.CID_A_CELL.isin(MJ_gdsc.columns) & g_breast7.CID_B_CELL.isin(MJ_gdsc.columns)]
g_breast8 = g_breast8.reset_index(drop=True) # row :41682, ccc : 32522




def get_synergy_data(cid_cid_cell):
	syn_list = list(set(g_breast8[g_breast8.cid_cid_cell==cid_cid_cell]['Synergy?']))
	if len(syn_list) == 1 : 
		syn_res = syn_list[0]
	else :
		print('a')
	return syn_res

max_len = 50

MY_chem_A_feat = torch.empty(size=(g_breast8.shape[0], max_len, 64))
MY_chem_B_feat= torch.empty(size=(g_breast8.shape[0], max_len, 64))
MY_chem_A_adj = torch.empty(size=(g_breast8.shape[0], max_len, max_len))
MY_chem_B_adj= torch.empty(size=(g_breast8.shape[0], max_len, max_len))
MY_syn =  torch.empty(size=(g_breast8.shape[0],1))

MY_g_EXP_A = torch.empty(size=(g_breast8.shape[0], 349, 1))##############
MY_g_EXP_B = torch.empty(size=(g_breast8.shape[0], 349, 1))##############


Fail_ind = []
from datetime import datetime

for IND in range(0, g_breast8.shape[0]): #  100
	if IND%100 == 0 : 
		print(str(IND)+'/'+str(g_breast8.shape[0]) )
		Fail_ind
		datetime.now()
	#
	cid_cid_cell = g_breast8.cid_cid_cell[IND]
	DrugA_CID = g_breast8['Anchor CID'][IND]
	DrugB_CID = g_breast8['Library CID'][IND]
	CELL = g_breast8['ccle_name'][IND]
	dat_type = g_breast8.type[IND]
	DrugA_CID_CELL = g_breast8.CID_A_CELL[IND]
	DrugB_CID_CELL = g_breast8.CID_B_CELL[IND]
	#
	k=1
	DrugA_Feat, DrugA_ADJ = check_drug_f_ts(DrugA_CID)
	DrugB_Feat, DrugB_ADJ = check_drug_f_ts(DrugB_CID)
	# 
	if dat_type == 'AOBO' :
		mean_ind_A = BETA_BIND_M_SIG_df_CID[BETA_BIND_M_SIG_df_CID.CID__CELL == DrugA_CID_CELL].index.item()
		EXP_A = BETA_BIND_MEAN[mean_ind_A]
		mean_ind_B = BETA_BIND_M_SIG_df_CID[BETA_BIND_M_SIG_df_CID.CID__CELL == DrugB_CID_CELL].index.item()
		EXP_B = BETA_BIND_MEAN[mean_ind_B]
	#
	else :
		DrugA_check = BETA_BIND_M_SIG_df_CID[BETA_BIND_M_SIG_df_CID.CID__CELL == DrugA_CID_CELL]
		DrugB_check = BETA_BIND_M_SIG_df_CID[BETA_BIND_M_SIG_df_CID.CID__CELL == DrugB_CID_CELL]
	#
		if len(DrugA_check) == 0 :
			EXP_A, OX = get_MJ_data(DrugA_CID_CELL)
			if 'X' in OX :
				Fail_ind.append(IND)
			EXP_A = torch.Tensor(EXP_A).unsqueeze(1)
		else : 
			mean_ind_A = BETA_BIND_M_SIG_df_CID[BETA_BIND_M_SIG_df_CID.CID__CELL == DrugA_CID_CELL].index.item()
			EXP_A = BETA_BIND_MEAN[mean_ind_A]
	#
		if len(DrugB_check) == 0 :
			EXP_B, OX = get_MJ_data(DrugB_CID_CELL)
			if 'X' in OX :
				Fail_ind.append(IND)
			EXP_B = torch.Tensor(EXP_B).unsqueeze(1)
		else : 
			mean_ind_B = BETA_BIND_M_SIG_df_CID[BETA_BIND_M_SIG_df_CID.CID__CELL == DrugB_CID_CELL].index.item()
			EXP_B = BETA_BIND_MEAN[mean_ind_B]
	#
	# 
	# 
	AB_SYN = get_synergy_data(cid_cid_cell)
	# 
	MY_chem_A_feat[IND] = torch.Tensor(DrugA_Feat)
	MY_chem_B_feat[IND] = torch.Tensor(DrugB_Feat)
	MY_chem_A_adj[IND] = torch.Tensor(DrugA_ADJ)
	MY_chem_B_adj[IND] = torch.Tensor(DrugB_ADJ)
	MY_g_EXP_A[IND] = torch.Tensor(EXP_A)
	MY_g_EXP_B[IND] = torch.Tensor(EXP_B)
	MY_syn[IND] = torch.Tensor([AB_SYN])


PRJ_NAME = 'GDSC_BREAST' # save the original ver 

SAVE_PATH = gdsc_c_path

torch.save(MY_chem_A_feat, SAVE_PATH+'{}.MY_chem_A_feat.pt'.format(PRJ_NAME))
torch.save(MY_chem_B_feat, SAVE_PATH+'{}.MY_chem_B_feat.pt'.format(PRJ_NAME))
torch.save(MY_chem_A_adj, SAVE_PATH+'{}.MY_chem_A_adj.pt'.format(PRJ_NAME))
torch.save(MY_chem_B_adj, SAVE_PATH+'{}.MY_chem_B_adj.pt'.format(PRJ_NAME))
torch.save(MY_g_EXP_A, SAVE_PATH+'{}.MY_g_EXP_A.pt'.format(PRJ_NAME))
torch.save(MY_g_EXP_B, SAVE_PATH+'{}.MY_g_EXP_B.pt'.format(PRJ_NAME))
torch.save(MY_syn, SAVE_PATH+'{}.MY_syn.pt'.format(PRJ_NAME))

g_breast8.to_csv(SAVE_PATH+'{}.A_B_C_S_SET.csv'.format(PRJ_NAME))
g_breast8 = pd.read_csv(SAVE_PATH+'{}.A_B_C_S_SET.csv'.format(PRJ_NAME))
g_breast8[['cid_cid_cell','type']].drop_duplicates() # 32522 







# colon 대상 
# colon 대상 
# colon 대상 
# colon 대상 
# colon 대상 


gg_tmp = g_colon5[['cid_cid_cell','Synergy?']].drop_duplicates() 
# 14530


1) # synergy 갈리는 애들은 빼버리는 경우 -> 아예 없애버리는 경우 
dup_ccc = list(gg_tmp[gg_tmp.cid_cid_cell.duplicated()]['cid_cid_cell']) 
g_colon5 = g_colon5[g_colon5.cid_cid_cell.isin(dup_ccc) == False] 
# row 24890 # dup_ccc 1031


2) 아예 synergy 있는걸로 밀어버리는 경우 -> 더 살릴 수 있을지도 
dup_ccc = list(gg_tmp[gg_tmp.cid_cid_cell.duplicated()]['cid_cid_cell'])
tmp_change = g_colon5[g_colon5.cid_cid_cell.isin(dup_ccc)]
tmp_change['Synergy?'] = 1 
g_breast_ok = g_colon5[g_colon5.cid_cid_cell.isin(dup_ccc)==False] 
g_colon5 = pd.concat([g_breast_ok, tmp_change])
# row 26952 # dup_ccc 1031



#### 위에 lincs 에서 가져옴 
filter8.columns = ['long_id_A' , 'CID_A_CELL']
g_colon6 = pd.merge(g_colon5, filter8, on = 'CID_A_CELL', how = 'left') # row 61804, ccc : 47706

filter8.columns = ['long_id_B' , 'CID_B_CELL']
g_colon7 = pd.merge(g_colon6, filter8, on = 'CID_B_CELL', how = 'left') # 63800, ccc : 47706

g_long_A = list(g_colon7.long_id_A)
g_long_B = list(g_colon7.long_id_B)

ttype = [] 
for a in range(g_colon7.shape[0]) :
	type_a = type(g_long_A[a])
	type_b = type(g_long_B[a])
	if (type_a == str) & (type_b == str) : 
		ttype.append('AOBO')
	elif (type_a != str) & (type_b != str) : 
		ttype.append('AXBX')
	else : 
		ttype.append('AXBO')



g_colon7['type'] = ttype

g_colon7[g_colon7['type'] =='AOBO'] # 284
g_colon7[g_colon7['type'] =='AXBO'] # 484
g_colon7[g_colon7['type'] =='AXBX'] # 24122


g_colon8 = g_colon7[g_colon7.CID_A_CELL.isin(MJ_gdsc.columns) & g_colon7.CID_B_CELL.isin(MJ_gdsc.columns)]
g_colon8 = g_colon8.reset_index(drop=True) # row :41682, ccc : 32522






def get_synergy_data(cid_cid_cell):
	syn_list = list(set(g_colon8[g_colon8.cid_cid_cell==cid_cid_cell]['Synergy?']))
	if len(syn_list) == 1 : 
		syn_res = syn_list[0]
	else :
		print('a')
	return syn_res



max_len = 50

MY_chem_A_feat = torch.empty(size=(g_colon8.shape[0], max_len, 64))
MY_chem_B_feat= torch.empty(size=(g_colon8.shape[0], max_len, 64))
MY_chem_A_adj = torch.empty(size=(g_colon8.shape[0], max_len, max_len))
MY_chem_B_adj= torch.empty(size=(g_colon8.shape[0], max_len, max_len))
MY_syn =  torch.empty(size=(g_colon8.shape[0],1))

MY_g_EXP_A = torch.empty(size=(g_colon8.shape[0], 349, 1))##############
MY_g_EXP_B = torch.empty(size=(g_colon8.shape[0], 349, 1))##############


Fail_ind = []
from datetime import datetime

for IND in range(0, g_colon8.shape[0]): #  100
	if IND%100 == 0 : 
		print(str(IND)+'/'+str(g_colon8.shape[0]) )
		Fail_ind
		datetime.now()
	#
	cid_cid_cell = g_colon8.cid_cid_cell[IND]
	DrugA_CID = g_colon8['Anchor CID'][IND]
	DrugB_CID = g_colon8['Library CID'][IND]
	CELL = g_colon8['ccle_name'][IND]
	dat_type = g_colon8.type[IND]
	DrugA_CID_CELL = g_colon8.CID_A_CELL[IND]
	DrugB_CID_CELL = g_colon8.CID_B_CELL[IND]
	#
	k=1
	DrugA_Feat, DrugA_ADJ = check_drug_f_ts(DrugA_CID)
	DrugB_Feat, DrugB_ADJ = check_drug_f_ts(DrugB_CID)
	# 
	if dat_type == 'AOBO' :
		mean_ind_A = BETA_BIND_M_SIG_df_CID[BETA_BIND_M_SIG_df_CID.CID__CELL == DrugA_CID_CELL].index.item()
		EXP_A = BETA_BIND_MEAN[mean_ind_A]
		mean_ind_B = BETA_BIND_M_SIG_df_CID[BETA_BIND_M_SIG_df_CID.CID__CELL == DrugB_CID_CELL].index.item()
		EXP_B = BETA_BIND_MEAN[mean_ind_B]
	#
	else :
		DrugA_check = BETA_BIND_M_SIG_df_CID[BETA_BIND_M_SIG_df_CID.CID__CELL == DrugA_CID_CELL]
		DrugB_check = BETA_BIND_M_SIG_df_CID[BETA_BIND_M_SIG_df_CID.CID__CELL == DrugB_CID_CELL]
	#
		if len(DrugA_check) == 0 :
			EXP_A, OX = get_MJ_data(DrugA_CID_CELL)
			if 'X' in OX :
				Fail_ind.append(IND)
			EXP_A = torch.Tensor(EXP_A).unsqueeze(1)
		else : 
			mean_ind_A = BETA_BIND_M_SIG_df_CID[BETA_BIND_M_SIG_df_CID.CID__CELL == DrugA_CID_CELL].index.item()
			EXP_A = BETA_BIND_MEAN[mean_ind_A]
	#
		if len(DrugB_check) == 0 :
			EXP_B, OX = get_MJ_data(DrugB_CID_CELL)
			if 'X' in OX :
				Fail_ind.append(IND)
			EXP_B = torch.Tensor(EXP_B).unsqueeze(1)
		else : 
			mean_ind_B = BETA_BIND_M_SIG_df_CID[BETA_BIND_M_SIG_df_CID.CID__CELL == DrugB_CID_CELL].index.item()
			EXP_B = BETA_BIND_MEAN[mean_ind_B]
	#
	# 
	# 
	AB_SYN = get_synergy_data(cid_cid_cell)
	# 
	MY_chem_A_feat[IND] = torch.Tensor(DrugA_Feat)
	MY_chem_B_feat[IND] = torch.Tensor(DrugB_Feat)
	MY_chem_A_adj[IND] = torch.Tensor(DrugA_ADJ)
	MY_chem_B_adj[IND] = torch.Tensor(DrugB_ADJ)
	MY_g_EXP_A[IND] = torch.Tensor(EXP_A)
	MY_g_EXP_B[IND] = torch.Tensor(EXP_B)
	MY_syn[IND] = torch.Tensor([AB_SYN])


PRJ_NAME = 'GDSC_COLON' # save the original ver 

SAVE_PATH = gdsc_c_path

torch.save(MY_chem_A_feat, SAVE_PATH+'{}.MY_chem_A_feat.pt'.format(PRJ_NAME))
torch.save(MY_chem_B_feat, SAVE_PATH+'{}.MY_chem_B_feat.pt'.format(PRJ_NAME))
torch.save(MY_chem_A_adj, SAVE_PATH+'{}.MY_chem_A_adj.pt'.format(PRJ_NAME))
torch.save(MY_chem_B_adj, SAVE_PATH+'{}.MY_chem_B_adj.pt'.format(PRJ_NAME))
torch.save(MY_g_EXP_A, SAVE_PATH+'{}.MY_g_EXP_A.pt'.format(PRJ_NAME))
torch.save(MY_g_EXP_B, SAVE_PATH+'{}.MY_g_EXP_B.pt'.format(PRJ_NAME))
torch.save(MY_syn, SAVE_PATH+'{}.MY_syn.pt'.format(PRJ_NAME))

g_colon8.to_csv(SAVE_PATH+'{}.A_B_C_S_SET.csv'.format(PRJ_NAME))

g_colon8[['cid_cid_cell','type']].drop_duplicates() #  















# pancreas 대상 
# pancreas 대상 
# pancreas 대상 
# pancreas 대상 
# pancreas 대상 


gg_tmp = g_pancreas5[['cid_cid_cell','Synergy?']].drop_duplicates() # 10097


1) # synergy 갈리는 애들은 빼버리는 경우 -> 아예 없애버리는 경우 

dup_ccc = list(gg_tmp[gg_tmp.cid_cid_cell.duplicated()]['cid_cid_cell']) 
g_pancreas5 = g_pancreas5[g_pancreas5.cid_cid_cell.isin(dup_ccc) == False] 
#row 17250 #dup_ccc 700


2) 아예 synergy 있는걸로 밀어버리는 경우 -> 더 살릴 수 있을지도 
dup_ccc = list(gg_tmp[gg_tmp.cid_cid_cell.duplicated()]['cid_cid_cell'])
tmp_change = g_pancreas5[g_pancreas5.cid_cid_cell.isin(dup_ccc)]
tmp_change['Synergy?'] = 1 
g_breast_ok = g_pancreas5[g_pancreas5.cid_cid_cell.isin(dup_ccc)==False] 
g_pancreas5 = pd.concat([g_breast_ok, tmp_change])
#row 18650 #dup_ccc 700






#### 위에 lincs 에서 가져옴 
filter8.columns = ['long_id_A' , 'CID_A_CELL']
g_pancreas6 = pd.merge(g_pancreas5, filter8, on = 'CID_A_CELL', how = 'left') # row 17250, ccc : 8697

filter8.columns = ['long_id_B' , 'CID_B_CELL']
g_pancreas7 = pd.merge(g_pancreas6, filter8, on = 'CID_B_CELL', how = 'left') # row 17250, ccc : 8697

g_long_A = list(g_pancreas7.long_id_A)
g_long_B = list(g_pancreas7.long_id_B)

ttype = [] 
for a in range(g_pancreas7.shape[0]) :
	type_a = type(g_long_A[a])
	type_b = type(g_long_B[a])
	if (type_a == str) & (type_b == str) : 
		ttype.append('AOBO')
	elif (type_a != str) & (type_b != str) : 
		ttype.append('AXBX')
	else : 
		ttype.append('AXBO')



g_pancreas7['type'] = ttype

g_pancreas7[g_pancreas7['type'] =='AOBO'] # 290
g_pancreas7[g_pancreas7['type'] =='AXBO'] # 276
g_pancreas7[g_pancreas7['type'] =='AXBX'] # 16684


g_pancreas8 = g_pancreas7[g_pancreas7.CID_A_CELL.isin(MJ_gdsc.columns) & g_pancreas7.CID_B_CELL.isin(MJ_gdsc.columns)]
g_pancreas8 = g_pancreas8.reset_index(drop=True) # row :41682, ccc : 32522




def get_synergy_data(cid_cid_cell):
	syn_list = list(set(g_pancreas8[g_pancreas8.cid_cid_cell==cid_cid_cell]['Synergy?']))
	if len(syn_list) == 1 : 
		syn_res = syn_list[0]
	else :
		print('a')
	return syn_res



max_len = 50

MY_chem_A_feat = torch.empty(size=(g_pancreas8.shape[0], max_len, 64))
MY_chem_B_feat= torch.empty(size=(g_pancreas8.shape[0], max_len, 64))
MY_chem_A_adj = torch.empty(size=(g_pancreas8.shape[0], max_len, max_len))
MY_chem_B_adj= torch.empty(size=(g_pancreas8.shape[0], max_len, max_len))
MY_syn =  torch.empty(size=(g_pancreas8.shape[0],1))

MY_g_EXP_A = torch.empty(size=(g_pancreas8.shape[0], 349, 1))##############
MY_g_EXP_B = torch.empty(size=(g_pancreas8.shape[0], 349, 1))##############


Fail_ind = []
from datetime import datetime

for IND in range(0, g_pancreas8.shape[0]): #  100
	if IND%100 == 0 : 
		print(str(IND)+'/'+str(g_pancreas8.shape[0]) )
		Fail_ind
		datetime.now()
	#
	cid_cid_cell = g_pancreas8.cid_cid_cell[IND]
	DrugA_CID = g_pancreas8['Anchor CID'][IND]
	DrugB_CID = g_pancreas8['Library CID'][IND]
	CELL = g_pancreas8['ccle_name'][IND]
	dat_type = g_pancreas8.type[IND]
	DrugA_CID_CELL = g_pancreas8.CID_A_CELL[IND]
	DrugB_CID_CELL = g_pancreas8.CID_B_CELL[IND]
	#
	k=1
	DrugA_Feat, DrugA_ADJ = check_drug_f_ts(DrugA_CID)
	DrugB_Feat, DrugB_ADJ = check_drug_f_ts(DrugB_CID)
	# 
	if dat_type == 'AOBO' :
		mean_ind_A = BETA_BIND_M_SIG_df_CID[BETA_BIND_M_SIG_df_CID.CID__CELL == DrugA_CID_CELL].index.item()
		EXP_A = BETA_BIND_MEAN[mean_ind_A]
		mean_ind_B = BETA_BIND_M_SIG_df_CID[BETA_BIND_M_SIG_df_CID.CID__CELL == DrugB_CID_CELL].index.item()
		EXP_B = BETA_BIND_MEAN[mean_ind_B]
	#
	else :
		DrugA_check = BETA_BIND_M_SIG_df_CID[BETA_BIND_M_SIG_df_CID.CID__CELL == DrugA_CID_CELL]
		DrugB_check = BETA_BIND_M_SIG_df_CID[BETA_BIND_M_SIG_df_CID.CID__CELL == DrugB_CID_CELL]
	#
		if len(DrugA_check) == 0 :
			EXP_A, OX = get_MJ_data(DrugA_CID_CELL)
			if 'X' in OX :
				Fail_ind.append(IND)
			EXP_A = torch.Tensor(EXP_A).unsqueeze(1)
		else : 
			mean_ind_A = BETA_BIND_M_SIG_df_CID[BETA_BIND_M_SIG_df_CID.CID__CELL == DrugA_CID_CELL].index.item()
			EXP_A = BETA_BIND_MEAN[mean_ind_A]
	#
		if len(DrugB_check) == 0 :
			EXP_B, OX = get_MJ_data(DrugB_CID_CELL)
			if 'X' in OX :
				Fail_ind.append(IND)
			EXP_B = torch.Tensor(EXP_B).unsqueeze(1)
		else : 
			mean_ind_B = BETA_BIND_M_SIG_df_CID[BETA_BIND_M_SIG_df_CID.CID__CELL == DrugB_CID_CELL].index.item()
			EXP_B = BETA_BIND_MEAN[mean_ind_B]
	#
	# 
	# 
	AB_SYN = get_synergy_data(cid_cid_cell)
	# 
	MY_chem_A_feat[IND] = torch.Tensor(DrugA_Feat)
	MY_chem_B_feat[IND] = torch.Tensor(DrugB_Feat)
	MY_chem_A_adj[IND] = torch.Tensor(DrugA_ADJ)
	MY_chem_B_adj[IND] = torch.Tensor(DrugB_ADJ)
	MY_g_EXP_A[IND] = torch.Tensor(EXP_A)
	MY_g_EXP_B[IND] = torch.Tensor(EXP_B)
	MY_syn[IND] = torch.Tensor([AB_SYN])


PRJ_NAME = 'GDSC_PANCRE' # save the original ver 

SAVE_PATH = gdsc_c_path

torch.save(MY_chem_A_feat, SAVE_PATH+'{}.MY_chem_A_feat.pt'.format(PRJ_NAME))
torch.save(MY_chem_B_feat, SAVE_PATH+'{}.MY_chem_B_feat.pt'.format(PRJ_NAME))
torch.save(MY_chem_A_adj, SAVE_PATH+'{}.MY_chem_A_adj.pt'.format(PRJ_NAME))
torch.save(MY_chem_B_adj, SAVE_PATH+'{}.MY_chem_B_adj.pt'.format(PRJ_NAME))
torch.save(MY_g_EXP_A, SAVE_PATH+'{}.MY_g_EXP_A.pt'.format(PRJ_NAME))
torch.save(MY_g_EXP_B, SAVE_PATH+'{}.MY_g_EXP_B.pt'.format(PRJ_NAME))
torch.save(MY_syn, SAVE_PATH+'{}.MY_syn.pt'.format(PRJ_NAME))

g_pancreas8.to_csv(SAVE_PATH+'{}.A_B_C_S_SET.csv'.format(PRJ_NAME))

g_pancreas8[['cid_cid_cell','type']].drop_duplicates() #  




###################################################################



이제 우리꺼로 돌려보기 GDSC 


LOCAL 


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




OLD_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V8_W403_349_MIS2/'
ANA_DF_CSV = pd.read_csv(os.path.join(OLD_PATH,'RAY_ANA_DF.{}.csv'.format('M3V8_W403_349_MIS2')))

my_config = ANA_DF_CSV.loc[0]

CKP_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V8_W404_349_MIS2/M_404'


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
			G_chem_layer, 64 , G_chem_hdim,
			G_exp_layer, 3, G_exp_hdim,
			dsn_layers, dsn_layers, snp_layers, 
			1,
			inDrop, Drop
			) 



device = "cuda:0" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
	state_dict = torch.load(CKP_PATH) #### change ! 
else:
	state_dict = torch.load(CKP_PATH, map_location=torch.device('cpu'))



print("state_dict_done", flush = True)

if type(state_dict) == tuple:
	best_model.load_state_dict(state_dict[0])
else : 
	best_model.load_state_dict(state_dict)


print("state_load_done", flush = True)
#
#
best_model.to(device)
best_model.eval()





# DATA


gdsc_c_path = '/st06/jiyeonH/11.TOX/DR_SPRING/val_data/'

1) BREAST

PRJ_NAME = 'GDSC_BREAST'
g_breast8 = pd.read_csv(gdsc_c_path+'{}.A_B_C_S_SET.csv'.format(PRJ_NAME))
g_breast8[['cid_cid_cell','type']].drop_duplicates() # 32522 

MY_chem_A_feat = torch.load(gdsc_c_path+'{}.MY_chem_A_feat.pt'.format(PRJ_NAME))
MY_chem_B_feat = torch.load(  gdsc_c_path+'{}.MY_chem_B_feat.pt'.format(PRJ_NAME))
MY_chem_A_adj = torch.load( gdsc_c_path+'{}.MY_chem_A_adj.pt'.format(PRJ_NAME))
MY_chem_B_adj = torch.load( gdsc_c_path+'{}.MY_chem_B_adj.pt'.format(PRJ_NAME))
MY_g_EXP_A = torch.load( gdsc_c_path+'{}.MY_g_EXP_A.pt'.format(PRJ_NAME))
MY_g_EXP_B = torch.load( gdsc_c_path+'{}.MY_g_EXP_B.pt'.format(PRJ_NAME))
MY_syn = torch.load( gdsc_c_path+'{}.MY_syn.pt'.format(PRJ_NAME))

g_breast8_tuple = [(g_breast8['Anchor CID'][a], g_breast8['Library CID'][a], g_breast8['ccle_name'][a]) for a in range(g_breast8.shape[0])]



2) colon

PRJ_NAME = 'GDSC_COLON'

g_colon8 = pd.read_csv(gdsc_c_path+'{}.A_B_C_S_SET.csv'.format(PRJ_NAME))
g_colon8[['cid_cid_cell','type']].drop_duplicates() # 32522 

MY_chem_A_feat = torch.load(gdsc_c_path+'{}.MY_chem_A_feat.pt'.format(PRJ_NAME))
MY_chem_B_feat = torch.load(  gdsc_c_path+'{}.MY_chem_B_feat.pt'.format(PRJ_NAME))
MY_chem_A_adj = torch.load( gdsc_c_path+'{}.MY_chem_A_adj.pt'.format(PRJ_NAME))
MY_chem_B_adj = torch.load( gdsc_c_path+'{}.MY_chem_B_adj.pt'.format(PRJ_NAME))
MY_g_EXP_A = torch.load( gdsc_c_path+'{}.MY_g_EXP_A.pt'.format(PRJ_NAME))
MY_g_EXP_B = torch.load( gdsc_c_path+'{}.MY_g_EXP_B.pt'.format(PRJ_NAME))
MY_syn = torch.load( gdsc_c_path+'{}.MY_syn.pt'.format(PRJ_NAME))

g_colon8_tuple = [(g_colon8['Anchor CID'][a], g_colon8['Library CID'][a], g_colon8['ccle_name'][a]) for a in range(g_colon8.shape[0])]



3) pancreas

PRJ_NAME = 'GDSC_PANCRE'

g_pancreas8 = pd.read_csv(gdsc_c_path+'{}.A_B_C_S_SET.csv'.format(PRJ_NAME))
g_pancreas8[['cid_cid_cell','type']].drop_duplicates() # 32522 

MY_chem_A_feat = torch.load(gdsc_c_path+'{}.MY_chem_A_feat.pt'.format(PRJ_NAME))
MY_chem_B_feat = torch.load(  gdsc_c_path+'{}.MY_chem_B_feat.pt'.format(PRJ_NAME))
MY_chem_A_adj = torch.load( gdsc_c_path+'{}.MY_chem_A_adj.pt'.format(PRJ_NAME))
MY_chem_B_adj = torch.load( gdsc_c_path+'{}.MY_chem_B_adj.pt'.format(PRJ_NAME))
MY_g_EXP_A = torch.load( gdsc_c_path+'{}.MY_g_EXP_A.pt'.format(PRJ_NAME))
MY_g_EXP_B = torch.load( gdsc_c_path+'{}.MY_g_EXP_B.pt'.format(PRJ_NAME))
MY_syn = torch.load( gdsc_c_path+'{}.MY_syn.pt'.format(PRJ_NAME))

g_pancreas8_tuple = [(g_pancreas8['Anchor CID'][a], g_pancreas8['Library CID'][a], g_pancreas8['ccle_name'][a]) for a in range(g_pancreas8.shape[0])]







# target & basal exp 데이터 가져갈 수 있게 해주기 

TARGET_PATH = '/st06/jiyeonH/13.DD_SESS/01.PRJ2/'
TARGET_DB = pd.read_csv(TARGET_PATH+'TARGET_CID_ENTREZ.csv', sep ='\t', index_col = 0)

target_cids = list(TARGET_DB.CID)
gene_ids = list(BETA_ORDER_DF.gene_id)


def get_targets(CID): # 데려 가기로 함 
	if CID in target_cids:
		tmp_df2 = TARGET_DB[TARGET_DB.CID == CID]
		targets = list(set(tmp_df2.EntrezID))
		vec = [1 if a in targets else 0 for a in gene_ids ]
	else :
		vec = [0] * 349
	return vec






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

avail_cell_list =  ['VCAP_PROSTATE', 'NIHOVCAR3_OVARY', 'SW620_LARGE_INTESTINE', 'OVCAR4_OVARY', 'BT549_BREAST', 'A549_LUNG', 'SKMEL5_SKIN', 'A427_LUNG', 'BT474_BREAST', 'HOP92_LUNG', 'T98G_CENTRAL_NERVOUS_SYSTEM', 'NCIH23_LUNG', 'HT144_SKIN', 'RVH421_SKIN', 'MDAMB361_BREAST', 'LOVO_LARGE_INTESTINE', 'MDAMB231_BREAST', 'SKMEL28_SKIN', 'NCIH1650_LUNG', 'RKO_LARGE_INTESTINE', 'OVCAR5_OVARY', 'UACC812_BREAST', 'KPL1_BREAST', 'MSTO211H_PLEURA', 'KM12_LARGE_INTESTINE', 'IGROV1_OVARY', 'UHO1_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'NCIH520_LUNG', 'OVCAR8_OVARY', 'HCT15_LARGE_INTESTINE', 'A375_SKIN', 'CAKI1_KIDNEY', 'MDAMB468_BREAST', 'K562_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'A101D_SKIN', 'PA1_OVARY', 'UO31_KIDNEY', 'HOP62_LUNG', 'SF539_CENTRAL_NERVOUS_SYSTEM', 'MDAMB175VII_BREAST', 'U251MG_CENTRAL_NERVOUS_SYSTEM', 'HCC1500_BREAST', 'L1236_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'HCC1419_BREAST', 'NCIH460_LUNG', 'NCIH2122_LUNG', 'COLO792_SKIN', 'SR786_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'UACC257_SKIN', 'SNB75_CENTRAL_NERVOUS_SYSTEM', 'HCT116_LARGE_INTESTINE', 'PC3_PROSTATE', 'NCIH226_LUNG', 'RPMI8226_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'EKVX_LUNG', 'COLO800_SKIN', 'HT29_LARGE_INTESTINE', 'UWB1289_OVARY', 'MDAMB436_BREAST', 'SKOV3_OVARY', 'ZR751_BREAST', 'MEWO_SKIN', 'MELHO_SKIN', 'A2058_SKIN', 'RPMI7951_SKIN', 'SF268_CENTRAL_NERVOUS_SYSTEM', 'ACHN_KIDNEY', 'IPC298_SKIN', 'MALME3M_SKIN', 'A673_BONE', 'SF295_CENTRAL_NERVOUS_SYSTEM', 'CAOV3_OVARY', 'A498_KIDNEY', 'SKMEL2_SKIN', 'UACC62_SKIN', 'ES2_OVARY', 'LOXIMVI_SKIN', '786O_KIDNEY', 'MCF7_BREAST', 'WM115_SKIN', 'A2780_OVARY', 'DLD1_LARGE_INTESTINE', 'HS578T_BREAST', 'SKMES1_LUNG', 'T47D_BREAST', 'OV90_OVARY', 'G361_SKIN', 'SKMEL30_SKIN', 'COLO829_SKIN', 'SW837_LARGE_INTESTINE', 'NCIH522_LUNG', 'CAMA1_BREAST']


def get_ccle_data(cello) :
	if cello in ccle_cello_names : 
		ccle_exp_df = ccle_exp3[ccle_exp3.DrugCombCCLE==cello][BETA_ENTREZ_ORDER]
		ccle_exp_vector = ccle_exp_df.values[0].tolist()
	else : # no worries here. 
		# ccle_exp_vector = [0]*978
		ccle_exp_vector = [0]*349
	return ccle_exp_vector


set(g_breast8.ccle_name) - set(ccle_cello_names)
set(g_colon8.ccle_name) - set(ccle_cello_names)
set(g_pancreas8.ccle_name) - set(ccle_cello_names)


# (11152667, 84691, 'CAL120_BREAST')


					# LINCS 값을 우선시 하는 버전 (마치 MISS 2)
					def check_exp_f_ts(A, CID, CELLO) :
						if A == 'A' : 
							indexx = g_breast8[ (g_breast8['Anchor CID'] == CID) & (g_breast8['ccle_name'] == CELLO)].index[0]
							EXP_vector = MY_g_EXP_A[indexx]
						else :
							indexx = g_breast8[ (g_breast8['Library CID'] == CID) & (g_breast8['ccle_name'] == CELLO)].index[0]
							EXP_vector = MY_g_EXP_B[indexx]
						#
						# TARGET 
						TG_vector = get_targets(CID)
						#
						# BASAL EXP 
						B_vector = get_ccle_data(CELLO)
						#
						#
						FEAT = torch.Tensor(np.array([ EXP_vector.squeeze().tolist() , TG_vector, B_vector]).T)
						return FEAT.view(-1,3)




				# LINCS 값을 우선시 하는 버전 (마치 MISS 2)
				def check_exp_f_ts(A, CID, CELLO) :
					if A == 'A' : 
						indexx = g_colon8[ (g_colon8['Anchor CID'] == CID) & (g_colon8['ccle_name'] == CELLO)].index[0]
						EXP_vector = MY_g_EXP_A[indexx]
					else :
						indexx = g_colon8[ (g_colon8['Library CID'] == CID) & (g_colon8['ccle_name'] == CELLO)].index[0]
						EXP_vector = MY_g_EXP_B[indexx]
					#
					# TARGET 
					TG_vector = get_targets(CID)
					#
					# BASAL EXP 
					B_vector = get_ccle_data(CELLO)
					#
					#
					FEAT = torch.Tensor(np.array([ EXP_vector.squeeze().tolist() , TG_vector, B_vector]).T)
					return FEAT.view(-1,3)



					# LINCS 값을 우선시 하는 버전 (마치 MISS 2)
					def check_exp_f_ts(A, CID, CELLO) :
						if A == 'A' : 
							indexx = g_pancreas8[ (g_pancreas8['Anchor CID'] == CID) & (g_pancreas8['ccle_name'] == CELLO)].index[0]
							EXP_vector = MY_g_EXP_A[indexx]
						else :
							indexx = g_pancreas8[ (g_pancreas8['Library CID'] == CID) & (g_pancreas8['ccle_name'] == CELLO)].index[0]
							EXP_vector = MY_g_EXP_B[indexx]
						#
						# TARGET 
						TG_vector = get_targets(CID)
						#
						# BASAL EXP 
						B_vector = get_ccle_data(CELLO)
						#
						#
						FEAT = torch.Tensor(np.array([ EXP_vector.squeeze().tolist() , TG_vector, B_vector]).T)
						return FEAT.view(-1,3)











chem_A_pre = []
for i in range(MY_chem_A_adj.shape[0]):
	adj_pre = MY_chem_A_adj[i]
	adj_proc = adj_pre.long().to_sparse().indices()
	chem_A_pre.append(adj_proc)

chem_B_pre = []
for i in range(MY_chem_B_adj.shape[0]):
	adj_pre = MY_chem_B_adj[i]
	adj_proc = adj_pre.long().to_sparse().indices()
	chem_B_pre.append(adj_proc)


class GDSC_Dataset(Dataset): 
	def __init__(self, tuple_list):
		self.tuple_list = tuple_list
	#
	def __len__(self): 
		return len(self.tuple_list)
	#
	def __getitem__(self, idx): 
		ROW_CID, COL_CID, CELLO = self.tuple_list[idx]
		#
		#
		drug1_f , drug1_a = MY_chem_A_feat[idx], chem_A_pre[idx]
		drug2_f , drug2_a = MY_chem_B_feat[idx], chem_B_pre[idx]
		adj = copy.deepcopy(JY_ADJ_IDX).long()
		adj_w = torch.Tensor(JY_IDX_WEIGHT).squeeze()
		expA = check_exp_f_ts("A", ROW_CID, CELLO)
		expB = check_exp_f_ts("B", COL_CID, CELLO)
		cell = torch.zeros(size = (1, 25)) # no need
		y = torch.Tensor([0]).float().unsqueeze(1)
		#
		return ROW_CID, COL_CID, CELLO, drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y







a_c_tups = [(a,c) for a,b,c in g_pancreas8_tuple]
a_c_tups = list(set(a_c_tups))
b_c_tups = [(b,c) for a,b,c in g_pancreas8_tuple]
b_c_tups = list(set(b_c_tups))

expA_all = []

for a,c in a_c_tups :
	a
	resres = check_exp_f_ts("A", a, c)
	expA_all.append(resres)


expB_all = []

for b,c in b_c_tups :
	b
	resres = check_exp_f_ts("B", b, c)
	expB_all.append(resres)














def graph_collate_fn(batch):
	tup_list = []
	#
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
	for ROW_CID, COL_CID, CELLO, drug1_f, drug2_f, drug1_adj, drug2_adj, expA, expB, exp_adj, exp_adj_w, cell, y in batch :
		tup_list.append( (ROW_CID, COL_CID, CELLO) )
		#
		drug1_f_list.append(drug1_f)
		drug2_f_list.append(drug2_f)
		drug1_adj_list.append(drug1_adj+DRUG_1_num_nodes_seen)
		drug2_adj_list.append(drug2_adj+DRUG_2_num_nodes_seen)
		expA_list.append(expA)
		expB_list.append(expB)
		exp_adj_list.append(exp_adj+EXP_num_nodes_seen)
		exp_adj_w_list.append(exp_adj_w.unsqueeze(0))
		y_list.append(torch.Tensor(y))
		cell_list.append(cell)
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
	return tup_list, drug1_f_new, drug2_f_new, drug1_adj_new, drug2_adj_new, expA_new, expB_new, exp_adj_new, exp_adj_w_new, cell_new, y_new




dataset = GDSC_Dataset(g_breast8_tuple)
dataset = GDSC_Dataset(g_colon8_tuple)
dataset = GDSC_Dataset(g_pancreas8_tuple)


dataloader = torch.utils.data.DataLoader(dataset, batch_size = 64 , collate_fn = graph_collate_fn, shuffle = False , num_workers = 8)# , num_workers=my_config['config/n_workers'].item()
#

CELL_PRED_DF = pd.DataFrame(columns = ['PRED','ROW_CID','COL_CID','CCLE','Y'])
CELL_PRED_DF.to_csv(gdsc_c_path+'PRED_{}.FINAL_ing2.csv'.format(PRJ_NAME), index=False)

with torch.no_grad():
	for batch_idx_t, (tup_list, drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y) in enumerate(dataloader):
		print("{} / {}".format(batch_idx_t, len(dataloader)) , flush = True)
		print(datetime.now(), flush = True)
		list_ROW_CID = [a[0] for a in tup_list]
		list_COL_CID = [a[1] for a in tup_list]
		list_CELLO = [a[2] for a in tup_list]
		#
		output = best_model(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w.squeeze(), cell.squeeze(), y) 
		outputs = output.squeeze().tolist() # [output.squeeze().item()]
		#
		tmp_df = pd.DataFrame({
		'PRED': outputs,
		'ROW_CID' : list_ROW_CID,
		'COL_CID' : list_COL_CID,
		'CCLE' : list_CELLO,
		'Y' : y.squeeze().tolist()
		})
		CELL_PRED_DF = pd.concat([CELL_PRED_DF, tmp_df])
		tmp_df.to_csv(gdsc_c_path+'PRED_{}.FINAL_ing2.csv'.format(PRJ_NAME), mode='a', index=False, header = False)
		


g_breast8['PRED'] = list(CELL_PRED_DF['PRED'])
g_breast8.to_csv(gdsc_c_path+'FINAL_{}.csv'.format(PRJ_NAME))



g_colon8['PRED'] = list(CELL_PRED_DF['PRED'])
g_colon8.to_csv(gdsc_c_path+'FINAL_{}.csv'.format(PRJ_NAME))


g_pancreas8['PRED'] = list(CELL_PRED_DF['PRED'])
g_pancreas8.to_csv(gdsc_c_path+'FINAL_{}.csv'.format(PRJ_NAME))






# 그래서 예측한 결과

1) breast 
CELL_PRED_DF = pd.read_csv(gdsc_c_path+'PRED_GDSC_BREAST.FINAL_ing2.csv')
cid_a = list(CELL_PRED_DF['ROW_CID'])
cid_b = list(CELL_PRED_DF['COL_CID'])
cell = list(CELL_PRED_DF['CCLE'])

CELL_PRED_DF['cid_cid_cell'] = [str(int(cid_a[i])) + '___' + str(int(cid_b[i]))+ '___' + cell[i] if cid_a[i] < cid_b[i] else str(int(cid_b[i])) + '___' + str(int(cid_a[i]))+ '___' + cell[i] for i in range(CELL_PRED_DF.shape[0])]


CELL_PRED_DF_re = pd.DataFrame(CELL_PRED_DF.groupby('cid_cid_cell').mean())[['PRED']]
CELL_PRED_DF_re['cid_cid_cell'] = CELL_PRED_DF_re.index
CELL_PRED_DF_re.index = [a for a in range(CELL_PRED_DF_re.shape[0])]

g_breast9 = pd.merge(g_breast8, CELL_PRED_DF_re, on = 'cid_cid_cell', how ='left') # 42778


1) colon 
CELL_PRED_DF = pd.read_csv(gdsc_c_path+'PRED_GDSC_COLON.FINAL_ing2.csv')
cid_a = list(CELL_PRED_DF['ROW_CID'])
cid_b = list(CELL_PRED_DF['COL_CID'])
cell = list(CELL_PRED_DF['CCLE'])

CELL_PRED_DF['cid_cid_cell'] = [str(int(cid_a[i])) + '___' + str(int(cid_b[i]))+ '___' + cell[i] if cid_a[i] < cid_b[i] else str(int(cid_b[i])) + '___' + str(int(cid_a[i]))+ '___' + cell[i] for i in range(CELL_PRED_DF.shape[0])]


CELL_PRED_DF_re = pd.DataFrame(CELL_PRED_DF.groupby('cid_cid_cell').mean())[['PRED']]
CELL_PRED_DF_re['cid_cid_cell'] = CELL_PRED_DF_re.index
CELL_PRED_DF_re.index = [a for a in range(CELL_PRED_DF_re.shape[0])]

g_colon9 = pd.merge(g_colon8, CELL_PRED_DF_re, on = 'cid_cid_cell', how ='left') # 19690



1) pancreas  
CELL_PRED_DF = pd.read_csv(gdsc_c_path+'PRED_GDSC_PANCRE.FINAL_ing2.csv')
cid_a = list(CELL_PRED_DF['ROW_CID'])
cid_b = list(CELL_PRED_DF['COL_CID'])
cell = list(CELL_PRED_DF['CCLE'])

CELL_PRED_DF['cid_cid_cell'] = [str(int(cid_a[i])) + '___' + str(int(cid_b[i]))+ '___' + cell[i] if cid_a[i] < cid_b[i] else str(int(cid_b[i])) + '___' + str(int(cid_a[i]))+ '___' + cell[i] for i in range(CELL_PRED_DF.shape[0])]


CELL_PRED_DF_re = pd.DataFrame(CELL_PRED_DF.groupby('cid_cid_cell').mean())[['PRED']]
CELL_PRED_DF_re['cid_cid_cell'] = CELL_PRED_DF_re.index
CELL_PRED_DF_re.index = [a for a in range(CELL_PRED_DF_re.shape[0])]

g_pancreas9 = pd.merge(g_pancreas8, CELL_PRED_DF_re, on = 'cid_cid_cell', how ='left')


#####################
####################
그림그리기 



PRJ_NAME= 'GDSC_BREAST' # #025669
#g_breast_result = pd.read_csv(gdsc_c_path+'FINAL_{}.csv'.format(PRJ_NAME))


PRJ_NAME= 'GDSC_COLON' # #ffcd36
#g_colon_result = pd.read_csv(gdsc_c_path+'FINAL_{}.csv'.format(PRJ_NAME))


PRJ_NAME= 'GDSC_PANCRE' # #ffcd36
#g_pancreas_result = pd.read_csv(gdsc_c_path+'FINAL_{}.csv'.format(PRJ_NAME))



# training 에 들어간 애들은 제거해야지 
from matplotlib import colors as mcolors


SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V8_349_FULL/'

file_name = 'M3V8_349_MISS2_FULL' # 0608
A_B_C_S_SET_ADD = pd.read_csv(SAVE_PATH+'{}.A_B_C_S_SET_ADD.csv'.format(file_name), low_memory=False)
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
A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.type.isin(MISS_filter)]

CELL_92 = ['VCAP_PROSTATE', 'NIHOVCAR3_OVARY', 'SW620_LARGE_INTESTINE', 'OVCAR4_OVARY', 'BT549_BREAST', 'A549_LUNG', 'SKMEL5_SKIN', 'A427_LUNG', 'BT474_BREAST', 'HOP92_LUNG', 'T98G_CENTRAL_NERVOUS_SYSTEM', 'NCIH23_LUNG', 'HT144_SKIN', 'RVH421_SKIN', 'MDAMB361_BREAST', 'LOVO_LARGE_INTESTINE', 'MDAMB231_BREAST', 'SKMEL28_SKIN', 'NCIH1650_LUNG', 'RKO_LARGE_INTESTINE', 'OVCAR5_OVARY', 'UACC812_BREAST', 'KPL1_BREAST', 'MSTO211H_PLEURA', 'KM12_LARGE_INTESTINE', 'IGROV1_OVARY', 'UHO1_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'NCIH520_LUNG', 'OVCAR8_OVARY', 'HCT15_LARGE_INTESTINE', 'A375_SKIN', 'CAKI1_KIDNEY', 'MDAMB468_BREAST', 'K562_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'A101D_SKIN', 'PA1_OVARY', 'UO31_KIDNEY', 'HOP62_LUNG', 'SF539_CENTRAL_NERVOUS_SYSTEM', 'MDAMB175VII_BREAST', 'U251MG_CENTRAL_NERVOUS_SYSTEM', 'HCC1500_BREAST', 'L1236_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'HCC1419_BREAST', 'NCIH460_LUNG', 'NCIH2122_LUNG', 'COLO792_SKIN', 'SR786_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'UACC257_SKIN', 'SNB75_CENTRAL_NERVOUS_SYSTEM', 'HCT116_LARGE_INTESTINE', 'PC3_PROSTATE', 'NCIH226_LUNG', 'RPMI8226_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'EKVX_LUNG', 'COLO800_SKIN', 'HT29_LARGE_INTESTINE', 'UWB1289_OVARY', 'MDAMB436_BREAST', 'SKOV3_OVARY', 'ZR751_BREAST', 'MEWO_SKIN', 'MELHO_SKIN', 'A2058_SKIN', 'RPMI7951_SKIN', 'SF268_CENTRAL_NERVOUS_SYSTEM', 'ACHN_KIDNEY', 'IPC298_SKIN', 'MALME3M_SKIN', 'A673_BONE', 'SF295_CENTRAL_NERVOUS_SYSTEM', 'CAOV3_OVARY', 'A498_KIDNEY', 'SKMEL2_SKIN', 'UACC62_SKIN', 'ES2_OVARY', 'LOXIMVI_SKIN', '786O_KIDNEY', 'MCF7_BREAST', 'WM115_SKIN', 'A2780_OVARY', 'DLD1_LARGE_INTESTINE', 'HS578T_BREAST', 'SKMES1_LUNG', 'T47D_BREAST', 'OV90_OVARY', 'G361_SKIN', 'SKMEL30_SKIN', 'COLO829_SKIN', 'SW837_LARGE_INTESTINE', 'NCIH522_LUNG', 'CAMA1_BREAST']
A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.CELL.isin(CELL_92)]


DC_CELL_DF2 = pd.read_csv(DC_PATH+'DC_CELL_INFO.csv', sep = '\t')
DC_CELL_DF2 = pd.concat([
	DC_CELL_DF2, 
	pd.DataFrame({'cell_line_id' : [1],'DC_cellname' : ['786O'],'DrugCombCello' : ['CVCL_1051'],'DrugCombCCLE':['786O_KIDNEY']})])

DC_CELL_info_filt = DC_CELL_DF2[DC_CELL_DF2.DrugCombCCLE.isin(A_B_C_S_SET.CELL)] # 38

DC_CELL_info_filt = DC_CELL_info_filt.drop(['Unnamed: 0'], axis = 1)
DC_CELL_info_filt.columns = ['cell_line_id', 'DC_cellname', 'DrugCombCello', 'CELL']
DC_CELL_info_filt = DC_CELL_info_filt[['CELL','DC_cellname']]

A_B_C_S_SET_COH = pd.merge(A_B_C_S_SET, DC_CELL_info_filt, on = 'CELL', how = 'left'  )






known = list(A_B_C_S_SET_COH.cid_cid_cell) # 294444


g_colon_result = g_colon9
g_breast_result = g_breast9
g_pancreas_result = g_pancreas9

# _1 의 경우 
g_colon_result_filt = g_colon_result[g_colon_result.cid_cid_cell.isin(known)==False] # 18416
g_colon_result_filt = g_colon_result_filt[['PRED','Synergy?']]
g_colon_result_filt['Tissue'] = 'LARGE_INTESTINE'
g_colon_result_filt['CLASS'] = g_colon_result_filt['Synergy?'].apply(lambda x : 'O' if x==1 else 'X')
g_colon_result_filt['CT'] = g_colon_result_filt['Tissue'] + '_' + g_colon_result_filt['CLASS']

g_breast_result_filt = g_breast_result[g_breast_result.cid_cid_cell.isin(known)==False] # 40947
g_breast_result_filt = g_breast_result_filt[['PRED','Synergy?']]
g_breast_result_filt['Tissue'] = 'BREAST'
g_breast_result_filt['CLASS'] = g_breast_result_filt['Synergy?'].apply(lambda x : 'O' if x==1 else 'X')
g_breast_result_filt['CT'] = g_breast_result_filt['Tissue'] + '_' + g_breast_result_filt['CLASS']

g_pancreas_result_filt = g_pancreas_result[g_pancreas_result.cid_cid_cell.isin(known)==False] # 12278
g_pancreas_result_filt = g_pancreas_result_filt[['PRED','Synergy?']]
g_pancreas_result_filt['Tissue'] = 'PANCREAS'
g_pancreas_result_filt['CLASS'] = g_pancreas_result_filt['Synergy?'].apply(lambda x : 'O' if x==1 else 'X')
g_pancreas_result_filt['CT'] = g_pancreas_result_filt['Tissue'] + '_' + g_pancreas_result_filt['CLASS']


# 이건 _2 의 경우 
g_colon_result2 = g_colon_result.groupby('cid_cid_cell').mean()
g_breast_result2 = g_breast_result.groupby('cid_cid_cell').mean()
g_pancreas_result2 = g_pancreas_result.groupby('cid_cid_cell').mean()

g_colon_result_filt = g_colon_result2[['PRED','Synergy?']]
g_colon_result_filt['Tissue'] = 'LARGE_INTESTINE'
g_colon_result_filt['CLASS'] = g_colon_result_filt['Synergy?'].apply(lambda x : 'O' if x==1 else 'X')
g_colon_result_filt['CT'] = g_colon_result_filt['Tissue'] + '_' + g_colon_result_filt['CLASS']

g_breast_result_filt = g_breast_result2[['PRED','Synergy?']]
g_breast_result_filt['Tissue'] = 'BREAST'
g_breast_result_filt['CLASS'] = g_breast_result_filt['Synergy?'].apply(lambda x : 'O' if x==1 else 'X')
g_breast_result_filt['CT'] = g_breast_result_filt['Tissue'] + '_' + g_breast_result_filt['CLASS']

g_pancreas_result_filt = g_pancreas_result2[['PRED','Synergy?']]
g_pancreas_result_filt['Tissue'] = 'PANCREAS'
g_pancreas_result_filt['CLASS'] = g_pancreas_result_filt['Synergy?'].apply(lambda x : 'O' if x==1 else 'X')
g_pancreas_result_filt['CT'] = g_pancreas_result_filt['Tissue'] + '_' + g_pancreas_result_filt['CLASS']



1_1 : 18438 / 41682 / 12278
1_2 : 9240 / 32522 / 6139


2_1 : 19664 / 42021 / 13156
2_2 : 9866 / 33070 / 6578





test_num = '2_2'



gdsc_all = pd.concat([g_colon_result_filt, g_breast_result_filt, g_pancreas_result_filt])
from statannot import add_stat_annotation


# order=my_order,
fig, ax = plt.subplots(figsize=(13, 10)) 
sns.boxplot(ax = ax , data = gdsc_all, x = 'CT', y = 'PRED', width = 0.3, linewidth=1, hue = 'Tissue', palette = {'LARGE_INTESTINE' : '#ffcd36' , 'BREAST' : '#025669', 'PANCREAS' : '#ff009bff'} ) # width = 3,
ax, test_results = add_stat_annotation(ax = ax, data=gdsc_all, x='CT', y='PRED',
                                   box_pairs=[("LARGE_INTESTINE_O", "LARGE_INTESTINE_X"),  ("BREAST_O", "BREAST_X"), ("PANCREAS_O", "PANCREAS_X")],
                                   test='t-test_ind', text_format='star', loc='outside', verbose=2)

# test='Mann-Whitney'
# t-test_ind, t-test_welch, t-test_paired, 
# Mann-Whitney, Mann-Whitney-gt, Mann-Whitney-ls, 
# Levene, Wilcoxon, Kruskal

sns.despine()
ax.set_xlabel('tissue, OX', fontsize=10)
ax.set_ylabel('Predicted value', fontsize=10)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
ax.tick_params(axis='both', which='major', labelsize=10 )
plt.legend(bbox_to_anchor=(1.02, 0.15), loc='upper left', borderaxespad=0)
#plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join('/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/gdsc.0922.box{}.png'.format(test_num)), dpi = 300)
plt.savefig('/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/gdsc.0922.box{}.pdf'.format(test_num), format="pdf", bbox_inches = 'tight')

plt.close()




# order=my_order,
fig, ax = plt.subplots(figsize=(13, 10)) 
sns.boxplot(ax = ax , data = gdsc_all, x = 'CT', y = 'PRED', width = 0.3, linewidth=1, hue = 'Tissue', palette = {'LARGE_INTESTINE' : '#ffcd36' , 'BREAST' : '#025669', 'PANCREAS' : '#ff009bff'} ) # width = 3,
ax, test_results = add_stat_annotation(ax = ax, data=gdsc_all, x='CT', y='PRED',
                                   box_pairs=[("LARGE_INTESTINE_O", "LARGE_INTESTINE_X"),  ("BREAST_O", "BREAST_X"), ("PANCREAS_O", "PANCREAS_X")],
                                   test='Mann-Whitney', text_format='star', loc='outside', verbose=2)

# test=''
# t-test_ind, t-test_welch, t-test_paired, 
# Mann-Whitney, Mann-Whitney-gt, Mann-Whitney-ls, 
# Levene, Wilcoxon, Kruskal

sns.despine()
ax.set_xlabel('tissue, OX', fontsize=10)
ax.set_ylabel('Predicted value', fontsize=10)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
ax.tick_params(axis='both', which='major', labelsize=10 )
plt.legend(bbox_to_anchor=(1.02, 0.15), loc='upper left', borderaxespad=0)
#plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join('/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/gdsc.0922.MWbox{}.png'.format(test_num)), dpi = 300)
plt.savefig('/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/gdsc.0922.MWbox{}.pdf'.format(test_num), format="pdf", bbox_inches = 'tight')

plt.close()




# order=my_order,
fig, ax = plt.subplots(figsize=(13, 10)) 
sns.boxplot(ax = ax , data = gdsc_all, x = 'CT', y = 'PRED', width = 0.3, linewidth=1, hue = 'Tissue', palette = {'LARGE_INTESTINE' : '#ffcd36' , 'BREAST' : '#025669', 'PANCREAS' : '#ff009bff'} ) # width = 3,
ax, test_results = add_stat_annotation(ax = ax, data=gdsc_all, x='CT', y='PRED',
                                   box_pairs=[("LARGE_INTESTINE_O", "LARGE_INTESTINE_X"),  ("BREAST_O", "BREAST_X"), ("PANCREAS_O", "PANCREAS_X")],
                                   test='t-test_welch', text_format='star', loc='outside', verbose=2)

# test=''
# t-test_ind, t-test_welch, t-test_paired, 
# Mann-Whitney, Mann-Whitney-gt, Mann-Whitney-ls, 
# Levene, Wilcoxon, Kruskal

sns.despine()
ax.set_xlabel('tissue, OX', fontsize=10)
ax.set_ylabel('Predicted value', fontsize=10)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
ax.tick_params(axis='both', which='major', labelsize=10 )
plt.legend(bbox_to_anchor=(1.02, 0.15), loc='upper left', borderaxespad=0)
#plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join('/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/gdsc.0922.TWbox{}.png'.format(test_num)), dpi = 300)
plt.savefig('/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/gdsc.0922.TWbox{}.pdf'.format(test_num), format="pdf", bbox_inches = 'tight')

plt.close()






# order=my_order,
fig, ax = plt.subplots(figsize=(13, 10)) 
sns.boxplot(ax = ax , data = gdsc_all, x = 'CT', y = 'PRED', width = 0.3, linewidth=1, hue = 'Tissue', palette = {'LARGE_INTESTINE' : '#ffcd36' , 'BREAST' : '#025669', 'PANCREAS' : '#ff009bff'} ) # width = 3,
ax, test_results = add_stat_annotation(ax = ax, data=gdsc_all, x='CT', y='PRED',
                                   box_pairs=[("LARGE_INTESTINE_O", "LARGE_INTESTINE_X"),  ("BREAST_O", "BREAST_X"), ("PANCREAS_O", "PANCREAS_X")],
                                   test='Kruskal', text_format='star', loc='outside', verbose=2)

# test=''
# t-test_ind, t-test_welch, t-test_paired, 
# Mann-Whitney, Mann-Whitney-gt, Mann-Whitney-ls, 
# Levene, Wilcoxon, Kruskal

sns.despine()
ax.set_xlabel('tissue, OX', fontsize=10)
ax.set_ylabel('Predicted value', fontsize=10)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
ax.tick_params(axis='both', which='major', labelsize=10 )
plt.legend(bbox_to_anchor=(1.02, 0.15), loc='upper left', borderaxespad=0)
#plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join('/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/gdsc.0922.Krubox{}.png'.format(test_num)), dpi = 300)
plt.savefig('/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/gdsc.0922.Krubox{}.pdf'.format(test_num), format="pdf", bbox_inches = 'tight')

plt.close()


















































############################### HEATMAP
############################### HEATMAP
############################### HEATMAP


PRJ_NAME= 'GDSC_BREAST' # #025669
g_breast_result = pd.read_csv(gdsc_c_path+'FINAL_{}.csv'.format(PRJ_NAME))


c_c_list = list(g_breast_result.cid_cid)
c_c_list_set = list(set(g_breast_result.cid_cid))
c_c_list_set.sort()

c_c_c_list = list(g_breast_result.cid_cid_cell)
c_c_c_list_set = list(set(g_breast_result.cid_cid_cell))
c_c_c_list_set.sort()

g_breast_result_O = g_breast_result[g_breast_result['Synergy?'] == 1] # 1015
g_breast_result_X = g_breast_result[g_breast_result['Synergy?'] == 0] # 40667


result_O_c_c_c_list = list(g_breast_result_O.cid_cid_cell)
result_O_c_c_c_list_set = list(set(g_breast_result_O.cid_cid_cell))
result_O_c_c_c_list_set.sort()

result_O_c_c_list = list(g_breast_result_O.cid_cid)
result_O_c_c_list_set = list(set(g_breast_result_O.cid_cid))
result_O_c_c_list_set.sort()


result_X_c_c_c_list = list(g_breast_result_X.cid_cid_cell)
result_X_c_c_c_list_set = list(set(g_breast_result_X.cid_cid_cell))
result_X_c_c_c_list_set.sort()

result_X_c_c_list = list(g_breast_result_X.cid_cid)
result_X_c_c_list_set = list(set(g_breast_result_X.cid_cid))
result_X_c_c_list_set.sort()



my_heatmap_dot_O = pd.DataFrame(
	data = "NA", #np.inf,
	columns=list(set(g_breast_result_O.ccle_name)),
	index=c_c_list_set
)

for c_ind in range(len(result_O_c_c_c_list_set)) : # 대충 18만줄 대상으로 30분 소요 
	if c_ind%1000 == 0 : 
		print(str(c_ind)+'/'+str(len(result_O_c_c_c_list_set)) )
		datetime.now() 
	c_c_c = result_O_c_c_c_list_set[c_ind]
	tmp_res = g_breast_result_O[g_breast_result_O.cid_cid_cell==c_c_c]
	c_c = c_c_c.split('___')[0] + '___'+ c_c_c.split('___')[1]
	c = c_c_c.split('___')[2]
	if tmp_res.shape[0] ==1 :
		my_heatmap_dot_O.at[c_c, c] = tmp_res.PRED.item()
	elif  tmp_res.shape[0] >1 :
		my_heatmap_dot_O.at[c_c, c] = np.mean(tmp_res.PRED)
	else :
		my_heatmap_dot_O.at[c_c, c] = 0
	
# my_heatmap_dot_O.to_csv( gdsc_c_path + 'FINAL_BREAST_HEAT_O.csv')
my_heatmap_dot_O = pd.read_csv(gdsc_c_path + 'FINAL_BREAST_HEAT_O.csv', index_col = 0 )
my_heatmap_dot_O.columns = ['O_'+ col for col in list(my_heatmap_dot_O.columns)]


my_heatmap_dot_X = pd.DataFrame(
	data = "NA", #np.inf,
	columns=list(set(g_breast_result_X.ccle_name)),
	index=c_c_list_set
)

for c_ind in range(len(result_X_c_c_c_list)) : # 대충 18만줄 대상으로 30분 소요 
	if c_ind%1000 == 0 : 
		print(str(c_ind)+'/'+str(len(result_X_c_c_c_list)) )
		datetime.now() 
	c_c_c = result_X_c_c_c_list[c_ind]
	tmp_res = g_breast_result_X[g_breast_result_X.cid_cid_cell==c_c_c]
	c_c = c_c_c.split('___')[0] + '___'+ c_c_c.split('___')[1]
	c = c_c_c.split('___')[2]
	if tmp_res.shape[0] ==1 :
		my_heatmap_dot_X.at[c_c, c] = tmp_res.PRED.item()
	elif  tmp_res.shape[0] >1 :
		my_heatmap_dot_X.at[c_c, c] = np.mean(tmp_res.PRED)
	else :
		my_heatmap_dot_X.at[c_c, c] = 0


# my_heatmap_dot_X.to_csv( gdsc_c_path + 'FINAL_BREAST_HEAT_X.csv')

my_heatmap_dot_X = pd.read_csv(gdsc_c_path + 'FINAL_BREAST_HEAT_X.csv', index_col = 0 )
my_heatmap_dot_X.columns = ['X_'+ col for col in list(my_heatmap_dot_X.columns)]

my_heatmap_dot_M = pd.concat([my_heatmap_dot_X, my_heatmap_dot_O], axis= 1)





tmp = my_heatmap_dot_M.replace('NA', 0 )
gg = sns.clustermap(
	tmp, center=0, cmap="vlag", vmin=-70, vmax=2,
	figsize=(20,20),
	row_cluster=True, col_cluster = False, 
	metric = 'correlation', method = 'complete',
	dendrogram_ratio=0.2, yticklabels=False) 

plt.tight_layout()
plt.savefig("/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/example.BREAST.pdf", bbox_inches='tight')
plt.close()


# O 기준으로만 가져오기 
#cmap = cm.get_cmap('PiYG', 11)
cmap = plt.get_cmap('RdBu_r', 11)

dot_mini = my_heatmap_dot_M.loc[result_O_c_c_list_set]

tmp = dot_mini.replace('NA', 0 )
# tmp = dot_mini.replace(np.nan, 0 )
gg = sns.clustermap(
	tmp, center=0, cmap=cmap, #vmin=-50, vmax=2,
	figsize=(20,20),
	row_cluster=True, col_cluster = False, 
	metric = 'correlation', method = 'complete',
	dendrogram_ratio=0.2, yticklabels=False) 

plt.tight_layout()
plt.savefig("/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/example.BREAST.mini.pdf", bbox_inches='tight')
plt.close()














PRJ_NAME= 'GDSC_COLON' # #ffcd36
g_colon_result = pd.read_csv(gdsc_c_path+'FINAL_{}.csv'.format(PRJ_NAME))

c_c_list = list(g_colon_result.cid_cid)
c_c_list_set = list(set(g_colon_result.cid_cid))
c_c_list_set.sort()

c_c_c_list = list(g_colon_result.cid_cid_cell)
c_c_c_list_set = list(set(g_colon_result.cid_cid_cell))
c_c_c_list_set.sort()

g_colon_result_O = g_colon_result[g_colon_result['Synergy?'] == 1] # 278
g_colon_result_X = g_colon_result[g_colon_result['Synergy?'] == 0] # 18160


result_O_c_c_c_list = list(g_colon_result_O.cid_cid_cell)
result_O_c_c_c_list_set = list(set(g_colon_result_O.cid_cid_cell))
result_O_c_c_c_list_set.sort()

result_O_c_c_list = list(g_colon_result_O.cid_cid)
result_O_c_c_list_set = list(set(g_colon_result_O.cid_cid))
result_O_c_c_list_set.sort()


result_X_c_c_c_list = list(g_colon_result_X.cid_cid_cell)
result_X_c_c_c_list_set = list(set(g_colon_result_X.cid_cid_cell))
result_X_c_c_c_list_set.sort()

result_X_c_c_list = list(g_colon_result_X.cid_cid)
result_X_c_c_list_set = list(set(g_colon_result_X.cid_cid))
result_X_c_c_list_set.sort()



my_heatmap_dot_O = pd.DataFrame(
	data = "NA", #np.inf,
	columns=list(set(g_colon_result_O.ccle_name)),
	index=c_c_list_set
)

for c_ind in range(len(result_O_c_c_c_list_set)) : # 대충 18만줄 대상으로 30분 소요 
	if c_ind%1000 == 0 : 
		print(str(c_ind)+'/'+str(len(result_O_c_c_c_list_set)) )
		datetime.now() 
	c_c_c = result_O_c_c_c_list_set[c_ind]
	tmp_res = g_colon_result_O[g_colon_result_O.cid_cid_cell==c_c_c]
	c_c = c_c_c.split('___')[0] + '___'+ c_c_c.split('___')[1]
	c = c_c_c.split('___')[2]
	if tmp_res.shape[0] ==1 :
		my_heatmap_dot_O.at[c_c, c] = tmp_res.PRED.item()
	elif  tmp_res.shape[0] >1 :
		my_heatmap_dot_O.at[c_c, c] = np.mean(tmp_res.PRED)
	else :
		my_heatmap_dot_O.at[c_c, c] = 0
	
# my_heatmap_dot_O.to_csv( gdsc_c_path + 'FINAL_COLON_HEAT_O.csv')


my_heatmap_dot_O = pd.read_csv(gdsc_c_path + 'FINAL_COLON_HEAT_O.csv', index_col = 0  )
my_heatmap_dot_O.columns = ['O_'+ col for col in list(my_heatmap_dot_O.columns)]



my_heatmap_dot_X = pd.DataFrame(
	data = "NA", #np.inf,
	columns=list(set(g_colon_result_X.ccle_name)),
	index=c_c_list_set
)

for c_ind in range(len(result_X_c_c_c_list)) : # 대충 18만줄 대상으로 30분 소요 
	if c_ind%1000 == 0 : 
		print(str(c_ind)+'/'+str(len(result_X_c_c_c_list)) )
		datetime.now() 
	c_c_c = result_X_c_c_c_list[c_ind]
	tmp_res = g_colon_result_X[g_colon_result_X.cid_cid_cell==c_c_c]
	c_c = c_c_c.split('___')[0] + '___'+ c_c_c.split('___')[1]
	c = c_c_c.split('___')[2]
	if tmp_res.shape[0] ==1 :
		my_heatmap_dot_X.at[c_c, c] = tmp_res.PRED.item()
	elif  tmp_res.shape[0] >1 :
		my_heatmap_dot_X.at[c_c, c] = np.mean(tmp_res.PRED)
	else :
		my_heatmap_dot_X.at[c_c, c] = 0


# my_heatmap_dot_X.to_csv( gdsc_c_path + 'FINAL_COLON_HEAT_X.csv')


my_heatmap_dot_X = pd.read_csv(gdsc_c_path + 'FINAL_COLON_HEAT_X.csv', index_col = 0  )
my_heatmap_dot_X.columns = ['X_'+ col for col in list(my_heatmap_dot_X.columns)]

my_heatmap_dot_M = pd.concat([my_heatmap_dot_X, my_heatmap_dot_O], axis= 1)



tmp = my_heatmap_dot_M.replace('NA', 0 )
gg = sns.clustermap(
	tmp, center=0, cmap="vlag", vmin=-70, vmax=2,
	figsize=(20,20),
	row_cluster=True, col_cluster = False, 
	metric = 'correlation', method = 'complete',
	dendrogram_ratio=0.2, yticklabels=False) 



plt.tight_layout()
plt.savefig("/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/example.COLON.pdf", bbox_inches='tight')
plt.close()




# O 기준으로만 가져오기 

dot_mini = my_heatmap_dot_M.loc[result_O_c_c_list_set]

tmp = dot_mini.replace('NA', 0 )
# tmp = dot_mini.replace(np.nan, 0 )

cmap = plt.get_cmap('RdBu_r', 11)

gg = sns.clustermap(
	tmp, center=0, cmap=cmap, #vmin=-70, vmax=2,
	figsize=(20,20),
	row_cluster=True, col_cluster = False, 
	metric = 'correlation', method = 'complete',
	dendrogram_ratio=0.2, yticklabels=False) 

plt.tight_layout()
plt.savefig("/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/example.COLON.mini.pdf", bbox_inches='tight')
plt.close()




# colon cell 이름 확인 

avail_cell_dict = {'PROSTATE': ['VCAP', 'PC3'], 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE': ['UHO1', 'K562', 'L1236', 'SR786', 'RPMI8226'], 'BREAST': ['BT549', 'BT474', 'MDAMB361', 'MDAMB231', 'UACC812', 'KPL1', 'MDAMB468', 'MDAMB175VII', 'HCC1500', 'HCC1419', 'MDAMB436', 'ZR751', 'MCF7', 'HS578T', 'T47D', 'CAMA1'], 'CENTRAL_NERVOUS_SYSTEM': ['T98G', 'SF539', 'U251MG', 'SNB75', 'SF268', 'SF295'], 'LUNG': ['A549', 'A427', 'HOP92', 'NCIH23', 'NCIH1650', 'NCIH520', 'HOP62', 'NCIH460', 'NCIH2122', 'NCIH226', 'EKVX', 'SKMES1', 'NCIH522'], 'LARGE_INTESTINE': ['SW620', 'LOVO', 'RKO', 'KM12', 'HCT15', 'HCT116', 'HT29', 'DLD1', 'SW837'], 'OVARY': ['NIHOVCAR3', 'OVCAR4', 'OVCAR5', 'IGROV1', 'OVCAR8', 'PA1', 'UWB1289', 'SKOV3', 'CAOV3', 'ES2', 'A2780', 'OV90'], 'SKIN': ['SKMEL5', 'HT144', 'RVH421', 'SKMEL28', 'A375', 'A101D', 'COLO792', 'UACC257', 'COLO800', 'MEWO', 'MELHO', 'A2058', 'RPMI7951', 'IPC298', 'MALME3M', 'SKMEL2', 'UACC62', 'LOXIMVI', 'WM115', 'G361', 'SKMEL30', 'COLO829'], 'BONE': ['A673'], 'KIDNEY': ['CAKI1', 'UO31', 'ACHN', 'A498', '786O'], 'PLEURA': ['MSTO211H']}
my_colon = [a+'_LARGE_INTESTINE' for a in avail_cell_dict['LARGE_INTESTINE']]
set(my_colon) & set(g_colon_result.ccle_name)
{'HCT116_LARGE_INTESTINE', 'HT29_LARGE_INTESTINE', 'SW837_LARGE_INTESTINE', 'RKO_LARGE_INTESTINE', 'LOVO_LARGE_INTESTINE', 'HCT15_LARGE_INTESTINE', 'SW620_LARGE_INTESTINE', 'KM12_LARGE_INTESTINE'}
filter_cell = [
	'O_HCT116_LARGE_INTESTINE', 'O_HT29_LARGE_INTESTINE', 'O_SW837_LARGE_INTESTINE', 'O_RKO_LARGE_INTESTINE', 'O_HCT15_LARGE_INTESTINE', 'O_SW620_LARGE_INTESTINE', 'O_KM12_LARGE_INTESTINE',
	'X_HCT116_LARGE_INTESTINE', 'X_HT29_LARGE_INTESTINE', 'X_SW837_LARGE_INTESTINE', 'X_RKO_LARGE_INTESTINE', 'X_LOVO_LARGE_INTESTINE', 'X_HCT15_LARGE_INTESTINE', 'X_SW620_LARGE_INTESTINE', 'X_KM12_LARGE_INTESTINE'
]



dot_mini = my_heatmap_dot_M.loc[result_O_c_c_list_set]
dot_mini = dot_mini[filter_cell]

tmp = dot_mini.replace(np.nan, 0 )

cmap = plt.get_cmap('RdBu_r', 11)

gg = sns.clustermap(
	tmp, center=0, cmap=cmap, #vmin=-70, vmax=2,
	figsize=(20,20),
	row_cluster=True, col_cluster = False, 
	metric = 'correlation', method = 'complete',
	dendrogram_ratio=0.2, yticklabels=False) 

plt.tight_layout()
plt.savefig("/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/example.COLON.mini2.pdf", bbox_inches='tight')
plt.close()
















PRJ_NAME= 'GDSC_PANCRE' # #ffcd36
g_pancreas_result = pd.read_csv(gdsc_c_path+'FINAL_{}.csv'.format(PRJ_NAME))

c_c_list = list(g_pancreas_result.cid_cid)
c_c_list_set = list(set(g_pancreas_result.cid_cid))
c_c_list_set.sort()

c_c_c_list = list(g_pancreas_result.cid_cid_cell)
c_c_c_list_set = list(set(g_pancreas_result.cid_cid_cell))
c_c_c_list_set.sort()

g_pancreas_result_O = g_pancreas_result[g_pancreas_result['Synergy?'] == 1] # 358
g_pancreas_result_X = g_pancreas_result[g_pancreas_result['Synergy?'] == 0] # 11920


result_O_c_c_c_list = list(g_pancreas_result_O.cid_cid_cell)
result_O_c_c_c_list_set = list(set(g_pancreas_result_O.cid_cid_cell))
result_O_c_c_c_list_set.sort()

result_O_c_c_list = list(g_pancreas_result_O.cid_cid)
result_O_c_c_list_set = list(set(g_pancreas_result_O.cid_cid))
result_O_c_c_list_set.sort()


result_X_c_c_c_list = list(g_pancreas_result_X.cid_cid_cell)
result_X_c_c_c_list_set = list(set(g_pancreas_result_X.cid_cid_cell))
result_X_c_c_c_list_set.sort()

result_X_c_c_list = list(g_pancreas_result_X.cid_cid)
result_X_c_c_list_set = list(set(g_pancreas_result_X.cid_cid))
result_X_c_c_list_set.sort()



my_heatmap_dot_O = pd.DataFrame(
	data = "NA", #np.inf,
	columns=list(set(g_pancreas_result_O.ccle_name)),
	index=c_c_list_set
)

for c_ind in range(len(result_O_c_c_c_list_set)) : # 대충 18만줄 대상으로 30분 소요 
	if c_ind%1000 == 0 : 
		print(str(c_ind)+'/'+str(len(result_O_c_c_c_list_set)) )
		datetime.now() 
	c_c_c = result_O_c_c_c_list_set[c_ind]
	tmp_res = g_pancreas_result_O[g_pancreas_result_O.cid_cid_cell==c_c_c]
	c_c = c_c_c.split('___')[0] + '___'+ c_c_c.split('___')[1]
	c = c_c_c.split('___')[2]
	if tmp_res.shape[0] ==1 :
		my_heatmap_dot_O.at[c_c, c] = tmp_res.PRED.item()
	elif  tmp_res.shape[0] >1 :
		my_heatmap_dot_O.at[c_c, c] = np.mean(tmp_res.PRED)
	else :
		my_heatmap_dot_O.at[c_c, c] = 0
	
# my_heatmap_dot_O.to_csv( gdsc_c_path + 'FINAL_PANCRE_HEAT_O.csv')


my_heatmap_dot_O = pd.read_csv(gdsc_c_path + 'FINAL_PANCRE_HEAT_O.csv', index_col = 0  )
my_heatmap_dot_O.columns = ['O_'+ col for col in list(my_heatmap_dot_O.columns)]



my_heatmap_dot_X = pd.DataFrame(
	data = "NA", #np.inf,
	columns=list(set(g_pancreas_result_X.ccle_name)),
	index=c_c_list_set
)

for c_ind in range(len(result_X_c_c_c_list)) : # 대충 18만줄 대상으로 30분 소요 
	if c_ind%1000 == 0 : 
		print(str(c_ind)+'/'+str(len(result_X_c_c_c_list)) )
		datetime.now() 
	c_c_c = result_X_c_c_c_list[c_ind]
	tmp_res = g_pancreas_result_X[g_pancreas_result_X.cid_cid_cell==c_c_c]
	c_c = c_c_c.split('___')[0] + '___'+ c_c_c.split('___')[1]
	c = c_c_c.split('___')[2]
	if tmp_res.shape[0] ==1 :
		my_heatmap_dot_X.at[c_c, c] = tmp_res.PRED.item()
	elif  tmp_res.shape[0] >1 :
		my_heatmap_dot_X.at[c_c, c] = np.mean(tmp_res.PRED)
	else :
		my_heatmap_dot_X.at[c_c, c] = 0


# my_heatmap_dot_X.to_csv( gdsc_c_path + 'FINAL_PANCRE_HEAT_X.csv')


my_heatmap_dot_X = pd.read_csv(gdsc_c_path + 'FINAL_PANCRE_HEAT_X.csv', index_col = 0  )
my_heatmap_dot_X.columns = ['X_'+ col for col in list(my_heatmap_dot_X.columns)]

my_heatmap_dot_M = pd.concat([my_heatmap_dot_X, my_heatmap_dot_O], axis= 1)



tmp = my_heatmap_dot_M.replace('NA', 0 )
gg = sns.clustermap(
	tmp, center=0, cmap="vlag", #vmin=-40, vmax=,
	figsize=(20,20),
	row_cluster=True, col_cluster = False, 
	metric = 'correlation', method = 'complete',
	dendrogram_ratio=0.2, yticklabels=False) 



plt.tight_layout()
plt.savefig("/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/example.PANCRE.pdf", bbox_inches='tight')
plt.close()




# O 기준으로만 가져오기 

dot_mini = my_heatmap_dot_M.loc[result_O_c_c_list_set]

tmp = dot_mini.replace('NA', 0 )
# tmp = dot_mini.replace(np.nan, 0 )

cmap = plt.get_cmap('RdBu_r', 11)

gg = sns.clustermap(
	tmp, center=0, cmap=cmap, #vmin=-70, vmax=2,
	figsize=(20,20),
	row_cluster=True, col_cluster = False, 
	metric = 'correlation', method = 'complete',
	dendrogram_ratio=0.2, yticklabels=False) 

plt.tight_layout()
plt.savefig("/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/example.COLON.mini.pdf", bbox_inches='tight')
plt.close()









































아니면......
그냥 violin plot 으로 나타내야 하나 




g_colon_result_O[g_colon_result_O.type=='AOBO']['PRED'].describe()
g_colon_result_O[g_colon_result_O.type=='AXBO']['PRED'].describe()
g_colon_result_O[g_colon_result_O.type=='AXBX']['PRED'].describe()


g_colon_result_X[g_colon_result_X.type=='AOBO']['PRED'].describe()
g_colon_result_X[g_colon_result_X.type=='AXBO']['PRED'].describe()
g_colon_result_X[g_colon_result_X.type=='AXBX']['PRED'].describe()


g_breast_result_O[g_breast_result_O.type=='AOBO']['PRED'].describe()
g_breast_result_O[g_breast_result_O.type=='AXBO']['PRED'].describe()
g_breast_result_O[g_breast_result_O.type=='AXBX']['PRED'].describe()

g_breast_result_X[g_breast_result_X.type=='AOBO']['PRED'].describe()
g_breast_result_X[g_breast_result_X.type=='AXBO']['PRED'].describe()
g_breast_result_X[g_breast_result_X.type=='AXBX']['PRED'].describe()


g_pancreas_result_O[g_pancreas_result_O.type=='AOBO']['PRED'].describe()
g_pancreas_result_O[g_pancreas_result_O.type=='AXBO']['PRED'].describe()
g_pancreas_result_O[g_pancreas_result_O.type=='AXBX']['PRED'].describe()

g_pancreas_result_X[g_pancreas_result_X.type=='AOBO']['PRED'].describe()
g_pancreas_result_X[g_pancreas_result_X.type=='AXBO']['PRED'].describe()
g_pancreas_result_X[g_pancreas_result_X.type=='AXBX']['PRED'].describe()









# 해보자 violin plot 

# violin plot for tissue 
from matplotlib import colors as mcolors

g_colon_result
g_breast_result





SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V8_349_FULL/'

file_name = 'M3V8_349_MISS2_FULL' # 0608
A_B_C_S_SET_ADD = pd.read_csv(SAVE_PATH+'{}.A_B_C_S_SET_ADD.csv'.format(file_name), low_memory=False)
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
A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.type.isin(MISS_filter)]

CELL_92 = ['VCAP_PROSTATE', 'NIHOVCAR3_OVARY', 'SW620_LARGE_INTESTINE', 'OVCAR4_OVARY', 'BT549_BREAST', 'A549_LUNG', 'SKMEL5_SKIN', 'A427_LUNG', 'BT474_BREAST', 'HOP92_LUNG', 'T98G_CENTRAL_NERVOUS_SYSTEM', 'NCIH23_LUNG', 'HT144_SKIN', 'RVH421_SKIN', 'MDAMB361_BREAST', 'LOVO_LARGE_INTESTINE', 'MDAMB231_BREAST', 'SKMEL28_SKIN', 'NCIH1650_LUNG', 'RKO_LARGE_INTESTINE', 'OVCAR5_OVARY', 'UACC812_BREAST', 'KPL1_BREAST', 'MSTO211H_PLEURA', 'KM12_LARGE_INTESTINE', 'IGROV1_OVARY', 'UHO1_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'NCIH520_LUNG', 'OVCAR8_OVARY', 'HCT15_LARGE_INTESTINE', 'A375_SKIN', 'CAKI1_KIDNEY', 'MDAMB468_BREAST', 'K562_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'A101D_SKIN', 'PA1_OVARY', 'UO31_KIDNEY', 'HOP62_LUNG', 'SF539_CENTRAL_NERVOUS_SYSTEM', 'MDAMB175VII_BREAST', 'U251MG_CENTRAL_NERVOUS_SYSTEM', 'HCC1500_BREAST', 'L1236_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'HCC1419_BREAST', 'NCIH460_LUNG', 'NCIH2122_LUNG', 'COLO792_SKIN', 'SR786_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'UACC257_SKIN', 'SNB75_CENTRAL_NERVOUS_SYSTEM', 'HCT116_LARGE_INTESTINE', 'PC3_PROSTATE', 'NCIH226_LUNG', 'RPMI8226_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'EKVX_LUNG', 'COLO800_SKIN', 'HT29_LARGE_INTESTINE', 'UWB1289_OVARY', 'MDAMB436_BREAST', 'SKOV3_OVARY', 'ZR751_BREAST', 'MEWO_SKIN', 'MELHO_SKIN', 'A2058_SKIN', 'RPMI7951_SKIN', 'SF268_CENTRAL_NERVOUS_SYSTEM', 'ACHN_KIDNEY', 'IPC298_SKIN', 'MALME3M_SKIN', 'A673_BONE', 'SF295_CENTRAL_NERVOUS_SYSTEM', 'CAOV3_OVARY', 'A498_KIDNEY', 'SKMEL2_SKIN', 'UACC62_SKIN', 'ES2_OVARY', 'LOXIMVI_SKIN', '786O_KIDNEY', 'MCF7_BREAST', 'WM115_SKIN', 'A2780_OVARY', 'DLD1_LARGE_INTESTINE', 'HS578T_BREAST', 'SKMES1_LUNG', 'T47D_BREAST', 'OV90_OVARY', 'G361_SKIN', 'SKMEL30_SKIN', 'COLO829_SKIN', 'SW837_LARGE_INTESTINE', 'NCIH522_LUNG', 'CAMA1_BREAST']
A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.CELL.isin(CELL_92)]


DC_CELL_DF2 = pd.read_csv(DC_PATH+'DC_CELL_INFO.csv', sep = '\t')
DC_CELL_DF2 = pd.concat([
	DC_CELL_DF2, 
	pd.DataFrame({'cell_line_id' : [1],'DC_cellname' : ['786O'],'DrugCombCello' : ['CVCL_1051'],'DrugCombCCLE':['786O_KIDNEY']})])

DC_CELL_info_filt = DC_CELL_DF2[DC_CELL_DF2.DrugCombCCLE.isin(A_B_C_S_SET.CELL)] # 38

DC_CELL_info_filt = DC_CELL_info_filt.drop(['Unnamed: 0'], axis = 1)
DC_CELL_info_filt.columns = ['cell_line_id', 'DC_cellname', 'DrugCombCello', 'CELL']
DC_CELL_info_filt = DC_CELL_info_filt[['CELL','DC_cellname']]

A_B_C_S_SET_COH = pd.merge(A_B_C_S_SET, DC_CELL_info_filt, on = 'CELL', how = 'left'  )





known = list(A_B_C_S_SET_COH.cid_cid_cell)


g_colon_result = g_colon9
g_breast_result = g_breast9
g_pancreas_result = g_pancreas9



g_colon_result_filt = g_colon_result[g_colon_result.cid_cid_cell.isin(known)==False] # 18416
g_colon_result_filt = g_colon_result_filt[['PRED','Synergy?']]
g_colon_result_filt['Tissue'] = 'LARGE_INTESTINE'
g_colon_result_filt['CLASS'] = g_colon_result_filt['Synergy?'].apply(lambda x : 'O' if x==1 else 'X')
g_colon_result_filt['CT'] = g_colon_result_filt['Tissue'] + '_' + g_colon_result_filt['CLASS']

g_breast_result_filt = g_breast_result[g_breast_result.cid_cid_cell.isin(known)==False] # 40947
g_breast_result_filt = g_breast_result_filt[['PRED','Synergy?']]
g_breast_result_filt['Tissue'] = 'BREAST'
g_breast_result_filt['CLASS'] = g_breast_result_filt['Synergy?'].apply(lambda x : 'O' if x==1 else 'X')
g_breast_result_filt['CT'] = g_breast_result_filt['Tissue'] + '_' + g_breast_result_filt['CLASS']

g_pancreas_result_filt = g_pancreas_result[g_pancreas_result.cid_cid_cell.isin(known)==False] # 12278
g_pancreas_result_filt = g_pancreas_result_filt[['PRED','Synergy?']]
g_pancreas_result_filt['Tissue'] = 'PANCREAS'
g_pancreas_result_filt['CLASS'] = g_pancreas_result_filt['Synergy?'].apply(lambda x : 'O' if x==1 else 'X')
g_pancreas_result_filt['CT'] = g_pancreas_result_filt['Tissue'] + '_' + g_pancreas_result_filt['CLASS']



g_colon_result2 = g_colon_result.groupby('cid_cid_cell').mean()
g_breast_result2 = g_breast_result.groupby('cid_cid_cell').mean()
g_pancreas_result2 = g_pancreas_result.groupby('cid_cid_cell').mean()

g_colon_result_filt = g_colon_result2[['PRED','Synergy?']]
g_colon_result_filt['Tissue'] = 'LARGE_INTESTINE'
g_colon_result_filt['CLASS'] = g_colon_result_filt['Synergy?'].apply(lambda x : 'O' if x==1 else 'X')
g_colon_result_filt['CT'] = g_colon_result_filt['Tissue'] + '_' + g_colon_result_filt['CLASS']


g_breast_result_filt = g_breast_result2[['PRED','Synergy?']]
g_breast_result_filt['Tissue'] = 'BREAST'
g_breast_result_filt['CLASS'] = g_breast_result_filt['Synergy?'].apply(lambda x : 'O' if x==1 else 'X')
g_breast_result_filt['CT'] = g_breast_result_filt['Tissue'] + '_' + g_breast_result_filt['CLASS']

g_pancreas_result_filt = g_pancreas_result2[['PRED','Synergy?']]
g_pancreas_result_filt['Tissue'] = 'PANCREAS'
g_pancreas_result_filt['CLASS'] = g_pancreas_result_filt['Synergy?'].apply(lambda x : 'O' if x==1 else 'X')
g_pancreas_result_filt['CT'] = g_pancreas_result_filt['Tissue'] + '_' + g_pancreas_result_filt['CLASS']



gdsc_all = pd.concat([g_colon_result_filt, g_breast_result_filt, g_pancreas_result_filt])

# my_order = test_cell_df.groupby(by=["tissue"])["P_COR"].mean().sort_values().iloc[::-1].index

# order=my_order,
fig, ax = plt.subplots(figsize=(25, 15)) 
sns.violinplot(ax = ax, data  = gdsc_all, x = 'CT', y = 'PRED', linewidth=1,  edgecolor="dimgrey",  inner = 'point') # width = 3,
violins = [c for i,c in enumerate(ax.collections) if i%2 == 0] # pollycollection 가져오기 위함 


# 하나밖에 없는애들은 violin 안그려져서 문제 
violins[0].set_facecolor(mcolors.to_rgba(color_dict['LARGE_INTESTINE'], 0.7))
violins[1].set_facecolor(mcolors.to_rgba(color_dict['PROSTATE'], 0.7))
violins[2].set_facecolor(mcolors.to_rgba(color_dict['OVARY'], 0.7))
violins[3].set_facecolor(mcolors.to_rgba(color_dict['LUNG'], 0.7))
violins[4].set_facecolor(mcolors.to_rgba(color_dict['SKIN'], 0.7))
violins[5].set_facecolor(mcolors.to_rgba(color_dict['KIDNEY'], 0.7))


ax.set_xlabel('tissue, OX', fontsize=10)
ax.set_ylabel('Predicted value', fontsize=10)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
ax.tick_params(axis='both', which='major', labelsize=10 )
plt.tight_layout()
plt.grid(True)
plt.savefig(os.path.join('/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/gdsc.violin.png'), dpi = 300)

plt.close()


# 아...생각보다 좀 구림. bar plot 으로 나타낼까 
$ pip install statannot
from statannot import add_stat_annotation


# order=my_order,
fig, ax = plt.subplots(figsize=(13, 10)) 
sns.boxplot(ax = ax , data = gdsc_all, x = 'CT', y = 'PRED', width = 0.3, linewidth=1, hue = 'Tissue', palette = {'LARGE_INTESTINE' : '#ffcd36' , 'BREAST' : '#025669', 'PANCREAS' : '#ff009bff'} ) # width = 3,
ax, test_results = add_stat_annotation(ax = ax, data=gdsc_all, x='CT', y='PRED',
                                   box_pairs=[("LARGE_INTESTINE_O", "LARGE_INTESTINE_X"),  ("BREAST_O", "BREAST_X"), ("PANCREAS_O", "PANCREAS_X")],
                                   test='t-test_ind', text_format='star', loc='outside', verbose=2)
# test='Mann-Whitney'

sns.despine()
ax.set_xlabel('tissue, OX', fontsize=10)
ax.set_ylabel('Predicted value', fontsize=10)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
ax.tick_params(axis='both', which='major', labelsize=10 )
plt.legend(bbox_to_anchor=(1.02, 0.15), loc='upper left', borderaxespad=0)
#plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join('/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/gdsc.0921.box3.png'), dpi = 300)
plt.savefig('/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/gdsc.0921.box3.pdf', format="pdf", bbox_inches = 'tight')

plt.close()






























원래 - 가 많은지?
ABCS_test_result = pd.read_csv('/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V6_W203_349_MIS2/ABCS_test_result.csv')

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




PRJ_NAME= 'GDSC_PANCRE' # #ff009bff
g_pancreas_result = pd.read_csv(gdsc_c_path+'FINAL_{}.csv'.format(PRJ_NAME))










tiss_cell_dict ={'BREAST' : '#025669', 'LARGE_INTESTINE' : '#ffcd36', 'PANCREAS' : '#ff009bff', }



					tissue_map = pd.DataFrame(my_heatmap_dot.columns, columns= ['cell'])
					tissue_map['tissue'] = ['_'.join(a.split('_')[1:]) for a in tissue_map.cell]
					tissue_map['col'] = tissue_map['tissue'].map(color_dict)

syn_OX_map = pd.DataFrame(my_heatmap_dot.columns, columns = 'answer')


col_colors = list(tissue_map['col'])




# row color 1) 
tanimoto_map = pd.DataFrame(my_heatmap_dot.index, columns = ['cid_cid'])
tani_tmp = ABCS_test_concat[['cid_cid_cell','tani_Q']]
tani_tmp['cid_cid'] = tani_tmp.cid_cid_cell.apply(lambda x : '___'.join(x.split('___')[0:2]))
tani_tmp = tani_tmp[['cid_cid','tani_Q']].drop_duplicates()
tanimoto_map2 = pd.merge(tanimoto_map, tani_tmp, on = 'cid_cid', how = 'left' )
tanimoto_map2['col'] = ['#0DD9FE' if a == 'O' else '#383b3c' for a in list(tanimoto_map2.tani_Q)]

row_colors = list(tanimoto_map2['col'])


value_to_int = {j:i for i,j in enumerate(pd.unique(my_heatmap_dot.values.ravel()))} # like you did
n = len(value_to_int)     
cmap = [mcolors.to_rgba('#f5f8fb'),mcolors.to_rgba('#ffe28a'),mcolors.to_rgba("#ffc1cb"),mcolors.to_rgba("#ff647e")]


gg = sns.clustermap(
	my_heatmap_dot.replace(value_to_int),  cmap=cmap, 
	figsize=(20,20),
	row_cluster=True, col_cluster = True, 
	metric = 'correlation', method = 'complete',
	col_colors = col_colors, row_colors = row_colors, dendrogram_ratio=0.2, yticklabels=False) 


plt.tight_layout()
plt.savefig("/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/example.W124.pdf", bbox_inches='tight')
plt.close()




############33 other reference 

import pandas as pd
import seaborn as sns
sns.set()

# Load the brain networks example dataset
df = sns.load_dataset("brain_networks", header=[0, 1, 2], index_col=0)

# Select a subset of the networks
used_networks = [1, 5, 6, 7, 8, 12, 13, 17]
used_columns = (df.columns.get_level_values("network")
                          .astype(int)
                          .isin(used_networks))
df = df.loc[:, used_columns]

# Create a categorical palette to identify the networks
network_pal = sns.husl_palette(8, s=.45)
network_lut = dict(zip(map(str, used_networks), network_pal))

# Convert the palette to vectors that will be drawn on the side of the matrix
networks = df.columns.get_level_values("network")
network_colors = pd.Series(networks, index=df.columns).map(network_lut)

# Draw the full plot
sns.clustermap(df.corr(), center=0, cmap="vlag",
               row_colors=network_colors, col_colors=network_colors,
               linewidths=.75, figsize=(13, 13))




annotation : text format simple 로 바꿀 수는 있을듯 





예를 들면 AZD4547 같은 애들 

지금 내가 드는 생각 
-> 여기서 우리 train 에 없는 애들 대상으로 예측을 하고 
synergy 0 /1 로 왼쪽에 나타내고 
cell line 별로 나타낸 다음에 보면 
잘맞춘 경우가 위에 좌라락... 이게 내 목표인디 
validation set 에서 그렇게 보여주는게 어떤 의미일까 
그중에 잘 못맞추는 약물의 경우에는 뭐가 문제라고 나타낼 수 있을까? 

맞춘 결과를 그래서 row : AOBO / AXBO / AXBX 나 혹은 tanimoto 등으로 나눠서 나타내면 
네 구역으로 나타낼 수 있을듯? 


흠...
breast 가 그래도 먼저 끝날 것 같음 







##################

다시.
그래서 

NS_raw = pd.read_csv('/st06/jiyeonH/13.DD_SESS/GDSC^2/Original_screen_All_tissues_fitted_data.csv', low_memory=False)

tmp1 = NS_raw[NS_raw.COMBI_ID=='1190:1022']
tmp2 = NS_raw[NS_raw.COMBI_ID=='1022:1190']

tmp11 = tmp1[tmp1.CELL_LINE_NAME == 'UACC-893']
tmp22 = tmp2[tmp2.CELL_LINE_NAME == 'UACC-893']

tmptmp = gdsc_breast[gdsc_breast['Cell Line name']=='UACC-893']
tmptmp1 = tmptmp[tmptmp['Anchor Name']=='Gemcitabine'] # Gemcitabine
tmptmp11 = tmptmp1[tmptmp1['Library Name']=='AZD7762']

'Delta Xmid'
'Delta Emax'


tmptmp = gdsc_breast[gdsc_breast['Cell Line name']=='UACC-893']
tmptmp2 = tmptmp[tmptmp['Anchor Name']=='AZD7762'] # Gemcitabine
tmptmp22 = tmptmp2[tmptmp2['Library Name']=='Gemcitabine']



A_L_C = NS_raw[['ANCHOR_NAME','LIBRARY_NAME', 'CELL_LINE_NAME']].drop_duplicates()
A_L_C = A_L_C.reset_index(drop = True)



####################
re_re 


gdsc_breast = pd.read_csv(gdsc_c_path + 'breast_anchor_combo.csv') # 163470
gdsc_chem_list_re.columns = ['Anchor Name','Anchor CID']
gdsc_breast2 = pd.merge(gdsc_breast, gdsc_chem_list_re, on ='Anchor Name', how ='left')

gdsc_chem_list_re.columns = ['Library Name','Library CID']
gdsc_breast3 = pd.merge(gdsc_breast2, gdsc_chem_list_re, on ='Library Name', how ='left')
gdsc_breast4 = pd.merge(gdsc_breast3, gdsc_cell_ccle2, on = 'Cell Line name', how = 'left')

cid_a = list(gdsc_breast4['Anchor CID'])
cid_b = list(gdsc_breast4['Library CID'])
cell = list(gdsc_breast4['ccle_name'])

gdsc_breast4['cid_cid_cell'] = [str(int(cid_a[i])) + '___' + str(int(cid_b[i]))+ '___' + cell[i] if cid_a[i] < cid_b[i] else str(int(cid_b[i])) + '___' + str(int(cid_a[i]))+ '___' + cell[i] for i in range(gdsc_breast4.shape[0])]
gdsc_breast4['CID_A_CELL'] = gdsc_breast4['Anchor CID'].apply(lambda a : str(a)) + '__' + gdsc_breast4['ccle_name']
gdsc_breast4['CID_B_CELL'] = gdsc_breast4['Library CID'].apply(lambda b : str(b)) + '__' + gdsc_breast4['ccle_name']
gdsc_breast4['cid_cid'] = [str(int(cid_a[i])) + '___' + str(int(cid_b[i])) if cid_a[i] < cid_b[i] else str(int(cid_b[i])) + '___' + str(int(cid_a[i])) for i in range(gdsc_breast4.shape[0])]

len(gdsc_breast4.cid_cid_cell) # 63800
len(set(gdsc_breast4.cid_cid_cell)) # 48704
gg_tmp = gdsc_breast4[['cid_cid_cell','Synergy?']].drop_duplicates() # 49702 갈리는 애가 있네 

dup_ccc = list(gg_tmp[gg_tmp.cid_cid_cell.duplicated()]['cid_cid_cell'])# synergy 갈리는 애들은 빼버림 

g_breast5 = g_breast5[g_breast5.cid_cid_cell.isin(dup_ccc) == False] # 61804 , ccc: 47706



g_breast_result = pd.read_csv(gdsc_c_path+'FINAL_{}.csv'.format(PRJ_NAME))
g_breast8 = 








tmp1 = NS_raw[NS_raw.COMBI_ID=='1017:1594']
tmp2 = NS_raw[NS_raw.COMBI_ID=='1594:1017']

tmp11 = tmp1[tmp1.CELL_LINE_NAME == 'CAL-85-1']
tmp22 = tmp2[tmp2.CELL_LINE_NAME == 'CAL-85-1']






###################################


synergy_rank = pd.read_excel('/st06/jiyeonH/11.TOX/DR_SPRING/val_data/41586_2022_4437_MOESM7_ESM.xlsx')

synergy_rank_breast = synergy_rank[synergy_rank.Tissue == 'Breast']
synergy_rank_colon = synergy_rank[synergy_rank.Tissue == 'Colon']
synergy_rank_pancreas = synergy_rank[synergy_rank.Tissue == 'Pancreas']


rank_breast = synergy_rank_breast[['ANCHOR NAME', 'LIBRARY NAME','Synergy Rank', 'Synergy (%)']].drop_duplicates()
rank_breast.columns = ['Anchor Name','Library Name','Synergy Rank','Synergy (%)']

g_breast9_filt = pd.merge(g_breast9, rank_breast, on = ['Anchor Name', 'Library Name'], how ='left')
g_breast9_filt2 = g_breast9_filt[g_breast9_filt.PRED.apply(lambda x : np.isnan(x) ==False)]

fig, ax = plt.subplots(1 , 1 ,figsize=(12, 12))

g = sns.scatterplot(ax = ax, data=g_breast9_filt2, x="PRED", y="Synergy (%)", hue="strip_name", size = 'type', alpha = 0.5) # , palette=color_dict

plt.tight_layout()
plt.savefig("/st06/jiyeonH/11.TOX/DR_SPRING/val_data/breast_rank.pdf", bbox_inches='tight')




g_colon9 


rank_colon = synergy_rank_colon[['ANCHOR NAME', 'LIBRARY NAME','Synergy Rank', 'Synergy (%)']].drop_duplicates()
rank_colon.columns = ['Anchor Name','Library Name','Synergy Rank','Synergy (%)']

g_colon9_filt = pd.merge(g_colon9, rank_colon, on = ['Anchor Name', 'Library Name'], how ='left')
g_colon9_filt2 = g_colon9_filt[g_colon9_filt.PRED.apply(lambda x : np.isnan(x) ==False)]

fig, ax = plt.subplots(1 , 1 ,figsize=(12, 12))

g = sns.scatterplot(ax = ax, data=g_colon9_filt2, x="PRED", y="Synergy (%)", hue="strip_name", size = 'type', alpha = 0.5) # , palette=color_dict

plt.tight_layout()
plt.savefig("/st06/jiyeonH/11.TOX/DR_SPRING/val_data/colon_rank.pdf", bbox_inches='tight')






g_pancreas9



rank_panc = synergy_rank_pancreas[['ANCHOR NAME', 'LIBRARY NAME','Synergy Rank', 'Synergy (%)']].drop_duplicates()
rank_panc.columns = ['Anchor Name','Library Name','Synergy Rank','Synergy (%)']

g_pancreas9_filt = pd.merge(g_pancreas9, rank_panc, on = ['Anchor Name', 'Library Name'], how ='left')
g_pancreas9_filt2 = g_pancreas9_filt[g_pancreas9_filt.PRED.apply(lambda x : np.isnan(x) ==False)]

fig, ax = plt.subplots(1 , 1 ,figsize=(12, 12))

g = sns.scatterplot(ax = ax, data=g_pancreas9_filt2, x="PRED", y="Synergy (%)", hue="strip_name", size = 'type', alpha = 0.5) # , palette=color_dict

plt.tight_layout()
plt.savefig("/st06/jiyeonH/11.TOX/DR_SPRING/val_data/pancreas_rank.pdf", bbox_inches='tight')




#######################
#######################
check new pred

new_pred_b_BT549 = g_breast3[g_breast3['Cell Line name'] == 'BT-549']


new_pred_b_BT549[new_pred_b_BT549['Anchor CID']==11707110]
new_pred_b_BT549[new_pred_b_BT549['Library CID']==11707110]

new_pred_b_BT549[new_pred_b_BT549['Anchor CID']==135449332]
new_pred_b_BT549[new_pred_b_BT549['Library CID']==135449332]



new_pred_b_BT549[(new_pred_b_BT549['Anchor CID']==11707110) & (new_pred_b_BT549['Library CID']==5330286)]
new_pred_b_BT549[(new_pred_b_BT549['Anchor CID']==5330286) & (new_pred_b_BT549['Library CID']==11707110)]


new_pred_b_BT549[(new_pred_b_BT549['Anchor CID']==11707110) & (new_pred_b_BT549['Library CID']==5330286)]
new_pred_b_BT549[(new_pred_b_BT549['Anchor CID']==5330286) & (new_pred_b_BT549['Library CID']==11707110)]
