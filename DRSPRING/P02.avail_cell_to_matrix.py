


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



# LOAD Cells 

MJ_NAME = 'M3'
MISS_NAME = 'MIS2'

SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/{}_FULL/'.format(MJ_NAME)
file_name = 'M3_MISS2_FULL'

A_B_C_S_SET_ADD = pd.read_csv(SAVE_PATH+'{}.A_B_C_S_SET_ADD.csv'.format(file_name), low_memory=False)

avail_cell_list = list(set(A_B_C_S_SET_ADD.DrugCombCello))






# NETWORK ORDER
hunet_dir = '/st06/jiyeonH/13.DD_SESS/HumanNetV3/'

hunet_gsp = pd.read_csv(hunet_dir+'HumanNet-GSP.tsv', sep = '\t', header = None)
hunet_gsp.columns = ['G_A','G_B']

BETA_GENE = pd.read_table('/st06/jiyeonH/11.TOX/LINCS/L_2020/geneinfo_beta.txt')
BETA_lm_genes = BETA_GENE[BETA_GENE.feature_space=='landmark'] # 978
BETA_lm_genes = BETA_lm_genes.reset_index()
lm_entrezs = list(BETA_lm_genes.gene_id)

hnet_L1 = hunet_gsp[hunet_gsp['G_A'].isin(BETA_lm_genes.gene_id)]
hnet_L2 = hnet_L1[hnet_L1['G_B'].isin(BETA_lm_genes.gene_id)] # 3885

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


# 유전자 이름으로 붙이기 
LINCS_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/' 
LINCS_gene_file = pd.read_csv(LINCS_PATH+'geneinfo_beta.txt', sep = '\t')
LINCS_978 = LINCS_gene_file[LINCS_gene_file.feature_space == 'landmark']
LINCS_978 = LINCS_978[['gene_id','gene_symbol']]
LINCS_978['new_node'] = [str(list(LINCS_978.gene_id)[i]) + "__" + list(LINCS_978.gene_symbol)[i] for i in range(978)]
LINCS_978 = LINCS_978.reset_index(drop=True)


new_node_names = []
for a in ID_G.nodes():
	tmp_name = LINCS_978[LINCS_978.gene_id == a ]['gene_symbol'].item() # 6118
	new_node_name = str(a) + '__' + tmp_name
	new_node_names = new_node_names + [new_node_name]

mapping = {list(ID_G.nodes())[a]:new_node_names[a] for a in range(len(new_node_names))}

ID_G_RE = nx.relabel_nodes(ID_G, mapping)

MY_G = ID_G_RE 
MY_WEIGHT_SCORE = ID_WEIGHT_SCORE # SCORE


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







CCLE_PATH = '/st06/jiyeonH/13.DD_SESS/CCLE.22Q1/'


ccle_exp = pd.read_csv(CCLE_PATH+'CCLE_expression.csv', low_memory=False)
ccle_info= pd.read_csv(CCLE_PATH+'sample_info.csv', low_memory=False)


ori_col = list( ccle_exp.columns ) # entrez!
for_gene = ori_col[1:]
for_gene2 = [int(a.split('(')[1].split(')')[0]) for a in for_gene]
new_col = ['DepMap_ID']+for_gene2 
ccle_exp.columns = new_col


ccle_cell_info = ccle_info[['DepMap_ID','RRID']]
ccle_cell_info.columns = ['DepMap_ID','DrugCombCello']
ccle_exp2 = pd.merge(ccle_exp, ccle_cell_info, on = 'DepMap_ID' , how='left')
ccle_exp3 = ccle_exp2[['DepMap_ID','DrugCombCello']+BETA_ENTREZ_ORDER]
ccle_cello_names = [a for a in ccle_exp3.DrugCombCello if type(a) == str]





# DC Cell info 
with open(DC_PATH+'cell_lines.json') as json_file :
	DC_CELL =json.load(json_file)

DC_CELL_K = list(DC_CELL[0].keys())
DC_CELL_DF = pd.DataFrame(columns=DC_CELL_K)

for DD in range(1,len(DC_CELL)):
	tmpdf = pd.DataFrame({k:[DC_CELL[DD][k]] for k in DC_CELL_K})
	DC_CELL_DF = pd.concat([DC_CELL_DF, tmpdf], axis = 0)

DC_CELL_DF2 = DC_CELL_DF[['id','name','cellosaurus_accession', 'ccle_name']] # 2319
DC_CELL_DF2.columns = ['cell_line_id', 'DC_cellname','DrugCombCello', 'DrugCombCCLE']



DC_CELL_DF_ids = set(DC_CELL_DF.cellosaurus_accession) # 1659
ccle_cell_ids = set(ccle_cell_info.DrugCombCello) # 1672
# DC_CELL_DF_ids - ccle_cell_ids = 205
# ccle_cell_ids - DC_CELL_DF_ids = 218





DC_CELL_DF3 = DC_CELL_DF2[DC_CELL_DF2.DrugCombCello.isin(avail_cell_list)]
DC_CELL_DF3 = DC_CELL_DF3.reset_index(drop = True)

cell_df = ccle_exp3[ccle_exp3.DrugCombCello.isin(avail_cell_list)]

# ccle supporting cvcl : len(set(cell_df.DrugCombCello)) 38 차이가 있음. 4개 는 전부 0 으로 들어간거 


cell_basal_exp_list = []
# give vector 
for i in range(DC_CELL_DF3.shape[0]) :
    if i%100 == 0 :
        print(str(i)+'/'+str(DC_CELL_DF3.shape[0]) )
        datetime.now()
    cello = DC_CELL_DF3['DrugCombCello'][i]
    if cello in ccle_cello_names : 
        ccle_exp_df = cell_df[cell_df.DrugCombCello==cello][BETA_ENTREZ_ORDER]
        ccle_exp_vector = ccle_exp_df.values[0].tolist()
        cell_basal_exp_list.append(ccle_exp_vector)
    else : # 'CVCL_A442', 'CVCL_0395' # remove?
        ccle_exp_vector = [0]*978
        cell_basal_exp_list.append(ccle_exp_vector)

cell_base_tensor = torch.Tensor(cell_basal_exp_list)

# 
SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/FINAL_VALIDATION/'
torch.save(cell_base_tensor, SAVE_PATH+'AVAIL_CLL_MY_CellBase.pt')

DC_CELL_DF3.to_csv(SAVE_PATH + 'AVAIL_CELL_DF.csv')





# {'CVCL_A442', 'CVCL_0105', 'CVCL_0395', 'CVCL_7151'}

TMD8
DU145
LNCaP
TC-32


















# 다시 써먹을수는 있을듯 
MJ_DIR = '/st06/jiyeonH/13.DD_SESS/01.PRJ2/'

# MJ_request_ANS_M3 = pd.read_csv(MJ_DIR+'PRJ2_EXP_fugcn_vers1.csv')
# MJ_request_ANS_M33 = pd.read_csv(MJ_DIR+'PRJ2_EXP_fugcn_versa2_1.csv')

set(MJ_request_ANS_M3.columns) - set(MJ_request_ANS_M33.columns) 
set(MJ_request_ANS_M33.columns) -set(MJ_request_ANS_M3.columns)
















