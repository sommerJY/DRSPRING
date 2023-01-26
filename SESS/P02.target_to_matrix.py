
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



DC_PATH = '/st06/jiyeonH/13.DD_SESS/DrugComb.1.5/' 
LINCS_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/' 
TARGET_PATH = '/st06/jiyeonH/13.DD_SESS/01.PRJ2/'
PC_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/'



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










# DC drug but no combination data 
# -> ranking please 


# Drug Comb 데이터 가져오기 
# synergy info 
DC_tmp_DF1 = pd.read_csv(DC_PATH+'TOTAL_BLOCK.csv', low_memory=False) # 1469182
DC_tmp_DF1_re = DC_tmp_DF1[['drug_row_id','drug_col_id','cell_line_id','synergy_loewe','quality']]
DC_tmp_DF1_re['drug_row_id_re'] = [float(a) for a in list(DC_tmp_DF1_re['drug_row_id'])]
DC_tmp_DF1_re['drug_col_id_re'] = [float(a) for a in list(DC_tmp_DF1_re['drug_col_id'])]

# drug len check : drug_ids = list(set(list(DC_tmp_DF1_re.drug_row_id_re) + list(DC_tmp_DF1_re.drug_col_id_re))) # 15811
# drug len check : drug_ids = [a for a in drug_ids if a>0] $ 8368
# cell len check : cell_ids = list(set(DC_tmp_DF1.cell_line_id)) # 2040

DC_tmp_DF2 = DC_tmp_DF1_re[DC_tmp_DF1_re['quality'] != 'bad'] # 1457561
# drug len check : drug_ids = list(set(list(DC_tmp_DF2.drug_row_id_re) + list(DC_tmp_DF2.drug_row_id_re))) # 8333
# cell len check : cell_ids = list(set(DC_tmp_DF2.cell_line_id)) # 2040


DC_tmp_DF3 = DC_tmp_DF2[(DC_tmp_DF2.drug_row_id_re > 0 ) & (DC_tmp_DF2.drug_col_id_re > 0 )] # 740932
DC_tmp_DF4 = DC_tmp_DF3[DC_tmp_DF3.cell_line_id>0].drop_duplicates() # 740884

# DC_tmp_DF4[['drug_row_id_re','drug_col_id_re', 'cell_line_id', 'synergy_loewe']].drop_duplicates() # 740884
# DC_tmp_DF4[['drug_row_id_re','drug_col_id_re', 'cell_line_id']].drop_duplicates() # 648516



# Drug info 
with open(DC_PATH+'drugs.json') as json_file :
	DC_DRUG =json.load(json_file)

DC_DRUG_K = list(DC_DRUG[0].keys())
DC_DRUG_DF = pd.DataFrame(columns=DC_DRUG_K)

for DD in range(1,len(DC_DRUG)):
	tmpdf = pd.DataFrame({k:[DC_DRUG[DD][k]] for k in DC_DRUG_K})
	DC_DRUG_DF = pd.concat([DC_DRUG_DF, tmpdf], axis = 0)

DC_DRUG_DF['id_re'] = [float(a) for a in list(DC_DRUG_DF['id'])]

DC_DRUG_DF_FULL = pd.read_csv('/st06/jiyeonH/11.TOX/MY_TRIAL_6/DC_DRUG_DF_PC.csv', sep ='\t')
# pubchem canonical smiles 쓰고 length 값 들어간 버전 


DC_lengs = list(DC_DRUG_DF_FULL.leng)
DC_lengs2 = [int(a) for a in DC_lengs if a!= 'error']


#plot_dir = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/'
#plt.hist(DC_lengs2 , range = (0, 295), bins = 295)
#plt.legend()
#plt.savefig(plot_dir+"DrugComb_CIDs_LENG.png", format = 'png', dpi=300)



DC_TOT_CIDS = DC_DRUG_DF_FULL[['CID','leng']]
DC_TOT_CIDS = DC_TOT_CIDS[DC_TOT_CIDS.CID>0]
total_DC_CIDs = set(DC_TOT_CIDS['CID'])

DC_TOT_CIDS = DC_TOT_CIDS.reset_index(drop = True)





# TARGET (2) 

OLD_TARGET_PATH = '/st06/jiyeonH/13.DD_SESS/merged_target/'
IDK_PATH = '/st06/jiyeonH/13.DD_SESS/ideker/' 


TARGET_DB_ori = pd.read_csv(OLD_TARGET_PATH+'combined_target.csv', low_memory=False, index_col = 0)
TARGET_DB_ori.columns = ['CID','gene_symbol','DB']

L_matching_list = pd.read_csv(IDK_PATH+'L_12_string.csv', sep = '\t')
L_check = L_matching_list[['L_gene_symbol','entrez']]

TARGET_DB_ori2 = pd.merge(TARGET_DB_ori, L_check, left_on = 'gene_symbol', right_on = 'L_gene_symbol', how = 'left')
TARGET_DB_ori3 = TARGET_DB_ori2[TARGET_DB_ori2.entrez>0]

TARGET_DB_ori3.columns= [ 'CID_RE','gene_symbol','DB','L_gene_symbol','EntrezID' ]
TARGET_DB_ori3['CID'] = list(TARGET_DB_ori3.CID_RE)

TARGET_DB = copy.deepcopy(TARGET_DB_ori3)




target_cids = copy.deepcopy(total_DC_CIDs)
gene_ids = list(BETA_ORDER_DF.gene_id)


def get_targets(CID): # 이건 지금 필터링 한 경우임 #
	if CID in target_cids:
		tmp_df2 = TARGET_DB[TARGET_DB.CID == CID]
		targets = list(set(tmp_df2.EntrezID))
		vec = [1 if a in targets else 0 for a in gene_ids ]
	else :
		vec = [0] * 978
	return vec


TARGETs = []

for IND in range(DC_TOT_CIDS.shape[0]) :
    if IND%100 == 0 :
        print(str(IND)+'/'+str(DC_TOT_CIDS.shape[0]) )
        datetime.now()
    CID = DC_TOT_CIDS['CID'][IND]
    target_vec = get_targets(CID)
    TARGETs.append(target_vec)
    

TARGET_TENSOR = torch.Tensor(TARGETs)

SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/FINAL_VALIDATION/'
torch.save(TARGET_TENSOR, SAVE_PATH+'DC_ALL_TARGET.pt')

DC_TOT_CIDS.to_csv(SAVE_PATH+'DC_ALL_TARGET.csv')












