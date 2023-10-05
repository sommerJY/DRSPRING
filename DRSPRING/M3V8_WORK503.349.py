
503 실험 
trial 1 : 원래 네트워크에 새로운 민지 데이터 사용
trial 2 : 새로운 네트워크에 새로운 민지 데이터 사용 



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
from matplotlib import colors as mcolors

MJ_DIR = '/st06/jiyeonH/13.DD_SESS/01.PRJ2/'

MJ_request_ANS = pd.read_csv(MJ_DIR+'PRJ2_EXP_ccle_all_fugcn_hwf2dttf3.csv')


entrez_id = list(MJ_request_ANS['entrez_id'])
MJ_request_ANS = MJ_request_ANS.drop(['entrez_id','Unnamed: 0','CID__CELL'], axis =1)
MJ_request_ANS['entrez_id'] = entrez_id

ord = [list(MJ_request_ANS.entrez_id).index(a) for a in BETA_ENTREZ_ORDER] # ordering ok 
MJ_request_ANS_re = MJ_request_ANS.loc[ord] 


A_B_C_S_SET_MJ = A_B_C_S_SET[A_B_C_S_SET.ROWCHECK.isin(MJ_request_ANS.columns)]
A_B_C_S_SET_MJ = A_B_C_S_SET_MJ[A_B_C_S_SET_MJ.COLCHECK.isin(MJ_request_ANS.columns)]
A_B_C_S_SET_MJ = A_B_C_S_SET_MJ.reset_index(drop = True)



def get_MJ_data( CHECK ): 
	if CHECK in list(MJ_request_ANS_re.columns) :
		RES = MJ_request_ANS_re[CHECK]
		OX = 'O'
	else : 
		RES = [0]*349        ##############
		OX = 'X'
	return list(RES), OX



A_B_C_S_SET_UNIQ = copy.deepcopy(A_B_C_S_SET_MJ) # 613961 or 455856
A_B_C_S_SET_UNIQ['type'] = [ 'AXBO' if a == 'AOBX' else a for a in A_B_C_S_SET_UNIQ['type']]
A_B_C_S_SET_UNIQ_2 = A_B_C_S_SET_UNIQ[['cid_cid_cell','sig_sig_cell','type']].drop_duplicates()
A_B_C_S_SET_UNIQ_3 = A_B_C_S_SET_UNIQ_2[['cid_cid_cell','type']].drop_duplicates()
A_B_C_S_SET_UNIQ_3 = A_B_C_S_SET_UNIQ_3.reset_index(drop=True)


A_B_C_S_SET_UNIQ_3['CID_A'] = A_B_C_S_SET_UNIQ_3.cid_cid_cell.apply(lambda x : int(x.split('___')[0]))
A_B_C_S_SET_UNIQ_3['CID_B'] = A_B_C_S_SET_UNIQ_3.cid_cid_cell.apply(lambda x : int(x.split('___')[1]))
A_B_C_S_SET_UNIQ_3['CELL'] = A_B_C_S_SET_UNIQ_3.cid_cid_cell.apply(lambda x : x.split('___')[2])



max_len = 50



MY_chem_A_feat = torch.empty(size=(A_B_C_S_SET_UNIQ_3.shape[0], max_len, 64))
MY_chem_B_feat= torch.empty(size=(A_B_C_S_SET_UNIQ_3.shape[0], max_len, 64))
MY_chem_A_adj = torch.empty(size=(A_B_C_S_SET_UNIQ_3.shape[0], max_len, max_len))
MY_chem_B_adj= torch.empty(size=(A_B_C_S_SET_UNIQ_3.shape[0], max_len, max_len))
MY_syn =  torch.empty(size=(A_B_C_S_SET_UNIQ_3.shape[0],1))

MY_g_EXP_A = torch.empty(size=(A_B_C_S_SET_UNIQ_3.shape[0], 349, 1))##############
MY_g_EXP_B = torch.empty(size=(A_B_C_S_SET_UNIQ_3.shape[0], 349, 1))##############




Fail_ind = []
from datetime import datetime

for IND in range(0, A_B_C_S_SET_UNIQ_3.shape[0]): #  100
	if IND%100 == 0 : 
		print(str(IND)+'/'+str(A_B_C_S_SET_UNIQ_3.shape[0]) )
		Fail_ind
		datetime.now()
	#
	cid_cid_cell = A_B_C_S_SET_UNIQ_3.cid_cid_cell[IND]
	DrugA_CID = A_B_C_S_SET_UNIQ_3['CID_A'][IND]
	DrugB_CID = A_B_C_S_SET_UNIQ_3['CID_B'][IND]
	CELL = A_B_C_S_SET_UNIQ_3['CELL'][IND]
	dat_type = A_B_C_S_SET_UNIQ_3.type[IND]
	DrugA_CID_CELL = str(DrugA_CID) + '__' + CELL
	DrugB_CID_CELL = str(DrugB_CID) + '__' + CELL
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



PRJ_NAME = 'M3V9_349_MISS2_FULL' # SNF 349

SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V9_349_FULL/'


torch.save(MY_chem_A_feat, SAVE_PATH+'{}.MY_chem_A_feat.pt'.format(PRJ_NAME))
torch.save(MY_chem_B_feat, SAVE_PATH+'{}.MY_chem_B_feat.pt'.format(PRJ_NAME))
torch.save(MY_chem_A_adj, SAVE_PATH+'{}.MY_chem_A_adj.pt'.format(PRJ_NAME))
torch.save(MY_chem_B_adj, SAVE_PATH+'{}.MY_chem_B_adj.pt'.format(PRJ_NAME))
torch.save(MY_g_EXP_A, SAVE_PATH+'{}.MY_g_EXP_A.pt'.format(PRJ_NAME))
torch.save(MY_g_EXP_B, SAVE_PATH+'{}.MY_g_EXP_B.pt'.format(PRJ_NAME))
torch.save(MY_syn, SAVE_PATH+'{}.MY_syn.pt'.format(PRJ_NAME))

A_B_C_S_SET_UNIQ_3.to_csv(SAVE_PATH+'{}.A_B_C_S_SET.csv'.format(PRJ_NAME))
A_B_C_S.to_csv(SAVE_PATH+'{}.A_B_C_S.csv'.format(PRJ_NAME))


MJ_DIR = '/st06/jiyeonH/13.DD_SESS/01.PRJ2/'
MJ_new_net = pd.read_csv(MJ_DIR+'fugcn_hwf2dttf3/fusion3_hsi20_349.csv')

ID_G = nx.from_pandas_edgelist(MJ_new_net, 'gene_id1', 'gene_id2')


JY_GRAPH = nx.from_pandas_edgelist(hnet_L3, 'G_A', 'G_B')

NEW_G = nx.Graph()
NEW_G.add_nodes_from(JY_GRAPH.nodes())
NEW_G.add_edges_from(ID_G.edges(data=True))


# edge 3871
ID_GENE_ORDER_mini = list(ID_G.nodes()) # 978
ID_ADJ = nx.adjacency_matrix(ID_G)
ID_ADJ_tmp = torch.LongTensor(ID_ADJ.toarray())
ID_ADJ_IDX = ID_ADJ_tmp.to_sparse().indices()  # [2, 7742]
ID_WEIGHT = [] # len : 3871 -> 7742

ID_WEIGHT_SCORE = [1 for a in range(ID_ADJ_IDX.shape[1])]

