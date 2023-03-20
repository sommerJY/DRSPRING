
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




# LOAD Cells 

SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V4_349_FULL/'

A_B_C_S_SET_ADD_EX = pd.read_csv(SAVE_PATH+'M3V4_349_MISS2_FULL.A_B_C_S_SET_ADD.csv', low_memory=False)



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

DC_DATA_filter = DC_tmp_DF2[['drug_row_id_re','drug_col_id_re','cell_line_id','synergy_loewe']] # 1457561
DC_DATA_filter2 = DC_DATA_filter.drop_duplicates() # 1374958 -> 1363698

DC_DATA_filter3 = DC_DATA_filter2[DC_DATA_filter2.drug_row_id_re > 0] # 1374958 -> 1363698
DC_DATA_filter4 = DC_DATA_filter3[DC_DATA_filter3.drug_col_id_re > 0] # 751450 -> 740884
DC_DATA_filter4.cell_line_id # unique 295
DC_DATA_filter4[['drug_row_id_re','drug_col_id_re','cell_line_id']].drop_duplicates() # 648516
DC_DATA_filter4[['drug_row_id_re','drug_col_id_re']].drop_duplicates() # 75174
len(list(set(list(DC_DATA_filter4.drug_row_id_re) + list(DC_DATA_filter4.drug_col_id_re)))) # 4327

DC_DATA_filter4 = DC_DATA_filter4.reset_index(drop = False)

DC_DATA_filter5 = pd.merge(DC_DATA_filter4, DC_CELL_DF2[['cell_line_id','DrugCombCCLE']], on = 'cell_line_id', how ='left' )



# 그래서 drugcomb 에서 일단 사용된 내용들 CID 기준 - CCLE 
DC_DATA_filter6 = DC_DATA_filter5[DC_DATA_filter5.DrugCombCCLE.isin(avail_cell_list)]

good_ind = [a for a in range(DC_DRUG_DF_FULL.shape[0]) if type(DC_DRUG_DF_FULL.CAN_SMILES[a]) == str ]
DC_DRUG_DF_FULL_filt = DC_DRUG_DF_FULL.loc[good_ind]

DC_DRUG_DF_FULL_filt['leng2'] = [int(a) for a in list(DC_DRUG_DF_FULL_filt.leng)]
DC_DRUG_DF_FULL_filt = DC_DRUG_DF_FULL_filt[DC_DRUG_DF_FULL_filt.leng2 <=50] # 7775

DC_DRUG_DF_FULL_cut = DC_DRUG_DF_FULL_filt[['id','CID','CAN_SMILES']] # DrugComb 에서 combi 할 수 있는 총 CID : 7775개 cid 
DC_DRUG_DF_FULL_cut.columns = ['drug_row_id_re','ROW_CID','ROW_CAN_SMILES']

# 있는 combi 에 대한 CID 붙이기 
DC_re_1 = pd.merge(DC_DATA_filter6, DC_DRUG_DF_FULL_cut, on = 'drug_row_id_re', how = 'left') # 146942

DC_DRUG_DF_FULL_cut.columns = ['drug_col_id_re','COL_CID', 'COL_CAN_SMILES']
DC_re_2 = pd.merge(DC_re_1, DC_DRUG_DF_FULL_cut, on = 'drug_col_id_re', how = 'left')

DC_DRUG_DF_FULL_cut.columns = ['id','CID','CAN_SMILES']

DC_re_3 = DC_re_2[['ROW_CID','COL_CID','DrugCombCCLE']].drop_duplicates()
DC_re_4 = DC_re_3.reset_index(drop = True)


from itertools import combinations
from itertools import product
from itertools import permutations

DC_all_cids = list(set(DC_DRUG_DF_FULL_cut[DC_DRUG_DF_FULL_cut.CID>0]['CID'])) # 7775개 (SM 있고, 50 이하에 leng 붙는 애들 )
DC_pairs = list(combinations(DC_all_cids, 2)) 
# permutation 으로 하면 66512180 -> 50 filter -> 60442850 모든 cid - cid  양면 
# combination 으로 하면 66512180 -> 50 filter -> 30221425 그냥 unique 한 cid - cid 


# 119522
# 그러고 나서 DC 안에 있는 모든 CID - CID - ccle triads 조사
IN_DC_pairs_1 = [(DC_re_4.ROW_CID[a] ,DC_re_4.COL_CID[a], DC_re_4.DrugCombCCLE[a]) for a in range(DC_re_4.shape[0])]
IN_DC_pairs_2 = [(DC_re_4.COL_CID[a] ,DC_re_4.ROW_CID[a], DC_re_4.DrugCombCCLE[a]) for a in range(DC_re_4.shape[0])]
IN_DC_pairs = IN_DC_pairs_1 + IN_DC_pairs_2 # 187734






# 8000 * 42 

DC_FULL_CID = [a for a in DC_DRUG_DF_FULL.CID if a > 0 ]

avail_cell_list

aa = product(DC_FULL_CID, avail_cell_list)
aaa = list(product(DC_FULL_CID, avail_cell_list))

# save as json

with open('/st06/jiyeonH/13.DD_SESS/01.PRJ2/'+'MJ_REQUEST_VER2.CCLE1.json', 'w') as f:
    json.dump(aaa,f)




# 아예 가능한 모든 cell line 달라고 하면? 
tot_cell_list_1 = list(set(DC_DATA_filter5.DrugCombCCLE))
tot_cell_list_2 = tot_cell_list_1[1:]
tot_cell_list_3 = [a for a in tot_cell_list_2 if a != 'NA']

aa = product(DC_FULL_CID, tot_cell_list_3)
aaa = list(product(DC_FULL_CID, tot_cell_list_3))


# save as json

with open('/st06/jiyeonH/13.DD_SESS/01.PRJ2/'+'MJ_REQUEST_VER2.CCLE2.json', 'w') as f:
    json.dump(aaa,f)

