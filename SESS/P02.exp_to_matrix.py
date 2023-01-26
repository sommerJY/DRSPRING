
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







# 다시 써먹을수는 있을듯 
MJ_DIR = '/st06/jiyeonH/13.DD_SESS/01.PRJ2/'

# MJ_request_ANS_M3 = pd.read_csv(MJ_DIR+'PRJ2_EXP_fugcn_vers1.csv')
# MJ_request_ANS_M33 = pd.read_csv(MJ_DIR+'PRJ2_EXP_fugcn_versa2_1.csv')
MJ_request_ANS_M3 = pd.read_csv(MJ_DIR+'PRJ2ver2_EXP_fugcn_vers1.csv', low_memory = False)

set(MJ_request_ANS_M3.columns) - set(MJ_request_ANS_M33.columns) 
set(MJ_request_ANS_M33.columns) -set(MJ_request_ANS_M3.columns)

ORD = [list(MJ_request_ANS_M3.entrez_id).index(a) for a in BETA_ENTREZ_ORDER]
MJ_request_ANS_M3_ORD = MJ_request_ANS_M3.iloc[ORD]

MJ_tuples_1 = [a for a in list(MJ_request_ANS_M3_ORD.columns) if 'CVCL' in a]
MJ_tuples_2 = [(a.split('__')[0],a.split('__')[1])  for a in list(MJ_request_ANS_M3_ORD.columns) if 'CVCL' in a]

MJ_tup_df = pd.DataFrame()

MJ_tup_df['sample'] = MJ_tuples_1 
MJ_tup_df['tuple'] = MJ_tuples_2

MJ_exp_list = []

for IND in range(MJ_tup_df.shape[0]) :
    if IND%100 == 0 :
        print(str(IND)+'/'+str(MJ_tup_df.shape[0]) )
        datetime.now()
    Column = MJ_tup_df['sample'][IND]
    MJ_vector = MJ_request_ANS_M3_ORD[Column].values.tolist()
    MJ_exp_list.append(MJ_vector)


MJ_TENSOR = torch.Tensor(MJ_exp_list)

SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/FINAL_VALIDATION/'

torch.save(MJ_TENSOR, SAVE_PATH+'MJ3_EXP_TOT.pt')

MJ_tup_df.to_csv(SAVE_PATH+'MJ3_EXP_TOT.csv')



#####################
TOTAL DC drug - cell 에 해당하는 LINCS 만 저장 

# 일단 모델별로 확인해야하니까 
# LOAD Cells 

MJ_NAME = 'M3'
MISS_NAME = 'MIS2'

SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/{}_FULL/'.format(MJ_NAME)
file_name = 'M3_MISS2_FULL'

A_B_C_S_SET_ADD = pd.read_csv(SAVE_PATH+'{}.A_B_C_S_SET_ADD.csv'.format(file_name), low_memory=False)

avail_cell_list = list(set(A_B_C_S_SET_ADD.DrugCombCello))





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

DC_DATA_filter5 = pd.merge(DC_DATA_filter4, DC_CELL_DF2[['cell_line_id','DrugCombCello']], on = 'cell_line_id', how ='left' )


# 그래서 drugcomb 에서 일단 사용된 내용들 CID 기준 
DC_DATA_filter6 = DC_DATA_filter5[DC_DATA_filter5.DrugCombCello.isin(avail_cell_list)]

good_ind = [a for a in range(DC_DRUG_DF_FULL.shape[0]) if type(DC_DRUG_DF_FULL.CAN_SMILES[a]) == str ]
DC_DRUG_DF_FULL_filt = DC_DRUG_DF_FULL.loc[good_ind]

DC_DRUG_DF_FULL_filt['leng2'] = [int(a) for a in list(DC_DRUG_DF_FULL_filt.leng)]
DC_DRUG_DF_FULL_filt = DC_DRUG_DF_FULL_filt[DC_DRUG_DF_FULL_filt.leng2 <=50] # 7775

DC_DRUG_DF_FULL_cut = DC_DRUG_DF_FULL_filt[['id','CID','CAN_SMILES']] # DrugComb 에서 combi 할 수 있는 총 CID : 7775개 cid 

# 있는 combi 에 대한 CID 붙이기 
DC_DRUG_DF_FULL_cut.columns = ['drug_row_id_re','ROW_CID','ROW_CAN_SMILES']
DC_re_1 = pd.merge(DC_DATA_filter6, DC_DRUG_DF_FULL_cut, on = 'drug_row_id_re', how = 'left') # 146942

DC_DRUG_DF_FULL_cut.columns = ['drug_col_id_re','COL_CID', 'COL_CAN_SMILES']
DC_re_2 = pd.merge(DC_re_1, DC_DRUG_DF_FULL_cut, on = 'drug_col_id_re', how = 'left')

DC_DRUG_DF_FULL_cut.columns = ['id','CID','CAN_SMILES']

DC_re_3 = DC_re_2[['ROW_CID','COL_CID','DrugCombCello']].drop_duplicates()
DC_re_4 = DC_re_3[ (DC_re_3.ROW_CID>0) & (DC_re_3.COL_CID>0) ]
DC_re_4 = DC_re_4.reset_index(drop = True) # 97118



# DC_ALL_D_C_1 = [(int(DC_re_4.ROW_CID[a]), DC_re_4.DrugCombCello[a]) for a in range(DC_re_4.shape[0])]
# DC_ALL_D_C_2 = [(int(DC_re_4.COL_CID[a]), DC_re_4.DrugCombCello[a]) for a in range(DC_re_4.shape[0])]
# DC_ALL_D_C = list(set(DC_ALL_D_C_1+DC_ALL_D_C_2)) # 5161



# LINCS filter 

BETA_BIND = pd.read_csv('/st06/jiyeonH/11.TOX/MY_TRIAL_5/'+"BETA_DATA_for_20220705_978.csv")
#BETA_CP_info = pd.read_table('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+'compoundinfo_beta.txt') # pert 34419 , cansmiles : 
#BETA_CEL_info = pd.read_table('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+'cellinfo_beta.txt')
#BETA_SIG_info = pd.read_table('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+'siginfo_beta.txt', low_memory = False)

ORD = [list(BETA_BIND.id).index(a) for a in BETA_ENTREZ_ORDER]
BETA_BIND_ORD = BETA_BIND.iloc[ORD]

# prepare py 에서 확인하고 가져온거 
# BETA_CID_CELLO_SIG

BETA_CID_CELLO_SIG_re = BETA_CID_CELLO_SIG.reset_index(drop = True)

BETA_CID_CELLO_SIG_tup = [(BETA_CID_CELLO_SIG_re.pubchem_cid[a], BETA_CID_CELLO_SIG_re.cellosaurus_id[a]) for a in range(BETA_CID_CELLO_SIG_re.shape[0])]
BETA_CID_CELLO_SIG_re['tuple'] = BETA_CID_CELLO_SIG_tup
BETA_CID_CELLO_SIG_tup_re = [(str(BETA_CID_CELLO_SIG_re.pubchem_cid[a]), BETA_CID_CELLO_SIG_re.cellosaurus_id[a]) for a in range(BETA_CID_CELLO_SIG_re.shape[0])]
BETA_CID_CELLO_SIG_re['tuple_re'] = BETA_CID_CELLO_SIG_tup_re

BETA_CID_CELLO_SIG_re_re = BETA_CID_CELLO_SIG_re[BETA_CID_CELLO_SIG_re['cellosaurus_id'].isin(avail_cell_list)]

BETA_CID_CELLO_SIG_re_re = BETA_CID_CELLO_SIG_re_re.reset_index(drop=True)

LINCS_exp_list = []

for IND in range(BETA_CID_CELLO_SIG_re_re.shape[0]) :
    if IND%100 == 0 :
        print(str(IND)+'/'+str(BETA_CID_CELLO_SIG_re_re.shape[0]) )
        datetime.now()
    Column = BETA_CID_CELLO_SIG_re_re['sig_id'][IND]
    L_vector = BETA_BIND_ORD[Column].values.tolist()
    LINCS_exp_list.append(L_vector)


L_TENSOR = torch.Tensor(LINCS_exp_list)

SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/FINAL_VALIDATION/'

torch.save(L_TENSOR, SAVE_PATH+'LINCS_EXP_cell42.pt')

BETA_CID_CELLO_SIG_re_re.to_csv(SAVE_PATH+'LINCS_EXP_cell42.csv')














# LOAD Cells 

MJ_NAME = 'M3'
MISS_NAME = 'MIS2'

SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/{}_FULL/'.format(MJ_NAME)
file_name = 'M3_MISS2_FULL'

A_B_C_S_SET_ADD = pd.read_csv(SAVE_PATH+'{}.A_B_C_S_SET_ADD.csv'.format(file_name), low_memory=False)









def get_LINCS_data(DRUG_SIG):
	Drug_EXP = BETA_BIND[['id',DRUG_SIG]]
	BIND_ORDER =[list(BETA_BIND.id).index(a) for a in BETA_ORDER_DF.gene_id]
	Drug_EXP_ORD = Drug_EXP.iloc[BIND_ORDER]
	#
	return list(Drug_EXP_ORD[DRUG_SIG])



# fu (M3 & M33)
def get_MJ_data( CHECK ): 
	if CHECK in list(MJ_request_ANS.columns) :
		MJ_DATA = MJ_request_ANS[['entrez_id', CHECK]]
		ord = [list(MJ_DATA.entrez_id).index(a) for a in BETA_ENTREZ_ORDER]
		MJ_DATA_re = MJ_DATA.loc[ord] 
		RES = MJ_DATA_re[CHECK]
		OX = 'O'
	else : 
		RES = [0]*978
		OX = 'X'
	return list(RES), OX


