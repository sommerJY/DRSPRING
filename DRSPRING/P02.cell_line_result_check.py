
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



# LOAD MODEL

MJ_NAME = 'M3'
MISS_NAME = 'MIS2'

SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/{}_FULL/'.format(MJ_NAME)
file_name = 'M3_MISS2_FULL'

A_B_C_S_SET_ADD = pd.read_csv(SAVE_PATH+'{}.A_B_C_S_SET_ADD.csv'.format(file_name), low_memory=False)



# A_B_C_S SET filter check
WORK_NAME = 'WORK_0'

#MISS_filter = ['AOBO']
#MISS_filter = ['AOBO','AXBO','AOBX']
MISS_filter = ['AOBO','AXBO','AOBX','AXBX']

A_B_C_S_SET = A_B_C_S_SET_ADD[A_B_C_S_SET_ADD.SYN_OX == 'O']
A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.T2OX == 'O']
A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.type.isin(MISS_filter)]

data_ind = list(A_B_C_S_SET.index)

A_B_C_S_SET = A_B_C_S_SET.reset_index(drop = True)


A_B_C_S_SET_SM = copy.deepcopy(A_B_C_S_SET)

A_B_C_S_SET_SM['ori_index'] = list(A_B_C_S_SET_SM.index)
aa = list(A_B_C_S_SET_SM['ROW_CAN_SMILES'])
bb = list(A_B_C_S_SET_SM['COL_CAN_SMILES'])
cc = list(A_B_C_S_SET_SM['DrugCombCello'])
A_B_C_S_SET_SM['SM_C_CHECK'] = [aa[i] + '___' + bb[i]+ '___' + cc[i] for i in range(A_B_C_S_SET_SM.shape[0])]


# get unique values, remove duplicates, but keep original counts
data_no_dup, counts = np.unique(list(A_B_C_S_SET_SM['SM_C_CHECK']), return_counts=True)
data_no_dup_cells = [setset.split('___')[2] for setset in data_no_dup]
data_nodup_df = pd.DataFrame({'setset' : data_no_dup.tolist(), 'cell' : data_no_dup_cells })





# cell line vector 
DC_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/'

DC_CELL_DF2 = pd.read_csv(DC_PATH+'DC_CELL_INFO.csv', sep = '\t')

DC_CELL_info_filt = DC_CELL_DF2[DC_CELL_DF2.DrugCombCello.isin(A_B_C_S_SET.DrugCombCello)]

DC_CELL_info_filt['cell_onehot'] = [a for a in range(len(set(A_B_C_S_SET.DrugCombCello)))]
DC_CELL_info_filt = DC_CELL_info_filt.reset_index(drop = True)

cell_one_hot_vec = torch.nn.functional.one_hot(torch.Tensor(DC_CELL_info_filt.cell_onehot).long())




# 8 : 1 : 1 

grouped_df = data_nodup_df.groupby('cell')

CV_1_list = []; CV_2_list = []; CV_3_list = []; CV_4_list = []; CV_5_list = []
CV_6_list = []; CV_7_list = []; CV_8_list = []; CV_9_list = []; CV_10_list = []

for i, g in grouped_df:
	if len(g) > 10 :
		nums = int(.1 * len(g))
		bins = []
		g2 = sklearn.utils.shuffle(g, random_state=42)
		for ii in list(range(0, len(g2), nums)):
			if len(bins)<= 9 :
				bins.append(ii)
		#
		bins = bins[1:]
		res = np.split(g2, bins)
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







# 모델 가져오기 

import pandas as pd 
import matplotlib.pyplot as plt
from ray.tune import ExperimentAnalysis
from ray.tune import Analysis
import pickle
import math
import torch
import os 


WORK_DATE = '23.12.25'
PRJ_NAME = 'M3'
WORK_NAME = 'WORK_0'
MISS_NAME = 'MIS2'




print('NETWORK')

NETWORK_PATH = '/st06/jiyeonH/13.DD_SESS/HumanNetV3/'

hunet_gsp = pd.read_csv(NETWORK_PATH+'HumanNet-GSP.tsv', sep = '\t', header = None)
hunet_gsp.columns = ['G_A','G_B']

LINCS_gene_file = pd.read_csv(LINCS_PATH+'geneinfo_beta.txt', sep = '\t')
LINCS_978 = LINCS_gene_file[LINCS_gene_file.feature_space == 'landmark']
LINCS_978 = LINCS_978[['gene_id','gene_symbol']]
LINCS_978['new_node'] = [str(list(LINCS_978.gene_id)[i]) + "__" + list(LINCS_978.gene_symbol)[i] for i in range(978)]
LINCS_978 = LINCS_978.reset_index(drop=True)
lm_entrezs = list(LINCS_978.gene_id)


hnet_L1 = hunet_gsp[hunet_gsp['G_A'].isin(lm_entrezs)]
hnet_L2 = hnet_L1[hnet_L1['G_B'].isin(lm_entrezs)] # 3885

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
















PRJ_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/{}_{}_W0/'.format(MJ_NAME, MISS_NAME)

ANA_DF = pd.read_csv(PRJ_PATH+'RAY_ANA_DF.{}_{}_{}.csv'.format(MJ_NAME, MISS_NAME, WORK_NAME))
with open(PRJ_PATH+'RAY_ANA_DF.{}_{}_{}.pickle'.format(MJ_NAME, MISS_NAME, WORK_NAME), 'rb') as f:
	ANA_ALL_DF = pickle.load(f)

# M3 MIS2 W0 list 
trial_list = ["4cef1a8e","763ac58a","2df50b20","852ed9b4","370ff2d0","eaf0de8c","f258543a","55550dce","91f3b472","448ef8d4","2644c294","e290e766","adfb3c80","ae46ad40","ec9e5d20","ba3d661e","66db7a7a","c8b31198","ef1205d8","ee27fb32","c3ba2e4a","3ceec234","babd315c","b2db1ffe","167cb48e","74763da4","953a09ec","9e33ae9e","3c4a5d96","22023ddc","453110f4","e16fb7de","a8ac9014","a80a7866","d40964a6","ab79c5f8","5143303a","cab12408","b1060c54","5eefb3a0","6dc113de","94be52a0","26a6fbce","de9b4924","9c3d0d84","95fecb6c","792c79e4","726cc86c","e07d1aaa","711f007c","5faa2210","da67ed3c","2e163b38","becb936a","47f20492","1a1f3488","f11f43da","4028a41e","9de9f854","4d984f8e","96b97d54","bef7c6e6","f9bfc7ec","a2c9dcfe","30d2333e","5ba1d0f2","340967fc","daff9558","07b2ad90","e4c35ab8","b93f4e56","025128b2","cc0fd08c","cec36b1e","bef320dc","21e0ce14","ce3da6d2","293e3478","9af94750","c84e6b1e","8a52a9ee","873194b8","11ccb92a","bf8483ec","bedef0d0","bf1ec96c","beb4fc1c","b2920c54","be8871e2","bfd650be","bfabc52e","bf595640","bf451e46","bec86130","bfeabbc6","bfc0c096","bf979004","bf6f4e6e","bf317440","bf0b0b52","4d591058","4d837b72","4dcf12d0","4e9560b6","4e69f2be","4d28306e","4df9f496","4e1130ac","4e7f4f4c","4dba1ed4","4de52ed0","4d6dcc82","4e2b3fc4","4e3f34ac","4e53c5c0","4ead1512","4d9abb0c"]

ANA_DF = ANA_DF[ANA_DF.trial_id.isin(trial_list)]
all_keys = list(ANA_ALL_DF.keys())
all_keys_trials = [a.split('/')[5].split('_')[3] for a in all_keys]
all_keys_trials_ind = [all_keys_trials.index(a) for a in trial_list if a in list(ANA_DF.trial_id)]
new_keys = [all_keys[a] for a in all_keys_trials_ind]

TOPVAL_PATH = PRJ_PATH



# M3_MIS2_W0 모델 
import numpy as np
TOT_max = -np.Inf
TOT_key = ""
for key in new_keys:
	trial_max = max(ANA_ALL_DF[key]['SCOR'])
	if trial_max > TOT_max :
		TOT_max = trial_max
		TOT_key = key

print('best cor', flush=True)
print(TOT_key, flush=True)
mini_df = ANA_ALL_DF[TOT_key]
#TOPVAL_PATH = TOT_key
my_config = ANA_DF[ANA_DF.logdir==TOT_key]
cck_num =mini_df[mini_df.SCOR==max(mini_df.SCOR)].index.item()
checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
#TOPVAL_PATH = TOT_key + checkpoint
print('best cor check', flush=True)
print(TOPVAL_PATH, flush=True)
R_6_V = max(mini_df.SCOR)
R_6_V
model_name = 'M4_checkpoint'




# read the cell line file

Cell_name = 'CVCL_0035'
Cell_name = 'CVCL_0179'
Cell_name = 'CVCL_0178'
Cell_name = 'CVCL_A442'
Cell_name = 'CVCL_0359'

CELVAL_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/FINAL_VALIDATION/CELLS_VAL/'

with open(CELVAL_PATH+'{}.json'.format(Cell_name)) as f:
   lst_check = [tuple(x) for x in json.load(f)]



# 데이터 가지고 돌리기 
use_cuda = False # False #   #  #  #  #
#
G_chem_layer = my_config['config/G_chem_layer'].item()
G_chem_hdim = my_config['config/G_chem_hdim'].item()
G_exp_layer = my_config['config/G_exp_layer'].item()
G_exp_hdim = my_config['config/G_exp_hdim'].item()
dsn1_layers = [my_config['config/feat_size_0'].item(), my_config['config/feat_size_1'].item(), my_config['config/feat_size_2'].item()]
dsn2_layers = [my_config['config/feat_size_0'].item(), my_config['config/feat_size_1'].item(), my_config['config/feat_size_2'].item()] 
snp_layers = [my_config['config/feat_size_3'].item() , my_config['config/feat_size_4'].item()]
inDrop = my_config['config/dropout_1'].item()
Drop = my_config['config/dropout_2'].item()
#       
best_model = MY_expGCN_parallel_model(
			G_chem_layer, 64, G_chem_hdim,
			G_exp_layer, 3, G_exp_hdim,
			dsn1_layers, dsn2_layers, snp_layers, 
			len(set(A_B_C_S_SET_SM.DrugCombCello)), 1,
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
	state_dict = torch.load(os.path.join(PRJ_PATH, model_name))
else:
	state_dict = torch.load(os.path.join(PRJ_PATH, model_name), map_location=torch.device('cpu'))
# 


print("state_dict_done", flush = True)

if type(state_dict) == tuple:
	best_model.load_state_dict(state_dict[0])
else : 
	best_model.load_state_dict(state_dict)	#


print("state_load_done", flush = True)
#


SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/FINAL_VALIDATION/'

all_chem_DF = pd.read_csv(SAVE_PATH+'DC_ALL_7555_ORDER.csv')
all_chem_feat_TS = torch.load(SAVE_PATH+'DC_ALL.MY_chem_feat.pt')
all_chem_feat_adj = torch.load(SAVE_PATH+'DC_ALL.MY_chem_adj.pt')


def check_drug_f_ts(CID) :
	INDEX = all_chem_DF[all_chem_DF.CID == CID].index.item()
	adj_pre = all_chem_feat_adj[INDEX]
	adj_proc = adj_pre.long().to_sparse().indices()
	return all_chem_feat_TS[INDEX], adj_proc


cell_42_LINCS_DF = pd.read_csv(SAVE_PATH+'LINCS_EXP_cell42.csv')
cell_42_LINCS_TS = torch.load(SAVE_PATH+'LINCS_EXP_cell42.pt')
cell_42_LINCS_DF['tuple'] = [(a.split('__')[0], a.split('__')[1]) for a in cell_42_LINCS_DF['IDCHECK']]
cell_42_LINCS_TPs = list(cell_42_LINCS_DF['tuple'])

mj_exp_DF = pd.read_csv(SAVE_PATH+'MJ3_EXP_TOT.csv')
mj_exp_TS = torch.load(SAVE_PATH+'MJ3_EXP_TOT.pt')
mj_exp_DF['tuple'] = [(a.split('__')[0], a.split('__')[1]) for a in mj_exp_DF['sample']]
mj_exp_TPs = list(mj_exp_DF['tuple'])



targets_DF = pd.read_csv(SAVE_PATH+'DC_ALL_TARGET.csv')
targets_TS = torch.load(SAVE_PATH+'DC_ALL_TARGET.pt')

all_CellBase_DF = pd.read_csv(SAVE_PATH+'AVAIL_CELL_DF.csv')
all_CellBase_TS = torch.load(SAVE_PATH+'AVAIL_CLL_MY_CellBase.pt')

TPs_all = cell_42_LINCS_TPs + mj_exp_TPs
TPs_all_2 = [a[0]+"__"+a[1] for a in TPs_all]




# tt = [a for a in lst_check if ((str(int(a[0])),a[2]) in TPs_all) and ((str(int(a[1])),a[2]) in TPs_all)]
# tt = [a for a in lst_test if ((str(int(a[0])),a[2]) in TPs_all) and ((str(int(a[1])),a[2]) in TPs_all)]

tt_df = pd.DataFrame()
tt_df['tuple'] = lst_check
tt_df['cid1'] = [str(int(a[0])) for a in lst_check]
tt_df['cid2'] = [str(int(a[1])) for a in lst_check]
tt_df['cello'] = [a[2] for a in lst_check]

tt_df['cid1_celo'] = tt_df.cid1 +'__' +tt_df.cello
tt_df['cid2_celo'] = tt_df.cid2 +'__' +tt_df.cello


tt_df_re1 = tt_df[tt_df.cid1_celo.isin(TPs_all_2)] # 18326925
tt_df_re2 = tt_df_re1[tt_df_re1.cid2_celo.isin(TPs_all_2)] # 10624097

# targets 
tg_lists = [str(a) for a in list(set(TARGET_DB.CID))]
tt_df_re3 = tt_df_re2[tt_df_re2.cid1.isin(tg_lists)] # 5289702
tt_df_re4 = tt_df_re3[tt_df_re3.cid2.isin(tg_lists)] # 2359767

tt_df_re4 = tt_df_re4.reset_index(drop=True)

tuple_list = tt_df_re4['tuple']





# target 없어도 데려가기 -> nope 필터링 해야겠다고 마음먹음 

def check_exp_f_ts(CID, CELLO) :
	TUPLE = (str(int(CID)), CELLO)
	# Gene EXP
	if TUPLE in cell_42_LINCS_TPs:
		L_index = cell_42_LINCS_DF[cell_42_LINCS_DF['tuple'] == TUPLE].index[0].item() # 이건 나중에 고쳐야해 
		EXP_vector = cell_42_LINCS_TS[L_index]
	elif TUPLE in mj_exp_TPs:
		M_index = mj_exp_DF[mj_exp_DF['tuple'] == TUPLE].index.item()
		EXP_vector = mj_exp_TS[M_index]
	else :
		print('error')
	#
	# TARGET 
	T_index = targets_DF[targets_DF['CID'] == CID].index.item()
	TG_vector = targets_TS[T_index]
	#
	# BASAL EXP 
	B_index = all_CellBase_DF[all_CellBase_DF.DrugCombCello == CELLO].index.item()
	B_vector = all_CellBase_TS[B_index]
	#
	#
	FEAT = torch.Tensor(np.array([ EXP_vector.squeeze().tolist() , TG_vector.squeeze().tolist(), B_vector.squeeze().tolist()]).T)
	return FEAT.view(-1,3)



def check_cell_oh(CELLO) :
	oh_index = DC_CELL_info_filt[DC_CELL_info_filt.DrugCombCello == CELLO].index.item()
	cell_vec = cell_one_hot_vec[oh_index]
	return cell_vec







CELL_PRED_DF = pd.DataFrame(columns = ['ROW_CID','COL_CID','CELLO','PRED_RES'])

PRED_list = []

# ROW_CID, COL_CID, CELLO = lst_test[0]

for IND in range(len(tuple_list)) :
	if IND%100 == 0 : 
		print(str(IND)+'/'+ str(len(tuple_list)))
		datetime.now()
	# 
	if IND%1000 == 0 : 
		# set(PRED_list)
		CELL_PRED_DF.to_csv(SAVE_PATH+'PRED_{}.csv'.format(Cell_name),  index=False, header=False) # mode='a',
	# 
	ROW_CID, COL_CID, CELLO = tuple_list[IND]
	#
	TUP_1 = (str(int(ROW_CID)), CELLO)
	TUP_2 = (str(int(COL_CID)), CELLO)
	#
	if (TUP_1 in TPs_all) & (TUP_2 in TPs_all) : 
		drug1_f , drug1_a = check_drug_f_ts(ROW_CID)
		drug2_f , drug2_a = check_drug_f_ts(COL_CID)
		#
		expA = check_exp_f_ts(ROW_CID, CELLO)
		expB = check_exp_f_ts(COL_CID, CELLO)
		#
		adj = copy.deepcopy(JY_ADJ_IDX).long()
		adj_w = torch.Tensor(JY_IDX_WEIGHT).squeeze()
		#
		cell = check_cell_oh(CELLO)
		cell = cell.unsqueeze(0)
		#
		#
		best_model.eval()
		with torch.no_grad():
			y = torch.Tensor([1]).float().unsqueeze(1)
			output= best_model(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y) 
		outputs = [output.squeeze().item()]
		print(outputs)
		PRED_list = PRED_list+outputs
	else :
		PRED_list = PRED_list+ ['NA']
	#
	tmp_df = pd.DataFrame({
		'ROW_CID' : [ROW_CID],
		'COL_CID' : [COL_CID],
		'CELLO' : [CELLO],
		'PRED_RES' : outputs,
		})
	CELL_PRED_DF = pd.concat([CELL_PRED_DF, tmp_df])


CELL_PRED_DF.to_csv(SAVE_PATH+'PRED_{}.FINAL.csv'.format(Cell_name), index=False)

df.to_csv(‘existing.csv’, mode=’a’, index=False, header=False)

CELL_PRED_DF['ROW_CID'] = [a[0] for a in tuple_list]
CELL_PRED_DF['COL_CID'] = [a[1] for a in tuple_list]
CELL_PRED_DF['CELLO'] = [a[2] for a in tuple_list]
CELL_PRED_DF['PRED_RES'] = PRED_list
CELL_PRED_DF.to_csv('PRED_{}.csv'.format(Cell_name))


# 결과 확인은? 

import glob

CVCL_files = glob.glob('/st06/jiyeonH/11.TOX/DR_SPRING/trials/FINAL_VALIDATION/PRED_*')

fifis = []
for fifi in CVCL_files :
	pred_df = pd.read_csv(fifi, header =None)
	fifis.append(pred_df)


merged_CVCL = pd.concat(fifis, axis = 0)
merged_CVCL.columns = ['ROW_CID','COL_CID','CELLO','PRED_RES']
merged_CVCL = merged_CVCL.drop_duplicates()


import seaborn
    
fig = plt.figure(figsize=(30,15))
seaborn.set(style = 'whitegrid') 
seaborn.violinplot(x ='CELLO', y ='PRED_RES', data = merged_CVCL)
plt.tight_layout()
fig.savefig('{}/{}.png'.format(cell_path, 'test_cell_violin'), bbox_inches = 'tight')
plt.close()




merged_CVCL.sort_values('PRED_RES')




##################################################



for IND in range(len(DC_pairs)) : 
	if IND%100 == 0 : 
		print(str(IND)+'/'+ str(len(DC_pairs)))
		datetime.now()
	a,b = DC_pairs[IND]
	tmp_avail = [(a,b,c) for c in avail_cell_list]
	VAL_LIST = [mini_set for mini_set in tmp_avail if mini_set not in IN_DC_pairs]
	NOT_in_DC_pairs =  NOT_in_DC_pairs + VAL_LIST

