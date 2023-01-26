

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



#NETWORK_PATH = '/st06/jiyeonH/13.DD_SESS/HumanNetV3/'
#LINCS_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/' 
#DATA_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V2_FULL/'
#DC_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/'


NETWORK_PATH = '/home01/k020a01/01.Data/HumanNet/'
LINCS_PATH = '/home01/k020a01/01.Data/LINCS/'
DC_PATH = '/home01/k020a01/01.Data/DrugComb/'





print('NETWORK')

# NETWORK_PATH = '/st06/jiyeonH/13.DD_SESS/HumanNetV3/'

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








MJ_NAME = 'M3V2'
WORK_DATE = '23.01.12'
MISS_NAME = 'MIS2'


SAVE_PATH = '/home01/k020a01/02.VER3/{}_FULL_DATA/'.format(MJ_NAME)
# SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/{}_FULL/'.format(MJ_NAME)



file_name = 'M3V2_MISS2_FULL'
A_B_C_S_SET_ADD = pd.read_csv(SAVE_PATH+'{}.A_B_C_S_SET_ADD.csv'.format(file_name), low_memory=False)
MY_chem_A_feat = torch.load(SAVE_PATH+'{}.MY_chem_A_feat.pt'.format(file_name))
MY_chem_B_feat = torch.load(SAVE_PATH+'{}.MY_chem_B_feat.pt'.format(file_name))
MY_chem_A_adj = torch.load(SAVE_PATH+'{}.MY_chem_A_adj.pt'.format(file_name))
MY_chem_B_adj = torch.load(SAVE_PATH+'{}.MY_chem_B_adj.pt'.format(file_name))
MY_g_EXP_A = torch.load(SAVE_PATH+'{}.MY_g_EXP_A.pt'.format(file_name))
MY_g_EXP_B = torch.load(SAVE_PATH+'{}.MY_g_EXP_B.pt'.format(file_name))
MY_Target_1_A = torch.load(SAVE_PATH+'{}.MY_Target_1_A.pt'.format(file_name))
MY_Target_1_B = torch.load(SAVE_PATH+'{}.MY_Target_1_B.pt'.format(file_name))
MY_Target_2_A = torch.load(SAVE_PATH+'{}.MY_Target_2_A.pt'.format(file_name))
MY_Target_2_B = torch.load(SAVE_PATH+'{}.MY_Target_2_B.pt'.format(file_name))
MY_CellBase = torch.load(SAVE_PATH+'{}.MY_CellBase.pt'.format(file_name))
MY_syn = torch.load(SAVE_PATH+'{}.MY_syn.pt'.format(file_name))






# A_B_C_S SET filter check
WORK_NAME = 'WORK_7'


#MISS_filter = ['AOBO']
#MISS_filter = ['AOBO','AXBO','AOBX']
MISS_filter = ['AOBO','AXBO','AOBX','AXBX']


A_B_C_S_SET = A_B_C_S_SET_ADD[A_B_C_S_SET_ADD.SYN_OX == 'O']
A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.T2OX == 'O'] # old targets 
A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.type.isin(MISS_filter)]



# basal node exclude -> CCLE match 만 사용
CCLE_PATH = '/home01/k020a01/01.Data/CCLE/'
# CCLE_PATH = '/st06/jiyeonH/13.DD_SESS/CCLE.22Q1/'
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

A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.DrugCombCello.isin(ccle_cello_names)]



data_ind = list(A_B_C_S_SET.index)

MY_chem_A_feat_RE = MY_chem_A_feat[data_ind]
MY_chem_B_feat_RE = MY_chem_B_feat[data_ind]
MY_chem_A_adj_RE = MY_chem_A_adj[data_ind]
MY_chem_B_adj_RE = MY_chem_B_adj[data_ind]
MY_g_EXP_A_RE = MY_g_EXP_A[data_ind]
MY_g_EXP_B_RE = MY_g_EXP_B[data_ind]
MY_Target_A = copy.deepcopy(MY_Target_2_A)[data_ind]
MY_Target_B = copy.deepcopy(MY_Target_2_B)[data_ind]
MY_CellBase_RE = MY_CellBase[data_ind]
MY_syn_RE = MY_syn[data_ind]


A_B_C_S_SET = A_B_C_S_SET.reset_index(drop = True)











# cell line vector 

DC_CELL_DF2 = pd.read_csv(DC_PATH+'DC_CELL_INFO.csv', sep = '\t')

DC_CELL_info_filt = DC_CELL_DF2[DC_CELL_DF2.DrugCombCello.isin(A_B_C_S_SET.DrugCombCello)] # 38

A_B_C_S_SET_COH = pd.merge(A_B_C_S_SET, DC_CELL_info_filt[['DrugCombCello','DC_cellname']], on = 'DrugCombCello', how = 'left'  )




# sample number filter 

# 빈도 확인 

C_names = list(set(A_B_C_S_SET_COH.DC_cellname))

C_freq = [list(A_B_C_S_SET_COH.DC_cellname).count(a) for a in C_names]

C_df = pd.DataFrame({'cell' : C_names, 'freq' : C_freq })
C_df = C_df.sort_values('freq')


					fig, ax = plt.subplots(figsize=(30, 15))
					## fig, ax = plt.subplots(figsize=(40, 15))

					x_pos = [a*3 for a in range(C_df.shape[0])]
					ax.bar(x_pos, list(C_df['freq']))

					plt.xticks(x_pos, list(C_df['cell']), rotation=90, fontsize=18)

					for i in range(C_df.shape[0]):
						plt.annotate(str(int(list(C_df['freq'])[i])), xy=(x_pos[i], list(C_df['freq'])[i]), ha='center', va='bottom', fontsize=18)

					ax.set_ylabel('cell nums')
					ax.set_title('used cells')
					plt.tight_layout()

					plotname = 'total_cells'
					path = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V2_MIS2_W7/'
					path = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V2_MIS2_W10/'
					fig.savefig('{}/{}.png'.format(path, plotname), bbox_inches = 'tight')
					plt.close()

						# C_MJ = pd.merge(C_df, DC_CELL_info_filt[['DC_cellname','DrugCombCello','DrugCombCCLE']], left_on = 'cell', right_on = 'DC_cellname', how = 'left')
						# C_MJ.to_csv('/st06/jiyeonH/13.DD_SESS/01.PRJ2/cell_cut_example.csv')



# 그래서 결론적으로 cell one hot 은? 

C_freq_filter = C_df[C_df.freq > 200] ############ WORK 7 ##############
# C_freq_filter = C_df[C_df.freq > 500] ############ WORK 8 ##############
# C_freq_filter = C_df[C_df.freq > 1000] ############ WORK 9 ##############

A_B_C_S_SET_COH = A_B_C_S_SET_COH[A_B_C_S_SET_COH.DC_cellname.isin(C_freq_filter.cell)]

DC_CELL_info_filt_re = DC_CELL_info_filt[DC_CELL_info_filt.DC_cellname.isin(C_freq_filter.cell)]
DC_CELL_info_filt_re['cell_onehot'] = [a for a in range(len(set(DC_CELL_info_filt_re.DrugCombCello)))]

DC_CELL_info_filt_re = DC_CELL_info_filt_re.reset_index(drop = True)


data_ind = list(A_B_C_S_SET_COH.index)

MY_chem_A_feat_RE2 = MY_chem_A_feat_RE[data_ind]
MY_chem_B_feat_RE2 = MY_chem_B_feat_RE[data_ind]
MY_chem_A_adj_RE2 = MY_chem_A_adj_RE[data_ind]
MY_chem_B_adj_RE2 = MY_chem_B_adj_RE[data_ind]
MY_g_EXP_A_RE2 = MY_g_EXP_A_RE[data_ind]
MY_g_EXP_B_RE2 = MY_g_EXP_B_RE[data_ind]
MY_Target_A2 = copy.deepcopy(MY_Target_A)[data_ind]
MY_Target_B2 = copy.deepcopy(MY_Target_B)[data_ind]
MY_CellBase_RE2 = MY_CellBase_RE[data_ind]
MY_syn_RE2 = MY_syn_RE[data_ind]

# merge 전 후로 index 달라지므로 뒤에 넣어줬음 
A_B_C_S_SET_COH2 = pd.merge(A_B_C_S_SET_COH, DC_CELL_info_filt_re[['DC_cellname','cell_onehot']], on = 'DC_cellname', how='left')
cell_one_hot = torch.nn.functional.one_hot(torch.Tensor(A_B_C_S_SET_COH2['cell_onehot']).long())








###########################################################################################
###########################################################################################
###########################################################################################

print("LEARNING")


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





A_B_C_S_SET_SM = copy.deepcopy(A_B_C_S_SET_COH2)

A_B_C_S_SET_SM['ori_index'] = list(A_B_C_S_SET_SM.index)
aa = list(A_B_C_S_SET_SM['ROW_CAN_SMILES'])
bb = list(A_B_C_S_SET_SM['COL_CAN_SMILES'])
cc = list(A_B_C_S_SET_SM['DrugCombCello'])
A_B_C_S_SET_SM['SM_C_CHECK'] = [aa[i] + '___' + bb[i]+ '___' + cc[i] for i in range(A_B_C_S_SET_SM.shape[0])]


# get unique values, remove duplicates, but keep original counts
data_no_dup, counts = np.unique(list(A_B_C_S_SET_SM['SM_C_CHECK']), return_counts=True)
data_no_dup_cells = [setset.split('___')[2] for setset in data_no_dup]
data_nodup_df = pd.DataFrame({'setset' : data_no_dup.tolist(), 'cell' : data_no_dup_cells })





# 8 : 1 : 1 
# cell line 500개도 안되면 빼버리는 경우 
# cell line 500개도 안되면 빼버리는 경우 

grouped_df = data_nodup_df.groupby('cell')

CV_1_list = []; CV_2_list = []; CV_3_list = []; CV_4_list = []; CV_5_list = []
CV_6_list = []; CV_7_list = []; CV_8_list = []; CV_9_list = []; CV_10_list = []

for i, g in grouped_df:
	if len(g) > 200 :
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
		print(i)




CV_ND_INDS = {'CV0_train' : CV_1_list+ CV_2_list+CV_3_list+CV_4_list+ CV_5_list+CV_6_list+CV_7_list+CV_8_list, 'CV0_val' : CV_9_list,'CV0_test' : CV_10_list,
			'CV1_train' : CV_3_list+CV_4_list+ CV_5_list+CV_6_list+CV_7_list+CV_8_list+CV_9_list+CV_10_list, 'CV1_val' : CV_1_list,'CV1_test' : CV_2_list,
			'CV2_train' : CV_5_list+CV_6_list+CV_7_list+CV_8_list+CV_9_list+CV_10_list+CV_1_list+ CV_2_list, 'CV2_val' : CV_3_list,'CV2_test' : CV_4_list,
			'CV3_train' : CV_7_list+CV_8_list+CV_9_list+CV_10_list+CV_1_list+ CV_2_list+CV_3_list+CV_4_list, 'CV3_val' : CV_5_list,'CV3_test' : CV_6_list,
			'CV4_train' : CV_9_list+CV_10_list+CV_1_list+ CV_2_list+CV_3_list+CV_4_list+CV_5_list+CV_6_list, 'CV4_val' : CV_7_list,'CV4_test' : CV_8_list }




# use just index 
# normalizing 과정에 대한 고민 필요 ... GCN 이니까 상관 없지 않을까
def prepare_data_GCN(A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, 
MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, 
cell_one_hot, MY_syn_RE2, norm ) : 
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
	chem_feat_A_train = MY_chem_A_feat_RE2[train_ind]; chem_feat_A_val = MY_chem_A_feat_RE2[val_ind]; chem_feat_A_test = MY_chem_A_feat_RE2[test_ind]
	chem_feat_B_train = MY_chem_B_feat_RE2[train_ind]; chem_feat_B_val = MY_chem_B_feat_RE2[val_ind]; chem_feat_B_test = MY_chem_B_feat_RE2[test_ind]
	chem_adj_A_train = MY_chem_A_adj_RE2[train_ind]; chem_adj_A_val = MY_chem_A_adj_RE2[val_ind]; chem_adj_A_test = MY_chem_A_adj_RE2[test_ind]
	chem_adj_B_train = MY_chem_B_adj_RE2[train_ind]; chem_adj_B_val = MY_chem_B_adj_RE2[val_ind]; chem_adj_B_test = MY_chem_B_adj_RE2[test_ind]
	gene_A_train = MY_g_EXP_A_RE2[train_ind]; gene_A_val = MY_g_EXP_A_RE2[val_ind]; gene_A_test = MY_g_EXP_A_RE2[test_ind]
	gene_B_train = MY_g_EXP_B_RE2[train_ind]; gene_B_val = MY_g_EXP_B_RE2[val_ind]; gene_B_test = MY_g_EXP_B_RE2[test_ind]
	target_A_train = MY_Target_A2[train_ind]; target_A_val = MY_Target_A2[val_ind]; target_A_test = MY_Target_A2[test_ind]
	target_B_train = MY_Target_B2[train_ind]; target_B_val = MY_Target_B2[val_ind]; target_B_test = MY_Target_B2[test_ind]
	cell_basal_train = MY_CellBase_RE2[train_ind]; cell_basal_val = MY_CellBase_RE2[val_ind]; cell_basal_test = MY_CellBase_RE2[test_ind]
	cell_train = cell_one_hot[train_ind]; cell_val = cell_one_hot[val_ind]; cell_test = cell_one_hot[test_ind]
	syn_train = MY_syn_RE2[train_ind]; syn_val = MY_syn_RE2[val_ind]; syn_test = MY_syn_RE2[test_ind]
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
	train_data['GENE_A'] = torch.concat([gene_A_train, gene_B_train], axis = 0)
	val_data['GENE_A'] = gene_A_val
	test_data['GENE_A'] = gene_A_test
	#
	train_data['GENE_B'] = torch.concat([gene_B_train, gene_A_train], axis = 0)
	val_data['GENE_B'] = gene_B_val
	test_data['GENE_B'] = gene_B_test
	#
	train_data['TARGET_A'] = torch.concat([target_A_train, target_B_train], axis = 0)
	val_data['TARGET_A'] = target_A_val
	test_data['TARGET_A'] = target_A_test
	#
	train_data['TARGET_B'] = torch.concat([target_B_train, target_A_train], axis = 0)
	val_data['TARGET_B'] = target_B_val
	test_data['TARGET_B'] = target_B_test
	#   #
	train_data['cell_BASAL'] = torch.concat((cell_basal_train, cell_basal_train), axis=0)
	val_data['cell_BASAL'] = cell_basal_val
	test_data['cell_BASAL'] = cell_basal_test
	##
	train_data['cell'] = torch.concat((cell_train, cell_train), axis=0)
	val_data['cell'] = cell_val
	test_data['cell'] = cell_test
	#            
	train_data['y'] = torch.concat((syn_train, syn_train), axis=0)
	val_data['y'] = syn_val
	test_data['y'] = syn_test
	#
	print(train_data['drug1_feat'].shape, flush=True)
	print(val_data['drug1_feat'].shape, flush=True)
	print(test_data['drug1_feat'].shape, flush=True)
	return train_data, val_data, test_data







class DATASET_GCN_W_FT(Dataset):
	def __init__(self, gcn_drug1_F, gcn_drug2_F, gcn_drug1_ADJ, gcn_drug2_ADJ, 
	gcn_gene_A, gcn_gene_B, target_A, target_B, cell_basal, gcn_adj, gcn_adj_weight, 
	cell_info, syn_ans ):
		self.gcn_drug1_F = gcn_drug1_F
		self.gcn_drug2_F = gcn_drug2_F
		self.gcn_drug1_ADJ = gcn_drug1_ADJ
		self.gcn_drug2_ADJ = gcn_drug2_ADJ
		self.gcn_gene_A = gcn_gene_A
		self.gcn_gene_B = gcn_gene_B
		self.target_A = target_A
		self.target_B = target_B
		self.cell_basal = cell_basal
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
		adj_re_A = self.gcn_drug1_ADJ[index].long().to_sparse().indices()
		adj_re_B = self.gcn_drug2_ADJ[index].long().to_sparse().indices()
		#
		FEAT_A = torch.Tensor(np.array([ self.gcn_gene_A[index].squeeze().tolist() , self.target_A[index].tolist(), self.cell_basal[index].tolist()]).T)
		FEAT_B = torch.Tensor(np.array([ self.gcn_gene_B[index].squeeze().tolist() , self.target_B[index].tolist(), self.cell_basal[index].tolist()]).T)
		#
		return self.gcn_drug1_F[index], self.gcn_drug2_F[index], adj_re_A, adj_re_B, FEAT_A, FEAT_B, self.gcn_adj, self.gcn_adj_weight , self.cell_info[index], self.syn_ans[index]



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
	#
	for drug1_f, drug2_f, drug1_adj, drug2_adj, expA, expB, exp_adj, exp_adj_w, cell, y in batch :
		drug1_f_list.append(drug1_f)
		drug2_f_list.append(drug2_f)
		drug1_adj_list.append(drug1_adj+DRUG_1_num_nodes_seen)
		drug2_adj_list.append(drug2_adj+DRUG_2_num_nodes_seen)
		expA_list.append(expA)
		expB_list.append(expB)
		exp_adj_list.append(exp_adj+EXP_num_nodes_seen)
		exp_adj_w_list.append(exp_adj_w)
		y_list.append(torch.Tensor(y))
		cell_list.append(torch.Tensor(cell))
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
train_data, val_data, test_data = prepare_data_GCN(A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, 
MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, 
cell_one_hot, MY_syn_RE2, norm)


# WEIGHT 
ys = train_data['y'].squeeze().tolist()
min_s = np.amin(ys)
loss_weight = np.log(train_data['y'] - min_s + np.e)
JY_IDX_WEIGHT_T = torch.Tensor(JY_IDX_WEIGHT).view(1,-1)








# DATA check 
T_train = DATASET_GCN_W_FT(
	torch.Tensor(train_data['drug1_feat']), torch.Tensor(train_data['drug2_feat']), 
	torch.Tensor(train_data['drug1_adj']), torch.Tensor(train_data['drug2_adj']),
	torch.Tensor(train_data['GENE_A']), torch.Tensor(train_data['GENE_B']), 
	torch.Tensor(train_data['TARGET_A']), torch.Tensor(train_data['TARGET_B']), torch.Tensor(train_data['cell_BASAL']), 
	JY_ADJ_IDX, JY_IDX_WEIGHT_T, 
	train_data['cell'].float(),
	torch.Tensor(train_data['y'])
	)

T_val = DATASET_GCN_W_FT(
	torch.Tensor(val_data['drug1_feat']), torch.Tensor(val_data['drug2_feat']), 
	torch.Tensor(val_data['drug1_adj']), torch.Tensor(val_data['drug2_adj']),
	torch.Tensor(val_data['GENE_A']), torch.Tensor(val_data['GENE_B']), 
	torch.Tensor(val_data['TARGET_A']), torch.Tensor(val_data['TARGET_B']), torch.Tensor(val_data['cell_BASAL']), 
	JY_ADJ_IDX, JY_IDX_WEIGHT_T, 
	val_data['cell'].float(),
	torch.Tensor(val_data['y'])
	)
	
T_test = DATASET_GCN_W_FT(
	torch.Tensor(test_data['drug1_feat']), torch.Tensor(test_data['drug2_feat']), 
	torch.Tensor(test_data['drug1_adj']), torch.Tensor(test_data['drug2_adj']),
	torch.Tensor(test_data['GENE_A']), torch.Tensor(test_data['GENE_B']), 
	torch.Tensor(test_data['TARGET_A']), torch.Tensor(test_data['TARGET_B']), torch.Tensor(test_data['cell_BASAL']), 
	JY_ADJ_IDX, JY_IDX_WEIGHT_T, 
	test_data['cell'].float(),
	torch.Tensor(test_data['y'])
	)

RAY_train = ray.put(T_train)
RAY_val = ray.put(T_val)
RAY_test = ray.put(T_test)
RAY_loss_weight = ray.put(loss_weight)





							# DATA check 
							T_train = DATASET_GCN_W_FT(
								torch.Tensor(train_data['drug1_feat'][0:128]), torch.Tensor(train_data['drug2_feat'][0:128]), 
								torch.Tensor(train_data['drug1_adj'][0:128]), torch.Tensor(train_data['drug2_adj'][0:128]),
								torch.Tensor(train_data['GENE_A'][0:128]), torch.Tensor(train_data['GENE_B'][0:128]), 
								torch.Tensor(train_data['TARGET_A'][0:128]), torch.Tensor(train_data['TARGET_B'][0:128]), torch.Tensor(train_data['cell_BASAL'][0:128]), 
								JY_ADJ_IDX, JY_IDX_WEIGHT_T, 
								train_data['cell'][0:128].float(),
								torch.Tensor(train_data['y'][0:128])
								)

							T_val = DATASET_GCN_W_FT(
								torch.Tensor(val_data['drug1_feat'][0:128]), torch.Tensor(val_data['drug2_feat'][0:128]), 
								torch.Tensor(val_data['drug1_adj'][0:128]), torch.Tensor(val_data['drug2_adj'][0:128]),
								torch.Tensor(val_data['GENE_A'][0:128]), torch.Tensor(val_data['GENE_B'][0:128]), 
								torch.Tensor(val_data['TARGET_A'][0:128]), torch.Tensor(val_data['TARGET_B'][0:128]), torch.Tensor(val_data['cell_BASAL'][0:128]), 
								JY_ADJ_IDX, JY_IDX_WEIGHT_T, 
								val_data['cell'][0:128].float(),
								torch.Tensor(val_data['y'][0:128])
								)
								
							T_test = DATASET_GCN_W_FT(
								torch.Tensor(test_data['drug1_feat'][0:128]), torch.Tensor(test_data['drug2_feat'][0:128]), 
								torch.Tensor(test_data['drug1_adj'][0:128]), torch.Tensor(test_data['drug2_adj'][0:128]),
								torch.Tensor(test_data['GENE_A'][0:128]), torch.Tensor(test_data['GENE_B'][0:128]), 
								torch.Tensor(test_data['TARGET_A'][0:128]), torch.Tensor(test_data['TARGET_B'][0:128]), torch.Tensor(test_data['cell_BASAL'][0:128]), 
								JY_ADJ_IDX, JY_IDX_WEIGHT_T, 
								test_data['cell'][0:128].float(),
								)

							RAY_train = ray.put(T_train)
							RAY_val = ray.put(T_val)
							RAY_test = ray.put(T_test)
							RAY_loss_weight = ray.put(loss_weight)

















print('MAIN')

class MY_expGCN_parallel_model(torch.nn.Module):
	def __init__(self, G_layer_chem, G_indim_chem, G_hiddim_chem, 
	G_layer_exp, G_indim_exp, G_hiddim_exp, 
	layers_1, layers_2, layers_3, cell_dim ,
	out_dim, inDrop, drop):
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
		for conv in self.G_convs_1_exp :
			conv.reset_parameters()
		for bns in self.G_bns_1_exp :
			bns.reset_parameters()
		for conv in self.G_convs_2_chem :
			conv.reset_parameters()
		for bns in self.G_bns_2_chem :
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
	use_cuda = False  # ,=#  #  #  #  #  # True
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
	snp_layers = [config["feat_size_3"] ,config["feat_size_4"]]
	inDrop = config["dropout_1"]
	Drop = config["dropout_2"]
	#
	#  
	MM_MODEL = MY_expGCN_parallel_model(
			config["G_chem_layer"], T_train.gcn_drug1_F.shape[-1] , config["G_chem_hdim"],      # G_layer_chem, G_indim_chem, G_hiddim_chem, 
			config["G_exp_layer"], 3 , config["G_exp_hdim"],      # G_layer_exp, G_indim_exp, G_hiddim_exp, 
			dsn1_layers, dsn2_layers, snp_layers,      # drug 1 layers, drug 2 layers, merged layer, 
			len(set(A_B_C_S_SET_SM.DrugCombCello)), 1,      # cell_dim ,out_dim,
			inDrop, Drop      # inDrop, drop
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
	val_pearson_corr_all = []
	val_spearman_corr_all = []
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
			expA = expA.view(-1,3)#### 다른점 
			expB = expB.view(-1,3)#### 다른점 
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
		ans_list = []
		pred_list = []
		for batch_idx_v, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y) in enumerate(loaders['eval']):
			expA = expA.view(-1,3)#### 다른점 
			expB = expB.view(-1,3)#### 다른점 
			adj_w = adj_w.squeeze()
			# move to GPU
			if use_cuda:
				drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell = drug1_f.cuda(), drug2_f.cuda(), drug1_a.cuda(), drug2_a.cuda(), expA.cuda(), expB.cuda(), adj.cuda(), adj_w.cuda(), y.cuda(), cell.cuda()
			## update the average validation loss
			output = MM_MODEL(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y)
			MSE = torch.nn.MSELoss()
			loss = MSE(output, y) # train 이 아니라서 weight 안넣어줌. 그냥 nn.MSE 넣어주기 
			# update average validation loss 
			valid_loss = valid_loss + loss.item()
			pred_list = pred_list + output.squeeze().tolist()
			ans_list = ans_list + y.squeeze().tolist()
		#
		# calculate average losses
		TRAIN_LOSS = train_loss/(batch_idx_t+1)
		train_loss_all.append(TRAIN_LOSS)
		VAL_LOSS = valid_loss/(batch_idx_v+1)
		valid_loss_all.append(VAL_LOSS)
		VAL_SCOR, _ = stats.spearmanr(pred_list, ans_list)
		VAL_PCOR, _ = stats.pearsonr(pred_list, ans_list)
		val_spearman_corr_all.append(VAL_SCOR) 
		val_pearson_corr_all.append(VAL_PCOR)
		#
		# print training/validation statistics 
		#
		done = datetime.now()
		time_spent = done-now
		#
		with tune.checkpoint_dir(step = epoch) as checkpoint_dir:
			trial_name = '_'.join(checkpoint_dir.split('/')[-2].split('_')[0:4])
			print('trial : {}, epoch : {}, TrainLoss : {}, ValLoss : {}'.format(trial_name, epoch, TRAIN_LOSS, VAL_LOSS), flush=True)
			path = os.path.join(checkpoint_dir, "checkpoint")
			torch.save((MM_MODEL.state_dict(), optimizer.state_dict()), path)
			torch.save(MM_MODEL.state_dict(), './model.pth')
		tune.report(TrainLoss= TRAIN_LOSS,  ValLoss=VAL_LOSS, SCOR = VAL_SCOR, PCOR = VAL_PCOR )
	#
	print("Finished Training")
 






def RAY_TEST_MODEL(my_config, model_path, model_name, PRJ_ATH, PRJ_NAME, MISS_NAME, number): 
	use_cuda = False #  #   #  #  #  #True
	T_test = ray.get(RAY_test)
	Test_loader = torch.utils.data.DataLoader(T_test, batch_size = my_config['config/batch_size'].item(), collate_fn = graph_collate_fn, shuffle =False, num_workers=my_config['config/n_workers'].item())
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
				G_chem_layer, T_test.gcn_drug1_F.shape[-1] , G_chem_hdim,
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
		state_dict = torch.load(os.path.join(model_path, model_name))
	else:
		state_dict = torch.load(os.path.join(model_path, model_name), map_location=torch.device('cpu'))
	#
	print("state_dict_done", flush = True)
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
	Y_list = T_test.syn_ans.squeeze().tolist()
	with torch.no_grad():
		best_model.eval()
		for batch_idx_t, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y) in enumerate(Test_loader):
			expA = expA.view(-1,3)
			expB = expB.view(-1,3)
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
	R__1 , R__2 = jy_corrplot(PRED_list, Y_list, PRJ_PATH,'P.{}.{}.{}_model'.format(PRJ_NAME, MISS_NAME, number) )
	return  R__T, R__1, R__2










def plot_corr(train_corr, valid_corr, path, plotname):
	fig = plt.figure(figsize=(10,8))
	plt.plot(range(1,len(train_corr)+1),train_corr, label='Training Corr')
	plt.plot(range(1,len(valid_corr)+1),valid_corr,label='Validation Corr')
	plt.xlabel('epochs')
	plt.ylabel('corr')
	plt.ylim(0, math.ceil(max(train_corr+valid_corr))) # 일정한 scale
	plt.xlim(0, len(train_corr)+1) # 일정한 scale
	plt.grid(True)
	plt.legend()
	plt.tight_layout()
	fig.savefig('{}/{}.corr_plot.png'.format(path, plotname), bbox_inches = 'tight')



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






def final_result(R_1_V, R_1_T, R_1_1, R_1_2, R_2_V, R_2_T, R_2_1, R_2_2, R_3_V, R_3_T, R_3_1, R_3_2,
R_4_V, R_4_T, R_4_1, R_4_2, R_5_V, R_5_T, R_5_1, R_5_2, R_6_V, R_6_T, R_6_1, R_6_2) :
	print('---1---', flush=True)
	print('- Val MSE : {:.2f}'.format(R_1_V), flush=True)
	print('- Test MSE : {:.2f}'.format(R_1_T), flush=True)
	print('- Test Pearson : {:.2f}'.format(R_1_1), flush=True)
	print('- Test Spearman : {:.2f}'.format(R_1_2), flush=True)
	print('---2---', flush=True)
	print('- Val MSE : {:.2f}'.format(R_2_V), flush=True)
	print('- Test MSE : {:.2f}'.format(R_2_T), flush=True)
	print('- Test Pearson : {:.2f}'.format(R_2_1), flush=True)
	print('- Test Spearman : {:.2f}'.format(R_2_2), flush=True)
	print('---3---', flush=True)
	print('- Val MSE : {:.2f}'.format(R_3_V), flush=True)
	print('- Test MSE : {:.2f}'.format(R_3_T), flush=True)
	print('- Test Pearson : {:.2f}'.format(R_3_1), flush=True)
	print('- Test Spearman : {:.2f}'.format(R_3_2), flush=True)
	print('---4---', flush=True)
	print('- Val SCOR : {:.2f}'.format(R_4_V), flush=True)
	print('- Test MSE : {:.2f}'.format(R_4_T), flush=True)
	print('- Test Pearson : {:.2f}'.format(R_4_1), flush=True)
	print('- Test Spearman : {:.2f}'.format(R_4_2), flush=True)
	print('---5---', flush=True)
	print('- Val SCOR : {:.2f}'.format(R_5_V), flush=True)
	print('- Test MSE : {:.2f}'.format(R_5_T), flush=True)
	print('- Test Pearson : {:.2f}'.format(R_5_1), flush=True)
	print('- Test Spearman : {:.2f}'.format(R_5_2), flush=True)
	print('---6---', flush=True)
	print('- Val SCOR : {:.2f}'.format(R_6_V), flush=True)
	print('- Test MSE : {:.2f}'.format(R_6_T), flush=True)
	print('- Test Pearson : {:.2f}'.format(R_6_1), flush=True)
	print('- Test Spearman : {:.2f}'.format(R_6_2), flush=True)



from ray.tune import Analysis
# from ray.tune import ExperimentAnalysis


def MAIN(ANAL_name, WORK_PATH, PRJ_PATH, PRJ_NAME, MISS_NAME, num_samples= 10, max_num_epochs=1000, grace_period = 150, cpus_per_trial = 16, gpus_per_trial = 1):
	# OPTUNA CONFIGURE 
	CONFIG={
		'n_workers' : tune.choice([cpus_per_trial]),
		"epoch" : tune.choice([max_num_epochs]),
		"G_chem_layer" : tune.choice([3]), # 
		"G_exp_layer" : tune.choice([3]), # 
		"G_chem_hdim" : tune.choice([32]), # 
		"G_exp_hdim" : tune.choice([32]), # 
		"batch_size" : tune.choice([ 128]), # CPU 니까 # 256, 
		"feat_size_0" : tune.choice([ 256  ]), # 4096, 2048, 1024, 512, 256, 128, 64, 32
		"feat_size_1" : tune.choice([ 128 ]),# 4096, 2048, 1024, 512, 256, 128, 64, 32
		"feat_size_2" : tune.choice([ 64 ]),# 4096, 2048, 1024, 512, 256, 128, 64, 32
		"feat_size_3" : tune.choice([ 128 ]),# 4096, 2048, 1024, 512, 256, 128, 64, 32
		"feat_size_4" : tune.choice([ 32 ]),# 4096, 2048, 1024, 512, 256, 128, 64, 32
		"dropout_1" : tune.choice([0.2]), # 0.01, 0.2, 0.5, 0.8
		"dropout_2" : tune.choice([0.2]), # 0.01, 0.2, 0.5, 0.8
		"lr" : tune.choice([ 0.0001]),# 0.00001, 0.0001, 0.001
	}
	#
	#pickle.dumps(trainable)
	reporter = CLIReporter(
		metric_columns=["TrainLoss", "ValLoss", 'SCOR','PCOR', "training_iteration"])
	#
	optuna_search = OptunaSearch(metric="ValLoss", mode="min")
	#
	ASHA_scheduler = tune.schedulers.ASHAScheduler(
		time_attr='training_iteration', metric="SCOR", mode="max", max_t= max_num_epochs, grace_period = grace_period )
	#
	#
	ANALYSIS = tune.run( # 끝내지 않음 
		tune.with_parameters(RAY_MY_train),
		name = ANAL_name,
		num_samples=num_samples,
		config=CONFIG,
		resources_per_trial={'cpu': cpus_per_trial}, # , ,'gpu' : gpus_per_trial 
		progress_reporter = reporter,
		search_alg = optuna_search,
		scheduler = ASHA_scheduler,
		resume = True
	)
	best_trial = ANALYSIS.get_best_trial("ValLoss", "min", "last")
	print("Best trial config: {}".format(best_trial.config), flush=True)
	print("Best trial final validation loss: {}".format(
	best_trial.last_result["ValLoss"]), flush=True)
	#
	#
	# TEST
	anal_dir = "/home01/k020a01/ray_results/{}/".format(ANAL_name)
	# anal_df = Analysis(anal_dir)
	anal_df = ExperimentAnalysis(anal_dir)
	ANA_DF = anal_df.dataframe()
	ANA_ALL_DF = anal_df.trial_dataframes
	#
	#
	# (1) MSE Min
	#	
	#
	min(ANA_DF.sort_values('ValLoss')['ValLoss'])
	DF_KEY = list(ANA_DF.sort_values('ValLoss')['logdir'])[0]
	print("MS MIN DF_KEY : ")
	print(DF_KEY, flush=True)
	mini_df = ANA_ALL_DF[DF_KEY]
	my_config = ANA_DF[ANA_DF.logdir==DF_KEY]
	R_1_V = list(ANA_DF.sort_values('ValLoss')['ValLoss'])[0]
	R_1_T, R_1_1, R_1_2 = RAY_TEST_MODEL(my_config, DF_KEY, 'model.pth', PRJ_PATH, MJ_NAME, MISS_NAME+'_'+WORK_NAME, 'M1')
	plot_loss(list(mini_df.TrainLoss), list(mini_df.ValLoss), PRJ_PATH, '{}_{}_{}_VAL_BestLast'.format(MJ_NAME, MISS_NAME, WORK_NAME)  )
	#
	#
	# (2) best final's checkpoint
	#	
	#
	cck_num =mini_df[mini_df.ValLoss==min(mini_df.ValLoss)].index.item()
	checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
	TOPVAL_PATH = DF_KEY + checkpoint
	print('best final check', flush=True)
	print(TOPVAL_PATH, flush=True)
	R_2_V = min(mini_df.ValLoss)
	R_2_V
	R_2_T, R_2_1, R_2_2 = RAY_TEST_MODEL(my_config, TOPVAL_PATH, 'checkpoint', PRJ_PATH, MJ_NAME, MISS_NAME+'_'+WORK_NAME, 'M2')
	#
	#
	# 3) total checkpoint best 
	#	
	#
	import numpy as np
	TOT_min = np.Inf
	TOT_key = ""
	for key in ANA_ALL_DF.keys():
		trial_min = min(ANA_ALL_DF[key]['ValLoss'])
		if trial_min < TOT_min :
			TOT_min = trial_min
			TOT_key = key
	#
	print('best val', flush=True)
	print(TOT_key, flush=True)
	mini_df = ANA_ALL_DF[TOT_key]
	my_config = ANA_DF[ANA_DF.logdir==TOT_key]
	cck_num =mini_df[mini_df.ValLoss==min(mini_df.ValLoss)].index.item()
	checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
	TOPVAL_PATH = TOT_key + checkpoint
	print('best val check', flush=True)
	print(TOPVAL_PATH, flush=True)
	R_3_V = min(mini_df.ValLoss)
	R_3_V
	R_3_T, R_3_1, R_3_2 = RAY_TEST_MODEL(my_config, TOPVAL_PATH, 'checkpoint', PRJ_PATH, MJ_NAME, MISS_NAME+'_'+WORK_NAME, 'M4')
	plot_loss(list(mini_df.TrainLoss), list(mini_df.ValLoss), PRJ_PATH, '{}_{}_{}_VAL_TOT'.format(MJ_NAME, MISS_NAME, WORK_NAME)  )
	#
	#
	# (4) best SCOR final
	#	
	#
	max_cor = max(ANA_DF.sort_values('SCOR')['SCOR'])
	DF_KEY = ANA_DF[ANA_DF.SCOR == max_cor]['logdir'].item()
	print('best SCOR final', flush=True)
	print(DF_KEY, flush=True)
	mini_df = ANA_ALL_DF[DF_KEY]
	my_config = ANA_DF[ANA_DF.logdir==DF_KEY]
	R_4_V = max_cor
	R_4_V
	R_4_T, R_4_1, R_4_2 = RAY_TEST_MODEL(my_config, DF_KEY, 'model.pth', PRJ_PATH, MJ_NAME, MISS_NAME+'_'+WORK_NAME, 'C1')
	plot_loss(list(mini_df.TrainLoss), list(mini_df.ValLoss), PRJ_PATH, '{}_{}_{}_SCOR_BestLast'.format(MJ_NAME, MISS_NAME, WORK_NAME)  )
	#
	#
	# (5) BEST cor final 내에서의 max cor 
	#	
	#
	cck_num =mini_df[mini_df.SCOR==max(mini_df.SCOR)].index.item()
	checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
	TOPCOR_PATH = DF_KEY + checkpoint
	print('best final check', flush=True)
	print(TOPCOR_PATH, flush=True)
	R_5_V = max(mini_df.SCOR)
	R_5_V
	R_5_T, R_5_1, R_5_2 = RAY_TEST_MODEL(my_config, TOPCOR_PATH, 'checkpoint', PRJ_PATH, MJ_NAME, MISS_NAME+'_'+WORK_NAME, 'C2')
	#
	#
	# (6) 그냥 최고 corr 
	#	
	#
	import numpy as np
	TOT_max = -np.Inf
	TOT_key = ""
	for key in ANA_ALL_DF.keys():
		trial_max = max(ANA_ALL_DF[key]['SCOR'])
		if trial_max > TOT_max :
			TOT_max = trial_max
			TOT_key = key
	#
	print('best cor', flush=True)
	print(TOT_key, flush=True)
	mini_df = ANA_ALL_DF[TOT_key]
	my_config = ANA_DF[ANA_DF.logdir==TOT_key]
	cck_num =mini_df[mini_df.SCOR==max(mini_df.SCOR)].index.item()
	checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
	TOPCOR_PATH = TOT_key + checkpoint
	print('best cor check', flush=True)
	print(TOPCOR_PATH, flush=True)
	R_6_V = max(mini_df.SCOR)
	R_6_V
	R_6_T, R_6_1, R_6_2 = RAY_TEST_MODEL(my_config, TOPCOR_PATH, "checkpoint", PRJ_PATH, MJ_NAME, MISS_NAME+'_'+WORK_NAME, 'C4')
	#
	plot_loss(list(mini_df.TrainLoss), list(mini_df.ValLoss), PRJ_PATH, '{}_{}_{}_SCOR_TOT'.format(MJ_NAME, MISS_NAME, WORK_NAME)   )
	final_result(R_1_V, R_1_T, R_1_1, R_1_2, R_2_V, R_2_T, R_2_1, R_2_2, R_3_V, R_3_T, R_3_1, R_3_2,
R_4_V, R_4_T, R_4_1, R_4_2, R_5_V, R_5_T, R_5_1, R_5_2, R_6_V, R_6_T, R_6_1, R_6_2)
	return ANALYSIS




CONFIG={
		'n_workers' : tune.choice([cpus_per_trial]),
		"epoch" : tune.choice([max_num_epochs]),
		"G_chem_layer" : tune.choice([2, 3, 4]), # 
		"G_exp_layer" : tune.choice([2, 3, 4]), # 
		"G_chem_hdim" : tune.choice([256, 128, 64, 32, 16, 8]), # 
		"G_exp_hdim" : tune.choice([64, 32, 16, 8, 4]), # 
		"batch_size" : tune.choice([  128, 64, 32, 16]), # CPU 니까 # 256, 
		"feat_size_0" : tune.choice([ 256, 128, 64, 32, 16, 8 ]), # 4096, 2048, 1024, 512, 256, 128, 64, 32
		"feat_size_1" : tune.choice([ 256, 128, 64, 32, 16, 8 ]),# 4096, 2048, 1024, 512, 256, 128, 64, 32
		"feat_size_2" : tune.choice([ 256, 128, 64, 32, 16, 8 ]),# 4096, 2048, 1024, 512, 256, 128, 64, 32
		"feat_size_3" : tune.choice([ 256, 128, 64, 32, 16, 8 ]),# 4096, 2048, 1024, 512, 256, 128, 64, 32
		"feat_size_4" : tune.choice([ 256, 128, 64, 32, 16, 8 ]),# 4096, 2048, 1024, 512, 256, 128, 64, 32
		"dropout_1" : tune.choice([0.01, 0.2, 0.5, 0.8]), # 0.01, 0.2, 0.5, 0.8
		"dropout_2" : tune.choice([0.01, 0.2, 0.5, 0.8]), # 0.01, 0.2, 0.5, 0.8
		"lr" : tune.choice([0.00001, 0.0001, 0.001]),# 0.00001, 0.0001, 0.001
	}


WORK_PATH = '/home01/k020a01/02.VER3/{}_{}_{}/'.format(MJ_NAME, MISS_NAME, WORK_NAME )
# WORK_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/{}_{}_{}/'.format(MJ_NAME, MISS_NAME, WORK_NAME )



# ANAL_name, WORK_PATH, PRJ_PATH, PRJ_NAME, MISS_NAME, num_samples= 10, max_num_epochs=1000, grace_period = 150, cpus_per_trial = 16, gpus_per_trial = 1
# cpu test (INT1 64)
MAIN('PRJ02.{}.{}.{}.{}'.format(WORK_DATE, MJ_NAME, MISS_NAME, WORK_NAME), WORK_PATH, WORK_PATH, WORK_NAME, MISS_NAME, 4, 10, 1, 16, 1)
# MAIN('PRJ02.{}.{}.{}.{}'.format(WORK_DATE, MJ_NAME, MISS_NAME, WORK_NAME), WORK_PATH, WORK_PATH, WORK_NAME, MISS_NAME, 4, 1000, 100, 8, 1)
# MAIN('PRJ02.{}.{}.{}.{}'.format(WORK_DATE, MJ_NAME, MISS_NAME, WORK_NAME), WORK_PATH, WORK_PATH, WORK_NAME, MISS_NAME, 1, 1000, 100, 32, 1)


# GPU 8 real
MAIN('PRJ02.{}.{}.{}.{}'.format(WORK_DATE, MJ_NAME, MISS_NAME, WORK_NAME), WORK_PATH, WORK_PATH, WORK_NAME, MISS_NAME, 100, 1000, 150, 16, 1)


# GPU 8 real
MAIN('PRJ02.{}.{}.{}.{}'.format(WORK_DATE, MJ_NAME, MISS_NAME, WORK_NAME), WORK_PATH, WORK_PATH, WORK_NAME, MISS_NAME, 100, 1000, 150, 16, 1)





################## GPU ###################
################## GPU ###################
################## GPU ###################
################## GPU ###################


import pandas as pd 
import matplotlib.pyplot as plt
from ray.tune import ExperimentAnalysis
from ray.tune import Analysis
import pickle
import math
import torch
import os 


WORK_DATE = '23.01.16'
PRJ_NAME = 'M3V2'
MISS_NAME = 'MIS2'
WORK_NAME = 'WORK_7'

WORK_DATE = '23.01.18'
PRJ_NAME = 'M3V2'
MISS_NAME = 'MIS2'
WORK_NAME = 'WORK_8'

WORK_DATE = '23.01.20'
PRJ_NAME = 'M3V2'
MISS_NAME = 'MIS2'
WORK_NAME = 'WORK_10'


































