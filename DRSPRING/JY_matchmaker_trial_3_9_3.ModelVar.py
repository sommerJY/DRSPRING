
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

import numpy as np
from rdkit import Chem
from rdkit.Chem.QED import qed
import sys
import os
import pandas as pd


seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


WORK_PATH = '/home01/k006a01/PRJ.01/TRIAL_3.9/'
DC_PATH = '/home01/k006a01/01.DATA/DrugComb/'
IDK_PATH = '/home01/k006a01/01.DATA/IDK/'
LINCS_PATH = '/home01/k006a01/01.DATA/LINCS/' #
TARGET_PATH = '/home01/k006a01/01.DATA/TARGET/'


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
DC_tmp_DF2=pd.read_csv(DC_PATH+'TOTAL_BLOCK.csv', low_memory=False)
DC_DATA_filter = DC_tmp_DF2[['drug_row_id','drug_col_id','cell_line_id','synergy_loewe']]
DC_DATA_filter2 = DC_DATA_filter.drop_duplicates()
DC_DATA_filter3 = DC_DATA_filter2[DC_DATA_filter2.drug_col_id>0]
DC_DATA_filter4 = DC_DATA_filter3[DC_DATA_filter3.drug_row_id>0]
DC_DATA_filter4.cell_line_id # unique 295

# Drug DATA
with open(DC_PATH+'drugs.json') as json_file :
	DC_DRUG =json.load(json_file)

DC_DRUG_K = list(DC_DRUG[0].keys())
DC_DRUG_DF = pd.DataFrame(columns=DC_DRUG_K)

for DD in range(1,len(DC_DRUG)):
	tmpdf = pd.DataFrame({k:[DC_DRUG[DD][k]] for k in DC_DRUG_K})
	DC_DRUG_DF = pd.concat([DC_DRUG_DF, tmpdf], axis = 0)



# cid filter 
DC_DRUG_DF2 = DC_DRUG_DF[['id','dname','cid']]
# dnames = list(DC_DRUG_DF2['dname']) # 그래 이부분때문에 다시 다운로드 받은거였음. 이게 문제가 drug 마다 CID 가 여러개 붙는 애들이 있어서
DC_DRUG_DF2.columns = ['drug_row_id','drug_row','drug_row_cid']
DC_DATA5 = pd.merge(DC_DATA_filter4, DC_DRUG_DF2, on ='drug_row_id', how='left' )

DC_DRUG_DF2.columns = ['drug_col_id','drug_col','drug_col_cid']
DC_DATA6 = pd.merge(DC_DATA5, DC_DRUG_DF2, on ='drug_col_id', how='left')



# Cell DATA
with open(DC_PATH+'cell_lines.json') as json_file :
	DC_CELL =json.load(json_file)


DC_CELL_K = list(DC_CELL[0].keys())
DC_CELL_DF = pd.DataFrame(columns=DC_CELL_K)

for DD in range(1,len(DC_CELL)):
	tmpdf = pd.DataFrame({k:[DC_CELL[DD][k]] for k in DC_CELL_K})
	DC_CELL_DF = pd.concat([DC_CELL_DF, tmpdf], axis = 0)

DC_CELL_DF2 = DC_CELL_DF[['id','name','cellosaurus_accession', 'ccle_name']] # 751450
DC_CELL_DF2.columns = ['cell_line_id', 'DC_cellname','DrugCombCello', 'DrugCombCCLE']



# check DC triads (DC drug, cell line data )
DC_DATA7_1 = pd.merge(DC_DATA6, DC_CELL_DF2 , on= 'cell_line_id', how = 'left')
DC_DATA7_2 = DC_DATA7_1[DC_DATA7_1.drug_row_cid>0]
DC_DATA7_3 = DC_DATA7_2[DC_DATA7_2.drug_col_cid>0] # 735595
cello_t=[True if type(a)==str else False for a in list(DC_DATA7_3.DrugCombCello)]
ccle_t=[True if type(a)==str else False for a in list(DC_DATA7_3.DrugCombCCLE)]

DC_DATA7_4_cello = DC_DATA7_3[cello_t] # 730348
DC_cello_final = DC_DATA7_4_cello[['drug_row_cid','drug_col_cid','DrugCombCello']].drop_duplicates() # 563367
DC_cello_final_dup = DC_DATA7_4_cello[['drug_row_cid','drug_col_cid','DrugCombCello', 'synergy_loewe']].drop_duplicates() # 730348


print('DC and LINCS')
BETA_CID_CELLO_SIG.columns=['pert_id', 'drug_row_cid', 'DrugCombCello', 'BETA_sig_id']
CELLO_DC_BETA_1 = pd.merge(DC_cello_final_dup, BETA_CID_CELLO_SIG, on = ['drug_row_cid','DrugCombCello'], how = 'left') # 731051

BETA_CID_CELLO_SIG.columns=['pert_id', 'drug_col_cid', 'DrugCombCello', 'BETA_sig_id']
CELLO_DC_BETA_2 = pd.merge(CELLO_DC_BETA_1, BETA_CID_CELLO_SIG, on = ['drug_col_cid','DrugCombCello'], how = 'left') # 731644

BETA_CID_CELLO_SIG.columns=['pert_id', 'pubchem_cid', 'cellosaurus_id', 'sig_id']

FILTER = [a for a in range(CELLO_DC_BETA_2.shape[0]) if (type(CELLO_DC_BETA_2.BETA_sig_id_x[a]) == str) & (type(CELLO_DC_BETA_2.BETA_sig_id_y[a]) == str)]
CELLO_DC_BETA = CELLO_DC_BETA_2.loc[FILTER] # 11742
FILTER2 = [True if type(a)==float else False for a in CELLO_DC_BETA.synergy_loewe]
CELLO_DC_BETA = CELLO_DC_BETA.loc[FILTER2] # 11742 ??? 
FILTER3 = [True if np.isnan(a)==False else False for a in CELLO_DC_BETA.synergy_loewe]
CELLO_DC_BETA = CELLO_DC_BETA.loc[FILTER3] # 11701 
CELLO_DC_BETA[['BETA_sig_id_x','BETA_sig_id_y','DrugCombCello']].drop_duplicates() # 9230
CELLO_DC_BETA_cids = list(set(list(CELLO_DC_BETA.drug_row_cid) + list(CELLO_DC_BETA.drug_col_cid))) # 176 




# 3985

print('NETWORK')

G_NAME='VAR_3' # Diff Pooling


hunet_dir = '/home01/k006a01/01.DATA/HumanNet/'
# hunet_dir = '/st06/jiyeonH/13.DD_SESS/HumanNetV3/'

hunet_gsp = pd.read_csv(hunet_dir+'HumanNet-GSP.tsv', sep = '\t', header = None)
hunet_gsp.columns = ['G_A','G_B']

L_matching_list = pd.read_csv(IDK_PATH+'L_12_string.csv', sep = '\t')

hnet_IAS_L1 = hunet_gsp[hunet_gsp['G_A'].isin(L_matching_list.entrez)]
hnet_IAS_L2 = hnet_IAS_L1[hnet_IAS_L1['G_B'].isin(L_matching_list.entrez)] # 20232

len(set(list(hnet_IAS_L2['G_A']) + list(hnet_IAS_L2['G_B']))) # 611
ID_G = nx.from_pandas_edgelist(hnet_IAS_L2, 'G_A', 'G_B')

ESSEN_NAMES = list(set(L_matching_list['entrez']))

MSSNG = [a for  a in ESSEN_NAMES if a not in list(ID_G.nodes)]
# MSSNG = L_matching_list[L_matching_list['PPI_name'].isin(list(ID_G.nodes))==False]['PPI_name']
for nn in list(MSSNG):
	ID_G.add_node(nn)

# edge 3871
ID_GENE_ORDER_mini = list(ID_G.nodes()) # 978
ID_ADJ = nx.adjacency_matrix(ID_G)
ID_ADJ_tmp = torch.LongTensor(ID_ADJ.toarray())
ID_ADJ_IDX = ID_ADJ_tmp.to_sparse().indices()  # [2, 7742]
ID_WEIGHT = [] # len : 3871 -> 7742

ID_WEIGHT_SCORE = [1 for a in range(ID_ADJ_IDX.shape[1])]


new_node_name = []
for a in ID_G.nodes():
	tmp_name = list(set(L_matching_list[L_matching_list.entrez == a ]['L_gene_symbol'])) # 6118
	new_node_name = new_node_name + tmp_name

mapping = {list(ID_G.nodes())[a]:new_node_name[a] for a in range(len(new_node_name))}

ID_G_RE = nx.relabel_nodes(ID_G, mapping)

MY_G = ID_G_RE 
MY_WEIGHT_SCORE = ID_WEIGHT_SCORE # SCORE







# TARGET # 민지가 다시 올려준다고 함 

TARGET_DB = pd.read_csv(TARGET_PATH+'combined_target.csv', low_memory=False)



###########################################################################################
###########################################################################################
###########################################################################################
print("LEARNING")

# Graph 확인 


JY_GRAPH = MY_G
JY_GRAPH_ORDER = MY_G.nodes()
JY_ADJ = nx.adjacency_matrix(JY_GRAPH)

JY_ADJ_tmp = torch.LongTensor(JY_ADJ.toarray())
JY_ADJ_IDX = JY_ADJ_tmp.to_sparse().indices()
JY_IDX_WEIGHT = MY_WEIGHT_SCORE




# DC set 확인 
A_B_C_S = CELLO_DC_BETA.reset_index()
# drug target filter 
#A_B_C_S_row = A_B_C_S[A_B_C_S.drug_row_cid.isin(list(TARGET_FILTER_re.cid))]
#A_B_C_S_col = A_B_C_S_row[A_B_C_S_row.drug_col_cid.isin(list(TARGET_FILTER_re.cid))]
#
A_B_C_S_SET = A_B_C_S[['BETA_sig_id_x','BETA_sig_id_y','DrugCombCello']].drop_duplicates()
A_B_C_S_SET = A_B_C_S_SET.reset_index()



# LINCS 확인 
BETA_ORDER_pre =[list(L_matching_list.L_gene_symbol).index(a) for a in JY_GRAPH_ORDER]
BETA_ORDER_DF = L_matching_list.iloc[BETA_ORDER_pre] # 어차피 ref 다르고 같은 애들이라 괜춘 
BETA_ORDER = list(BETA_ORDER_DF.entrez)


# filter under 50 
def check_under_number(sig_id, num):
	tf = []
	z=0
	smiles = BETA_EXM2[BETA_EXM2.sig_id == sig_id]['canonical_smiles_re'].item()
	maxNumAtoms = num
	iMol = Chem.MolFromSmiles(smiles.strip())
	#Adj
	try:
		NUM = iMol.GetNumAtoms()
		if( NUM < maxNumAtoms):
			tf.append("T")
		else:
			tf.append("F")
	except:
		tf.append("error")
		print("error",z,i)
	return tf



# 50개 필터로 data 쪼개기 

tf_list = []
for a in range(A_B_C_S_SET.shape[0]):
	tf_a = check_under_number(A_B_C_S_SET['BETA_sig_id_x'][a], 50)
	tf_b = check_under_number(A_B_C_S_SET['BETA_sig_id_y'][a], 50)
	if (tf_a[0] == 'T') & (tf_b[0] == 'T') :
		tf_list.append(True)
	else:
		tf_list.append(False)


A_B_C_S_SET = A_B_C_S_SET[tf_list]
A_B_C_S_SET = A_B_C_S_SET.reset_index(drop=True) # 


def get_LINCS_data(DrugA_SIG, DrugB_SIG):
	DrugA_EXP = BETA_BIND[['id',DrugA_SIG]]
	DrugB_EXP = BETA_BIND[['id',DrugB_SIG]]
	BIND_ORDER =[list(BETA_BIND.id).index(a) for a in BETA_ORDER_DF.entrez]
	DrugA_EXP_ORD = DrugA_EXP.iloc[BIND_ORDER]
	DrugB_EXP_ORD = DrugB_EXP.iloc[BIND_ORDER]
	#
	ARR = np.array([list(DrugA_EXP_ORD[DrugA_SIG]), list(DrugB_EXP_ORD[DrugB_SIG])])
	SUM = np.sum(ARR, axis = 0)
	return DrugA_EXP_ORD, DrugB_EXP_ORD, ARR.T, SUM



def get_synergy_data(DrugA_SIG, DrugB_SIG, Cell):
	ABCS1 = A_B_C_S[A_B_C_S.BETA_sig_id_x == DrugA_SIG]
	ABCS2 = ABCS1[ABCS1.BETA_sig_id_y == DrugB_SIG]
	ABCS3 = ABCS2[ABCS2.DrugCombCello == Cell]
	synergy_score = np.median(ABCS3.synergy_loewe)
	return synergy_score


def get_targets(sig_id): # 이걸 수정해야함? 아닌가 굳이 해야하나 아니지 해야지. CID 가 없는 경우를 나타내야지  
	tmp_df1 = BETA_CID_CELLO_SIG[BETA_CID_CELLO_SIG.sig_id == sig_id]
	CID = tmp_df1.pubchem_cid.item()
	#
	target_cids = list(TARGET_DB.cid)
	if CID in target_cids:
		tmp_df2 = TARGET_DB[TARGET_DB.cid == CID]
		targets = list(set(tmp_df2.target))
		gene_symbols = list(BETA_ORDER_DF.L_gene_symbol)
		vec = [1 if a in targets else 0 for a in gene_symbols ]
	else :
		gene_symbols = list(BETA_ORDER_DF.L_gene_symbol)
		vec = [0 for a in gene_symbols ]
	return vec





def get_CHEM(sig_id, k):
	maxNumAtoms = 50
	smiles = BETA_EXM2[BETA_EXM2.sig_id == sig_id]['canonical_smiles_re'].item()
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



# 176 개 cell line 
cello_list = ['CVCL_0332', 'CVCL_0035', 'CVCL_0139', 'CVCL_0023', 'CVCL_0320', 'CVCL_0178', 'CVCL_0062', 'CVCL_0291', 'CVCL_0031', 'CVCL_0004', 'CVCL_0132', 'CVCL_0033', 'CVCL_2235', 'CVCL_0336', 'CVCL_0527', 'CVCL_A442', 'CVCL_0395']
DC_CELL_DF3 = DC_CELL_DF2[DC_CELL_DF2.DrugCombCello.isin(cello_list)]
DC_CELL_DF3['cell_onehot'] = [a for a in range(17)]
DC_CELL_DF3['tissue'] = [ '_'.join(a.split('_')[1:]) for a in list(DC_CELL_DF3['DrugCombCCLE'])]
DC_CELL_DF3 = DC_CELL_DF3.reset_index()
DC_CELL_DF3.loc[6, 'tissue'] = 'PROSTATE'
DC_CELL_DF3.loc[12, 'tissue'] ='HAEMATOPOIETIC_AND_LYMPHOID_TISSUE'
TISSUE_SET = list(set(DC_CELL_DF3['tissue']))
DC_CELL_DF3['tissue_onehot'] = [TISSUE_SET.index(a) for a in list(DC_CELL_DF3['tissue'])]

A_B_C_S_SET_COH = pd.merge(A_B_C_S_SET, DC_CELL_DF3[['DrugCombCello','DC_cellname','cell_onehot']], on = 'DrugCombCello', how = 'left'  )
cell_one_hot = torch.nn.functional.one_hot(torch.Tensor(A_B_C_S_SET_COH['cell_onehot']).long())


def get_cell(DrugA_SIG, DrugB_SIG, Cell) : 
	ABCS1 = A_B_C_S_SET_COH[A_B_C_S_SET_COH.BETA_sig_id_x == DrugA_SIG]
	ABCS2 = ABCS1[ABCS1.BETA_sig_id_y == DrugB_SIG]
	ABCS3 = ABCS2[ABCS2.DrugCombCello == Cell]
	ABCS_index = ABCS3.index.item()
	cell_res = cell_one_hot[ABCS_index]
	return(cell_res)



MY_chem_A_feat = torch.empty(size=(A_B_C_S_SET.shape[0], 50, 64))
MY_chem_B_feat= torch.empty(size=(A_B_C_S_SET.shape[0], 50, 64))
MY_chem_A_adj = torch.empty(size=(A_B_C_S_SET.shape[0], 50, 50))
MY_chem_B_adj= torch.empty(size=(A_B_C_S_SET.shape[0], 50, 50))
MY_exp_F_A = torch.empty(size=(A_B_C_S_SET.shape[0], 978, 2))
MY_exp_F_B = torch.empty(size=(A_B_C_S_SET.shape[0], 978, 2))
MY_Cell = torch.empty(size=(A_B_C_S_SET.shape[0], cell_one_hot.shape[1]))
MY_syn =  torch.empty(size=(A_B_C_S_SET.shape[0],1))




for IND in range(MY_chem_A_feat.shape[0]): 
	DrugA_SIG = A_B_C_S_SET.iloc[IND,]['BETA_sig_id_x']
	DrugB_SIG = A_B_C_S_SET.iloc[IND,]['BETA_sig_id_y']
	Cell = A_B_C_S_SET.iloc[IND,]['DrugCombCello']
	#
	k=1
	DrugA_Feat, DrugA_ADJ = get_CHEM(DrugA_SIG, k)
	DrugB_Feat, DrugB_ADJ = get_CHEM(DrugB_SIG, k)
	#
	EXP_A, EXP_B, EXP_AB, LINCS = get_LINCS_data(DrugA_SIG, DrugB_SIG)
	EXP_A = list(EXP_A.iloc[:,1])
	EXP_B = list(EXP_B.iloc[:,1])
	#
	Cell_Vec = get_cell(DrugA_SIG, DrugB_SIG, Cell)
	#
	TGT_A = get_targets(DrugA_SIG)
	TGT_B = get_targets(DrugB_SIG)
	#
	AB_SYN = get_synergy_data(DrugA_SIG, DrugB_SIG, Cell)
	#
	FEAT_A = torch.Tensor(np.array([EXP_A, TGT_A]).T)
	FEAT_B = torch.Tensor(np.array([EXP_B, TGT_B]).T)
	#
	MY_chem_A_feat[IND] = torch.Tensor(DrugA_Feat)
	MY_chem_B_feat[IND] = torch.Tensor(DrugB_Feat)
	MY_chem_A_adj[IND] = torch.Tensor(DrugA_ADJ)
	MY_chem_B_adj[IND] = torch.Tensor(DrugB_ADJ)
	MY_Cell[IND] = Cell_Vec
	MY_syn[IND] = torch.Tensor([AB_SYN])
	MY_exp_F_A[IND] = FEAT_A
	MY_exp_F_B[IND] = FEAT_B




torch.save(MY_chem_A_feat, WORK_PATH+'0902.{}.MY_chem_A_feat.pt'.format(G_NAME))
torch.save(MY_chem_B_feat, WORK_PATH+'0902.{}.MY_chem_B_feat.pt'.format(G_NAME))
torch.save(MY_chem_A_adj, WORK_PATH+'0902.{}.MY_chem_A_adj.pt'.format(G_NAME))
torch.save(MY_chem_B_adj, WORK_PATH+'0902.{}.MY_chem_B_adj.pt'.format(G_NAME))
torch.save(MY_exp_F_A, WORK_PATH+'0902.{}.MY_exp_F_A.pt'.format(G_NAME))
torch.save(MY_exp_F_B, WORK_PATH+'0902.{}.MY_exp_F_B.pt'.format(G_NAME))
torch.save(MY_syn, WORK_PATH+'0902.{}.MY_syn.pt'.format(G_NAME))
torch.save(MY_Cell, WORK_PATH+'0902.{}.MY_Cell.pt'.format(G_NAME))



print('input ok')

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




# normalizing 과정에 대한 고민 필요 ... GCN 이니까 상관 없지 않을까
def prepare_data_GCN(MY_chem_A_feat, MY_chem_B_feat, MY_chem_A_adj, MY_chem_B_adj, MY_exp_F_A, MY_exp_F_B, MY_syn, MY_cell, norm ) :
	chem_feat_A_train, chem_feat_A_tv, chem_feat_B_train, chem_feat_B_tv, chem_adj_A_train, chem_adj_A_tv, chem_adj_B_train, chem_adj_B_tv, exp_F_A_train, exp_F_A_tv, exp_F_B_train, exp_F_B_tv, syn_train, syn_tv, cell_train, cell_tv = sklearn.model_selection.train_test_split(
			MY_chem_A_feat, MY_chem_B_feat, MY_chem_A_adj, MY_chem_B_adj, MY_exp_F_A, MY_exp_F_B, MY_syn, MY_Cell,
			test_size= 0.2 , random_state=42 )
	chem_feat_A_val, chem_feat_A_test, chem_feat_B_val, chem_feat_B_test, chem_adj_A_val, chem_adj_A_test, chem_adj_B_val, chem_adj_B_test, exp_F_A_val, exp_F_A_test, exp_F_B_val, exp_F_B_test, syn_val, syn_test, cell_val, cell_test  = sklearn.model_selection.train_test_split(
			chem_feat_A_tv, chem_feat_B_tv, chem_adj_A_tv, chem_adj_B_tv, exp_F_A_tv, exp_F_B_tv, syn_tv, cell_tv,
			test_size=0.5, random_state=42 )
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
	train_data['EXP_F_A'] = torch.concat([exp_F_A_train, exp_F_B_train], axis = 0)
	val_data['EXP_F_A'] = exp_F_A_val
	test_data['EXP_F_A'] = exp_F_A_test
	#
	train_data['EXP_F_B'] = torch.concat([exp_F_B_train, exp_F_A_train], axis = 0)
	val_data['EXP_F_B'] = exp_F_B_val
	test_data['EXP_F_B'] = exp_F_B_test
	#               
	train_data['y'] = np.concatenate((syn_train, syn_train), axis=0)
	val_data['y'] = syn_val
	test_data['y'] = syn_test
	#
	train_data['cell'] = np.concatenate((cell_train, cell_train), axis=0)
	val_data['cell'] = cell_val
	test_data['cell'] = cell_test
	#
	print(train_data['drug1_feat'].shape)
	print(val_data['drug1_feat'].shape)
	print(test_data['drug1_feat'].shape)
	return train_data, val_data, test_data





class DATASET_DIFF_W_FT(Dataset):
	def __init__(self, gcn_drug1_F, gcn_drug2_F, gcn_drug1_ADJ, gcn_drug2_ADJ, gcn_exp_A, gcn_exp_B, gcn_adj, gcn_adj_weight, syn_ans, cell_info):
		self.gcn_drug1_F = gcn_drug1_F
		self.gcn_drug2_F = gcn_drug2_F
		self.gcn_drug1_ADJ = gcn_drug1_ADJ
		self.gcn_drug2_ADJ = gcn_drug2_ADJ
		self.gcn_exp_A = gcn_exp_A
		self.gcn_exp_B = gcn_exp_B
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
		return self.gcn_drug1_F[index], self.gcn_drug2_F[index], self.gcn_drug1_ADJ[index], self.gcn_drug2_ADJ[index], self.gcn_exp_A[index], self.gcn_exp_B[index], self.gcn_adj, self.gcn_adj_weight, self.syn_ans[index] , self.cell_info[index]




def weighted_mse_loss(input, target, weight):
	return (weight * (input - target) ** 2).mean()


def result_pearson(y, pred):
	pear = stats.pearsonr(y, pred)
	pear_value = pear[0]
	pear_p_val = pear[1]
	print("Pearson correlation is {} and related p_value is {}".format(pear_value, pear_p_val))


def result_spearman(y, pred):
	spear = stats.spearmanr(y, pred)
	spear_value = spear[0]
	spear_p_val = spear[1]
	print("Spearman correlation is {} and related p_value is {}".format(spear_value, spear_p_val))


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


norm = 'tanh_norm'
train_data, val_data, test_data = prepare_data_GCN(MY_chem_A_feat, MY_chem_B_feat, MY_chem_A_adj, MY_chem_B_adj, MY_exp_F_A, MY_exp_F_B, MY_syn, MY_Cell, norm)



# WEIGHT 
ys = train_data['y'].squeeze().tolist()
min_s = np.amin(ys)
loss_weight = np.log(train_data['y'] - min_s + np.e)
JY_IDX_WEIGHT_T = torch.Tensor(JY_IDX_WEIGHT).view(1,-1)



# DATA check 
T_train = DATASET_DIFF_W_FT(
	torch.Tensor(train_data['drug1_feat']), torch.Tensor(train_data['drug2_feat']), 
	torch.Tensor(train_data['drug1_adj']), torch.Tensor(train_data['drug2_adj']),
	train_data['EXP_F_A'], train_data['EXP_F_B'], 
	JY_ADJ_tmp.unsqueeze(0), JY_IDX_WEIGHT_T, 
	torch.Tensor(train_data['y']),
	torch.Tensor(train_data['cell']))

T_val = DATASET_DIFF_W_FT(
	torch.Tensor(val_data['drug1_feat']), torch.Tensor(val_data['drug2_feat']), 
	torch.Tensor(val_data['drug1_adj']), torch.Tensor(val_data['drug2_adj']),
	val_data['EXP_F_A'], val_data['EXP_F_B'], 
	JY_ADJ_tmp.unsqueeze(0), JY_IDX_WEIGHT_T, 
	torch.Tensor(val_data['y']),
	torch.Tensor(val_data['cell']))
	
T_test = DATASET_DIFF_W_FT(
	torch.Tensor(test_data['drug1_feat']), torch.Tensor(test_data['drug2_feat']), 
	torch.Tensor(test_data['drug1_adj']), torch.Tensor(test_data['drug2_adj']),
	test_data['EXP_F_A'], test_data['EXP_F_B'], 
	JY_ADJ_tmp.unsqueeze(0), JY_IDX_WEIGHT_T, 
	torch.Tensor(test_data['y']),
	torch.Tensor(test_data['cell']))

RAY_train = ray.put(T_train)
RAY_val = ray.put(T_val)
RAY_test = ray.put(T_test)
RAY_loss_weight = ray.put(loss_weight)




############################ MAIN
print('MAIN')



class GNN(torch.nn.Module):
	def __init__(self, num_layer, in_dim, hid_dim, out_dim, normalize=False, lin=True): 
		super(GNN, self).__init__()
		self.num_layer = num_layer
		self.in_dim = in_dim
		self.hid_dim = hid_dim
		self.out_dim = out_dim
		self.normalize = normalize
		self.lin = lin
		#
		self.Convs = torch.nn.ModuleList([pyg_nn.DenseSAGEConv(self.in_dim, self.hid_dim, self.normalize)])
		self.Convs.extend([pyg_nn.DenseSAGEConv(self.hid_dim, self.hid_dim, self.normalize) for i in range(self.num_layer-2)])
		self.Convs.extend([pyg_nn.DenseSAGEConv(self.hid_dim, self.out_dim, self.normalize) ])
		self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(self.hid_dim) for i in range(self.num_layer-1)])
		self.bns.extend([torch.nn.BatchNorm1d(self.out_dim) ])
		#
		if lin is True : 
			self.linR = torch.nn.Linear((self.num_layer-1)*self.hid_dim+self.out_dim, self.out_dim )
		else : 
			self.linR = None
	#
	def bn (self, step , x ):
		batch_size , num_nodes , num_features = x.size()
		x = x.view(-1, num_features)
		x = self.bns[step](x)
		x = x.view(batch_size, num_nodes, num_features)
		return x 
	#
	def forward(self, x, adj, mask=None):
		batch_size, num_nodes, in_channels = x.size()
		layer_dict = {}
		key_list = ["x_"+str(i) for i in range(self.num_layer+1)]
		layer_dict['x_0'] = [x]
		for L in range(self.num_layer):
			key = key_list[L+1]
			layer_dict[key] = [self.bn(L, F.elu(self.Convs[L](layer_dict[key_list[L]][0], adj, mask)))]
		x = torch.cat([layer_dict[key][0] for key in key_list[1:]], dim= -1)
		if self.lin is True:
			x = F.elu(self.linR(x)) 
		return x




class DIFFPOOL(torch.nn.Module):
	def __init__(self, conv_num, input_dim, hidden_dim, output_dim, max_nodes, num_pool, ratio , dropout):
		super(DIFFPOOL, self).__init__()
		#
		self.conv_num = conv_num
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.max_nodes = max_nodes
		self.num_pool = num_pool
		self.ratio = ratio
		self.dropout = dropout
		#
		self.num_nodes = [ceil(self.ratio * self.max_nodes)]
		self.check_num_nodes()
		#
		self.POOLS = torch.nn.ModuleList([GNN(self.conv_num, self.input_dim, self.hidden_dim, self.num_nodes[1] )])
		self.POOLS.extend([GNN(self.conv_num, self.conv_num * self.hidden_dim, self.hidden_dim, self.num_nodes[i+2]) for i in range(self.num_pool-1)])
		#
		self.EMBED = torch.nn.ModuleList([GNN(self.conv_num, self.input_dim, self.hidden_dim, self.hidden_dim, lin=False)])
		self.EMBED.extend([GNN(self.conv_num, self.conv_num * self.hidden_dim, self.hidden_dim, self.hidden_dim, lin=False) for i in range(self.num_pool)])
		#
		self.lin1 = torch.nn.Linear(self.conv_num * self.hidden_dim, self.hidden_dim)
		self.lin2 = torch.nn.Linear(self.hidden_dim, self.output_dim)
		#
		self.reset_parameters()
	#
	def reset_parameters(self):
		self.lin1.reset_parameters()
		self.lin2.reset_parameters()
	#
	def check_num_nodes(self):
		for L in range(self.num_pool):
			this_node = ceil( self.ratio * self.num_nodes[-1])
			self.num_nodes.append(this_node)
	#	
	def forward(self, x, adj, mask = None):
		x = F.normalize(x)
		s = self.POOLS[0](x, adj, mask)
		x = self.EMBED[0](x, adj, mask)
		x, adj, l, e = pyg_nn.dense_diff_pool(x, adj, s, mask)
		#
		for i in range(self.num_pool-1):
			s = self.POOLS[i+1](x, adj)
			x = self.EMBED[i+1](x, adj)
			x, adj, l, e = pyg_nn.dense_diff_pool(x, adj, s)
		x = self.EMBED[-1](x, adj)
		#
		x = x.mean(dim=1)
		x = F.elu(self.lin2(self.lin1(x)))
		x = F.dropout(x, p=self.dropout, training=self.training)
		x = torch.tanh(x)
		return x




class MY_diffpool_parallel(torch.nn.Module) :
	def __init__(self, G_layer_chem, G_indim_chem, G_hiddim_chem, G_layer_exp, G_indim_exp, G_hiddim_exp, G_pool_chem, G_pool_exp, G_ratio_chem, G_ratio_exp, layers_1, layers_2, layers_3, cell_dim ,out_dim, inDrop, drop):
		super(MY_diffpool_parallel, self).__init__()
		#
		self.G_layer_chem = G_layer_chem
		self.G_indim_chem = G_indim_chem
		self.G_hiddim_chem = G_hiddim_chem
		self.G_layer_exp = G_layer_exp
		self.G_indim_exp = G_indim_exp
		self.G_hiddim_exp = G_hiddim_exp
		self.G_pool_chem = G_pool_chem
		self.G_pool_exp = G_pool_exp
		self.G_ratio_chem = G_ratio_chem
		self.G_ratio_exp = G_ratio_exp
		self.layers_1 = [int(a) for a in layers_1]
		self.layers_2 = [int(a) for a in layers_2]
		self.layers_3 = [int(a) for a in layers_3]
		self.cell_dim = cell_dim
		self.out_dim = out_dim
		self.inDrop = inDrop
		self.drop = drop
		self.tanh = torch.nn.Tanh()
		#
		self.G_1_C = DIFFPOOL(self.G_layer_chem, self.G_indim_chem, self.G_hiddim_chem, self.G_hiddim_chem, 50, self.G_pool_chem, self.G_ratio_chem, self.inDrop )
		self.G_2_C = DIFFPOOL(self.G_layer_chem, self.G_indim_chem, self.G_hiddim_chem, self.G_hiddim_chem, 50, self.G_pool_chem, self.G_ratio_chem, self.inDrop )
		self.G_1_E = DIFFPOOL(self.G_layer_exp, self.G_indim_exp, self.G_hiddim_exp, self.G_hiddim_exp, 978, self.G_pool_exp, self.G_ratio_exp, self.inDrop )
		self.G_2_E = DIFFPOOL(self.G_layer_exp, self.G_indim_exp, self.G_hiddim_exp, self.G_hiddim_exp, 978, self.G_pool_exp, self.G_ratio_exp, self.inDrop )
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
		#
	def forward (self, Drug1_F, Drug2_F, Drug1_ADJ, Drug2_ADJ, EXP1, EXP2, EXP_ADJ, EXP_ADJ_WGT, syn, cell ) :
		G_1_C_out = self.G_1_C(Drug1_F, Drug1_ADJ)
		G_2_C_out = self.G_2_C(Drug2_F, Drug2_ADJ)
		G_1_E_out = self.G_1_E(EXP1, EXP_ADJ)
		G_2_E_out = self.G_2_E(EXP2, EXP_ADJ)
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
	use_cuda = True # False
	#
	T_train = ray.get(RAY_train)
	T_val = ray.get(RAY_val)
	T_test = ray.get(RAY_test)
	T_loss_weight = ray.get(RAY_loss_weight)
	batch_cut_weight = [T_loss_weight[i:i+config["batch_size"]] for i in range(0,len(T_loss_weight), config["batch_size"])]
	#
	loaders = {
			'train' : torch.utils.data.DataLoader(T_train, batch_size = config["batch_size"],  shuffle =False, num_workers=config['n_workers']),
			'eval' : torch.utils.data.DataLoader(T_val, batch_size = config["batch_size"], shuffle =False, num_workers=config['n_workers']),
			'test' : torch.utils.data.DataLoader(T_test, batch_size = config["batch_size"], shuffle =False, num_workers=config['n_workers']),
	}
	#
	dsn1_layers = [config["feat_size_0"], config["feat_size_1"] , config["feat_size_2"] ]
	dsn2_layers = [config["feat_size_0"], config["feat_size_1"] , config["feat_size_2"] ]
	snp_layers = [config["feat_size_3"] ,config["feat_size_4"] ,]
	inDrop = config["dropout_1"]
	Drop = config["dropout_2"]
	#
	#
	MM_MODEL = MY_diffpool_parallel(
			config["G_layer"], T_train.gcn_drug1_F.shape[-1] , config["G_hiddim"],
			config["G_layer"], 2 , config["G_hiddim"],
			config["G_pool_1"], config["G_pool_2"], config["G_ratio_1"], config["G_ratio_2"], 
			dsn1_layers, dsn2_layers, snp_layers, cell_one_hot.shape[1], 1,
			inDrop, Drop
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
		for batch_idx_t, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell) in enumerate(loaders['train']):
			adj = adj.float()
			adj = adj.squeeze(1)
			# move to GPU
			if use_cuda:
				drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell = drug1_f.cuda(), drug2_f.cuda(), drug1_a.cuda(), drug2_a.cuda(), expA.cuda(), expB.cuda(), adj.cuda(), adj_w.cuda(), y.cuda(), cell.cuda() 
			## find the loss and update the model parameters accordingly
			# clear the gradients of all optimized variables
			optimizer.zero_grad()
			output = MM_MODEL(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell)
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
		for batch_idx_v, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell) in enumerate(loaders['eval']):
			adj = adj.float()
			adj = adj.squeeze(1)
			# move to GPU
			if use_cuda:
				drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell = drug1_f.cuda(), drug2_f.cuda(), drug1_a.cuda(), drug2_a.cuda(), expA.cuda(), expB.cuda(), adj.cuda(), adj_w.cuda(), y.cuda(), cell.cuda()
			## update the average validation loss
			output = MM_MODEL(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell)
			MSE = torch.nn.MSELoss()
			loss = MSE(output, y) # train 이 아니라서 weight 안넣어줌. 그냥 nn.MSE 넣어주기 
			# update average validation loss 
			valid_loss = valid_loss + loss.item()
		#
		# calculate average losses
		TRAIN_LOSS = train_loss/(batch_idx_t+1)
		train_loss_all.append(TRAIN_LOSS)
		VAL_LOSS = valid_loss/(batch_idx_v+1)
		valid_loss_all.append(VAL_LOSS)
		#
		# print training/validation statistics 
		#
		done = datetime.now()
		time_spent = done-now
		#
		with tune.checkpoint_dir(step = epoch) as checkpoint_dir:
			trial_name = '_'.join(checkpoint_dir.split('/')[-2].split('_')[0:4])
			print('trial : {}, epoch : {}, TrainLoss : {}, ValLoss : {}'.format(trial_name, epoch,TRAIN_LOSS,VAL_LOSS), flush=True)
			path = os.path.join(checkpoint_dir, "checkpoint")
			torch.save((MM_MODEL.state_dict(), optimizer.state_dict()), path)
			torch.save(MM_MODEL.state_dict(), './model.pth')
		tune.report(TrainLoss= TRAIN_LOSS,  ValLoss=VAL_LOSS )
	#
	print("Finished Training")
 


def RAY_TEST_MODEL(my_config, model_path, model_name, PRJ_PATH, Trial_name, G_NAME, number): 
	use_cuda =  False #True
	T_test = ray.get(RAY_test)
	Test_loader = torch.utils.data.DataLoader(T_test, batch_size = my_config['config/batch_size'].item(),  shuffle =False, num_workers=my_config['config/n_workers'].item())
	#
	G_layer = my_config['config/G_layer'].item()
	G_hiddim = my_config['config/G_hiddim'].item()
	G_pool_1 = my_config['config/G_pool_1'].item()
	G_pool_2 = my_config['config/G_pool_2'].item()
	G_ratio_1 = my_config['config/G_ratio_1'].item()
	G_ratio_2 = my_config['config/G_ratio_2'].item()
	dsn1_layers = [my_config['config/feat_size_0'].item(), my_config['config/feat_size_1'].item(), my_config['config/feat_size_2'].item()]
	dsn2_layers = [my_config['config/feat_size_0'].item(), my_config['config/feat_size_1'].item(), my_config['config/feat_size_2'].item()] 
	snp_layers = [my_config['config/feat_size_3'].item() , my_config['config/feat_size_4'].item()]
	inDrop = my_config['config/dropout_1'].item()
	Drop = my_config['config/dropout_2'].item()
	#       
	best_model = MY_diffpool_parallel(
				G_layer, T_test.gcn_drug1_F.shape[-1] , G_hiddim,
				G_layer, 2, G_hiddim,
				G_pool_1, G_pool_2, G_ratio_1, G_ratio_2, 
				dsn1_layers, dsn2_layers, snp_layers, 17, 1,
				inDrop, Drop
				)
	#
	if torch.cuda.is_available():
			best_model = best_model.cuda()
			if torch.cuda.device_count() > 1 :
				best_model = torch.nn.DataParallel(best_model)
	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	state_dict = torch.load(os.path.join(model_path, model_name))
	if type(state_dict) == tuple:
		best_model.load_state_dict(state_dict[0])
	else : 
		best_model.load_state_dict(state_dict)
	#
	#
	best_model.eval()
	test_loss = 0.0
	PRED_list = []
	Y_list = test_data['y'].squeeze().tolist()
	with torch.no_grad():
		best_model.eval()
		for batch_idx_t, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell) in enumerate(Test_loader):
			adj = adj.float()
			adj = adj.squeeze(1)
			if use_cuda:
				drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell = drug1_f.cuda(), drug2_f.cuda(), drug1_a.cuda(), drug2_a.cuda(), expA.cuda(), expB.cuda(), adj.cuda(), adj_w.cuda(), y.cuda(), cell.cuda()
			output = best_model(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell) 
			MSE = torch.nn.MSELoss()
			loss = MSE(output, y)
			test_loss = test_loss + loss.item()
			outputs = output.squeeze().tolist()
			PRED_list = PRED_list + outputs
	TEST_LOSS = test_loss/(batch_idx_t+1)
	R__T = TEST_LOSS
	R__1 , R__2 = jy_corrplot(PRED_list, Y_list, PRJ_PATH,'P.{}.{}.{}_model'.format(Trial_name, G_NAME, number) )
	return R__T, R__1, R__2



def final_result(R_1_V, R_1_T, R_1_1, R_1_2, R_2_V, R_2_T, R_2_1, R_2_2, R_3_V, R_3_T, R_3_1, R_3_2) :
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


from ray.tune import ExperimentAnalysis


def jy_corrplot(PRED_list, Y_list, path, plotname ):
	jplot = sns.jointplot(x=PRED_list, y=Y_list, ci=68, kind='reg')
	pr,pp = stats.pearsonr(PRED_list, Y_list)
	print("Pearson correlation is {} and related p_value is {}".format(pr, pp))
	sr,sp = stats.spearmanr(PRED_list, Y_list)
	print("Spearman correlation is {} and related p_value is {}".format(sr, sp))
	jplot.ax_joint.annotate(f'$pearson = {pr:.3f}, spearman = {sr:.3f}$',xy=(min(PRED_list)+ 0.01, max(Y_list)- 0.01 ), ha='left', va='center',)
	jplot.ax_joint.scatter(PRED_list, Y_list)
	jplot.set_axis_labels(xlabel='Predicted', ylabel='Answer', size=15)
	jplot.figure.savefig('{}/{}.corrplot.png'.format(path, plotname), bbox_inches = 'tight')
	return pr, sr




def MAIN(ANAL_name, WORK_PATH, PRJ_PATH, Trial_name, G_NAME, num_samples= 10, max_num_epochs=1000, grace_period = 150, cpus_per_trial = 16, gpus_per_trial = 1):
	# OPTUNA CONFIGURE 
	CONFIG={
		'n_workers' : tune.choice([cpus_per_trial]),
		"epoch" : tune.choice([max_num_epochs]),
		"G_layer" : tune.choice([2, 3, 4]), # 
		"G_hiddim" : tune.choice([512, 256, 128, 64, 32, 16]), #
		"G_pool_1" : tune.choice([2, 3, 4]), #
		"G_pool_2" : tune.choice([2, 3, 4]), #
		"G_ratio_1" : tune.choice([0.2, 0.5, 0.8]), # 
		"G_ratio_2" : tune.choice([0.2, 0.5, 0.8]), # 
		"batch_size" : tune.choice([ 64, 32, 16]), # CPU 니까 
		"feat_size_0" : tune.choice([2048, 1024, 512, 256, 128, 64, 32]), # 4096, 2048, 1024, 512, 256, 128, 64, 32
		"feat_size_1" : tune.choice([2048, 1024, 512, 256, 128, 64, 32]),# 4096, 2048, 1024, 512, 256, 128, 64, 32
		"feat_size_2" : tune.choice([2048, 1024, 512, 256, 128, 64, 32]),# 4096, 2048, 1024, 512, 256, 128, 64, 32
		"feat_size_3" : tune.choice([2048, 1024, 512, 256, 128, 64, 32]),# 4096, 2048, 1024, 512, 256, 128, 64, 32
		"feat_size_4" : tune.choice([2048, 1024, 512, 256, 128, 64, 32]),# 4096, 2048, 1024, 512, 256, 128, 64, 32
		"dropout_1" : tune.choice([0.01, 0.2, 0.5, 0.8]), # 0.01, 0.2, 0.5, 0.8
		"dropout_2" : tune.choice([0.01, 0.2, 0.5, 0.8]), # 0.01, 0.2, 0.5, 0.8
		"lr" : tune.choice([0.00001, 0.0001, 0.001]),# 0.00001, 0.0001, 0.001
	}
	#
	#
	reporter = CLIReporter(
		metric_columns=["TrainLoss", "ValLoss",  "training_iteration"])
	#
	optuna_search = OptunaSearch(metric="ValLoss", mode="min")
	#
	ASHA_scheduler = tune.schedulers.ASHAScheduler(
		time_attr='training_iteration', metric="ValLoss", mode="min", max_t= max_num_epochs, grace_period = grace_period )
	#
	#
	ANALYSIS = tune.run( # 끝내지 않음 
		tune.with_parameters(RAY_MY_train),
		name = ANAL_name,
		num_samples=num_samples,
		config=CONFIG,
		resources_per_trial={'cpu': cpus_per_trial, 'gpu' : gpus_per_trial }, # 
		progress_reporter = reporter,
		search_alg = optuna_search,
		scheduler = ASHA_scheduler
	)
	best_trial = ANALYSIS.get_best_trial("ValLoss", "min", "last")
	print("Best trial config: {}".format(best_trial.config), flush=True)
	print("Best trial final validation loss: {}".format(
	best_trial.last_result["ValLoss"]), flush=True)
	#
	#
	anal_df = ExperimentAnalysis("~/ray_results/{}".format(ANAL_name))
	#
	# 1) best final
	#
	ANA_DF = anal_df.dataframe()
	ANA_ALL_DF = anal_df.trial_dataframes
	#
	DF_KEY = list(ANA_DF.sort_values('ValLoss')['logdir'])[0]
	print('best final', flush=True)
	print(DF_KEY, flush=True)
	TOPVAL_PATH = DF_KEY
	mini_df = ANA_ALL_DF[DF_KEY]
	my_config = ANA_DF[ANA_DF.logdir==DF_KEY]
	R_1_V = list(ANA_DF.sort_values('ValLoss')['ValLoss'])[0]
	R_1_T, R_1_1, R_1_2 = RAY_TEST_MODEL(my_config, TOPVAL_PATH, 'model.pth', PRJ_PATH, Trial_name, G_NAME, 'M1')
	#
	# 2) best final's checkpoint
	# 
	cck_num =mini_df[mini_df.ValLoss==min(mini_df.ValLoss)].index.item()
	checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
	TOPVAL_PATH = DF_KEY + checkpoint
	print('best final check', flush=True)
	print(TOPVAL_PATH, flush=True)
	R_2_V = min(mini_df.ValLoss)
	R_2_T, R_2_1, R_2_2 = RAY_TEST_MODEL(my_config, TOPVAL_PATH, 'checkpoint', PRJ_PATH, Trial_name, G_NAME, 'M2')
	#
	# 3) total checkpoint best 
	#	
	import numpy as np
	TOT_min = np.Inf
	TOT_key = ""
	for key in ANA_ALL_DF.keys():
		trial_min = min(ANA_ALL_DF[key]['ValLoss'])
		if trial_min < TOT_min :
			TOT_min = trial_min
			TOT_key = key
	print('best val', flush=True)
	print(TOT_key, flush=True)
	mini_df = ANA_ALL_DF[TOT_key]
	TOPVAL_PATH = TOT_key
	my_config = ANA_DF[ANA_DF.logdir==TOT_key]
	cck_num =mini_df[mini_df.ValLoss==min(mini_df.ValLoss)].index.item()
	checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
	TOPVAL_PATH = TOT_key + checkpoint
	print('best val check', flush=True)
	print(TOPVAL_PATH, flush=True)
	R_3_V = min(mini_df.ValLoss)
	R_3_T, R_3_1, R_3_2 = RAY_TEST_MODEL(my_config, TOPVAL_PATH, 'checkpoint', PRJ_PATH, Trial_name, G_NAME, 'M4')
	#
	final_result(R_1_V, R_1_T, R_1_1, R_1_2, R_2_V, R_2_T, R_2_1, R_2_2, R_3_V, R_3_T, R_3_1, R_3_2)
	return ANALYSIS






MAIN('22.09.02.PRJ01.TRIAL3_9_3_pre', WORK_PATH, WORK_PATH, '3_9_3', 'Var3', 3, 3, 2, 10, 1)

MAIN('22.09.02.PRJ01.TRIAL3_9_3', WORK_PATH, PRJ_PATH, '3_9_3', 'Var3', 100, 1000, 150, 32, 1)







##############################
##############################
##############################
##############################


import pandas as pd 
import matplotlib.pyplot as plt
from ray.tune import ExperimentAnalysis
import pickle
import math
import torch



anal_df = ExperimentAnalysis("~/ray_results/22.09.02.PRJ01.TRIAL3_9_3")


ANA_DF = anal_df.dataframe()
ANA_ALL_DF = anal_df.trial_dataframes


ANA_DF.to_csv('/home01/k006a01/PRJ.01/TRIAL_3.9/RAY_ANA_DF.P01.3_9_3.csv')
import pickle
with open("/home01/k006a01/PRJ.01/TRIAL_3.9/RAY_ANA_DF.P01.3_9_3.pickle", "wb") as fp:
	pickle.dump(ANA_ALL_DF,fp) 


DF_KEY = list(ANA_DF.sort_values('ValLoss')['logdir'])[0]
DF_KEY
# get /home01/k006a01/ray_results/22.09.02.PRJ01.TRIAL3_9_3/RAY_MY_train_9434e5a2_28_G_hiddim=16,G_layer=3,G_pool_1=2,G_pool_2=2,G_ratio_1=0.5000,G_ratio_2=0.2000,batch_size=64,dropout_1=0.0_2022-09-07_09-43-41/model.pth M1_model.pth


TOPVAL_PATH = DF_KEY

mini_df = ANA_ALL_DF[DF_KEY]

cck_num =mini_df[mini_df.ValLoss==min(mini_df.ValLoss)].index.item()
checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
TOPVAL_PATH = DF_KEY + checkpoint
TOPVAL_PATH

# get /home01/k006a01/ray_results/22.09.02.PRJ01.TRIAL3_9_3/RAY_MY_train_9434e5a2_28_G_hiddim=16,G_layer=3,G_pool_1=2,G_pool_2=2,G_ratio_1=0.5000,G_ratio_2=0.2000,batch_size=64,dropout_1=0.0_2022-09-07_09-43-41/checkpoint_000804/checkpoint M2_checkpoint




import numpy as np
TOT_min = np.Inf
TOT_key = ""
for key in ANA_ALL_DF.keys():
	trial_min = min(ANA_ALL_DF[key]['ValLoss'])
	if trial_min < TOT_min :
		TOT_min = trial_min
		TOT_key = key

TOT_min
TOT_key

mini_df = ANA_ALL_DF[TOT_key]
TOPVAL_PATH = TOT_key

cck_num =mini_df[mini_df.ValLoss==min(mini_df.ValLoss)].index.item()
checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
TOPVAL_PATH = TOT_key + checkpoint
TOPVAL_PATH

# get /home01/k006a01/ray_results/22.09.02.PRJ01.TRIAL3_9_3/RAY_MY_train_419e70d0_26_G_hiddim=256,G_layer=2,G_pool_1=2,G_pool_2=2,G_ratio_1=0.5000,G_ratio_2=0.2000,batch_size=64,dropout_1=0._2022-09-06_18-19-32/checkpoint_000484/checkpoint M4_checkpoint





###########################################################
##################### LOCAL ###################################
###########################################################




print('input ok')

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





# normalizing 과정에 대한 고민 필요 ... GCN 이니까 상관 없지 않을까
def prepare_data_GCN(MY_chem_A_feat, MY_chem_B_feat, MY_chem_A_adj, MY_chem_B_adj, MY_exp_F_A, MY_exp_F_B, MY_syn, MY_cell, norm ) :
	chem_feat_A_train, chem_feat_A_tv, chem_feat_B_train, chem_feat_B_tv, chem_adj_A_train, chem_adj_A_tv, chem_adj_B_train, chem_adj_B_tv, exp_F_A_train, exp_F_A_tv, exp_F_B_train, exp_F_B_tv, syn_train, syn_tv, cell_train, cell_tv = sklearn.model_selection.train_test_split(
			MY_chem_A_feat, MY_chem_B_feat, MY_chem_A_adj, MY_chem_B_adj, MY_exp_F_A, MY_exp_F_B, MY_syn, MY_Cell,
			test_size= 0.2 , random_state=42 )
	chem_feat_A_val, chem_feat_A_test, chem_feat_B_val, chem_feat_B_test, chem_adj_A_val, chem_adj_A_test, chem_adj_B_val, chem_adj_B_test, exp_F_A_val, exp_F_A_test, exp_F_B_val, exp_F_B_test, syn_val, syn_test, cell_val, cell_test  = sklearn.model_selection.train_test_split(
			chem_feat_A_tv, chem_feat_B_tv, chem_adj_A_tv, chem_adj_B_tv, exp_F_A_tv, exp_F_B_tv, syn_tv, cell_tv,
			test_size=0.5, random_state=42 )
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
	train_data['EXP_F_A'] = torch.concat([exp_F_A_train, exp_F_B_train], axis = 0)
	val_data['EXP_F_A'] = exp_F_A_val
	test_data['EXP_F_A'] = exp_F_A_test
	#
	train_data['EXP_F_B'] = torch.concat([exp_F_B_train, exp_F_A_train], axis = 0)
	val_data['EXP_F_B'] = exp_F_B_val
	test_data['EXP_F_B'] = exp_F_B_test
	#               
	train_data['y'] = np.concatenate((syn_train, syn_train), axis=0)
	val_data['y'] = syn_val
	test_data['y'] = syn_test
	#
	train_data['cell'] = np.concatenate((cell_train, cell_train), axis=0)
	val_data['cell'] = cell_val
	test_data['cell'] = cell_test
	#
	print(train_data['drug1_feat'].shape)
	print(val_data['drug1_feat'].shape)
	print(test_data['drug1_feat'].shape)
	return train_data, val_data, test_data




class DATASET_DIFF_W_FT(Dataset):
	def __init__(self, gcn_drug1_F, gcn_drug2_F, gcn_drug1_ADJ, gcn_drug2_ADJ, gcn_exp_A, gcn_exp_B, gcn_adj, gcn_adj_weight, syn_ans, cell_info):
		self.gcn_drug1_F = gcn_drug1_F
		self.gcn_drug2_F = gcn_drug2_F
		self.gcn_drug1_ADJ = gcn_drug1_ADJ
		self.gcn_drug2_ADJ = gcn_drug2_ADJ
		self.gcn_exp_A = gcn_exp_A
		self.gcn_exp_B = gcn_exp_B
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
		return self.gcn_drug1_F[index], self.gcn_drug2_F[index], self.gcn_drug1_ADJ[index], self.gcn_drug2_ADJ[index], self.gcn_exp_A[index], self.gcn_exp_B[index], self.gcn_adj, self.gcn_adj_weight, self.syn_ans[index] , self.cell_info[index]




def weighted_mse_loss(input, target, weight):
	return (weight * (input - target) ** 2).mean()


def result_pearson(y, pred):
	pear = stats.pearsonr(y, pred)
	pear_value = pear[0]
	pear_p_val = pear[1]
	print("Pearson correlation is {} and related p_value is {}".format(pear_value, pear_p_val))


def result_spearman(y, pred):
	spear = stats.spearmanr(y, pred)
	spear_value = spear[0]
	spear_p_val = spear[1]
	print("Spearman correlation is {} and related p_value is {}".format(spear_value, spear_p_val))


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





class GNN(torch.nn.Module):
	def __init__(self, num_layer, in_dim, hid_dim, out_dim, normalize=False, lin=True): 
		super(GNN, self).__init__()
		self.num_layer = num_layer
		self.in_dim = in_dim
		self.hid_dim = hid_dim
		self.out_dim = out_dim
		self.normalize = normalize
		self.lin = lin
		#
		self.Convs = torch.nn.ModuleList([pyg_nn.DenseSAGEConv(self.in_dim, self.hid_dim, self.normalize)])
		self.Convs.extend([pyg_nn.DenseSAGEConv(self.hid_dim, self.hid_dim, self.normalize) for i in range(self.num_layer-2)])
		self.Convs.extend([pyg_nn.DenseSAGEConv(self.hid_dim, self.out_dim, self.normalize) ])
		self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(self.hid_dim) for i in range(self.num_layer-1)])
		self.bns.extend([torch.nn.BatchNorm1d(self.out_dim) ])
		#
		if lin is True : 
			self.linR = torch.nn.Linear((self.num_layer-1)*self.hid_dim+self.out_dim, self.out_dim )
		else : 
			self.linR = None
	#
	def bn (self, step , x ):
		batch_size , num_nodes , num_features = x.size()
		x = x.view(-1, num_features)
		x = self.bns[step](x)
		x = x.view(batch_size, num_nodes, num_features)
		return x 
	#
	def forward(self, x, adj, mask=None):
		batch_size, num_nodes, in_channels = x.size()
		layer_dict = {}
		key_list = ["x_"+str(i) for i in range(self.num_layer+1)]
		layer_dict['x_0'] = [x]
		for L in range(self.num_layer):
			key = key_list[L+1]
			layer_dict[key] = [self.bn(L, F.elu(self.Convs[L](layer_dict[key_list[L]][0], adj, mask)))]
		x = torch.cat([layer_dict[key][0] for key in key_list[1:]], dim= -1)
		if self.lin is True:
			x = F.elu(self.linR(x)) 
		return x




class DIFFPOOL(torch.nn.Module):
	def __init__(self, conv_num, input_dim, hidden_dim, output_dim, max_nodes, num_pool, ratio , dropout):
		super(DIFFPOOL, self).__init__()
		#
		self.conv_num = conv_num
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.max_nodes = max_nodes
		self.num_pool = num_pool
		self.ratio = ratio
		self.dropout = dropout
		#
		self.num_nodes = [ceil(self.ratio * self.max_nodes)]
		self.check_num_nodes()
		#
		self.POOLS = torch.nn.ModuleList([GNN(self.conv_num, self.input_dim, self.hidden_dim, self.num_nodes[1] )])
		self.POOLS.extend([GNN(self.conv_num, self.conv_num * self.hidden_dim, self.hidden_dim, self.num_nodes[i+2]) for i in range(self.num_pool-1)])
		#
		self.EMBED = torch.nn.ModuleList([GNN(self.conv_num, self.input_dim, self.hidden_dim, self.hidden_dim, lin=False)])
		self.EMBED.extend([GNN(self.conv_num, self.conv_num * self.hidden_dim, self.hidden_dim, self.hidden_dim, lin=False) for i in range(self.num_pool)])
		#
		self.lin1 = torch.nn.Linear(self.conv_num * self.hidden_dim, self.hidden_dim)
		self.lin2 = torch.nn.Linear(self.hidden_dim, self.output_dim)
		#
		self.reset_parameters()
	#
	def reset_parameters(self):
		self.lin1.reset_parameters()
		self.lin2.reset_parameters()
	#
	def check_num_nodes(self):
		for L in range(self.num_pool):
			this_node = ceil( self.ratio * self.num_nodes[-1])
			self.num_nodes.append(this_node)
	#	
	def forward(self, x, adj, mask = None):
		x = F.normalize(x)
		s = self.POOLS[0](x, adj, mask)
		x = self.EMBED[0](x, adj, mask)
		x, adj, l, e = pyg_nn.dense_diff_pool(x, adj, s, mask)
		#
		for i in range(self.num_pool-1):
			s = self.POOLS[i+1](x, adj)
			x = self.EMBED[i+1](x, adj)
			x, adj, l, e = pyg_nn.dense_diff_pool(x, adj, s)
		x = self.EMBED[-1](x, adj)
		#
		x = x.mean(dim=1)
		x = F.elu(self.lin2(self.lin1(x)))
		x = F.dropout(x, p=self.dropout, training=self.training)
		x = torch.tanh(x)
		return x




class MY_diffpool_parallel(torch.nn.Module) :
	def __init__(self, G_layer_chem, G_indim_chem, G_hiddim_chem, G_layer_exp, G_indim_exp, G_hiddim_exp, G_pool_chem, G_pool_exp, G_ratio_chem, G_ratio_exp, layers_1, layers_2, layers_3, cell_dim ,out_dim, inDrop, drop):
		super(MY_diffpool_parallel, self).__init__()
		#
		self.G_layer_chem = G_layer_chem
		self.G_indim_chem = G_indim_chem
		self.G_hiddim_chem = G_hiddim_chem
		self.G_layer_exp = G_layer_exp
		self.G_indim_exp = G_indim_exp
		self.G_hiddim_exp = G_hiddim_exp
		self.G_pool_chem = G_pool_chem
		self.G_pool_exp = G_pool_exp
		self.G_ratio_chem = G_ratio_chem
		self.G_ratio_exp = G_ratio_exp
		self.layers_1 = [int(a) for a in layers_1]
		self.layers_2 = [int(a) for a in layers_2]
		self.layers_3 = [int(a) for a in layers_3]
		self.cell_dim = cell_dim
		self.out_dim = out_dim
		self.inDrop = inDrop
		self.drop = drop
		self.tanh = torch.nn.Tanh()
		#
		self.G_1_C = DIFFPOOL(self.G_layer_chem, self.G_indim_chem, self.G_hiddim_chem, self.G_hiddim_chem, 50, self.G_pool_chem, self.G_ratio_chem, self.inDrop )
		self.G_2_C = DIFFPOOL(self.G_layer_chem, self.G_indim_chem, self.G_hiddim_chem, self.G_hiddim_chem, 50, self.G_pool_chem, self.G_ratio_chem, self.inDrop )
		self.G_1_E = DIFFPOOL(self.G_layer_exp, self.G_indim_exp, self.G_hiddim_exp, self.G_hiddim_exp, 978, self.G_pool_exp, self.G_ratio_exp, self.inDrop )
		self.G_2_E = DIFFPOOL(self.G_layer_exp, self.G_indim_exp, self.G_hiddim_exp, self.G_hiddim_exp, 978, self.G_pool_exp, self.G_ratio_exp, self.inDrop )
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
		#
	def forward (self, Drug1_F, Drug2_F, Drug1_ADJ, Drug2_ADJ, EXP1, EXP2, EXP_ADJ, EXP_ADJ_WGT, syn, cell ) :
		G_1_C_out = self.G_1_C(Drug1_F, Drug1_ADJ)
		G_2_C_out = self.G_2_C(Drug2_F, Drug2_ADJ)
		G_1_E_out = self.G_1_E(EXP1, EXP_ADJ)
		G_2_E_out = self.G_2_E(EXP2, EXP_ADJ)
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





결과 확인 
WORK_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_GPU/PJ01.TRIAL.3_9_3/'

G_NAME='VAR_3' # Diff Pooling

MY_chem_A_feat = torch.load(WORK_PATH+'0902.{}.MY_chem_A_feat.pt'.format(G_NAME))
MY_chem_B_feat = torch.load(WORK_PATH+'0902.{}.MY_chem_B_feat.pt'.format(G_NAME))
MY_chem_A_adj = torch.load(WORK_PATH+'0902.{}.MY_chem_A_adj.pt'.format(G_NAME))
MY_chem_B_adj= torch.load(WORK_PATH+'0902.{}.MY_chem_B_adj.pt'.format(G_NAME))
MY_exp_F_A = torch.load(WORK_PATH+'0902.{}.MY_exp_F_A.pt'.format(G_NAME))
MY_exp_F_B = torch.load(WORK_PATH+'0902.{}.MY_exp_F_B.pt'.format(G_NAME))
MY_Cell = torch.load(WORK_PATH+'0902.{}.MY_Cell.pt'.format(G_NAME))
MY_syn =  torch.load(WORK_PATH+'0902.{}.MY_syn.pt'.format(G_NAME))




DC_PATH = '/st06/jiyeonH/13.DD_SESS/DrugComb.1.5/' 
IDK_PATH = '/st06/jiyeonH/13.DD_SESS/ideker/' 
LINCS_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/' 
TARGET_PATH = '/st06/jiyeonH/13.DD_SESS/merged_target/'


hunet_dir = '/st06/jiyeonH/13.DD_SESS/HumanNetV3/'


hunet_gsp = pd.read_csv(hunet_dir+'HumanNet-GSP.tsv', sep = '\t', header = None)
hunet_gsp.columns = ['G_A','G_B']

L_matching_list = pd.read_csv(IDK_PATH+'L_12_string.csv', sep = '\t')

hnet_IAS_L1 = hunet_gsp[hunet_gsp['G_A'].isin(L_matching_list.entrez)]
hnet_IAS_L2 = hnet_IAS_L1[hnet_IAS_L1['G_B'].isin(L_matching_list.entrez)] # 20232

len(set(list(hnet_IAS_L2['G_A']) + list(hnet_IAS_L2['G_B']))) # 611
ID_G = nx.from_pandas_edgelist(hnet_IAS_L2, 'G_A', 'G_B')

ESSEN_NAMES = list(set(L_matching_list['entrez']))

MSSNG = [a for  a in ESSEN_NAMES if a not in list(ID_G.nodes)]
# MSSNG = L_matching_list[L_matching_list['PPI_name'].isin(list(ID_G.nodes))==False]['PPI_name']
for nn in list(MSSNG):
	ID_G.add_node(nn)

# edge 3871
ID_GENE_ORDER_mini = list(ID_G.nodes()) # 978
ID_ADJ = nx.adjacency_matrix(ID_G)
ID_ADJ_tmp = torch.LongTensor(ID_ADJ.toarray())
ID_ADJ_IDX = ID_ADJ_tmp.to_sparse().indices()  # [2, 7742]
ID_WEIGHT = [] # len : 3871 -> 7742

ID_WEIGHT_SCORE = [1 for a in range(ID_ADJ_IDX.shape[1])]


new_node_name = []
for a in ID_G.nodes():
	tmp_name = list(set(L_matching_list[L_matching_list.entrez == a ]['L_gene_symbol'])) # 6118
	new_node_name = new_node_name + tmp_name

mapping = {list(ID_G.nodes())[a]:new_node_name[a] for a in range(len(new_node_name))}

ID_G_RE = nx.relabel_nodes(ID_G, mapping)

MY_G = ID_G_RE 
MY_WEIGHT_SCORE = ID_WEIGHT_SCORE # SCORE





JY_GRAPH = MY_G
JY_GRAPH_ORDER = MY_G.nodes()
JY_ADJ = nx.adjacency_matrix(JY_GRAPH)

JY_ADJ_tmp = torch.LongTensor(JY_ADJ.toarray())
JY_ADJ_IDX = JY_ADJ_tmp.to_sparse().indices()
JY_IDX_WEIGHT = MY_WEIGHT_SCORE




seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


norm = 'tanh_norm'
train_data, val_data, test_data = prepare_data_GCN(MY_chem_A_feat, MY_chem_B_feat, MY_chem_A_adj, MY_chem_B_adj, MY_exp_F_A, MY_exp_F_B, MY_syn, MY_Cell, norm)




# WEIGHT 
ys = train_data['y'].squeeze().tolist()
min_s = np.amin(ys)
loss_weight = np.log(train_data['y'] - min_s + np.e)
JY_IDX_WEIGHT_T = torch.Tensor(JY_IDX_WEIGHT).view(1,-1)



# DATA check 
T_train = DATASET_DIFF_W_FT(
	torch.Tensor(train_data['drug1_feat']), torch.Tensor(train_data['drug2_feat']), 
	torch.Tensor(train_data['drug1_adj']), torch.Tensor(train_data['drug2_adj']),
	train_data['EXP_F_A'], train_data['EXP_F_B'], 
	JY_ADJ_tmp.unsqueeze(0), JY_IDX_WEIGHT_T, 
	torch.Tensor(train_data['y']),
	torch.Tensor(train_data['cell']))

T_val = DATASET_DIFF_W_FT(
	torch.Tensor(val_data['drug1_feat']), torch.Tensor(val_data['drug2_feat']), 
	torch.Tensor(val_data['drug1_adj']), torch.Tensor(val_data['drug2_adj']),
	val_data['EXP_F_A'], val_data['EXP_F_B'], 
	JY_ADJ_tmp.unsqueeze(0), JY_IDX_WEIGHT_T, 
	torch.Tensor(val_data['y']),
	torch.Tensor(val_data['cell']))
	
T_test = DATASET_DIFF_W_FT(
	torch.Tensor(test_data['drug1_feat']), torch.Tensor(test_data['drug2_feat']), 
	torch.Tensor(test_data['drug1_adj']), torch.Tensor(test_data['drug2_adj']),
	test_data['EXP_F_A'], test_data['EXP_F_B'], 
	JY_ADJ_tmp.unsqueeze(0), JY_IDX_WEIGHT_T, 
	torch.Tensor(test_data['y']),
	torch.Tensor(test_data['cell']))

RAY_train = ray.put(T_train)
RAY_val = ray.put(T_val)
RAY_test = ray.put(T_test)
RAY_loss_weight = ray.put(loss_weight)



##################### GPU 결과 확인 ##################
##################### GPU 결과 확인 ##################
##################### GPU 결과 확인 ##################
##################### GPU 결과 확인 ##################

import pandas as pd 
import matplotlib.pyplot as plt
from ray.tune import ExperimentAnalysis
import pickle
import math
import torch


def RAY_TEST_MODEL(my_config, model_path, model_name, PRJ_PATH, Trial_name, G_NAME, number): 
	use_cuda =  False
	T_test = ray.get(RAY_test)
	Test_loader = torch.utils.data.DataLoader(T_test, batch_size = my_config['config/batch_size'].item(),  shuffle =False, num_workers=my_config['config/n_workers'].item())
	#
	G_layer = my_config['config/G_layer'].item()
	G_hiddim = my_config['config/G_hiddim'].item()
	G_pool_1 = my_config['config/G_pool_1'].item()
	G_pool_2 = my_config['config/G_pool_2'].item()
	G_ratio_1 = my_config['config/G_ratio_1'].item()
	G_ratio_2 = my_config['config/G_ratio_2'].item()
	dsn1_layers = [my_config['config/feat_size_0'].item(), my_config['config/feat_size_1'].item(), my_config['config/feat_size_2'].item()]
	dsn2_layers = [my_config['config/feat_size_0'].item(), my_config['config/feat_size_1'].item(), my_config['config/feat_size_2'].item()] 
	snp_layers = [my_config['config/feat_size_3'].item() , my_config['config/feat_size_4'].item()]
	inDrop = my_config['config/dropout_1'].item()
	Drop = my_config['config/dropout_2'].item()
	#       
	best_model = MY_diffpool_parallel(
				G_layer, T_test.gcn_drug1_F.shape[-1] , G_hiddim,
				G_layer, 2, G_hiddim,
				G_pool_1, G_pool_2, G_ratio_1, G_ratio_2, 
				dsn1_layers, dsn2_layers, snp_layers, 17, 1,
				inDrop, Drop
				)
	#
	if torch.cuda.is_available():
		best_model = best_model.cuda()
		if torch.cuda.device_count() > 1 :
			best_model = torch.nn.DataParallel(best_model)
	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	if torch.cuda.is_available():
		state_dict = torch.load(os.path.join(model_path, model_name))
	else:
		state_dict = torch.load(os.path.join(model_path, model_name),map_location=torch.device('cpu'))
	if type(state_dict) == tuple:
		best_model.load_state_dict(state_dict[0])
	else : 
		best_model.load_state_dict(state_dict)	#
	#
	#
	best_model.eval()
	test_loss = 0.0
	PRED_list = []
	Y_list = test_data['y'].squeeze().tolist()
	with torch.no_grad():
		best_model.eval()
		for batch_idx_t, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell) in enumerate(Test_loader):
			adj = adj.float()
			adj = adj.squeeze(1)
			if use_cuda:
				drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell = drug1_f.cuda(), drug2_f.cuda(), drug1_a.cuda(), drug2_a.cuda(), expA.cuda(), expB.cuda(), adj.cuda(), adj_w.cuda(), y.cuda(), cell.cuda()
			output = best_model(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell) 
			MSE = torch.nn.MSELoss()
			loss = MSE(output, y)
			test_loss = test_loss + loss.item()
			outputs = output.squeeze().tolist()
			PRED_list = PRED_list + outputs
	TEST_LOSS = test_loss/(batch_idx_t+1)
	R__T = TEST_LOSS
	R__1 , R__2 = jy_corrplot(PRED_list, Y_list, PRJ_PATH,'P.{}.{}.{}_model'.format(Trial_name, G_NAME, number) )
	return  R__T, R__1, R__2




def jy_corrplot(PRED_list, Y_list, path, plotname ):
	jplot = sns.jointplot(x=PRED_list, y=Y_list, ci=68, kind='reg')
	pr,pp = stats.pearsonr(PRED_list, Y_list)
	print("Pearson correlation is {} and related p_value is {}".format(pr, pp))
	sr,sp = stats.spearmanr(PRED_list, Y_list)
	print("Spearman correlation is {} and related p_value is {}".format(sr, sp))
	jplot.ax_joint.annotate(f'$pearson = {pr:.3f}, spearman = {sr:.3f}$',xy=(min(PRED_list)+ 0.01, max(Y_list)- 0.01 ), ha='left', va='center',)
	jplot.ax_joint.scatter(PRED_list, Y_list)
	jplot.set_axis_labels(xlabel='Predicted', ylabel='Answer', size=15)
	jplot.figure.savefig('{}/{}.corrplot.png'.format(path, plotname), bbox_inches = 'tight')
	return pr, sr









PRJ_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_GPU/PJ01.TRIAL.3_9_3/'
ANA_DF = pd.read_csv(PRJ_PATH+'RAY_ANA_DF.P01.3_9_3.csv')
with open(PRJ_PATH+'RAY_ANA_DF.P01.3_9_3.pickle', 'rb') as f:
	ANA_ALL_DF = pickle.load(f)


(1) 마지막 모델 확인 

DF_KEY = list(ANA_DF.sort_values('ValLoss')['logdir'])[0]
DF_KEY
list(ANA_DF.sort_values('ValLoss')['ValLoss'])[0]

mini_df = ANA_ALL_DF[DF_KEY]
plot_loss(list(mini_df.TrainLoss), list(mini_df.ValLoss), 
PRJ_PATH, 'TRIAL.3_9_3.{}.BEST.loss'.format(G_NAME) )


TOPVAL_PATH = DF_KEY
mini_df = ANA_ALL_DF[DF_KEY]
my_config = ANA_DF[ANA_DF.logdir==DF_KEY]
R_1_V = list(ANA_DF.sort_values('ValLoss')['ValLoss'])[0]
Trial_name = '3_9_3'
R_1_T, R_1_1, R_1_2 = RAY_TEST_MODEL(my_config, PRJ_PATH, 'M1_model.pth', PRJ_PATH, Trial_name, G_NAME, 'M1')





(2) 중간 체크포인트 확인 

cck_num =mini_df[mini_df.ValLoss==min(mini_df.ValLoss)].index.item()
checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
TOPVAL_PATH = DF_KEY + checkpoint
TOPVAL_PATH

R_2_V = min(mini_df.ValLoss)
Trial_name = '3_9_2'
R_2_T, R_2_1, R_2_2 = RAY_TEST_MODEL(my_config, PRJ_PATH, 'M2_checkpoint', PRJ_PATH, Trial_name, G_NAME, 'M2')







(3) 최소 값 확인

import numpy as np
TOT_min = np.Inf
TOT_key = ""
for key in ANA_ALL_DF.keys():
	trial_min = min(ANA_ALL_DF[key]['ValLoss'])
	if trial_min < TOT_min :
		TOT_min = trial_min
		TOT_key = key

mini_df = ANA_ALL_DF[TOT_key]
TOPVAL_PATH = TOT_key
my_config = ANA_DF[ANA_DF.logdir==TOT_key]
cck_num =mini_df[mini_df.ValLoss==min(mini_df.ValLoss)].index.item()
checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
TOPVAL_PATH = TOT_key + checkpoint


plot_loss(list(mini_df.TrainLoss), list(mini_df.ValLoss), 
PRJ_PATH, 'TRIAL.3_9_3.{}.MIN.loss'.format(G_NAME) )


R_3_V = min(mini_df.ValLoss)
Trial_name = '3_9_3'
R_3_T, R_3_1, R_3_2 = RAY_TEST_MODEL(my_config, PRJ_PATH, 'M4_checkpoint', PRJ_PATH, Trial_name, G_NAME, 'M4')





def final_result(R_1_V, R_1_T, R_1_1, R_1_2, R_2_V, R_2_T, R_2_1, R_2_2, R_3_V, R_3_T, R_3_1, R_3_2) :
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


final_result(R_1_V, R_1_T, R_1_1, R_1_2, R_2_V, R_2_T, R_2_1, R_2_2, R_3_V, R_3_T, R_3_1, R_3_2)





















######### data testing 

MY_chem_A_feat = torch.empty(size=(30, 50, 64))
MY_chem_B_feat= torch.empty(size=(30, 50, 64))
MY_chem_A_adj = torch.empty(size=(30, 50, 50))
MY_chem_B_adj= torch.empty(size=(30, 50, 50))

MY_exp_A = torch.empty(size=(30, 978))
MY_exp_B = torch.empty(size=(30, 978))
MY_exp_AB = torch.empty(size=(30, 978, 2))

MY_exp_F_A = torch.empty(size=(30, 978, 2))
MY_exp_F_B = torch.empty(size=(30, 978, 2))


MY_Cell = torch.empty(size=(30, cell_one_hot.shape[1]))
MY_tgt_A = torch.empty(size=(30, 978))
MY_tgt_B = torch.empty(size=(30, 978))
MY_syn =  torch.empty(size=(30,1))






for IND in range(MY_chem_A_feat.shape[0]): 
	DrugA_SIG = A_B_C_S_SET.iloc[IND,]['BETA_sig_id_x']
	DrugB_SIG = A_B_C_S_SET.iloc[IND,]['BETA_sig_id_y']
	Cell = A_B_C_S_SET.iloc[IND,]['DrugCombCello']
	#
	k=1
	DrugA_Feat, DrugA_ADJ = get_CHEM(DrugA_SIG, k)
	DrugB_Feat, DrugB_ADJ = get_CHEM(DrugB_SIG, k)
	#
	EXP_A, EXP_B, EXP_AB, LINCS = get_LINCS_data(DrugA_SIG, DrugB_SIG)
	EXP_A = list(EXP_A.iloc[:,1])
	EXP_B = list(EXP_B.iloc[:,1])
	#
	Cell_Vec = get_cell(DrugA_SIG, DrugB_SIG, Cell)
	#
	TGT_A = get_targets(DrugA_SIG)
	TGT_B = get_targets(DrugB_SIG)
	#
	AB_SYN = get_synergy_data(DrugA_SIG, DrugB_SIG, Cell)
	#
	FEAT_A = torch.Tensor(np.array([EXP_A, TGT_A]).T)
	FEAT_B = torch.Tensor(np.array([EXP_B, TGT_B]).T)
	#
	MY_chem_A_feat[IND] = torch.Tensor(DrugA_Feat)
	MY_chem_B_feat[IND] = torch.Tensor(DrugB_Feat)
	MY_chem_A_adj[IND] = torch.Tensor(DrugA_ADJ)
	MY_chem_B_adj[IND] = torch.Tensor(DrugB_ADJ)
	MY_exp_A[IND] = torch.Tensor(EXP_A)
	MY_exp_B[IND] = torch.Tensor(EXP_B)
	MY_exp_AB[IND] = torch.Tensor(EXP_AB).unsqueeze(0)
	MY_Cell[IND] = Cell_Vec
	MY_tgt_A[IND] = torch.Tensor(TGT_A)
	MY_tgt_B[IND] = torch.Tensor(TGT_B)
	MY_syn[IND] = torch.Tensor([AB_SYN])
	MY_exp_F_A[IND] = FEAT_A
	MY_exp_F_B[IND] = FEAT_B







d_f = train_data['drug1_feat'][0:10]
d_a = train_data['drug1_adj'][0:10]



DrugA_SIG = 'MOAR001_A375_24H:H10'
DrugB_SIG = 'LJP009_A375_24H:O24'
Cell = 'CVCL_0132'
EXP_A, EXP_B, EXP_AB, LINCS = get_LINCS_data(DrugA_SIG, DrugB_SIG)

EXP_A = list(EXP_A.iloc[:,1])
TGT_A = get_targets(DrugA_SIG)

FEAT_A = torch.Tensor(np.array([EXP_A, TGT_A]).T)



self.G_convs_1_chem = torch.nn.ModuleList([pyg_nn.SAGEConv(self.G_indim_chem, self.G_hiddim_chem)])
self.G_convs_1_chem.extend([pyg_nn.SAGEConv(self.G_hiddim_chem, self.G_hiddim_chem) for i in range(self.G_layer_chem-2)])
self.G_convs_1_chem.extend([pyg_nn.SAGEConv(self.G_hiddim_chem, self.G_hiddim_chem)])
self.G_bns_1_chem = torch.nn.ModuleList([torch.nn.BatchNorm1d(self.G_hiddim_chem) for i in range(self.G_layer_chem-1)])




class GNN(torch.nn.Module):
	def __init__(self, num_layer, in_dim, hid_dim, out_dim, normalize=False, lin=True): 
		super(GNN, self).__init__()
		self.num_layer = num_layer
		self.in_dim = in_dim
		self.hid_dim = hid_dim
		self.out_dim = out_dim
		self.normalize = normalize
		self.lin = lin
		#
		self.Convs = torch.nn.ModuleList([pyg_nn.DenseSAGEConv(self.in_dim, self.hid_dim, self.normalize)])
		self.Convs.extend([pyg_nn.DenseSAGEConv(self.hid_dim, self.hid_dim, self.normalize) for i in range(self.num_layer-2)])
		self.Convs.extend([pyg_nn.DenseSAGEConv(self.hid_dim, self.out_dim, self.normalize) ])
		self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(self.hid_dim) for i in range(self.num_layer-1)])
		self.bns.extend([torch.nn.BatchNorm1d(self.out_dim) ])
		#
		if lin is True : 
			self.linR = torch.nn.Linear((self.num_layer-1)*self.hid_dim+self.out_dim, self.out_dim )
		else : 
			self.linR = None
	#
	def bn (self, step , x ):
		batch_size , num_nodes , num_features = x.size()
		x = x.view(-1, num_features)
		x = self.bns[step](x)
		x = x.view(batch_size, num_nodes, num_features)
		return x 
	#
	def forward(self, x, adj, mask=None):
		batch_size, num_nodes, in_channels = x.size()
		layer_dict = {}
		key_list = ["x_"+str(i) for i in range(self.num_layer+1)]
		layer_dict['x_0'] = [x]
		for L in range(self.num_layer):
			key = key_list[L+1]
			layer_dict[key] = [self.bn(L, F.elu(self.Convs[L](layer_dict[key_list[L]][0], adj, mask)))]
		x = torch.cat([layer_dict[key][0] for key in key_list[1:]], dim= -1)
		if self.lin is True:
			x = F.elu(self.linR(x)) 
		return x








class DIFFPOOL(torch.nn.Module):
	def __init__(self, conv_num, input_dim, hidden_dim, output_dim, max_nodes, num_pool, ratio , dropout):
		super(DIFFPOOL, self).__init__()
		#
		self.conv_num = conv_num
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.max_nodes = max_nodes
		self.num_pool = num_pool
		self.ratio = ratio
		self.dropout = dropout
		#
		self.num_nodes = [ceil(self.ratio * self.max_nodes)]
		self.check_num_nodes()
		#
		self.POOLS = torch.nn.ModuleList([GNN(self.conv_num, self.input_dim, self.hidden_dim, self.num_nodes[1] )])
		self.POOLS.extend([GNN(self.conv_num, self.conv_num * self.hidden_dim, self.hidden_dim, self.num_nodes[i+2]) for i in range(self.num_pool-1)])
		#
		self.EMBED = torch.nn.ModuleList([GNN(self.conv_num, self.input_dim, self.hidden_dim, self.hidden_dim, lin=False)])
		self.EMBED.extend([GNN(self.conv_num, self.conv_num * self.hidden_dim, self.hidden_dim, self.hidden_dim, lin=False) for i in range(self.num_pool)])
		#
		self.lin1 = torch.nn.Linear(self.conv_num * self.hidden_dim, self.hidden_dim)
		self.lin2 = torch.nn.Linear(self.hidden_dim, self.output_dim)
		#
		self.reset_parameters()
	#
	def reset_parameters(self):
		self.lin1.reset_parameters()
		self.lin2.reset_parameters()
	#
	def check_num_nodes(self):
		for L in range(self.num_pool):
			this_node = ceil( self.ratio * self.num_nodes[-1])
			self.num_nodes.append(this_node)
	#	
	def forward(self, x, adj, mask = None):
		x = F.normalize(x)
		s = self.POOLS[0](x, adj, mask)
		x = self.EMBED[0](x, adj, mask)
		x, adj, l, e = pyg_nn.dense_diff_pool(x, adj, s, mask)
		#
		for i in range(self.num_pool-1):
			s = self.POOLS[i+1](x, adj)
			x = self.EMBED[i+1](x, adj)
			x, adj, l, e = pyg_nn.dense_diff_pool(x, adj, s)
		x = self.EMBED[-1](x, adj)
		#
		x = x.mean(dim=1)
		x = F.elu(self.lin2(self.lin1(x)))
		x = F.dropout(x, p=self.dropout, training=self.training)
		x = torch.tanh(x)
		return x




class MY_diffpool_parallel(torch.nn.Module) :
	def __init__(self, G_layer_chem, G_indim_chem, G_hiddim_chem, G_layer_exp, G_indim_exp, G_hiddim_exp, G_pool_chem, G_pool_exp, G_ratio_chem, G_ratio_exp, layers_1, layers_2, layers_3, cell_dim ,out_dim, inDrop, drop):
		super(MY_diffpool_parallel, self).__init__()
		#
		self.G_layer_chem = G_layer_chem
		self.G_indim_chem = G_indim_chem
		self.G_hiddim_chem = G_hiddim_chem
		self.G_layer_exp = G_layer_exp
		self.G_indim_exp = G_indim_exp
		self.G_hiddim_exp = G_hiddim_exp
		self.G_pool_chem = G_pool_chem
		self.G_pool_exp = G_pool_exp
		self.G_ratio_chem = G_ratio_chem
		self.G_ratio_exp = G_ratio_exp
		self.layers_1 = [int(a) for a in layers_1]
		self.layers_2 = [int(a) for a in layers_2]
		self.layers_3 = [int(a) for a in layers_3]
		self.cell_dim = cell_dim
		self.out_dim = out_dim
		self.inDrop = inDrop
		self.drop = drop
		self.tanh = torch.nn.Tanh()
		#
		self.G_1_C = DIFFPOOL(self.G_layer_chem, self.G_indim_chem, self.G_hiddim_chem, self.G_hiddim_chem, 50, self.G_pool_chem, self.G_ratio_chem, self.inDrop )
		self.G_2_C = DIFFPOOL(self.G_layer_chem, self.G_indim_chem, self.G_hiddim_chem, self.G_hiddim_chem, 50, self.G_pool_chem, self.G_ratio_chem, self.inDrop )
		self.G_1_E = DIFFPOOL(self.G_layer_exp, self.G_indim_exp, self.G_hiddim_exp, self.G_hiddim_exp, 978, self.G_pool_exp, self.G_ratio_exp, self.inDrop )
		self.G_2_E = DIFFPOOL(self.G_layer_exp, self.G_indim_exp, self.G_hiddim_exp, self.G_hiddim_exp, 978, self.G_pool_exp, self.G_ratio_exp, self.inDrop )
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
		#
	def forward (self, Drug1_F, Drug2_F, Drug1_ADJ, Drug2_ADJ, EXP1, EXP2, EXP_ADJ, EXP_ADJ_WGT, syn, cell ) :
		G_1_C_out = self.G_1_C(Drug1_F, Drug1_ADJ)
		G_2_C_out = self.G_2_C(Drug2_F, Drug2_ADJ)
		G_1_E_out = self.G_1_E(EXP1, EXP_ADJ)
		G_2_E_out = self.G_2_E(EXP2, EXP_ADJ)
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




Drug1_F, Drug2_F, Drug1_ADJ, Drug2_ADJ, EXP1, EXP2, EXP_ADJ, EXP_ADJ_WGT, syn, cell

test_pall = MY_diffpool_parallel(G_layer_chem, G_indim_chem, G_hiddim_chem, 
G_layer_exp, G_indim_exp, G_hiddim_exp, 
G_pool_chem, G_pool_exp, G_ratio_chem, G_ratio_exp, 
layers_1, layers_2, layers_3, cell_dim ,out_dim, inDrop, drop)

test_pall(d1_f, d2_f, d1_a, d2_a, e1_f, e2_f, e1_a.float(), e1_a, y, torch.Tensor(cell)  )




class DATASET_DIFF_W_FT(Dataset):
	def __init__(self, gcn_drug1_F, gcn_drug2_F, gcn_drug1_ADJ, gcn_drug2_ADJ, gcn_exp_A, gcn_exp_B, gcn_adj, gcn_adj_weight, syn_ans, cell_info):
		self.gcn_drug1_F = gcn_drug1_F
		self.gcn_drug2_F = gcn_drug2_F
		self.gcn_drug1_ADJ = gcn_drug1_ADJ
		self.gcn_drug2_ADJ = gcn_drug2_ADJ
		self.gcn_exp_A = gcn_exp_A
		self.gcn_exp_B = gcn_exp_B
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
		return self.gcn_drug1_F[index], self.gcn_drug2_F[index], self.gcn_drug1_ADJ[index], self.gcn_drug2_ADJ[index], self.gcn_exp_A[index], self.gcn_exp_B[index], self.gcn_adj, self.gcn_adj_weight, self.syn_ans[index] , self.cell_info[index]



T_train = DATASET_DIFF_W_FT(
	MY_chem_A_feat, MY_chem_B_feat, 
	MY_chem_A_adj, MY_chem_B_adj,
	MY_exp_F_A, MY_exp_F_B,
	JY_ADJ_tmp.unsqueeze(0), JY_IDX_WEIGHT_T.unsqueeze(0), MY_syn,
	MY_Cell)




loaders = {
'train' : torch.utils.data.DataLoader(T_train, batch_size = 2,  shuffle =False),
}



# 이거 실험중이었음 
for batch_idx_t, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell) in enumerate(loaders['train']):
	adj = adj.squeeze(1)
	test_pall(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj.float(), adj_w, y, cell  )








def train(epoch):
	model.train()
	loss_all = 0
	for step, data in enumerate(train_loader):
		X = data[0]
		ADJ = data[1].float()
		ANS = data[2].flatten().long()
		#
		optimizer.zero_grad()
		output, _, _ = model(X, ADJ)
		loss = F.nll_loss(output, ANS.view(-1))
		loss.backward()
		loss_all += ANS.size(0) * loss.item()
		optimizer.step()
	return loss_all / len(train_loader.dataset.indices)

def test(loader):
	model.eval()
	correct = 0
	for step, data in enumerate(test_loader):
		with torch.no_grad():
			X = data[0]
			ADJ = data[1].float()
			ANS = data[2].flatten().long()
			#
			pred = model(X, ADJ)[0].max(dim=1)[1]
			correct += pred.eq(ANS.view(-1)).sum().item()
	RESULT = correct / len(test_loader.dataset.indices)
	print(pred)
	return RESULT









