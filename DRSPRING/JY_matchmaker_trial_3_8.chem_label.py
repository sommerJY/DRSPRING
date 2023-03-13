
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
from ray.tune import ExperimentAnalysis

import numpy as np
from rdkit import Chem
from rdkit.Chem.QED import qed
import sys
import os
import pandas as pd


WORK_PATH = '/home01/k006a01/PRJ.01/TRIAL_3.8/'
DC_PATH = '/home01/k006a01/01.DATA/DrugComb/'
IDK_PATH = '/home01/k006a01/01.DATA/IDK/'
LINCS_PATH = '/home01/k006a01/01.DATA/LINCS/' #
TARGET_PATH = '/home01/k006a01/01.DATA/TARGET/'
cid_PATH = '/home01/k006a01/01.DATA/PC_ATC_LIST/'


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
cid_PATH = '/st06/jiyeonH/11.TOX/Pubchem_Classi_176/'

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

G_NAME='HN_GSP'

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
A_B_C_S_SET = A_B_C_S[['drug_row_cid','drug_col_cid','BETA_sig_id_x','BETA_sig_id_y','DrugCombCello']].drop_duplicates()
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



# 50개 필터로 data 필터링 

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

# cid_PATH = '/st06/jiyeonH/11.TOX/Pubchem_Classi_176/'
# sig_id = 'LUNG001_A549_24H:BRD-K48935217:10'








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









# ATC 11701 -> 6587 이거보다 더 줄어들듯 -> 5362 -> 5128

pds_ATC = pd.DataFrame(columns=['CID','level1','level2','level3','level4','level5'])

for cid in CELLO_DC_BETA_cids:
	with open(cid_PATH+'PC_C.{}.json'.format(cid)) as json_file :
		tmp_json =json.load(json_file)
		tmp_re = tmp_json['Hierarchies']['Hierarchy']
		for dat in tmp_re :
			if dat['SourceID'] == 'ATCTree' : # 'NCI Thesaurus (NCIt)'
				print(cid)
				nodes = dat['Node']
				classes = [nn['Information']['Name'] for nn in nodes]
				ATC_keys = ['level1','level2','level3','level4','level5']
				tmp_ATC = {ATC_keys[i] : [classes[i]] for i in range(5)}
				tmp_ATC['CID'] = [cid]
				pds_ATC = pd.concat([pds_ATC, pd.DataFrame(tmp_ATC)])



A_B_C_S_filt = A_B_C_S_SET[A_B_C_S_SET.drug_row_cid.isin(list(pds_ATC.CID))]
A_B_C_S_filt = A_B_C_S_filt[A_B_C_S_filt.drug_col_cid.isin(list(pds_ATC.CID))]

A_B_C_S_filt = A_B_C_S_filt.reset_index(drop=True) # 5128



pds_ATC_L5 = pds_ATC[['CID','level5']]
dict_ATC_key = list(set(pds_ATC_L5['level5']))
dict_ATC_key.sort()
pds_ATC_L5['L5_OH'] = [dict_ATC_key.index(a) for a in list(pds_ATC_L5['level5'])]
pds_ATC_L5 = pds_ATC_L5.reset_index(drop=True)
atc_one_hot = torch.nn.functional.one_hot(torch.Tensor(list(pds_ATC_L5['L5_OH'])).long())
# [list(pds_ATC['level5']).count(a) for a in list(set(pds_ATC['level5']))]



def get_chem_category(sig_id) :
	tmp_df1 = BETA_CID_CELLO_SIG[BETA_CID_CELLO_SIG.sig_id == sig_id]
	CID = int(tmp_df1.pubchem_cid.item())
	ind = pds_ATC_L5[pds_ATC_L5['CID'] == CID].index.item()
	atc_oh = atc_one_hot[ind]
	return(atc_oh)



MY_chem_A_feat = torch.empty(size=(A_B_C_S_filt.shape[0], 50, 64))
MY_chem_B_feat= torch.empty(size=(A_B_C_S_filt.shape[0], 50, 64))
MY_chem_A_adj = torch.empty(size=(A_B_C_S_filt.shape[0], 50, 50))
MY_chem_B_adj= torch.empty(size=(A_B_C_S_filt.shape[0], 50, 50))
MY_exp_A = torch.empty(size=(A_B_C_S_filt.shape[0], 978))
MY_exp_B = torch.empty(size=(A_B_C_S_filt.shape[0], 978))
MY_exp_AB = torch.empty(size=(A_B_C_S_filt.shape[0], 978, 2))
MY_tgt_A = torch.empty(size=(A_B_C_S_filt.shape[0], 978))
MY_tgt_B = torch.empty(size=(A_B_C_S_filt.shape[0], 978))
MY_chem_cat_A = torch.empty(size=(A_B_C_S_filt.shape[0], 8))
MY_chem_cat_B = torch.empty(size=(A_B_C_S_filt.shape[0], 8))
MY_syn =  torch.empty(size=(A_B_C_S_filt.shape[0],1))



MY_chem_A_feat = torch.empty(size=(A_B_C_S_filt.shape[0], 50, 64))
MY_chem_B_feat= torch.empty(size=(A_B_C_S_filt.shape[0], 50, 64))
MY_chem_A_adj = torch.empty(size=(A_B_C_S_filt.shape[0], 50, 50))
MY_chem_B_adj= torch.empty(size=(A_B_C_S_filt.shape[0], 50, 50))
MY_exp_F_A = torch.empty(size=(A_B_C_S_filt.shape[0], 978, 2))
MY_exp_F_B = torch.empty(size=(A_B_C_S_filt.shape[0], 978, 2))
MY_chem_cat_A = torch.empty(size=(A_B_C_S_filt.shape[0], 8))
MY_chem_cat_B = torch.empty(size=(A_B_C_S_filt.shape[0], 8))
MY_Cell = torch.empty(size=(A_B_C_S_filt.shape[0], cell_one_hot.shape[1]))
MY_syn =  torch.empty(size=(A_B_C_S_filt.shape[0], 1))



for IND in range(MY_chem_A_feat.shape[0]): #  
	DrugA_SIG = A_B_C_S_filt.iloc[IND,]['BETA_sig_id_x']
	DrugB_SIG = A_B_C_S_filt.iloc[IND,]['BETA_sig_id_y']
	Cell = A_B_C_S_filt.iloc[IND,]['DrugCombCello']
	#
	k=1
	DrugA_Feat, DrugA_ADJ = get_CHEM(DrugA_SIG, k)
	DrugB_Feat, DrugB_ADJ = get_CHEM(DrugB_SIG, k)
	#
	EXP_A, EXP_B, EXP_AB, LINCS = get_LINCS_data(DrugA_SIG, DrugB_SIG)
	EXP_A = list(EXP_A.iloc[:,1])
	EXP_B = list(EXP_B.iloc[:,1])
	#
	TGT_A = get_targets(DrugA_SIG)
	TGT_B = get_targets(DrugB_SIG)
	#
	Cell_Vec = get_cell(DrugA_SIG, DrugB_SIG, Cell)
	#
	DrugA_CAT = get_chem_category(DrugA_SIG)
	DrugB_CAT = get_chem_category(DrugB_SIG)
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
	MY_exp_F_A[IND] = FEAT_A
	MY_exp_F_B[IND] = FEAT_B
	MY_chem_cat_A[IND] = torch.Tensor(DrugA_CAT.float())
	MY_chem_cat_B[IND] = torch.Tensor(DrugB_CAT.float())
	MY_Cell[IND] = Cell_Vec
	MY_syn[IND] = torch.Tensor([AB_SYN])



torch.save(MY_chem_A_feat, WORK_PATH+'0908.{}.MY_chem_A_feat.pt'.format(G_NAME))
torch.save(MY_chem_B_feat, WORK_PATH+'0908.{}.MY_chem_B_feat.pt'.format(G_NAME))
torch.save(MY_chem_A_adj, WORK_PATH+'0908.{}.MY_chem_A_adj.pt'.format(G_NAME))
torch.save(MY_chem_B_adj, WORK_PATH+'0908.{}.MY_chem_B_adj.pt'.format(G_NAME))
torch.save(MY_chem_cat_A, WORK_PATH+'0908.{}.MY_chem_cat_A.pt'.format(G_NAME))
torch.save(MY_chem_cat_B, WORK_PATH+'0908.{}.MY_chem_cat_B.pt'.format(G_NAME))
torch.save(MY_exp_F_A, WORK_PATH+'0908.{}.MY_exp_F_A.pt'.format(G_NAME))
torch.save(MY_exp_F_B, WORK_PATH+'0908.{}.MY_exp_F_B.pt'.format(G_NAME))
torch.save(MY_Cell, WORK_PATH+'0908.{}.MY_Cell.pt'.format(G_NAME))
torch.save(MY_syn, WORK_PATH+'0908.{}.MY_syn.pt'.format(G_NAME))






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
def prepare_data_GCN(MY_chem_A_feat, MY_chem_B_feat, MY_chem_A_adj, MY_chem_B_adj, MY_exp_F_A, MY_exp_F_B, MY_chem_cat_A, MY_chem_cat_B, MY_syn, MY_Cell, norm ) :
	chem_feat_A_train, chem_feat_A_tv, chem_feat_B_train, chem_feat_B_tv, chem_adj_A_train, chem_adj_A_tv, chem_adj_B_train, chem_adj_B_tv, exp_F_A_train, exp_F_A_tv, exp_F_B_train, exp_F_B_tv, c_cat_A_train, c_cat_A_tv, c_cat_B_train, c_cat_B_tv, syn_train, syn_tv, cell_train, cell_tv = sklearn.model_selection.train_test_split(
			MY_chem_A_feat, MY_chem_B_feat, MY_chem_A_adj, MY_chem_B_adj, MY_exp_F_A, MY_exp_F_B, MY_chem_cat_A, MY_chem_cat_B, MY_syn, MY_Cell,
			test_size= 0.2 , random_state=42 )
	chem_feat_A_val, chem_feat_A_test, chem_feat_B_val, chem_feat_B_test, chem_adj_A_val, chem_adj_A_test, chem_adj_B_val, chem_adj_B_test, exp_F_A_val, exp_F_A_test, exp_F_B_val, exp_F_B_test, c_cat_A_val, c_cat_A_test, c_cat_B_val, c_cat_B_test, syn_val, syn_test, cell_val, cell_test  = sklearn.model_selection.train_test_split(
			chem_feat_A_tv, chem_feat_B_tv, chem_adj_A_tv, chem_adj_B_tv, exp_F_A_tv, exp_F_B_tv, c_cat_A_tv, c_cat_B_tv, syn_tv, cell_tv,
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
	train_data['chem_cat_A'] = torch.concat((c_cat_A_train, c_cat_B_train), axis=0)
	val_data['chem_cat_A'] = c_cat_A_val
	test_data['chem_cat_A'] = c_cat_A_test
	#
	train_data['chem_cat_B'] = torch.concat((c_cat_B_train, c_cat_A_train), axis=0)
	val_data['chem_cat_B'] = c_cat_B_val
	test_data['chem_cat_B'] = c_cat_B_test
	#
	#
	print(train_data['drug1_feat'].shape)
	print(val_data['drug1_feat'].shape)
	print(test_data['drug1_feat'].shape)
	return train_data, val_data, test_data



class DATASET_GCN_W_FT(Dataset):
	def __init__(self, gcn_drug1_F, gcn_drug2_F, gcn_drug1_ADJ, gcn_drug2_ADJ, gcn_exp_f_A, gcn_exp_f_B, gcn_adj, gcn_adj_weight, syn_ans, cell_info, chem_cat_A, chem_cat_B):
		self.gcn_drug1_F = gcn_drug1_F
		self.gcn_drug2_F = gcn_drug2_F
		self.gcn_drug1_ADJ = gcn_drug1_ADJ
		self.gcn_drug2_ADJ = gcn_drug2_ADJ
		self.gcn_exp_f_A = gcn_exp_f_A
		self.gcn_exp_f_B = gcn_exp_f_B
		self.gcn_adj = gcn_adj
		self.gcn_adj_weight = gcn_adj_weight
		self.syn_ans = syn_ans
		self.cell_info = cell_info
		self.chem_cat_A = chem_cat_A
		self.chem_cat_B = chem_cat_B
		#
	#
	def __len__(self):
		return len(self.gcn_drug1_F)
			#
	def __getitem__(self, index):
		adj_re_A = self.gcn_drug1_ADJ[index].long().to_sparse().indices()
		adj_re_B = self.gcn_drug2_ADJ[index].long().to_sparse().indices()
		return self.gcn_drug1_F[index], self.gcn_drug2_F[index], adj_re_A, adj_re_B, self.gcn_exp_f_A[index], self.gcn_exp_f_B[index], self.gcn_adj, self.gcn_adj_weight, self.syn_ans[index] , self.cell_info[index], self.chem_cat_A[index], self.chem_cat_B[index]





def graph_collate_fn(batch):
	drug1_f_list = []
	drug2_f_list = []
	drug1_adj_list = []
	drug2_adj_list = []
	exp_f_A_list = []
	exp_f_B_list = []
	exp_adj_list = []
	exp_adj_w_list = []
	y_list = []
	cell_list = []
	c_cat_A_list = []
	c_cat_B_list = []
	EXP_num_nodes_seen = 0
	DRUG_1_num_nodes_seen = 0
	DRUG_2_num_nodes_seen = 0
	for drug1_f, drug2_f, drug1_adj, drug2_adj, exp_f_A, exp_f_B, exp_adj, exp_adj_w, y, cell, c_cat_A, c_cat_B in batch :
		drug1_f_list.append(drug1_f)
		drug2_f_list.append(drug2_f)
		drug1_adj_list.append(drug1_adj+DRUG_1_num_nodes_seen)
		drug2_adj_list.append(drug2_adj+DRUG_2_num_nodes_seen)
		exp_f_A_list.append(exp_f_A)
		exp_f_B_list.append(exp_f_B)
		exp_adj_list.append(exp_adj+EXP_num_nodes_seen)
		exp_adj_w_list.append(exp_adj_w)
		y_list.append(y)
		cell_list.append(cell)
		c_cat_A_list.append(c_cat_A)
		c_cat_B_list.append(c_cat_B)
		EXP_num_nodes_seen += exp_f_A.shape[0]
		DRUG_1_num_nodes_seen += drug1_f.shape[0]
		DRUG_2_num_nodes_seen += drug2_f.shape[0]
	drug1_f_new = torch.cat(drug1_f_list, 0)
	drug2_f_new = torch.cat(drug2_f_list, 0)
	drug1_adj_new = torch.cat(drug1_adj_list, 1)
	drug2_adj_new = torch.cat(drug2_adj_list, 1)
	exp_f_A_new = torch.cat(exp_f_A_list, 0)
	exp_f_B_new = torch.cat(exp_f_B_list, 0)
	exp_adj_new = torch.cat(exp_adj_list, 1)
	exp_adj_w_new = torch.cat(exp_adj_w_list, 1)
	y_new = torch.stack(y_list, 0)
	cell_new = torch.stack(cell_list, 0)
	c_cat_A_new = torch.stack(c_cat_A_list, 0)
	c_cat_B_new = torch.stack(c_cat_B_list, 0)
	return drug1_f_new, drug2_f_new, drug1_adj_new, drug2_adj_new, exp_f_A_new, exp_f_B_new, exp_adj_new, exp_adj_w_new, y_new, cell_new, c_cat_A_new, c_cat_B_new









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
train_data, val_data, test_data = prepare_data_GCN(MY_chem_A_feat, MY_chem_B_feat, MY_chem_A_adj, MY_chem_B_adj, MY_exp_F_A, MY_exp_F_B, MY_chem_cat_A, MY_chem_cat_B, MY_syn, MY_Cell, norm)






# WEIGHT 
ys = train_data['y'].squeeze().tolist()
min_s = np.amin(ys)
loss_weight = np.log(train_data['y'] - min_s + np.e)
JY_IDX_WEIGHT_T = torch.Tensor(JY_IDX_WEIGHT).view(1,-1)


# DATA check 
T_train = DATASET_GCN_W_FT(
	torch.Tensor(train_data['drug1_feat']), torch.Tensor(train_data['drug2_feat']), 
	torch.Tensor(train_data['drug1_adj']), torch.Tensor(train_data['drug2_adj']),
	train_data['EXP_F_A'], train_data['EXP_F_B'], 
	JY_ADJ_IDX, JY_IDX_WEIGHT_T, 
	torch.Tensor(train_data['y']),
	torch.Tensor(train_data['cell']), 
	torch.Tensor(train_data['chem_cat_A']), torch.Tensor(train_data['chem_cat_B']))

T_val = DATASET_GCN_W_FT(
	torch.Tensor(val_data['drug1_feat']), torch.Tensor(val_data['drug2_feat']), 
	torch.Tensor(val_data['drug1_adj']), torch.Tensor(val_data['drug2_adj']),
	val_data['EXP_F_A'], val_data['EXP_F_B'], 
	JY_ADJ_IDX, JY_IDX_WEIGHT_T, 
	torch.Tensor(val_data['y']),
	torch.Tensor(val_data['cell']), 
	torch.Tensor(val_data['chem_cat_A']), torch.Tensor(val_data['chem_cat_B']))
	
T_test = DATASET_GCN_W_FT(
	torch.Tensor(test_data['drug1_feat']), torch.Tensor(test_data['drug2_feat']), 
	torch.Tensor(test_data['drug1_adj']), torch.Tensor(test_data['drug2_adj']),
	test_data['EXP_F_A'], test_data['EXP_F_B'], 
	JY_ADJ_IDX, JY_IDX_WEIGHT_T, 
	torch.Tensor(test_data['y']),
	torch.Tensor(test_data['cell']), 
	torch.Tensor(test_data['chem_cat_A']), torch.Tensor(test_data['chem_cat_B']))

RAY_train = ray.put(T_train)
RAY_val = ray.put(T_val)
RAY_test = ray.put(T_test)
RAY_loss_weight = ray.put(loss_weight)




############################ MAIN GRAPH CHECK
print('MAIN', flush=True)




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





class MY_expGCN_parallel_model(torch.nn.Module):
	def __init__(self, G_layer_chem, G_indim_chem, G_hiddim_chem, G_layer_exp, G_indim_exp, G_hiddim_exp, layers_1, layers_2, layers_3, cell_dim , cat_dim, out_dim, inDrop, drop):
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
		self.cat_dim = cat_dim
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
		self.Convs_1 = torch.nn.ModuleList([torch.nn.Linear(self.G_hiddim_chem+self.G_hiddim_exp+self.cat_dim, self.layers_1[0] )])
		self.Convs_1.extend([torch.nn.Linear(self.layers_1[a], self.layers_1[a+1]) for a in range(len(self.layers_1)-1)])
		#
		self.Convs_2 = torch.nn.ModuleList([torch.nn.Linear(self.G_hiddim_chem+self.G_hiddim_exp+self.cat_dim, self.layers_2[0] )])
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
		for conv in self.G_convs_2_chem :
			conv.reset_parameters()
		for bns in self.G_bns_2_chem :
			bns.reset_parameters()
		for conv in self.G_convs_1_exp :
			conv.reset_parameters()
		for bns in self.G_bns_1_exp :
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
	def forward(self, Drug1_F, Drug2_F, Drug1_ADJ, Drug2_ADJ, EXP1, EXP2, EXP_ADJ, EXP_ADJ_WGT, syn, cell, C_cat_1, C_cat_2 ):
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
		input_drug1 = torch.concat( (G_1_C_out, G_1_E_out, C_cat_1), 1 ) # normalization 추가 해야할것 같은데 
		input_drug2 = torch.concat( (G_2_C_out, G_2_E_out, C_cat_2), 1 )
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
	use_cuda = True
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
	snp_layers = [config["feat_size_3"] ,config["feat_size_4"] ,]
	inDrop = config["dropout_1"]
	Drop = config["dropout_2"]
	#
	#
	MM_MODEL = MY_expGCN_parallel_model(
			config["G_layer"], T_train.gcn_drug1_F.shape[-1] , config["G_hiddim"],
			config["G_layer"], 2 , config["G_hiddim"],
			dsn1_layers, dsn2_layers, snp_layers, cell_one_hot.shape[1], T_train.chem_cat_A.shape[1] ,1,
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
		for batch_idx_t, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell, c_cat_a, c_cat_b) in enumerate(loaders['train']):
			adj_w = adj_w.squeeze()
			# move to GPU
			if use_cuda:
				drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell, c_cat_a, c_cat_b = drug1_f.cuda(), drug2_f.cuda(), drug1_a.cuda(), drug2_a.cuda(), expA.cuda(), expB.cuda(), adj.cuda(), adj_w.cuda(), y.cuda(), cell.cuda(), c_cat_a.cuda(), c_cat_b.cuda() 
			## find the loss and update the model parameters accordingly
			# clear the gradients of all optimized variables
			optimizer.zero_grad()
			output = MM_MODEL(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell, c_cat_a, c_cat_b)
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
		#
		######################    
		# validate the model #
		######################
		MM_MODEL.eval()
		for batch_idx_v, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell, c_cat_a, c_cat_b) in enumerate(loaders['eval']):
			adj_w = adj_w.squeeze()
			# move to GPU
			if use_cuda:
				drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell, c_cat_a, c_cat_b = drug1_f.cuda(), drug2_f.cuda(), drug1_a.cuda(), drug2_a.cuda(), expA.cuda(), expB.cuda(), adj.cuda(), adj_w.cuda(), y.cuda(), cell.cuda(), c_cat_a.cuda(), c_cat_b.cuda() 
			## update the average validation loss
			output = MM_MODEL(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell, c_cat_a, c_cat_b)
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
			print('trial : {}, epoch : {}, TrainLoss : {}, ValLoss : {}'.format(trial_name, epoch, TRAIN_LOSS, VAL_LOSS), flush=True)
			path = os.path.join(checkpoint_dir, "checkpoint")
			torch.save((MM_MODEL.state_dict(), optimizer.state_dict()), path)
			torch.save(MM_MODEL.state_dict(), './model.pth')
		tune.report(TrainLoss= TRAIN_LOSS,  ValLoss=VAL_LOSS )
	#
	print("Finished Training", flush=True)
 








def RAY_TEST_MODEL(my_config, model_path, model_name, PRJ_PATH, Trial_name, G_NAME, number): 
	use_cuda =  True
	T_test = ray.get(RAY_test)
	Test_loader = torch.utils.data.DataLoader(T_test, batch_size = my_config['config/batch_size'].item(), collate_fn = graph_collate_fn, shuffle =False, num_workers=my_config['config/n_workers'].item())
	#
	G_layer = my_config['config/G_layer'].item()
	G_hiddim = my_config['config/G_hiddim'].item()
	dsn1_layers = [my_config['config/feat_size_0'].item(), my_config['config/feat_size_1'].item(), my_config['config/feat_size_2'].item()]
	dsn2_layers = [my_config['config/feat_size_0'].item(), my_config['config/feat_size_1'].item(), my_config['config/feat_size_2'].item()] 
	snp_layers = [my_config['config/feat_size_3'].item() , my_config['config/feat_size_4'].item()]
	inDrop = my_config['config/dropout_1'].item()
	Drop = my_config['config/dropout_2'].item()
	#       
	best_model = MY_expGCN_parallel_model(
				G_layer, T_test.gcn_drug1_F.shape[-1] , G_hiddim,
				G_layer, 2, G_hiddim,
				dsn1_layers, dsn2_layers, snp_layers, 17, T_test.chem_cat_A.shape[1], 1,
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
		best_model.load_state_dict(state_dict)	#
	#
	#
	best_model.eval()
	test_loss = 0.0
	PRED_list = []
	Y_list = test_data['y'].squeeze().tolist()
	with torch.no_grad():
		best_model.eval()
		for batch_idx_t, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell, c_cat_a, c_cat_b) in enumerate(Test_loader):
			adj_w = adj_w.squeeze()
			if use_cuda:
				drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell, c_cat_a, c_cat_b = drug1_f.cuda(), drug2_f.cuda(), drug1_a.cuda(), drug2_a.cuda(), expA.cuda(), expB.cuda(), adj.cuda(), adj_w.cuda(), y.cuda(), cell.cuda(), c_cat_a.cuda(), c_cat_b.cuda()
			output = best_model(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell, c_cat_a, c_cat_b) 
			MSE = torch.nn.MSELoss()
			loss = MSE(output, y)
			test_loss = test_loss + loss.item()
			outputs = output.squeeze().tolist()
			PRED_list = PRED_list+outputs
	TEST_LOSS = test_loss/(batch_idx_t+1)
	R__T = TEST_LOSS
	R__1 , R__2 = jy_corrplot(PRED_list, Y_list, PRJ_PATH,'P.{}.{}.{}_model'.format(Trial_name, G_NAME, number) )
	return  R__T, R__1, R__2


def final_result(R_1_V, R_1_T, R_1_1, R_1_2, R_2_V, R_2_T, R_2_1, R_2_2, R_3_V, R_3_T, R_3_1, R_3_2) :
	print('---1---')
	print('- Val MSE : {:.2f}'.format(R_1_V))
	print('- Test MSE : {:.2f}'.format(R_1_T))
	print('- Test Pearson : {:.2f}'.format(R_1_1))
	print('- Test Spearman : {:.2f}'.format(R_1_2))
	print('---2---')
	print('- Val MSE : {:.2f}'.format(R_2_V))
	print('- Test MSE : {:.2f}'.format(R_2_T))
	print('- Test Pearson : {:.2f}'.format(R_2_1))
	print('- Test Spearman : {:.2f}'.format(R_2_2))
	print('---3---')
	print('- Val MSE : {:.2f}'.format(R_3_V))
	print('- Test MSE : {:.2f}'.format(R_3_T))
	print('- Test Pearson : {:.2f}'.format(R_3_1))
	print('- Test Spearman : {:.2f}'.format(R_3_2))






def MAIN(ANAL_name, WORK_PATH, PRJ_PATH, Trial_name, G_NAME, num_samples= 10, max_num_epochs=1000, grace_period = 150, cpus_per_trial = 16, gpus_per_trial = 1):
	# OPTUNA CONFIGURE 
	CONFIG={
		'n_workers' : tune.choice([cpus_per_trial]),
		"epoch" : tune.choice([max_num_epochs]),
		"G_layer" : tune.choice([2, 3, 4]), # 
		"G_hiddim" : tune.choice([512, 256, 128, 64, 32, 16]), #
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



MAIN('22.09.08.PRJ01.TRIAL3_8_loc', WORK_PATH, WORK_PATH, '3_8', 'ATC', 2, 3, 2, 16, 1)

MAIN('22.09.08.PRJ01.TRIAL3_8_pre', WORK_PATH, WORK_PATH, '3_8', 'ATC', 3, 3, 2, 10, 1)

MAIN('22.09.08.PRJ01.TRIAL3_8', WORK_PATH, WORK_PATH, '3_8', 'ATC', 100, 1000, 150, 32, 1)







#loaders = {
#			'train' : torch.utils.data.DataLoader(T_train, batch_size = 16, collate_fn = graph_collate_fn, shuffle =False),
#}

#mm = MY_expGCN_parallel_model(2, 64, 3, 2, 2, 3, [100,100,100], [100,100,100], [10,10], 17, 8, 1, 0.5, 0.5 )
			

#for a, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell, c_cat_a, c_cat_b) in enumerate(loaders['train']) :
#	adj_w = adj_w.squeeze()
#	mm(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell, c_cat_a, c_cat_b)






########################################################
########################################################
########################################################
########################################################
#######################################################
########################################################
# 에러나서 따로 확인해야함 

import pandas as pd 
import matplotlib.pyplot as plt
from ray.tune import ExperimentAnalysis
import pickle
import math
import torch





anal_df = ExperimentAnalysis("~/ray_results/22.09.08.PRJ01.TRIAL3_8")



ANA_DF = anal_df.dataframe()
ANA_ALL_DF = anal_df.trial_dataframes


ANA_DF.to_csv('/home01/k006a01/PRJ.01/TRIAL_3.8/RAY_ANA_DF.P01.3_8_1.csv')
import pickle
with open("/home01/k006a01/PRJ.01/TRIAL_3.8/RAY_ANA_DF.P01.3_8_1.pickle", "wb") as fp:
	pickle.dump(ANA_ALL_DF,fp) 


DF_KEY = list(ANA_DF.sort_values('ValLoss')['logdir'])[0]
DF_KEY
# get /home01/k006a01/ray_results/22.09.08.PRJ01.TRIAL3_8/RAY_MY_train_ffca25d4_83_G_hiddim=32,G_layer=4,batch_size=16,dropout_1=0.2000,dropout_2=0.2000,epoch=1000,feat_size_0=512,feat_siz_2022-09-12_05-09-15/model.pth M1_model.pth


TOPVAL_PATH = DF_KEY

mini_df = ANA_ALL_DF[DF_KEY]

cck_num =mini_df[mini_df.ValLoss==min(mini_df.ValLoss)].index.item()
checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
TOPVAL_PATH = DF_KEY + checkpoint
TOPVAL_PATH

# get /home01/k006a01/ray_results/22.09.08.PRJ01.TRIAL3_8/RAY_MY_train_ffca25d4_83_G_hiddim=32,G_layer=4,batch_size=16,dropout_1=0.2000,dropout_2=0.2000,epoch=1000,feat_size_0=512,feat_siz_2022-09-12_05-09-15/checkpoint_000636/checkpoint M2_checkpoint




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

# get /home01/k006a01/ray_results/22.09.08.PRJ01.TRIAL3_8/RAY_MY_train_3bf77b5e_67_G_hiddim=32,G_layer=4,batch_size=16,dropout_1=0.2000,dropout_2=0.2000,epoch=1000,feat_size_0=512,feat_siz_2022-09-11_19-28-59/checkpoint_000843/checkpoint M4_checkpoint






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
def prepare_data_GCN(MY_chem_A_feat, MY_chem_B_feat, MY_chem_A_adj, MY_chem_B_adj, MY_exp_F_A, MY_exp_F_B, MY_chem_cat_A, MY_chem_cat_B, MY_syn, MY_Cell, norm ) :
	chem_feat_A_train, chem_feat_A_tv, chem_feat_B_train, chem_feat_B_tv, chem_adj_A_train, chem_adj_A_tv, chem_adj_B_train, chem_adj_B_tv, exp_F_A_train, exp_F_A_tv, exp_F_B_train, exp_F_B_tv, c_cat_A_train, c_cat_A_tv, c_cat_B_train, c_cat_B_tv, syn_train, syn_tv, cell_train, cell_tv = sklearn.model_selection.train_test_split(
			MY_chem_A_feat, MY_chem_B_feat, MY_chem_A_adj, MY_chem_B_adj, MY_exp_F_A, MY_exp_F_B, MY_chem_cat_A, MY_chem_cat_B, MY_syn, MY_Cell,
			test_size= 0.2 , random_state=42 )
	chem_feat_A_val, chem_feat_A_test, chem_feat_B_val, chem_feat_B_test, chem_adj_A_val, chem_adj_A_test, chem_adj_B_val, chem_adj_B_test, exp_F_A_val, exp_F_A_test, exp_F_B_val, exp_F_B_test, c_cat_A_val, c_cat_A_test, c_cat_B_val, c_cat_B_test, syn_val, syn_test, cell_val, cell_test  = sklearn.model_selection.train_test_split(
			chem_feat_A_tv, chem_feat_B_tv, chem_adj_A_tv, chem_adj_B_tv, exp_F_A_tv, exp_F_B_tv, c_cat_A_tv, c_cat_B_tv, syn_tv, cell_tv,
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
	train_data['chem_cat_A'] = torch.concat((c_cat_A_train, c_cat_B_train), axis=0)
	val_data['chem_cat_A'] = c_cat_A_val
	test_data['chem_cat_A'] = c_cat_A_test
	#
	train_data['chem_cat_B'] = torch.concat((c_cat_B_train, c_cat_A_train), axis=0)
	val_data['chem_cat_B'] = c_cat_B_val
	test_data['chem_cat_B'] = c_cat_B_test
	#
	#
	print(train_data['drug1_feat'].shape)
	print(val_data['drug1_feat'].shape)
	print(test_data['drug1_feat'].shape)
	return train_data, val_data, test_data





class DATASET_GCN_W_FT(Dataset):
	def __init__(self, gcn_drug1_F, gcn_drug2_F, gcn_drug1_ADJ, gcn_drug2_ADJ, gcn_exp_f_A, gcn_exp_f_B, gcn_adj, gcn_adj_weight, syn_ans, cell_info, chem_cat_A, chem_cat_B):
		self.gcn_drug1_F = gcn_drug1_F
		self.gcn_drug2_F = gcn_drug2_F
		self.gcn_drug1_ADJ = gcn_drug1_ADJ
		self.gcn_drug2_ADJ = gcn_drug2_ADJ
		self.gcn_exp_f_A = gcn_exp_f_A
		self.gcn_exp_f_B = gcn_exp_f_B
		self.gcn_adj = gcn_adj
		self.gcn_adj_weight = gcn_adj_weight
		self.syn_ans = syn_ans
		self.cell_info = cell_info
		self.chem_cat_A = chem_cat_A
		self.chem_cat_B = chem_cat_B
		#
	#
	def __len__(self):
		return len(self.gcn_drug1_F)
			#
	def __getitem__(self, index):
		adj_re_A = self.gcn_drug1_ADJ[index].long().to_sparse().indices()
		adj_re_B = self.gcn_drug2_ADJ[index].long().to_sparse().indices()
		return self.gcn_drug1_F[index], self.gcn_drug2_F[index], adj_re_A, adj_re_B, self.gcn_exp_f_A[index], self.gcn_exp_f_B[index], self.gcn_adj, self.gcn_adj_weight, self.syn_ans[index] , self.cell_info[index], self.chem_cat_A[index], self.chem_cat_B[index]



def graph_collate_fn(batch):
	drug1_f_list = []
	drug2_f_list = []
	drug1_adj_list = []
	drug2_adj_list = []
	exp_f_A_list = []
	exp_f_B_list = []
	exp_adj_list = []
	exp_adj_w_list = []
	y_list = []
	cell_list = []
	c_cat_A_list = []
	c_cat_B_list = []
	EXP_num_nodes_seen = 0
	DRUG_1_num_nodes_seen = 0
	DRUG_2_num_nodes_seen = 0
	for drug1_f, drug2_f, drug1_adj, drug2_adj, exp_f_A, exp_f_B, exp_adj, exp_adj_w, y, cell, c_cat_A, c_cat_B in batch :
		drug1_f_list.append(drug1_f)
		drug2_f_list.append(drug2_f)
		drug1_adj_list.append(drug1_adj+DRUG_1_num_nodes_seen)
		drug2_adj_list.append(drug2_adj+DRUG_2_num_nodes_seen)
		exp_f_A_list.append(exp_f_A)
		exp_f_B_list.append(exp_f_B)
		exp_adj_list.append(exp_adj+EXP_num_nodes_seen)
		exp_adj_w_list.append(exp_adj_w)
		y_list.append(y)
		cell_list.append(cell)
		c_cat_A_list.append(c_cat_A)
		c_cat_B_list.append(c_cat_B)
		EXP_num_nodes_seen += exp_f_A.shape[0]
		DRUG_1_num_nodes_seen += drug1_f.shape[0]
		DRUG_2_num_nodes_seen += drug2_f.shape[0]
	drug1_f_new = torch.cat(drug1_f_list, 0)
	drug2_f_new = torch.cat(drug2_f_list, 0)
	drug1_adj_new = torch.cat(drug1_adj_list, 1)
	drug2_adj_new = torch.cat(drug2_adj_list, 1)
	exp_f_A_new = torch.cat(exp_f_A_list, 0)
	exp_f_B_new = torch.cat(exp_f_B_list, 0)
	exp_adj_new = torch.cat(exp_adj_list, 1)
	exp_adj_w_new = torch.cat(exp_adj_w_list, 1)
	y_new = torch.stack(y_list, 0)
	cell_new = torch.stack(cell_list, 0)
	c_cat_A_new = torch.stack(c_cat_A_list, 0)
	c_cat_B_new = torch.stack(c_cat_B_list, 0)
	return drug1_f_new, drug2_f_new, drug1_adj_new, drug2_adj_new, exp_f_A_new, exp_f_B_new, exp_adj_new, exp_adj_w_new, y_new, cell_new, c_cat_A_new, c_cat_B_new





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





class MY_expGCN_parallel_model(torch.nn.Module):
	def __init__(self, G_layer_chem, G_indim_chem, G_hiddim_chem, G_layer_exp, G_indim_exp, G_hiddim_exp, layers_1, layers_2, layers_3, cell_dim , cat_dim, out_dim, inDrop, drop):
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
		self.cat_dim = cat_dim
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
		self.Convs_1 = torch.nn.ModuleList([torch.nn.Linear(self.G_hiddim_chem+self.G_hiddim_exp+self.cat_dim, self.layers_1[0] )])
		self.Convs_1.extend([torch.nn.Linear(self.layers_1[a], self.layers_1[a+1]) for a in range(len(self.layers_1)-1)])
		#
		self.Convs_2 = torch.nn.ModuleList([torch.nn.Linear(self.G_hiddim_chem+self.G_hiddim_exp+self.cat_dim, self.layers_2[0] )])
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
		for conv in self.G_convs_2_chem :
			conv.reset_parameters()
		for bns in self.G_bns_2_chem :
			bns.reset_parameters()
		for conv in self.G_convs_1_exp :
			conv.reset_parameters()
		for bns in self.G_bns_1_exp :
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
	def forward(self, Drug1_F, Drug2_F, Drug1_ADJ, Drug2_ADJ, EXP1, EXP2, EXP_ADJ, EXP_ADJ_WGT, syn, cell, C_cat_1, C_cat_2 ):
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
		input_drug1 = torch.concat( (G_1_C_out, G_1_E_out, C_cat_1), 1 ) # normalization 추가 해야할것 같은데 
		input_drug2 = torch.concat( (G_2_C_out, G_2_E_out, C_cat_2), 1 )
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
WORK_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_GPU/PJ01.TRIAL.3_8_1/'

G_NAME= 'HN_GSP'




MY_chem_A_feat = torch.load(WORK_PATH+'0908.{}.MY_chem_A_feat.pt'.format(G_NAME))
MY_chem_B_feat = torch.load(WORK_PATH+'0908.{}.MY_chem_B_feat.pt'.format(G_NAME))
MY_chem_A_adj = torch.load(WORK_PATH+'0908.{}.MY_chem_A_adj.pt'.format(G_NAME))
MY_chem_B_adj = torch.load(WORK_PATH+'0908.{}.MY_chem_B_adj.pt'.format(G_NAME))
MY_chem_cat_A = torch.load(WORK_PATH+'0908.{}.MY_chem_cat_A.pt'.format(G_NAME))
MY_chem_cat_B = torch.load(WORK_PATH+'0908.{}.MY_chem_cat_B.pt'.format(G_NAME))
MY_exp_F_A = torch.load(WORK_PATH+'0908.{}.MY_exp_F_A.pt'.format(G_NAME))
MY_exp_F_B = torch.load(WORK_PATH+'0908.{}.MY_exp_F_B.pt'.format(G_NAME))
MY_Cell = torch.load(WORK_PATH+'0908.{}.MY_Cell.pt'.format(G_NAME))
MY_syn = torch.load(WORK_PATH+'0908.{}.MY_syn.pt'.format(G_NAME))


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
train_data, val_data, test_data = prepare_data_GCN(MY_chem_A_feat, MY_chem_B_feat, MY_chem_A_adj, MY_chem_B_adj, MY_exp_F_A, MY_exp_F_B, MY_chem_cat_A, MY_chem_cat_B, MY_syn, MY_Cell, norm)



# WEIGHT 
ys = train_data['y'].squeeze().tolist()
min_s = np.amin(ys)
loss_weight = np.log(train_data['y'] - min_s + np.e)
JY_IDX_WEIGHT_T = torch.Tensor(JY_IDX_WEIGHT).view(1,-1)


# DATA check 
T_train = DATASET_GCN_W_FT(
	torch.Tensor(train_data['drug1_feat']), torch.Tensor(train_data['drug2_feat']), 
	torch.Tensor(train_data['drug1_adj']), torch.Tensor(train_data['drug2_adj']),
	train_data['EXP_F_A'], train_data['EXP_F_B'], 
	JY_ADJ_IDX, JY_IDX_WEIGHT_T, 
	torch.Tensor(train_data['y']),
	torch.Tensor(train_data['cell']), 
	torch.Tensor(train_data['chem_cat_A']), torch.Tensor(train_data['chem_cat_B']))

T_val = DATASET_GCN_W_FT(
	torch.Tensor(val_data['drug1_feat']), torch.Tensor(val_data['drug2_feat']), 
	torch.Tensor(val_data['drug1_adj']), torch.Tensor(val_data['drug2_adj']),
	val_data['EXP_F_A'], val_data['EXP_F_B'], 
	JY_ADJ_IDX, JY_IDX_WEIGHT_T, 
	torch.Tensor(val_data['y']),
	torch.Tensor(val_data['cell']), 
	torch.Tensor(val_data['chem_cat_A']), torch.Tensor(val_data['chem_cat_B']))
	
T_test = DATASET_GCN_W_FT(
	torch.Tensor(test_data['drug1_feat']), torch.Tensor(test_data['drug2_feat']), 
	torch.Tensor(test_data['drug1_adj']), torch.Tensor(test_data['drug2_adj']),
	test_data['EXP_F_A'], test_data['EXP_F_B'], 
	JY_ADJ_IDX, JY_IDX_WEIGHT_T, 
	torch.Tensor(test_data['y']),
	torch.Tensor(test_data['cell']), 
	torch.Tensor(test_data['chem_cat_A']), torch.Tensor(test_data['chem_cat_B']))

RAY_train = ray.put(T_train)
RAY_val = ray.put(T_val)
RAY_test = ray.put(T_test)
RAY_loss_weight = ray.put(loss_weight)



################# 결과 확인


import pandas as pd 
import matplotlib.pyplot as plt
from ray.tune import ExperimentAnalysis
import pickle
import math
import torch


def RAY_TEST_MODEL(my_config, model_path, model_name, PRJ_PATH, Trial_name, G_NAME, number): 
	use_cuda =  False
	T_test = ray.get(RAY_test)
	Test_loader = torch.utils.data.DataLoader(T_test, batch_size = my_config['config/batch_size'].item(), collate_fn = graph_collate_fn, shuffle =False, num_workers=my_config['config/n_workers'].item())
	#
	G_layer = my_config['config/G_layer'].item()
	G_hiddim = my_config['config/G_hiddim'].item()
	dsn1_layers = [my_config['config/feat_size_0'].item(), my_config['config/feat_size_1'].item(), my_config['config/feat_size_2'].item()]
	dsn2_layers = [my_config['config/feat_size_0'].item(), my_config['config/feat_size_1'].item(), my_config['config/feat_size_2'].item()] 
	snp_layers = [my_config['config/feat_size_3'].item() , my_config['config/feat_size_4'].item()]
	inDrop = my_config['config/dropout_1'].item()
	Drop = my_config['config/dropout_2'].item()
	#       
	best_model = MY_expGCN_parallel_model(
				G_layer, T_test.gcn_drug1_F.shape[-1] , G_hiddim,
				G_layer, 2, G_hiddim,
				dsn1_layers, dsn2_layers, snp_layers, 17, T_test.chem_cat_A.shape[1], 1,
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
		for batch_idx_t, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell, c_cat_a, c_cat_b) in enumerate(Test_loader):
			adj_w = adj_w.squeeze()
			if use_cuda:
				drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell, c_cat_a, c_cat_b = drug1_f.cuda(), drug2_f.cuda(), drug1_a.cuda(), drug2_a.cuda(), expA.cuda(), expB.cuda(), adj.cuda(), adj_w.cuda(), y.cuda(), cell.cuda(), c_cat_a.cuda(), c_cat_b.cuda() 
			output = best_model(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell, c_cat_a, c_cat_b)
			MSE = torch.nn.MSELoss()
			loss = MSE(output, y)
			test_loss = test_loss + loss.item()
			outputs = sum(output.tolist(), [])
			PRED_list = PRED_list+outputs
	TEST_LOSS = test_loss/(batch_idx_t+1)
	R__T = TEST_LOSS
	R__1 , R__2 = jy_corrplot(PRED_list, Y_list, PRJ_PATH,'P.{}.{}.{}_model'.format(Trial_name, G_NAME, number) )
	return  R__T, R__1, R__2



#a = torch.Tensor([[-1.9493]])
#a.squeeze()
#sum(a.tolist(),[])

#b = torch.Tensor([[-1.9493] , [-10.1609] ])
#b.squeeze()
#sum(b.tolist(),[])



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




PRJ_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_GPU/PJ01.TRIAL.3_8_1/'
ANA_DF = pd.read_csv(PRJ_PATH+'RAY_ANA_DF.P01.3_8_1.csv')
with open(PRJ_PATH+'RAY_ANA_DF.P01.3_8_1.pickle', 'rb') as f:
	ANA_ALL_DF = pickle.load(f)

Trial_name = '3_8_1'
G_NAME = 'HN_GSP'
TOPVAL_PATH = PRJ_PATH



# 1) best final
#
#
DF_KEY = list(ANA_DF.sort_values('ValLoss')['logdir'])[0]
print('best final', flush=True)
print(DF_KEY, flush=True)
#TOPVAL_PATH = DF_KEY
mini_df = ANA_ALL_DF[DF_KEY]
my_config = ANA_DF[ANA_DF.logdir==DF_KEY]
R_1_V = list(ANA_DF.sort_values('ValLoss')['ValLoss'])[0]
R_1_T, R_1_1, R_1_2 = RAY_TEST_MODEL(my_config, TOPVAL_PATH, 'M1_model.pth', PRJ_PATH, Trial_name, G_NAME, 'M1')
#
# 2) best final's checkpoint
# 
cck_num =mini_df[mini_df.ValLoss==min(mini_df.ValLoss)].index.item()
checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
#TOPVAL_PATH = DF_KEY + checkpoint
print('best final check', flush=True)
print(TOPVAL_PATH, flush=True)
R_2_V = min(mini_df.ValLoss)
R_2_T, R_2_1, R_2_2 = RAY_TEST_MODEL(my_config, TOPVAL_PATH, 'M2_checkpoint', PRJ_PATH, Trial_name, G_NAME, 'M2')
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
#TOPVAL_PATH = TOT_key
my_config = ANA_DF[ANA_DF.logdir==TOT_key]
cck_num =mini_df[mini_df.ValLoss==min(mini_df.ValLoss)].index.item()
checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
#TOPVAL_PATH = TOT_key + checkpoint
print('best val check', flush=True)
print(TOPVAL_PATH, flush=True)
R_3_V = min(mini_df.ValLoss)
R_3_T, R_3_1, R_3_2 = RAY_TEST_MODEL(my_config, TOPVAL_PATH, 'M4_checkpoint', PRJ_PATH, Trial_name, G_NAME, 'M4')
#




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












##############################################################################
##############################################################################
##############################################################################






# 일단 NCI Thesaurus (NCIt) 로 실험해보기 -> class 없는 애들은 그냥 blank 로 둬야하지 않을까 
# 아니면 해당하는 애들 없애면 어떻게 되는지... 근데 그러면 물질이 129 개로 필터링 될텐데 
# 일단 176 -> 129 가 되면 얼마나 줄어드는지 확인부터 해보고 
# NCIt 말고 다른 애들도 어떻게 되는지 확인해보기.
# 전체 데이터 수가 얼마나 줄어드는지 한번 보고 one hot vector 만들어보기 


CELLO_DC_BETA_cids = [9826308, 11955716, 208908, 9884685, 46926350, 11494412, 24964624, 10074640, 4114, 56655374, 11977753, 24748573, 13342, 176167, 44607530, 3062316, 5328940, 11524144, 42611257, 23725625, 135418940, 52912189, 11626560, 11488320, 3025986, 11364421, 11640390, 3657, 2123, 135440466, 451668, 159324, 2141, 5730, 135398501, 2662, 126565, 46244454, 3690, 24978538, 36462, 5743, 135398510, 50905713, 11676786, 4211, 4212, 644213, 9934458, 57469, 3712, 16654980, 11964036, 11713159, 2187, 644241, 3081361, 9829523, 2708, 49867926, 148121, 49867930, 1691, 148124, 2719, 9933475, 4261, 5291, 24748204, 25227436, 216239, 5494449, 135410875, 5278396, 11683005, 5311, 24995524, 25262792, 5113032, 46943432, 11960529, 16736978, 71384, 4829, 46233311, 9826528, 82146, 148195, 9444, 176870, 11707110, 16038120, 56649450, 2747117, 25021165, 123631, 11213558, 25023738, 24776445, 156414, 216326, 10127622, 2723601, 5394, 11617559, 216345, 11152667, 60700, 135565596, 10341154, 3363, 104741, 60198, 71226662, 5426, 54708532, 3385, 5081913, 3902, 354624, 25033539, 11656518, 17754438, 5453, 60750, 657237, 11625818, 2907, 15983966, 667490, 1893730, 24821094, 11556711, 285033, 5330286, 11998575, 11167602, 24856436, 25262965, 387447, 51039095, 1401, 60795, 2942, 16720766, 16659841, 104842, 46907787, 4493, 25126798, 16722836, 42623900, 11683, 57519523, 5035, 17755052, 2478, 3503, 135401907, 449459, 300471, 10113978, 107970, 11327430, 25117126, 45375953, 31703, 11485656, 36314, 126941, 766949, 44551660, 24180719, 9549298, 10302451, 4091]
CELLO_DC_BETA_cids.sort()

#for cid in CELLO_DC_BETA_cids:
#    os.system('wget https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/classification/JSON -O /st06/jiyeonH/11.TOX/Pubchem_Classi_176/PC_C.{}.json'.format(cid, cid))
#    os.system('sleep 1')

# 따로 다운받은 
36462
845084
135401907



# 테스트용
cid_PATH = '/st06/jiyeonH/11.TOX/Pubchem_Classi_176/'

cid = 4091

import json 
for cid in CELLO_DC_BETA_cids:
with open(cid_PATH+'PC_C.{}.json'.format(cid)) as json_file :
	tmp_json =json.load(json_file)

tmp_1 = tmp_json['Hierarchies']['Hierarchy']
tmp_1[1]
len(tmp_1) # 7


for cid in CELLO_DC_BETA_cids:
with open(cid_PATH+'PC_C.{}.json'.format(cid)) as json_file :
	tmp_json =json.load(json_file)

tmp_1 = tmp_json['Hierarchies']['Hierarchy']
for dat in tmp_1 :
	s_tmp_df = pd.DataFrame({ 'SourceName' : SourceName , 'SourceIDs' : SourceIDs})
	source_table = pd.concat([ source_table, s_tmp_df ])

[ dat['SourceName'] for dat in tmp_1 ]
[ dat['SourceID'] for dat in tmp_1 ]
[ dat['Information']['Name'] for dat in tmp_1 ]
[ dat['Information']['Name'] for dat in tmp_1 ]







keys = []
sources = []
s_ids = []  # 41

source_table = pd.DataFrame(columns = ['SourceName','SourceIDs','InfoName'])

for cid in CELLO_DC_BETA_cids :
	try :
		with open(cid_PATH+'PC_C.{}.json'.format(cid)) as json_file :
			tmp_json =json.load(json_file)
			tmp_re = tmp_json['Hierarchies']['Hierarchy']
			tmp_sources = []
			tmp_source_id = []
			for dat in tmp_re :
				keys = keys + list(dat.keys())
				keys = keys + list(dat['Information'].keys())
				SourceName = [dat['SourceName']]
				SourceIDs = [dat['SourceID']]
				InfoName = [dat['Information']['Name']]
				s_tmp_df = pd.DataFrame({ 'SourceName' : SourceName , 'SourceIDs' : SourceIDs, 'InfoName': InfoName})
				source_table = pd.concat([ source_table, s_tmp_df ])
				#tmp_sources = tmp_sources + SourceName
				#tmp_source_id = tmp_source_id + SourceIDs
			#sources = sources + list(set(tmp_sources))
			#s_ids = s_ids + list(set(tmp_source_id)) 
	except :
		print(cid)

source_table = source_table.drop_duplicates()




# NCI Thesaurus (NCIt) 11701 -> 9618 -> 7171

pds_NCIt = pd.DataFrame(columns=['CID','Class']) # class : 65개 

for cid in CELLO_DC_BETA_cids:
	with open(cid_PATH+'PC_C.{}.json'.format(cid)) as json_file :
		tmp_json =json.load(json_file)
		tmp_re = tmp_json['Hierarchies']['Hierarchy']
		for dat in tmp_re :
			if dat['SourceID'] == 'NCIt' : # 'NCI Thesaurus (NCIt)'
				print(cid)
				nodes = dat['Node']
				final_class = nodes[1]['Information']['Name']
				tmp_ddff = pd.DataFrame({'CID': [cid], 'Class': [final_class]})
				pds_NCIt = pd.concat([pds_NCIt, tmp_ddff])
				print(final_class)
				# print([no['Information']['Name'] for no in nodes ])
# tmp_re[-5]['Node'][1]['Information']['Name']


A_B_C_S_tmp = A_B_C_S[A_B_C_S.drug_row_cid.isin(list(pds_NCIt.CID))]
A_B_C_S_tmp = A_B_C_S_tmp[A_B_C_S_tmp.drug_col_cid.isin(list(pds_NCIt.CID))]
A_B_C_S_tmp[['drug_row_cid','drug_col_cid','BETA_sig_id_x','BETA_sig_id_y','DrugCombCello']].drop_duplicates()




# KEGG -> 못써먹을것 같음 

pds_KEGG = pd.DataFrame(columns=['CID','Class'])






# ATC 11701 -> 6587 이거보다 더 줄어들듯 -> 5362

pds_ATC = pd.DataFrame(columns=['CID','level1','level2','level3','level4','level5'])

for cid in CELLO_DC_BETA_cids:
	with open(cid_PATH+'PC_C.{}.json'.format(cid)) as json_file :
		tmp_json =json.load(json_file)
		tmp_re = tmp_json['Hierarchies']['Hierarchy']
		for dat in tmp_re :
			if dat['SourceID'] == 'ATCTree' : # 'NCI Thesaurus (NCIt)'
				print(cid)
				nodes = dat['Node']
				classes = [nn['Information']['Name'] for nn in nodes]
				ATC_keys = ['level1','level2','level3','level4','level5']
				tmp_ATC = {ATC_keys[i] : [classes[i]] for i in range(5)}
				tmp_ATC['CID'] = [cid]
				pds_ATC = pd.concat([pds_ATC, pd.DataFrame(tmp_ATC)])



A_B_C_S_tmp = A_B_C_S[A_B_C_S.drug_row_cid.isin(list(pds_ATC.CID))]
A_B_C_S_tmp = A_B_C_S_tmp[A_B_C_S_tmp.drug_col_cid.isin(list(pds_ATC.CID))]
A_B_C_S_tmp[['drug_row_cid','drug_col_cid','BETA_sig_id_x','BETA_sig_id_y','DrugCombCello']].drop_duplicates()




# NCI Thesaurus (NCIt) 11701 -> 9618 -> 7171 -> 6835

# cid_PATH = '/st06/jiyeonH/11.TOX/Pubchem_Classi_176/'

pds_NCIt = pd.DataFrame(columns=['CID','Class']) # class : 65개 

for cid in CELLO_DC_BETA_cids:
	with open(cid_PATH+'PC_C.{}.json'.format(cid)) as json_file :
		tmp_json =json.load(json_file)
		tmp_re = tmp_json['Hierarchies']['Hierarchy']
		for dat in tmp_re :
			if dat['SourceID'] == 'NCIt' : # 'NCI Thesaurus (NCIt)'
				print(cid)
				nodes = dat['Node']
				final_class = nodes[1]['Information']['Name']
				tmp_ddff = pd.DataFrame({'CID': [cid], 'Class': [final_class]})
				pds_NCIt = pd.concat([pds_NCIt, tmp_ddff])
				print(final_class)

				# print([no['Information']['Name'] for no in nodes ])
# tmp_re[-5]['Node'][1]['Information']['Name']



A_B_C_S_filt = A_B_C_S_SET[A_B_C_S_SET.drug_row_cid.isin(list(pds_NCIt.CID))]
A_B_C_S_filt = A_B_C_S_filt[A_B_C_S_filt.drug_col_cid.isin(list(pds_NCIt.CID))]

A_B_C_S_filt = A_B_C_S_filt.reset_index(drop=True) # 




# IUPHER 도 확인해볼 수 있을듯 11701 -> 4956 
# 이거 코드 짜봏기 

pds_IUPHER = pd.DataFrame(columns=['CID','node']) # class :150개 넘음 

for cid in CELLO_DC_BETA_cids:
	with open(cid_PATH+'PC_C.{}.json'.format(cid)) as json_file :
		tmp_json =json.load(json_file)
		tmp_re = tmp_json['Hierarchies']['Hierarchy']
		for dat in tmp_re :
			if dat['SourceID'] == 'Target Classification' :
				print(cid)
				tmp = copy.deepcopy(dat['Node'])
				len(tmp)
				for tt in tmp :
					nn = tt['Information']['Name']
					tmp_I = {'CID' : [cid], 'node' : [nn]}
					pds_IUPHER = pd.concat([pds_IUPHER, pd.DataFrame(tmp_I)])


A_B_C_S_tmp = A_B_C_S[A_B_C_S.drug_row_cid.isin(list(pds_IUPHER.CID))]
A_B_C_S_tmp = A_B_C_S_tmp[A_B_C_S_tmp.drug_col_cid.isin(list(pds_IUPHER.CID))]
A_B_C_S_tmp[['drug_row_cid','drug_col_cid','BETA_sig_id_x','BETA_sig_id_y','DrugCombCello']].drop_duplicates()






# new drugbank 
# new drugbank 
# new drugbank 
# new drugbank 
# new drugbank 


import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd 
import os 

doc = ET.parse('/st06/jiyeonH/11.TOX/DrugBank.5.1.9/full_database.xml')
# /home/jiyeon/Dropbox/Lab_computer/06.TOXIDB/DrugBank/full_database.xml
# /media/jiyeon/6CBCAD92BCAD56FC/JIYEON/drugbank/removal.xml

root = doc.getroot()

tot_tags = []
for i in root.iter():
	#print(i.tag)
	#print(i.text)
	#print('#'*30)
	tot_tags.append(i.tag)

tot_tags = list(set(tot_tags))



all_names = root.findall("./{http://www.drugbank.ca}drug/{http://www.drugbank.ca}name")
all_names_text = [drug.text for drug in all_names]





# test 
test_name = 'Metformin'
tmp_df = pd.DataFrame(columns = ["DrugBankID","Name","Resource","Identifier"])
tmp_a = root.findall("./{http://www.drugbank.ca}drug/[{http://www.drugbank.ca}name='"+test_name+"']")
print(len(tmp_a))
tmp_ids = tmp_a[0].findall("./{http://www.drugbank.ca}drugbank-id")
drugbankID=[a.text for a in tmp_ids if len(a.attrib)>0][0]

tmp_resource = tmp_a[0].findall("./{http://www.drugbank.ca}external-identifiers/{http://www.drugbank.ca}external-identifier/{http://www.drugbank.ca}resource")
tmp_iden = tmp_a[0].findall("./{http://www.drugbank.ca}external-identifiers/{http://www.drugbank.ca}external-identifier/{http://www.drugbank.ca}identifier")
tmp_resource_text = [a.text.replace('\r\n','(tag)') for a in tmp_resource]
tmp_iden_text = [a.text for a in tmp_iden]
tmp_df['Resource'] = tmp_resource_text
tmp_df['Identifier'] = tmp_iden_text
tmp_df['DrugBankID'] = drugbankID
tmp_df['Name'] = name



tmp_cat = tmp_a[0].findall("./{http://www.drugbank.ca}categories/{http://www.drugbank.ca}category/{http://www.drugbank.ca}category")
tmp_cat_text = [a.text for a in tmp_cat]
A,B = get_data_df(test_name)






# beautifull code


def get_data_df (name) :
	tmp_ID_df = pd.DataFrame(columns = ["DrugBankID","Name","Resource","Identifier"])
	tmp_Class_df = pd.DataFrame(columns = ["DrugBankID","Name","Dcat"])
	#
	# get id 
	tmp_a = root.findall("./{http://www.drugbank.ca}drug/[{http://www.drugbank.ca}name='"+name+"']")
	tmp_ids = tmp_a[0].findall("./{http://www.drugbank.ca}drugbank-id")
	drugbankID=[a.text for a in tmp_ids if len(a.attrib)>0][0]
	# get externals 
	tmp_resource = tmp_a[0].findall("./{http://www.drugbank.ca}external-identifiers/{http://www.drugbank.ca}external-identifier/{http://www.drugbank.ca}resource")
	tmp_iden = tmp_a[0].findall("./{http://www.drugbank.ca}external-identifiers/{http://www.drugbank.ca}external-identifier/{http://www.drugbank.ca}identifier")
	tmp_resource_text = [a.text.replace('\r\n','(tag)') for a in tmp_resource]
	tmp_iden_text = [a.text for a in tmp_iden]
	# tmp_ID_df
	for x in range(0,len(tmp_resource_text)):
		tmptmp = pd.DataFrame({'DrugBankID' : [drugbankID],'Name' : [name], 'Resource' : tmp_resource_text[x], 'Identifier' : tmp_iden_text[x] })
		tmp_ID_df = pd.concat([tmp_ID_df, tmptmp])
	#
	# get drug class
	tmp_cat = tmp_a[0].findall("./{http://www.drugbank.ca}categories/{http://www.drugbank.ca}category/{http://www.drugbank.ca}category")
	tmp_cat_text = [a.text for a in tmp_cat]
	#
	# tmp class df 
	for x in range(0,len(tmp_cat_text)):
		tmptmp = pd.DataFrame({'DrugBankID' : [drugbankID],'Name' : [name],'Dcat' : tmp_cat_text[x] })
		tmp_Class_df = pd.concat([tmp_Class_df, tmptmp])
	return(tmp_ID_df, tmp_Class_df)

ID_table = pd.DataFrame(columns = ["DrugBankID","Name","Resource","Identifier"])
CLAS_table = pd.DataFrame(columns = ["DrugBankID","Name","Dcat"])


all_names = root.findall("./{http://www.drugbank.ca}drug/{http://www.drugbank.ca}name")
all_names_text = [drug.text for drug in all_names]


fail_names = []

for drug in all_names :
	name = drug.text
	try :
		ID, CLAS = get_data_df(name)
		ID_table = pd.concat([ID_table, ID])
		CLAS_table = pd.concat([CLAS_table, CLAS])
	except :
		fail_names.append(name)

ID_table.to_csv('/st06/jiyeonH/11.TOX/DrugBank.5.1.9/IDs.csv')
CLAS_table.to_csv('/st06/jiyeonH/11.TOX/DrugBank.5.1.9/Category.csv')

ID_table2 = ID_table[['DrugBankID','Name','Resource','Identifier']]
ID_table2.to_csv('/st06/jiyeonH/11.TOX/DrugBank.5.1.9/IDs2.csv')











get_external_iden_df_fail = pd.DataFrame(columns = ["Name"])



for drug in all_names: 
	
	try : 
		get_external_iden_df_all = get_external_iden_df(name)
		get_external_iden_df_all.to_csv("DrugBank_ids.csv", mode = 'a', sep = '\t', index = True, header = False)
	except :
		get_external_iden_df_fail = pd.DataFrame({"Name": [name]})
		get_external_iden_df_fail.to_csv("DrugBank_ids_failnames.csv",mode = 'a', sep = '\t', index = True, header = False)












# make big table

all_names = root.findall("./{http://www.drugbank.ca}drug/{http://www.drugbank.ca}name")
all_names_text = [drug.text for drug in all_names]



get_external_iden_df_all.to_csv("DrugBank_ids.csv",sep = '\t', index = True, header = False)

get_external_iden_df_fail.to_csv("DrugBank_ids_failnames.csv",sep = '\t', index = True, header = False)

























for cid in CELLO_DC_BETA_cids:
	with open(cid_PATH+'PC_C.{}.json'.format(cid)) as json_file :
		for dat in tmp_re :
			dat['SourceName']
			dat['SourceID']
			""

			if  == 'Target Classification' :



# 일단 traids 가 얼마나 빠지는지 확인좀 해보자 
for cid in CELLO_DC_BETA_cids : 
	with open(cid_PATH+'PC_C.{}.json'.format(cid)) as json_file :
		tmp_json =json.load(json_file)
		tmp_re = tmp_json['Hierarchies']['Hierarchy']











for ss in list(set(sources)) :
	print(ss)
	sources.count(ss)









tmp_df = pd.DataFrame(columns = list(tmp_1[0].keys()))
tmp_1[0]['Information']
['Name', 'Description', 'HNID', 'ChildID', 'HasCountsOfType', 'Counts']

for kk in tmp_1[0].keys() :
	tmp_df['']





# pip install pygobo
from pygobo import OBOParser
parser = OBOParser()
with open('ontology.obo','r') as input:
   ontology = parser.parse(input)





