
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


WORK_PATH = '/home01/k006a01/PRJ.01/TRIAL_3.8/'
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

# cid_PATH = '/st06/jiyeonH/11.TOX/Pubchem_Classi_176/'
# sig_id = 'LUNG001_A549_24H:BRD-K48935217:10'

















def get_chem_category(sig_id) :
tmp_df1 = BETA_CID_CELLO_SIG[BETA_CID_CELLO_SIG.sig_id == sig_id]
CID = int(tmp_df1.pubchem_cid.item())
cid = str(CID)
with open(cid_PATH+'PC_C.{}.json'.format(CID)) as json_file :
	tmp_json =json.load(json_file)
	tmp_re = tmp_json['Hierarchies']['Hierarchy']









MY_chem_A_feat = torch.empty(size=(A_B_C_S_SET.shape[0], 50, 64))
MY_chem_B_feat= torch.empty(size=(A_B_C_S_SET.shape[0], 50, 64))
MY_chem_A_adj = torch.empty(size=(A_B_C_S_SET.shape[0], 50, 50))
MY_chem_B_adj= torch.empty(size=(A_B_C_S_SET.shape[0], 50, 50))
MY_exp_A = torch.empty(size=(A_B_C_S_SET.shape[0], 978))
MY_exp_B = torch.empty(size=(A_B_C_S_SET.shape[0], 978))
MY_exp_AB = torch.empty(size=(A_B_C_S_SET.shape[0], 978, 2))
MY_tgt_A = torch.empty(size=(A_B_C_S_SET.shape[0], 978))
MY_tgt_B = torch.empty(size=(A_B_C_S_SET.shape[0], 978))
MY_syn =  torch.empty(size=(A_B_C_S_SET.shape[0],1))



for IND in range(MY_chem_A_feat.shape[0]): #  
	DrugA_SIG = A_B_C_S_SET.iloc[IND,]['BETA_sig_id_x']
	DrugB_SIG = A_B_C_S_SET.iloc[IND,]['BETA_sig_id_y']
	Cell = A_B_C_S_SET.iloc[IND,]['DrugCombCello']
	#
	k=1
	DrugA_Feat, DrugA_ADJ = get_CHEM(DrugA_SIG, k)
	DrugB_Feat, DrugB_ADJ = get_CHEM(DrugB_SIG, k)
	#
	EXP_A, EXP_B, EXP_AB, LINCS = get_LINCS_data(DrugA_SIG, DrugB_SIG)
	#
	TGT_A = get_targets(DrugA_SIG)
	TGT_B = get_targets(DrugB_SIG)
	#
	AB_SYN = get_synergy_data(DrugA_SIG, DrugB_SIG, Cell)
	#
	MY_chem_A_feat[IND] = torch.Tensor(DrugA_Feat)
	MY_chem_B_feat[IND] = torch.Tensor(DrugB_Feat)
	MY_chem_A_adj[IND] = torch.Tensor(DrugA_ADJ)
	MY_chem_B_adj[IND] = torch.Tensor(DrugB_ADJ)
	MY_exp_A[IND] = torch.Tensor(EXP_A.iloc[:,1])
	MY_exp_B[IND] = torch.Tensor(EXP_B.iloc[:,1])
	MY_exp_AB[IND] = torch.Tensor(EXP_AB).unsqueeze(0)
	MY_tgt_A[IND] = torch.Tensor(TGT_A)
	MY_tgt_B[IND] = torch.Tensor(TGT_B)
	MY_syn[IND] = torch.Tensor([AB_SYN])



torch.save(MY_chem_A_feat, WORK_PATH+'0803.{}.MY_chem_A_feat.pt'.format(G_NAME))
torch.save(MY_chem_B_feat, WORK_PATH+'0803.{}.MY_chem_B_feat.pt'.format(G_NAME))
torch.save(MY_chem_A_adj, WORK_PATH+'0803.{}.MY_chem_A_adj.pt'.format(G_NAME))
torch.save(MY_chem_B_adj, WORK_PATH+'0803.{}.MY_chem_B_adj.pt'.format(G_NAME))
torch.save(MY_exp_A, WORK_PATH+'0803.{}.MY_exp_A.pt'.format(G_NAME))
torch.save(MY_exp_B, WORK_PATH+'0803.{}.MY_exp_B.pt'.format(G_NAME))
torch.save(MY_exp_AB, WORK_PATH+'0803.{}.MY_exp_AB.pt'.format(G_NAME))
torch.save(MY_tgt_A, WORK_PATH+'0803.{}.MY_tgt_A.pt'.format(G_NAME))
torch.save(MY_tgt_B, WORK_PATH+'0803.{}.MY_tgt_B.pt'.format(G_NAME))
torch.save(MY_syn, WORK_PATH+'0803.{}.MY_syn.pt'.format(G_NAME))




























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





