
graph2vec 사용 방법 비교

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


import os
import json
import glob
import pandas as pd
import networkx as nx
from joblib import Parallel, delayed
import smart_open
smart_open.open = smart_open.smart_open # for gensim and node2vec 
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import hashlib
import tqdm
from tqdm import tqdm
from tqdm import trange






seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


WORK_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/my_trial_3_2/'
DC_PATH = '/st06/jiyeonH/13.DD_SESS/DrugComb.1.5/'
IDK_PATH = '/st06/jiyeonH/13.DD_SESS/ideker/'
LINCS_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/' 
TARGET_PATH = '/st06/jiyeonH/13.DD_SESS/merged_target/'


WORK_PATH = '/home01/k006a01/PRJ.01/TRIAL_3.2/'
DC_PATH = '/home01/k006a01/01.DATA/DrugComb/'
IDK_PATH = '/home01/k006a01/01.DATA/IDK/'
LINCS_PATH = '/home01/k006a01/01.DATA/LINCS/'
TARGET_PATH = '/home01/k006a01/01.DATA/TARGET/'




# LINCS DATA
print("LINCS")


BETA_BIND = pd.read_csv(LINCS_PATH+"BETA_DATA_for_20220705_978.csv")
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






# NETWORK
print('NETWORK')
IDEKER_IAS = pd.read_csv(IDK_PATH+'IAS_score.tsv', sep = '\t')
IDEKER_TOT_GS = list(set(list(IDEKER_IAS['Protein 1'])+list(IDEKER_IAS['Protein 2']))) # 16840
L_matching_list = pd.read_csv(IDK_PATH+'L_12_string.csv', sep = '\t')

IDEKER_IAS_L1 = IDEKER_IAS[IDEKER_IAS['Protein 1'].isin(L_matching_list.PPI_name)]
IDEKER_IAS_L2 = IDEKER_IAS_L1[IDEKER_IAS_L1['Protein 2'].isin(L_matching_list.PPI_name)] # 20232

len(set(list(IDEKER_IAS_L2['Protein 1']) + list(IDEKER_IAS_L2['Protein 2'])))
ID_G = nx.from_pandas_edgelist(IDEKER_IAS_L2, 'Protein 1', 'Protein 2')

ESSEN_NAMES = list(set(L_matching_list['PPI_name']))

MSSNG = [a for a in ESSEN_NAMES if a not in list(ID_G.nodes)]
for nn in list(MSSNG):
		ID_G.add_node(nn)

ID_GENE_ORDER_mini = list(ID_G.nodes()) # 20232
ID_ADJ = nx.adjacency_matrix(ID_G)
ID_ADJ_tmp = torch.LongTensor(ID_ADJ.toarray())
ID_ADJ_IDX = ID_ADJ_tmp.to_sparse().indices()  # [2, 40464]
ID_WEIGHT = [] # len : 20232 -> 40464

ID_ADJ_IDX_T = pd.DataFrame(ID_ADJ_IDX.T.detach().tolist())
ID_ADJ_IDX_T.columns = ['Protein 1','Protein 2']
ID_ADJ_IDX_T['NodeA'] = [list(ID_GENE_ORDER_mini)[A] for A in list(ID_ADJ_IDX_T['Protein 1'])]
ID_ADJ_IDX_T['NodeB'] = [list(ID_GENE_ORDER_mini)[B] for B in list(ID_ADJ_IDX_T['Protein 2'])]
ID_ADJ_IDX_T['NAMESUM'] = ID_ADJ_IDX_T['NodeA'] +'__'+ID_ADJ_IDX_T['NodeB']

IAS_FILTER = IDEKER_IAS[['Protein 1', 'Protein 2', 'Integrated score']]
IAS_FILTER['NAMESUM_1'] = IAS_FILTER['Protein 1']+'__'+IAS_FILTER['Protein 2']
IAS_FILTER['NAMESUM_2'] = IAS_FILTER['Protein 2']+'__'+IAS_FILTER['Protein 1']
IAS_FILTER = IAS_FILTER[['NAMESUM_1','NAMESUM_2','Integrated score']]
IAS_FILTER1= IAS_FILTER[['NAMESUM_1', 'Integrated score']]
IAS_FILTER2= IAS_FILTER[['NAMESUM_2', 'Integrated score']]
IAS_FILTER1.columns = ['NAMESUM', 'Integrated score']
IAS_FILTER2.columns = ['NAMESUM', 'Integrated score']
IAS_FILTER = pd.concat([IAS_FILTER1, IAS_FILTER2],axis = 0)

ID_WEIGHT = pd.merge(ID_ADJ_IDX_T, IAS_FILTER, on = 'NAMESUM', how = 'left' )

ID_WEIGHT_SCORE = list(ID_WEIGHT['Integrated score'])



print('target') 
# 모든 데이터베이스 포함된 cid - target 파일
tot_target_df = pd.read_csv(TARGET_PATH+'combined_target_b_woprediction.csv', low_memory = False)
# drug 데이터베이스 포함된 cid - target 파일(stitch bindingdb snap interdecagon 포함안됨)
wo_target_df = pd.read_csv(TARGET_PATH+'combined_target.csv')

tot_target_df_re = tot_target_df[['cid','target', 'db_name']] # 3370390
wo_target_df_re = wo_target_df[['cid','target', 'db_name']]


TARGET_FILTER = tot_target_df_re[tot_target_df_re.cid.isin(CELLO_DC_BETA_cids)] # 176개 중에서 2개가 음슴 
TARGET_FILTER[TARGET_FILTER.target.isin(L_matching_list.L_gene_symbol)] # 11481
TARGET_FILTER[TARGET_FILTER.target.isin(L_matching_list.PPI_name)] # 11445

TARGET_FILTER_re = TARGET_FILTER[TARGET_FILTER.target.isin(L_matching_list.L_gene_symbol)]
target_cids = list(set(TARGET_FILTER_re.cid)) # 162 


TARGET_SCORE = pd.DataFrame(columns = ['cid', 'target', 'score'])

for cid in target_cids :
	tmp_df = TARGET_FILTER_re[TARGET_FILTER_re.cid == cid]
	tmp_df2 = tmp_df.groupby('target').count()['db_name']
	tmp_df3 = pd.DataFrame({'cid' : [cid]*len(tmp_df2), 'target' : tmp_df2.index.to_list() , 'score' : tmp_df2.to_list()})
	TARGET_SCORE = pd.concat([TARGET_SCORE, tmp_df3])



########################################################
#########################################################


print("LEARNING")

# Graph 확인 
ID_GENE_ORDER_mini = ID_G.nodes()
IAS_PPI = nx.adjacency_matrix(ID_G)

JY_GRAPH = ID_G
JY_GRAPH_ORDER = ID_G.nodes()
JY_ADJ = nx.adjacency_matrix(ID_G)

JY_ADJ_tmp = torch.LongTensor(JY_ADJ.toarray())
JY_ADJ_IDX = JY_ADJ_tmp.to_sparse().indices()
JY_IDX_WEIGHT = ID_WEIGHT_SCORE



# DC set 확인 
A_B_C_S = CELLO_DC_BETA.reset_index()
# drug target filter 
#A_B_C_S_row = A_B_C_S[A_B_C_S.drug_row_cid.isin(list(TARGET_FILTER_re.cid))]
#A_B_C_S_col = A_B_C_S_row[A_B_C_S_row.drug_col_cid.isin(list(TARGET_FILTER_re.cid))]
#
A_B_C_S_SET = A_B_C_S[['BETA_sig_id_x','BETA_sig_id_y','DrugCombCello']].drop_duplicates()
A_B_C_S_SET = A_B_C_S_SET.reset_index()


# LINCS 확인 
BETA_ORDER_pre =[list(L_matching_list.PPI_name).index(a) for a in JY_GRAPH_ORDER]
BETA_ORDER_DF = L_matching_list.iloc[BETA_ORDER_pre] # 어차피 ref 다르고 같은 애들이라 괜춘 
BETA_ORDER = list(BETA_ORDER_DF.entrez)




근데 여기서 이제 문제가, target filter 필요함? ㄴㄴ target 지금은 못넣어줌. exp 때문에 
일단 해보자 



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
A_B_C_S_SET = A_B_C_S_SET.reset_index(drop=True)



# Lincs G2V 에 넣기 
class WeisfeilerLehmanMachine:
	def __init__(self, graph, features, iterations):
		self.iterations = iterations
		self.graph = graph
		self.features = features
		self.nodes = self.graph.nodes()
		self.extracted_features = [str(v) for k, v in features.items()]
		self.node_dict = {list(self.nodes)[a]:BETA_ORDER[a] for a in range(978)}
		self.do_recursions()
	#
	def do_a_recursion(self):
		new_features = {}
		for node in self.nodes:
			nebs = self.graph.neighbors(node)
			degs = [self.features[self.node_dict[neb]] for neb in nebs]
			wl_features = [str(self.features[self.node_dict[node]])]+sorted([str(deg) for deg in degs])
			wl_features = "_".join(wl_features)
			hash_object = hashlib.md5(wl_features.encode()) # hash 값으로 그래프를 벡터화 
			hashing = hash_object.hexdigest()
			new_features[self.node_dict[node]] = hashing
		self.extracted_features = self.extracted_features + list(new_features.values())
		return new_features
	#
	def do_recursions(self):
		for _ in range(self.iterations):
			self.features = self.do_a_recursion()



def get_LINCS(sig_id):
	# sig_id = 'MOAR001_A375_24H:H10'
	graph = ID_G
	Drug_EXP = BETA_BIND[['id', sig_id]]
	BIND_ORDER = [list(BETA_BIND.id).index(a) for a in BETA_ORDER_DF.entrez] 
	Drug_EXP_ORD = Drug_EXP.iloc[BIND_ORDER]
	tmp_feature_pre = np.round(Drug_EXP_ORD[[sig_id]])
	tmp_feature = pd.concat([ Drug_EXP_ORD[['id']] , tmp_feature_pre ], axis = 1)
	tmp_feature_dict = {}
	#
	for indd in range(978) :
		gene_id = tmp_feature.loc[indd,'id']
		gene_exp = tmp_feature.loc[indd, sig_id]
		tmp_feature_dict[int(gene_id)] = gene_exp	
	return graph, tmp_feature_dict, sig_id


def feature_extractor(sig_id, rounds): 
	graph, features, sig_id = get_LINCS(sig_id) 
	machine = WeisfeilerLehmanMachine(graph, features, rounds) 
	doc = TaggedDocument(words=machine.extracted_features, tags=["g_" + sig_id]) # update 된거 말고 전부 
	return doc


def infer_from_model(model, data, dim):
	vec_A = []
	vec_B = []
	for line in data.g2v_A :
		tmp_a = model.infer_vector(line[0])
		vec_A.append(tmp_a)
	for line in data.g2v_B :
		tmp_b = model.infer_vector(line[0])
		vec_B.append(tmp_b)
	return torch.Tensor(vec_A).view(-1,dim), torch.Tensor(vec_B).view(-1,dim)


def get_from_model(model, data, dim):
	vec_A = []
	vec_B = []
	for line in data.g2v_A :
		tmp_a = model.get_vector(line[0])
		vec_A.append(tmp_a)
	for line in data.g2v_B :
		tmp_b = model.get_vector(line[0])
		vec_B.append(tmp_b)
	return torch.Tensor(vec_A).view(-1,dim), torch.Tensor(vec_B).view(-1,dim)



def get_synergy_data(DrugA_SIG, DrugB_SIG, Cell):
	ABCS1 = A_B_C_S[A_B_C_S.BETA_sig_id_x == DrugA_SIG]
	ABCS2 = ABCS1[ABCS1.BETA_sig_id_y == DrugB_SIG]
	ABCS3 = ABCS2[ABCS2.DrugCombCello == Cell]
	synergy_score = np.median(ABCS3.synergy_loewe)
	return synergy_score


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



MY_chem_A_feat = torch.empty(size=(A_B_C_S_SET.shape[0], 50, 64))
MY_chem_B_feat= torch.empty(size=(A_B_C_S_SET.shape[0], 50, 64))
MY_chem_A_adj = torch.empty(size=(A_B_C_S_SET.shape[0], 50, 50))
MY_chem_B_adj= torch.empty(size=(A_B_C_S_SET.shape[0], 50, 50))
# G2V feature 작성 생각해보기 
MY_g2v_A = []
MY_g2v_B = []
#
MY_syn =  torch.empty(size=(A_B_C_S_SET.shape[0],1))





MY_chem_A_feat = torch.empty(size=(100, 50, 64))
MY_chem_B_feat= torch.empty(size=(100, 50, 64))
MY_chem_A_adj = torch.empty(size=(100, 50, 50))
MY_chem_B_adj= torch.empty(size=(100, 50, 50))
# G2V feature 작성 생각해보기 
MY_g2v_A = []
MY_g2v_B = []
#
MY_syn =  torch.empty(size=(100,1))
### tt = feature_extractor('MOAR001_A375_24H:H10', 2)



for IND in range(A_B_C_S_SET.shape[0]):  # 
	# IND=0  
	DrugA_SIG = A_B_C_S_SET.iloc[IND,]['BETA_sig_id_x']
	DrugB_SIG = A_B_C_S_SET.iloc[IND,]['BETA_sig_id_y']
	Cell = A_B_C_S_SET.iloc[IND,]['DrugCombCello']
	#
	k=1
	DrugA_Feat, DrugA_ADJ = get_CHEM(DrugA_SIG, k)
	DrugB_Feat, DrugB_ADJ = get_CHEM(DrugB_SIG, k)
	#
	wl_it = 2 
	G2V_A = feature_extractor(DrugA_SIG, wl_it)
	G2V_B = feature_extractor(DrugB_SIG, wl_it)
	# EXP_A, EXP_B, EXP_AB, LINCS = get_LINCS_data(DrugA_SIG, DrugB_SIG)
	#
	AB_SYN = get_synergy_data(DrugA_SIG, DrugB_SIG, Cell)
	#
	MY_chem_A_feat[IND] = torch.Tensor(DrugA_Feat)
	MY_chem_B_feat[IND] = torch.Tensor(DrugB_Feat)
	MY_chem_A_adj[IND] = torch.Tensor(DrugA_ADJ)
	MY_chem_B_adj[IND] = torch.Tensor(DrugB_ADJ)
	MY_g2v_A.append(G2V_A)
	MY_g2v_B.append(G2V_B)
	MY_syn[IND] = torch.Tensor([AB_SYN])


torch.save(MY_chem_A_feat, WORK_PATH+'0713.MY_chem_A_feat.pt')
torch.save(MY_chem_B_feat, WORK_PATH+'0713.MY_chem_B_feat.pt')
torch.save(MY_chem_A_adj, WORK_PATH+'0713.MY_chem_A_adj.pt')
torch.save(MY_chem_B_adj, WORK_PATH+'0713.MY_chem_B_adj.pt')
with open(WORK_PATH+'0713.MY_chem_A_g2v.json', 'w') as f:
	json.dump(MY_g2v_A,f)
with open(WORK_PATH+'0713.MY_chem_B_g2v.json', 'w') as f:
	json.dump(MY_g2v_B,f)
torch.save(MY_syn, WORK_PATH+'0713.MY_syn.pt')



MY_chem_A_feat = torch.load( '/st06/jiyeonH/11.TOX/MY_TRIAL_GPU/PJ01.TRIAL.3_2/'+'0713.MY_chem_A_feat.pt')
MY_chem_B_feat = torch.load( '/st06/jiyeonH/11.TOX/MY_TRIAL_GPU/PJ01.TRIAL.3_2/'+'0713.MY_chem_B_feat.pt')
MY_chem_A_adj = torch.load( '/st06/jiyeonH/11.TOX/MY_TRIAL_GPU/PJ01.TRIAL.3_2/'+'0713.MY_chem_A_adj.pt')
MY_chem_B_adj = torch.load( '/st06/jiyeonH/11.TOX/MY_TRIAL_GPU/PJ01.TRIAL.3_2/'+'0713.MY_chem_B_adj.pt')
MY_syn = torch.load( '/st06/jiyeonH/11.TOX/MY_TRIAL_GPU/PJ01.TRIAL.3_2/'+'0713.MY_syn.pt')
with open('/st06/jiyeonH/11.TOX/MY_TRIAL_GPU/PJ01.TRIAL.3_2/'+'0713.MY_chem_A_g2v.json', 'r') as f:
	MY_g2v_A = json.load(f)

with open('/st06/jiyeonH/11.TOX/MY_TRIAL_GPU/PJ01.TRIAL.3_2/'+'0713.MY_chem_B_g2v.json', 'r') as f:
	MY_g2v_B = json.load(f)



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




def prepare_data_GCN(MY_chem_A_feat, MY_chem_B_feat, MY_chem_A_adj, MY_chem_B_adj, MY_g2v_A, MY_g2v_B, MY_syn, norm ) :
	chem_feat_A_train, chem_feat_A_tv, chem_feat_B_train, chem_feat_B_tv,chem_adj_A_train, chem_adj_A_tv, chem_adj_B_train, chem_adj_B_tv, g2v_A_train, g2v_A_tv, g2v_B_train, g2v_B_tv, syn_train, syn_tv = sklearn.model_selection.train_test_split(
			MY_chem_A_feat[0:100], MY_chem_B_feat[0:100], MY_chem_A_adj[0:100], MY_chem_B_adj[0:100], MY_g2v_A[0:100], MY_g2v_B[0:100], MY_syn[0:100], test_size= 0.2 , random_state=42 )
	chem_feat_A_val, chem_feat_A_test, chem_feat_B_val, chem_feat_B_test, chem_adj_A_val, chem_adj_A_test, chem_adj_B_val, chem_adj_B_test, g2v_A_val, g2v_A_test, g2v_B_val, g2v_B_test, syn_val, syn_test  = sklearn.model_selection.train_test_split(
			chem_feat_A_tv, chem_feat_B_tv, chem_adj_A_tv, chem_adj_B_tv, g2v_A_tv, g2v_B_tv, syn_tv, test_size=0.5, random_state=42 )
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
	train_data['G2V_A'] = g2v_A_train + g2v_B_train 
	val_data['G2V_A'] = g2v_A_val
	test_data['G2V_A'] = g2v_A_test
	#
	train_data['G2V_B'] = g2v_B_train + g2v_A_train
	val_data['G2V_B'] = g2v_B_val
	test_data['G2V_B'] = g2v_B_test
	#               
	train_data['y'] = np.concatenate((syn_train, syn_train), axis=0)
	val_data['y'] = syn_val
	test_data['y'] = syn_test
	#
	print(train_data['drug1_feat'].shape)
	print(val_data['drug1_feat'].shape)
	print(test_data['drug1_feat'].shape)
	return train_data, val_data, test_data




norm = 'tanh_norm'
train_data, val_data, test_data = prepare_data_GCN(MY_chem_A_feat, MY_chem_B_feat, MY_chem_A_adj, MY_chem_B_adj, MY_g2v_A, MY_g2v_B, MY_syn, norm)
wl_iterations = 2








class DATASET_GCN_G2V(Dataset):
	def __init__(self, gcn_drug1_F, gcn_drug2_F, gcn_drug1_ADJ, gcn_drug2_ADJ, g2v_A, g2v_B, syn_ans):
		self.gcn_drug1_F = gcn_drug1_F
		self.gcn_drug2_F = gcn_drug2_F
		self.gcn_drug1_ADJ = gcn_drug1_ADJ
		self.gcn_drug2_ADJ = gcn_drug2_ADJ
		self.g2v_A = g2v_A
		self.g2v_B = g2v_B
		self.syn_ans = syn_ans
		#
	#
	def __len__(self):
		return len(self.gcn_drug1_F)
	#
	def __getitem__(self, index):
		adj_re_A = self.gcn_drug1_ADJ[index].long().to_sparse().indices()
		adj_re_B = self.gcn_drug2_ADJ[index].long().to_sparse().indices()
		return self.gcn_drug1_F[index], self.gcn_drug2_F[index], adj_re_A, adj_re_B, self.g2v_A[index], self.g2v_B[index], self.syn_ans[index]



def graph_collate_fn(batch):
	drug1_f_list = []
	drug2_f_list = []
	drug1_adj_list = []
	drug2_adj_list = []
	g2v_A_list = []
	g2v_B_list = []
	y_list = []
	DRUG_1_num_nodes_seen = 0
	DRUG_2_num_nodes_seen = 0
	for drug1_f, drug2_f, drug1_adj, drug2_adj, g2v_A, g2v_B, y in batch :
		drug1_f_list.append(drug1_f)
		drug2_f_list.append(drug2_f)
		drug1_adj_list.append(drug1_adj+DRUG_1_num_nodes_seen)
		drug2_adj_list.append(drug2_adj+DRUG_2_num_nodes_seen)
		g2v_A_list.append(g2v_A)
		g2v_B_list.append(g2v_B)
		y_list.append(y)
		DRUG_1_num_nodes_seen += drug1_f.shape[0]
		DRUG_2_num_nodes_seen += drug2_f.shape[0]
	drug1_f_new = torch.cat(drug1_f_list, 0)
	drug2_f_new = torch.cat(drug2_f_list, 0)
	drug1_adj_new = torch.cat(drug1_adj_list, 1)
	drug2_adj_new = torch.cat(drug2_adj_list, 1)
	y_new = torch.stack(y_list, 0)
	return drug1_f_new, drug2_f_new, drug1_adj_new, drug2_adj_new, g2v_A_list, g2v_B_list, y_new



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




# WEIGHT 
ys = train_data['y'].squeeze().tolist()
min_s = np.amin(ys)
loss_weight = np.log(train_data['y'] - min_s + np.e)
JY_IDX_WEIGHT_T = torch.Tensor(JY_IDX_WEIGHT).view(1,-1)





# DATA check 
T_train = DATASET_GCN_G2V(
	torch.Tensor(train_data['drug1_feat']), torch.Tensor(train_data['drug2_feat']), 
	torch.Tensor(train_data['drug1_adj']), torch.Tensor(train_data['drug2_adj']),
	train_data['G2V_A'], train_data['G2V_B'], 
	torch.Tensor(train_data['y']))

T_val = DATASET_GCN_G2V(
	torch.Tensor(val_data['drug1_feat']), torch.Tensor(val_data['drug2_feat']), 
	torch.Tensor(val_data['drug1_adj']), torch.Tensor(val_data['drug2_adj']),
	val_data['G2V_A'], val_data['G2V_B'], 
	torch.Tensor(val_data['y']))
	
T_test = DATASET_GCN_G2V(
	torch.Tensor(test_data['drug1_feat']), torch.Tensor(test_data['drug2_feat']), 
	torch.Tensor(test_data['drug1_adj']), torch.Tensor(test_data['drug2_adj']),
	test_data['G2V_A'], test_data['G2V_B'], 
	torch.Tensor(test_data['y']))

RAY_train = ray.put(T_train)
RAY_val = ray.put(T_val)
RAY_test = ray.put(T_test)
RAY_loss_weight = ray.put(loss_weight)



class MY_GCN_G2V_MODEL(torch.nn.Module):
	def __init__(self, G_layer_chem, G_indim_chem, G_hiddim_chem, g2v_indim, layers_1, layers_2, layers_3, out_dim, inDrop, drop):
		super(MY_GCN_G2V_MODEL, self).__init__()
		self.G_layer_chem = G_layer_chem
		self.G_indim_chem = G_indim_chem
		self.G_hiddim_chem = G_hiddim_chem
		self.g2v_indim = g2v_indim
		self.layers_1 = [int(a) for a in layers_1]
		self.layers_2 = [int(a) for a in layers_2]
		self.layers_3 = [int(a) for a in layers_3]
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
		self.Convs_1 = torch.nn.ModuleList([torch.nn.Linear(self.G_hiddim_chem+self.g2v_indim, self.layers_1[0] )])
		self.Convs_1.extend([torch.nn.Linear(self.layers_1[a], self.layers_1[a+1]) for a in range(len(self.layers_1)-1)])
		#
		self.Convs_2 = torch.nn.ModuleList([torch.nn.Linear(self.G_hiddim_chem+self.g2v_indim, self.layers_2[0] )])
		self.Convs_2.extend([torch.nn.Linear(self.layers_2[a], self.layers_2[a+1]) for a in range(len(self.layers_2)-1)])
		#
		self.SNPs = torch.nn.ModuleList([torch.nn.Linear(self.layers_1[-1]+self.layers_2[-1], self.layers_3[0] )])
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
	def forward(self, Drug1_F, Drug2_F, Drug1_ADJ, Drug2_ADJ, G2V_A, G2V_B, syn ):
		Drug_batch_label = self.calc_batch_label(syn, Drug1_F)
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
		input_drug1 = torch.concat( (G_1_C_out, G2V_A), 1 ) # normalization 추가 해야할것 같은데 
		input_drug2 = torch.concat( (G_2_C_out, G2V_B), 1 )
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
		X = torch.cat((input_drug1,input_drug2),1)
		for L3 in range(len(self.SNPs)):
			if L3 != len(self.SNPs)-1 :
				X = self.SNPs[L3](X)
				X = F.dropout(X, p=self.drop, training = self.training)
				X = F.relu(X)
			else :
				X = self.SNPs[L3](X)
		return X



def RAY_GCN_D2V_Train(config, checkpoint_dir=None):
	use_cuda = False
	import smart_open
	smart_open.open = smart_open.smart_open 
	from gensim.models.doc2vec import Doc2Vec, TaggedDocument
	from gensim.test.utils import get_tmpfile
	#
	T_train = ray.get(RAY_train)
	T_val = ray.get(RAY_val)
	T_test = ray.get(RAY_test)
	T_loss_weight = ray.get(RAY_loss_weight)
	batch_cut_weight = [T_loss_weight[i:i+config["batch_size"]] for i in range(0,len(T_loss_weight), config["batch_size"])]
	#
	loaders = {
			'train' : torch.utils.data.DataLoader(T_train, batch_size = config["batch_size"], collate_fn = graph_collate_fn, shuffle =False, num_workers=config['n_workers']),
			'eval' : torch.utils.data.DataLoader(T_val, batch_size = config["batch_size"], collate_fn = graph_collate_fn, shuffle =False, num_workers=config['n_workers'])
	}
	#
	print('G2V train')
	g2v_model = Doc2Vec(T_train.g2v_A, # 한쪽만 해도 겹쳐서 해서 상관 없을듯 
					vector_size=config['g2v_vector_size'] , # 근데 plotting 하려면 어떻게 뿌려줘야하나 
					window=config['g2v_window'] ,
					min_count=config['g2v_min_count'],
					dm=config['g2v_dm'],
					sample=config['g2v_sample'],
					workers=config['n_workers'],
					epochs=config['g2v_epochs'],
					alpha=config['g2v_alpha'])
	#
	print('infer')
	Train_G2V_A, Train_G2V_B = infer_from_model(g2v_model, T_train , config['g2v_vector_size'])
	Val_G2V_A, Val_G2V_B = infer_from_model(g2v_model, T_val , config['g2v_vector_size'])
	torch.save(Train_G2V_A, './0713.g2v.train_vector_A.pt')
	torch.save(Train_G2V_B, './0713.g2v.train_vector_B.pt')
	torch.save(Val_G2V_A, './0713.g2v.val_vector_A.pt')
	torch.save(Val_G2V_B, './0713.g2v.val_vector_B.pt')
	#
	dsn1_layers = [config["GCN_feat_size_0"], config["GCN_feat_size_1"] , config["GCN_feat_size_2"] ]
	dsn2_layers = [config["GCN_feat_size_0"], config["GCN_feat_size_1"] , config["GCN_feat_size_2"] ]
	snp_layers = [config["GCN_feat_size_3"] ,config["GCN_feat_size_4"]]
	inDrop = config["GCN_dropout_1"]
	drop = config["GCN_dropout_2"]
	#
	GCN_model = MY_GCN_G2V_MODEL(
		config['GCN_layer'], T_train.gcn_drug1_F.shape[-1], config['GCN_hiddim'], 
		config['g2v_vector_size'], dsn1_layers, dsn2_layers, snp_layers, 1,
		inDrop, drop)
	#
	if torch.cuda.is_available():
		GCN_model = GCN_model.cuda()
		if torch.cuda.device_count() > 1 :
			GCN_model = torch.nn.DataParallel(GCN_model)
	#
	optimizer = torch.optim.Adam(GCN_model.parameters(), lr = config['GCN_lr'])
	criterion = weighted_mse_loss
	#
	if checkpoint_dir :
		checkpoint = os.path.join(checkpoint_dir, "checkpoint")
		model_state, optimizer_state = torch.load(checkpoint)
		GCN_model.load_state_dict(model_state)
		optimizer.load_state_dict(optimizer_state)
	#		
	train_loss_all = []
	valid_loss_all = []
	# 이제 epoch 짜면 됨 
	n_epochs = config["GCN_epoch"]
	for epoch in range(n_epochs):
		now=datetime.now()
		train_loss = 0
		valid_loss = 0
		#
		GCN_model.train()
		for batch_idx_t, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, y) in enumerate(loaders['train']): 
			if use_cuda:
				drug1_f, drug2_f, drug1_a, drug2_a, y = drug1_f.cuda(), drug2_f.cuda(), drug1_a.cuda(), drug2_a.cuda(), y.cuda()
			optimizer.zero_grad()
			g2v_a = Train_G2V_A[batch_idx_t*config["batch_size"]:(batch_idx_t+1)*config["batch_size"]]
			g2v_b = Train_G2V_B[batch_idx_t*config["batch_size"]:(batch_idx_t+1)*config["batch_size"]]
			if use_cuda:
				g2v_a, g2v_b = g2v_a.cuda(), g2v_b.cuda()
			output = GCN_model(drug1_f, drug2_f, drug1_a, drug2_a, g2v_a, g2v_b, y)
			wc = torch.Tensor(batch_cut_weight[batch_idx_t]).view(-1,1)
			if torch.cuda.is_available():
				wc = wc.cuda()
			loss = criterion(output, y, wc )
			loss.backward()
			optimizer.step()
			train_loss += loss
			train_loss = train_loss + loss.item()
		######################    
		# validate the model #
		######################
		GCN_model.eval()
		for batch_idx_v, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, y) in enumerate(loaders['eval']):
			with torch.no_grad():
				if use_cuda:
					drug1_f, drug2_f, drug1_a, drug2_a, y = drug1_f.cuda(), drug2_f.cuda(), drug1_a.cuda(), drug2_a.cuda(), y.cuda()
				g2v_a = Val_G2V_A[batch_idx_v*config["batch_size"]:(batch_idx_v+1)*config["batch_size"]]
				g2v_b = Val_G2V_B[batch_idx_v*config["batch_size"]:(batch_idx_v+1)*config["batch_size"]]
				if use_cuda:
					g2v_a, g2v_b = g2v_a.cuda(), g2v_b.cuda()
				output = GCN_model(drug1_f, drug2_f, drug1_a, drug2_a, g2v_a, g2v_b, y)
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
		done = datetime.now()
		time_spent = done-now
		# check point 
		with tune.checkpoint_dir(step = epoch) as checkpoint_dir:
			path = os.path.join(checkpoint_dir, "checkpoint")
			torch.save((GCN_model.state_dict(), optimizer.state_dict()), path)
			torch.save(GCN_model.state_dict(), './GCN_model.pth')
			g2v_model.save('./G2V_model')
		tune.report(TrainLoss= TRAIN_LOSS, ValLoss= VAL_LOSS)
	#
	print("Finished Training")



def RAY_GCN_D2V_Test(best_trial, g2v_model_path, use_cuda=False):
	from gensim.models.doc2vec import Doc2Vec, TaggedDocument
	from gensim.test.utils import get_tmpfile
	T_test = ray.get(RAY_test)
	Test_loader = torch.utils.data.DataLoader(T_test, batch_size = best_trial.config["batch_size"], collate_fn = graph_collate_fn, shuffle =False, num_workers=best_trial.config['n_workers'])
	g2v_model = Doc2Vec.load(g2v_model_path)
	Test_G2V_A, Test_G2V_B = infer_from_model(g2v_model, T_test , config['g2v_vector_size'])
	#
	dsn1_layers = [best_trial.config["GCN_feat_size_0"], best_trial.config["GCN_feat_size_1"], best_trial.config["GCN_feat_size_2"] ]
	dsn2_layers = [best_trial.config["GCN_feat_size_0"], best_trial.config["GCN_feat_size_1"], best_trial.config["GCN_feat_size_2"] ]
	snp_layers = [best_trial.config["GCN_feat_size_3"], best_trial.config["GCN_feat_size_4"]]
	inDrop = best_trial.config["GCN_dropout_1"]
	drop = best_trial.config["GCN_dropout_2"]
	#       
	best_trained_model = MY_GCN_G2V_MODEL(
		best_trial.config['GCN_layer'], T_test.gcn_drug1_F.shape[-1], best_trial.config['GCN_hiddim'], 
		best_trial.config['g2v_vector_size'], dsn1_layers, dsn2_layers, snp_layers, 1,
		inDrop, drop)
	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	best_trained_model.to(device)
	checkpoint_path = os.path.join(best_trial.checkpoint.value, "checkpoint")
	model_state, optimizer_state = torch.load(checkpoint_path)
	best_trained_model.load_state_dict(model_state)
	#
	#
	test_loss = 0.0
	PRED_list = []
	Y_list = test_data['y'].squeeze().tolist()
	with torch.no_grad():
		best_trained_model.eval()
	for batch_idx_t, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB,  y) in enumerate(Test_loader):
		if use_cuda:
			drug1_f, drug2_f, drug1_a, drug2_a, y = drug1_f.cuda(), drug2_f.cuda(), drug1_a.cuda(), drug2_a.cuda(), y.cuda()
		g2v_a = Test_G2V_A[batch_idx_t*config["batch_size"]:(batch_idx_t+1)*config["batch_size"]]
		g2v_b = Test_G2V_B[batch_idx_t*config["batch_size"]:(batch_idx_t+1)*config["batch_size"]]
		if use_cuda:
			g2v_a, g2v_b = g2v_a.cuda(), g2v_b.cuda()
		outout = best_trained_model(drug1_f, drug2_f, drug1_a, drug2_a, g2v_a, g2v_b, y)
		MSE = torch.nn.MSELoss()
		loss = MSE(output, y)
		test_loss = test_loss + loss.item()
		outputs = output.squeeze().tolist()
		PRED_list = PRED_list+outputs
	TEST_LOSS = test_loss/(batch_idx_t+1)
	result_pearson(PRED_list, Y_list)
	result_spearman(PRED_list, Y_list)
	print("Best model TEST loss: {}".format(TEST_LOSS))







def MAIN(ANAL_name, WORK_PATH, num_samples= 10, max_num_epochs=1000, grace_period = 150, cpus_per_trial = 16, gpus_per_trial = 1):
	CONFIG={
		"g2v_vector_size" : tune.randint(10,100), # 얘도 1024개니까  너무 feat 많음 
		"g2v_window" : tune.choice([0]),
		"g2v_min_count" : tune.choice([0]), # 다 고려해야지 
		"g2v_dm" : tune.choice([0]), # 메모리 관련 
		"g2v_sample" : tune.loguniform(1e-5, 1e-1)  , # high Freq down sample 
		#"g2v_workers" : tune.choice([3]), 
		"g2v_epochs" : tune.choice([10, 50, 100]),
		"g2v_alpha" : tune.loguniform(1e-5, 1e-2)  ,
		#
		'n_workers' : tune.choice([cpus_per_trial]),
		"batch_size" : tune.choice([64, 32, 16]), # The number of batch sizes should be a power of 2 to take full advantage of the GPUs processing #  
		"GCN_epoch" : tune.choice([max_num_epochs]),
		"GCN_layer" : tune.choice([2, 3, 4]),
		"GCN_hiddim" : tune.choice([512, 256, 128, 64, 32]),
		"GCN_feat_size_0" : tune.choice([4096, 2048, 1024, 512, 256, 128, 64, 32]), # 
		"GCN_feat_size_1" : tune.choice([4096, 2048, 1024, 512, 256, 128, 64, 32]),# 
		"GCN_feat_size_2" : tune.choice([4096, 2048, 1024, 512, 256, 128, 64, 32]),# 
		"GCN_feat_size_3" : tune.choice([4096, 2048, 1024, 512, 256, 128, 64, 32]),# 
		"GCN_feat_size_4" : tune.choice([4096, 2048, 1024, 512, 256, 128, 64, 32]),# 
		"GCN_dropout_1" : tune.choice([0.01, 0.2, 0.5, 0.8]),
		"GCN_dropout_2" : tune.choice([0.01, 0.2, 0.5, 0.8]),
		"GCN_lr" : tune.choice([0.00001, 0.0001, 0.001]),
	}
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
		tune.with_parameters(RAY_GCN_D2V_Train),
		name = ANAL_name,
		num_samples=num_samples,
		config=CONFIG,
		resources_per_trial={'cpu': cpus_per_trial},# , 'gpu' : gpus_per_trial
		progress_reporter = reporter,
		search_alg = optuna_search,
		scheduler = ASHA_scheduler
	)
	best_trial = ANALYSIS.get_best_trial("ValLoss", "min", "last")
	print("Best trial config: {}".format(best_trial.config))
	print("Best trial final validation loss: {}".format(
	best_trial.last_result["ValLoss"]))
	#
	ALL_DF = ANALYSIS.trial_dataframes
	TMP_DF = ALL_DF[best_trial.logdir]
	plot_loss(list(TMP_DF.TrainLoss), list(TMP_DF.ValLoss), WORK_PATH, 'MM_GCN_G2V_IDK')
	#
	print('start test with best model')
	if ray.util.client.ray.is_connected():
		from ray.util.ml_utils.node import force_on_current_node
		remote_fn = force_on_current_node(ray.remote(test_best_model))
		ray.get(remote_fn.remote(best_trial))
	else:
		RAY_GCN_D2V_Test(best_trial, best_trial.logdir+'/G2V_model' )




ANAL_name, WORK_PATH, num_samples, max_num_epochs, grace_period, cpus_per_trial, gpus_per_trial = 1
WORK_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/my_trial_3_2/'


MAIN('22.07.13.PRJ01.TRIAL3_2_pre', WORK_PATH, 2, 2, 1, 16, 1)

MAIN('22.07.13.PRJ01.TRIAL3_2_pre', WORK_PATH, 2, 2, 1, 32, 1)

MAIN('22.07.13.PRJ01.TRIAL3_2', WORK_PATH, 100, 1000, 150, 32, 1)














# 테스트용

loaders = {
			'train' : torch.utils.data.DataLoader(T_train, batch_size = 32, collate_fn = graph_collate_fn, shuffle =False, num_workers=32),
			'eval' : torch.utils.data.DataLoader(T_val, batch_size = 32, collate_fn = graph_collate_fn, shuffle =False, num_workers=32)
}
#
g2v_model = Doc2Vec(T_train.g2v_A, # 한쪽만 해도 겹쳐서 해서 상관 없을듯 
				vector_size=100 , # 근데 plotting 하려면 어떻게 뿌려줘야하나 
				window=0 ,
				min_count=0,
				dm=0,
				sample=0.01,
				workers=32,
				epochs=10,
				alpha=0.01)
#
Train_G2V_A, Train_G2V_B = infer_from_model(g2v_model, T_train , 100)
Val_G2V_A, Val_G2V_B = infer_from_model(g2v_model, T_val , 100)
#
#
dsn1_layers = [1024, 128 , 64 ]
dsn2_layers = [1024, 128 , 64]
snp_layers = [32,16]
inDrop = 0.5
drop = 0.01
#

GCN_model = MY_GCN_G2V_MODEL(
		config['g2v_vector_size'], T_train.gcn_drug1_F.shape[-1], config['GCN_hiddim'], 
		config['g2v_vector_size'], dsn1_layers, dsn2_layers, snp_layers, 1,
		inDrop, drop)



GCN_model = MY_GCN_G2V_MODEL(
		3, T_train.gcn_drug1_F.shape[-1], 50, 
		100, dsn1_layers, dsn2_layers, snp_layers, 1,
		inDrop, drop)

for batch_idx_t, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, y) in enumerate(loaders['train']): 
	batch_idx_t


batch_idx_t = 0
tmp_a = Train_G2V_A[batch_idx_t*32:(batch_idx_t+1)*32]
tmp_b = Train_G2V_B[batch_idx_t*32:(batch_idx_t+1)*32]

GCN_model(drug1_f, drug2_f, drug1_a, drug2_a, tmp_a, tmp_b, y)

tmp_g2v_model = Doc2Vec.load('/home/jiyeonH/ray_results/22.07.13.PRJ01.TRIAL3_2_pre/RAY_GCN_D2V_Train_40ff39a6_1_GCN_dropout_1=0.5,GCN_dropout_2=0.5,GCN_epoch=2,GCN_feat_size_0=128,GCN_feat_size_1=1024,GCN_feat_siz_2022-07-13_12-02-39/G2V_model')



############################################################################
############################ from GPU ############################
############################################################################






anal_df = ExperimentAnalysis("~/ray_results/22.07.13.PRJ01.TRIAL3_2")

ANA_DF = anal_df.dataframe()
ANA_ALL_DF = anal_df.trial_dataframes

ANA_DF.to_csv('/home01/k006a01/PRJ.01/TRIAL_3.2/RAY_ANA_DF.P01.3_2.csv')
import pickle
with open("/home01/k006a01/PRJ.01/TRIAL_3.2/RAY_ANA_DF.P01.3_2.pickle", "wb") as fp:
	pickle.dump(ANA_ALL_DF,fp) 


DF_KEY = list(ANA_DF.sort_values('ValLoss')['logdir'])[0]
DF_KEY

# get 
get /home01/k006a01/ray_results/22.07.13.PRJ01.TRIAL3_2/RAY_GCN_D2V_Train_b0ad3ca6_32_GCN_dropout_1=0.5000,GCN_dropout_2=0.2000,GCN_epoch=1000,GCN_feat_size_0=2048,GCN_feat_size_1=1024,G_2022-07-15_22-07-21/
/G2V_model M1.G2V_model
/G2V_model.syn1neg.npy M1.G2V_model.syn1neg.npy
/G2V_model.wv.vectors.npy M1.G2V_model.wv.vectors.npy
/GCN_model.pth M1.GCN_model.pth


TOPVAL_PATH = DF_KEY
mini_df = ANA_ALL_DF[DF_KEY]


cck_num =mini_df[mini_df.ValLoss==min(mini_df.ValLoss)].index.item()
checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
TOPVAL_PATH = DF_KEY + checkpoint
TOPVAL_PATH

get /home01/k006a01/ray_results/22.07.13.PRJ01.TRIAL3_2/RAY_GCN_D2V_Train_b0ad3ca6_32_GCN_dropout_1=0.5000,GCN_dropout_2=0.2000,GCN_epoch=1000,GCN_feat_size_0=2048,GCN_feat_size_1=1024,G_2022-07-15_22-07-21/checkpoint_000492/checkpoint M2_checkpoint




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


# get 


get /home01/k006a01/ray_results/22.07.13.PRJ01.TRIAL3_2/RAY_GCN_DV_Train_18a15f94_78_GCN_dropout_1=0.5000,GCN_dropout_2=0.2000,GCN_epoch=1000,GCN_feat_size_0=512,GCN_feat_size_1=512,GCN_2022-07-19_05-41-04/checkpoint_000351
/checkpoint M4.checkpoint

get /home01/k006a01/ray_results/22.07.13.PRJ01.TRIAL3_2/RAY_GCN_D2V_Train_18a15f94_78_GCN_dropout_1=0.5000,GCN_dropout_2=0.2000,GCN_epoch=1000,GCN_feat_size_0=512,GCN_feat_size_1=512,GCN_2022-07-19_05-41-04/
/G2V_model M4.G2V_model
/G2V_model.syn1neg.npy M4.G2V_model.syn1neg.npy
/G2V_model.wv.vectors.npy M4.G2V_model.wv.vectors.npy







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



def prepare_data_GCN(MY_chem_A_feat, MY_chem_B_feat, MY_chem_A_adj, MY_chem_B_adj, MY_g2v_A, MY_g2v_B, MY_syn, norm ) :
	chem_feat_A_train, chem_feat_A_tv, chem_feat_B_train, chem_feat_B_tv,chem_adj_A_train, chem_adj_A_tv, chem_adj_B_train, chem_adj_B_tv, g2v_A_train, g2v_A_tv, g2v_B_train, g2v_B_tv, syn_train, syn_tv = sklearn.model_selection.train_test_split(
			MY_chem_A_feat, MY_chem_B_feat, MY_chem_A_adj, MY_chem_B_adj, MY_g2v_A, MY_g2v_B, MY_syn, test_size= 0.2 , random_state=42 )
	chem_feat_A_val, chem_feat_A_test, chem_feat_B_val, chem_feat_B_test, chem_adj_A_val, chem_adj_A_test, chem_adj_B_val, chem_adj_B_test, g2v_A_val, g2v_A_test, g2v_B_val, g2v_B_test, syn_val, syn_test  = sklearn.model_selection.train_test_split(
			chem_feat_A_tv, chem_feat_B_tv, chem_adj_A_tv, chem_adj_B_tv, g2v_A_tv, g2v_B_tv, syn_tv, test_size=0.5, random_state=42 )
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
	train_data['G2V_A'] = g2v_A_train + g2v_B_train 
	val_data['G2V_A'] = g2v_A_val
	test_data['G2V_A'] = g2v_A_test
	#
	train_data['G2V_B'] = g2v_B_train + g2v_A_train
	val_data['G2V_B'] = g2v_B_val
	test_data['G2V_B'] = g2v_B_test
	#               
	train_data['y'] = np.concatenate((syn_train, syn_train), axis=0)
	val_data['y'] = syn_val
	test_data['y'] = syn_test
	#
	print(train_data['drug1_feat'].shape)
	print(val_data['drug1_feat'].shape)
	print(test_data['drug1_feat'].shape)
	return train_data, val_data, test_data




class DATASET_GCN_G2V(Dataset):
	def __init__(self, gcn_drug1_F, gcn_drug2_F, gcn_drug1_ADJ, gcn_drug2_ADJ, g2v_A, g2v_B, syn_ans):
		self.gcn_drug1_F = gcn_drug1_F
		self.gcn_drug2_F = gcn_drug2_F
		self.gcn_drug1_ADJ = gcn_drug1_ADJ
		self.gcn_drug2_ADJ = gcn_drug2_ADJ
		self.g2v_A = g2v_A
		self.g2v_B = g2v_B
		self.syn_ans = syn_ans
		#
	#
	def __len__(self):
		return len(self.gcn_drug1_F)
	#
	def __getitem__(self, index):
		adj_re_A = self.gcn_drug1_ADJ[index].long().to_sparse().indices()
		adj_re_B = self.gcn_drug2_ADJ[index].long().to_sparse().indices()
		return self.gcn_drug1_F[index], self.gcn_drug2_F[index], adj_re_A, adj_re_B, self.g2v_A[index], self.g2v_B[index], self.syn_ans[index]



def graph_collate_fn(batch):
	drug1_f_list = []
	drug2_f_list = []
	drug1_adj_list = []
	drug2_adj_list = []
	g2v_A_list = []
	g2v_B_list = []
	y_list = []
	DRUG_1_num_nodes_seen = 0
	DRUG_2_num_nodes_seen = 0
	for drug1_f, drug2_f, drug1_adj, drug2_adj, g2v_A, g2v_B, y in batch :
		drug1_f_list.append(drug1_f)
		drug2_f_list.append(drug2_f)
		drug1_adj_list.append(drug1_adj+DRUG_1_num_nodes_seen)
		drug2_adj_list.append(drug2_adj+DRUG_2_num_nodes_seen)
		g2v_A_list.append(g2v_A)
		g2v_B_list.append(g2v_B)
		y_list.append(y)
		DRUG_1_num_nodes_seen += drug1_f.shape[0]
		DRUG_2_num_nodes_seen += drug2_f.shape[0]
	drug1_f_new = torch.cat(drug1_f_list, 0)
	drug2_f_new = torch.cat(drug2_f_list, 0)
	drug1_adj_new = torch.cat(drug1_adj_list, 1)
	drug2_adj_new = torch.cat(drug2_adj_list, 1)
	y_new = torch.stack(y_list, 0)
	return drug1_f_new, drug2_f_new, drug1_adj_new, drug2_adj_new, g2v_A_list, g2v_B_list, y_new



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






class MY_GCN_G2V_MODEL(torch.nn.Module):
	def __init__(self, G_layer_chem, G_indim_chem, G_hiddim_chem, g2v_indim, layers_1, layers_2, layers_3, out_dim, inDrop, drop):
		super(MY_GCN_G2V_MODEL, self).__init__()
		self.G_layer_chem = G_layer_chem
		self.G_indim_chem = G_indim_chem
		self.G_hiddim_chem = G_hiddim_chem
		self.g2v_indim = g2v_indim
		self.layers_1 = [int(a) for a in layers_1]
		self.layers_2 = [int(a) for a in layers_2]
		self.layers_3 = [int(a) for a in layers_3]
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
		self.Convs_1 = torch.nn.ModuleList([torch.nn.Linear(self.G_hiddim_chem+self.g2v_indim, self.layers_1[0] )])
		self.Convs_1.extend([torch.nn.Linear(self.layers_1[a], self.layers_1[a+1]) for a in range(len(self.layers_1)-1)])
		#
		self.Convs_2 = torch.nn.ModuleList([torch.nn.Linear(self.G_hiddim_chem+self.g2v_indim, self.layers_2[0] )])
		self.Convs_2.extend([torch.nn.Linear(self.layers_2[a], self.layers_2[a+1]) for a in range(len(self.layers_2)-1)])
		#
		self.SNPs = torch.nn.ModuleList([torch.nn.Linear(self.layers_1[-1]+self.layers_2[-1], self.layers_3[0] )])
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
	def forward(self, Drug1_F, Drug2_F, Drug1_ADJ, Drug2_ADJ, G2V_A, G2V_B, syn ):
		Drug_batch_label = self.calc_batch_label(syn, Drug1_F)
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
		input_drug1 = torch.concat( (G_1_C_out, G2V_A), 1 ) # normalization 추가 해야할것 같은데 
		input_drug2 = torch.concat( (G_2_C_out, G2V_B), 1 )
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
		X = torch.cat((input_drug1,input_drug2),1)
		for L3 in range(len(self.SNPs)):
			if L3 != len(self.SNPs)-1 :
				X = self.SNPs[L3](X)
				X = F.dropout(X, p=self.drop, training = self.training)
				X = F.relu(X)
			else :
				X = self.SNPs[L3](X)
		return X





from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile

WORK_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_GPU/PJ01.TRIAL.3_2/'

MY_chem_A_feat = torch.load( WORK_PATH+'0713.MY_chem_A_feat.pt')
MY_chem_B_feat = torch.load( WORK_PATH+'0713.MY_chem_B_feat.pt')
MY_chem_A_adj = torch.load( WORK_PATH+'0713.MY_chem_A_adj.pt')
MY_chem_B_adj = torch.load( WORK_PATH+'0713.MY_chem_B_adj.pt')
MY_syn = torch.load(WORK_PATH+'0713.MY_syn.pt')
with open(WORK_PATH+'0713.MY_chem_A_g2v.json', 'r') as f:
	MY_g2v_A = json.load(f)

with open(WORK_PATH+'0713.MY_chem_B_g2v.json', 'r') as f:
	MY_g2v_B = json.load(f)


seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)



norm = 'tanh_norm'
train_data, val_data, test_data = prepare_data_GCN(MY_chem_A_feat, MY_chem_B_feat, MY_chem_A_adj, MY_chem_B_adj, MY_g2v_A, MY_g2v_B, MY_syn, norm)
wl_iterations = 2



# WEIGHT 
ys = train_data['y'].squeeze().tolist()
min_s = np.amin(ys)
loss_weight = np.log(train_data['y'] - min_s + np.e)
JY_IDX_WEIGHT_T = torch.Tensor(JY_IDX_WEIGHT).view(1,-1)


# DATA check 
T_train = DATASET_GCN_G2V(
	torch.Tensor(train_data['drug1_feat']), torch.Tensor(train_data['drug2_feat']), 
	torch.Tensor(train_data['drug1_adj']), torch.Tensor(train_data['drug2_adj']),
	train_data['G2V_A'], train_data['G2V_B'], 
	torch.Tensor(train_data['y']))

T_val = DATASET_GCN_G2V(
	torch.Tensor(val_data['drug1_feat']), torch.Tensor(val_data['drug2_feat']), 
	torch.Tensor(val_data['drug1_adj']), torch.Tensor(val_data['drug2_adj']),
	val_data['G2V_A'], val_data['G2V_B'], 
	torch.Tensor(val_data['y']))
	
T_test = DATASET_GCN_G2V(
	torch.Tensor(test_data['drug1_feat']), torch.Tensor(test_data['drug2_feat']), 
	torch.Tensor(test_data['drug1_adj']), torch.Tensor(test_data['drug2_adj']),
	test_data['G2V_A'], test_data['G2V_B'], 
	torch.Tensor(test_data['y']))

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



PRJ_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_GPU/PJ01.TRIAL.3_2/'
ANA_DF = pd.read_csv(PRJ_PATH+'RAY_ANA_DF.P01.3_2.csv')
with open(PRJ_PATH+'RAY_ANA_DF.P01.3_2.pickle', 'rb') as f:
	ANA_ALL_DF = pickle.load(f)


DF_KEY = list(ANA_DF.sort_values('ValLoss')['logdir'])[0]
DF_KEY
list(ANA_DF.sort_values('ValLoss')['ValLoss'])[0]


mini_df = ANA_ALL_DF[DF_KEY]
train_loss_1 = list(mini_df.TrainLoss)
train_loss_2 = [float(a.split('(')[1].split(',')[0]) for a in train_loss_1]
plot_loss(train_loss_2, list(mini_df.ValLoss), 
PRJ_PATH, 'TRIAL.3_2.BEST.loss' )


(1) 마지막 모델 확인 

TOPVAL_PATH = DF_KEY
my_config = ANA_DF[ANA_DF.logdir==DF_KEY]

G_layer = my_config['config/GCN_layer'].item()
G_hiddim = my_config['config/GCN_hiddim'].item()
dsn1_layers = [my_config['config/GCN_feat_size_0'].item(), my_config['config/GCN_feat_size_1'].item(), my_config['config/GCN_feat_size_2'].item()]
dsn2_layers = [my_config['config/GCN_feat_size_0'].item(), my_config['config/GCN_feat_size_1'].item(), my_config['config/GCN_feat_size_2'].item()] 
snp_layers = [my_config['config/GCN_feat_size_3'].item() , my_config['config/GCN_feat_size_4'].item()]
inDrop = my_config['config/GCN_dropout_1'].item()
Drop = my_config['config/GCN_dropout_2'].item()
batch_size = my_config['config/batch_size'].item()
g2v_vector_size =  my_config['config/g2v_vector_size'].item()
g2v_alpha =  my_config['config/g2v_alpha'].item()
g2v_epochs =  my_config['config/g2v_epochs'].item()
g2v_min_count =  my_config['config/g2v_min_count'].item()
g2v_sample =  my_config['config/g2v_sample'].item()
g2v_window =  my_config['config/g2v_window'].item()

g2v_model_path = '/st06/jiyeonH/11.TOX/MY_TRIAL_GPU/PJ01.TRIAL.3_2/M1.G2V_model'
g2v_model = Doc2Vec.load(g2v_model_path)



def infer_from_model(model, data, dim):
	vec_A = []
	vec_B = []
	for line in data.g2v_A :
		tmp_a = model.infer_vector(line[0])
		vec_A.append(tmp_a)
	for line in data.g2v_B :
		tmp_b = model.infer_vector(line[0])
		vec_B.append(tmp_b)
	return torch.Tensor(vec_A).view(-1,dim), torch.Tensor(vec_B).view(-1,dim)


Test_loader = torch.utils.data.DataLoader(T_test, batch_size = my_config['config/batch_size'].item(), collate_fn = graph_collate_fn, shuffle =False, num_workers=8)
Test_G2V_A, Test_G2V_B = infer_from_model(g2v_model, T_test , my_config['config/g2v_vector_size'].item())
#
#


best_trained_model = MY_GCN_G2V_MODEL(
	my_config['config/GCN_layer'].item(), T_test.gcn_drug1_F.shape[-1], my_config['config/GCN_hiddim'].item(), 
	my_config['config/g2v_vector_size'].item(), dsn1_layers, dsn2_layers, snp_layers, 1,
	inDrop, Drop)


state_dict = torch.load(os.path.join(PRJ_PATH, "M1.GCN_model.pth"), map_location=torch.device('cpu'))
best_trained_model.load_state_dict(state_dict)

Test_loader = torch.utils.data.DataLoader(T_test, collate_fn = graph_collate_fn, batch_size = my_config['config/batch_size'].item(), shuffle =False)

#

best_trained_model.eval()
test_loss = 0.0
PRED_list = []
Y_list = test_data['y'].squeeze().tolist()
with torch.no_grad():
	best_trained_model.eval()
	for batch_idx_t, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB,  y) in enumerate(Test_loader):
		g2v_a = Test_G2V_A[batch_idx_t*my_config['config/batch_size'].item():(batch_idx_t+1)*my_config['config/batch_size'].item()]
		g2v_b = Test_G2V_B[batch_idx_t*my_config['config/batch_size'].item():(batch_idx_t+1)*my_config['config/batch_size'].item()]
		output = best_trained_model(drug1_f, drug2_f, drug1_a, drug2_a, g2v_a, g2v_b, y)
		MSE = torch.nn.MSELoss()
		loss = MSE(output, y)
		test_loss = test_loss + loss.item()
		outputs = output.squeeze().tolist()
		PRED_list = PRED_list+outputs

TEST_LOSS = test_loss/(batch_idx_t+1)
result_pearson(PRED_list, Y_list)
result_spearman(PRED_list, Y_list)
print("Best model TEST loss: {}".format(TEST_LOSS))

R_1_V = list(ANA_DF.sort_values('ValLoss')['ValLoss'])[0]
R_1_T = TEST_LOSS
R_1_1 , R_1_2 = jy_corrplot(PRED_list, Y_list, PRJ_PATH,'P3_2.M1_model' )




(2) 중간 체크포인트 확인 

cck_num =mini_df[mini_df.ValLoss==min(mini_df.ValLoss)].index.item()
checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
TOPVAL_PATH = DF_KEY + checkpoint
TOPVAL_PATH
# get 
# /home01/k006a01/ray_results/22.07.06.PRJ01.TRIAL3_1/RAY_MY_train_2292abdc_88_G_hiddim=512,G_layer=2,batch_size=128,dropout_1=0.2000,dropout_2=0.0100,epoch=1000,feat_size_0=64,feat_si_2022-07-11_19-34-58/checkpoint_000473
# G_2.M2_checkpoint
min(mini_df.ValLoss)


state_dict = torch.load(os.path.join(PRJ_PATH, "M2_checkpoint"),map_location=torch.device('cpu'))
best_trained_model.load_state_dict(state_dict[0])

best_trained_model.eval()
test_loss = 0.0
PRED_list = []
Y_list = test_data['y'].squeeze().tolist()
with torch.no_grad():
	best_trained_model.eval()
	for batch_idx_t, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB,  y) in enumerate(Test_loader):
		g2v_a = Test_G2V_A[batch_idx_t*my_config['config/batch_size'].item():(batch_idx_t+1)*my_config['config/batch_size'].item()]
		g2v_b = Test_G2V_B[batch_idx_t*my_config['config/batch_size'].item():(batch_idx_t+1)*my_config['config/batch_size'].item()]
		output = best_trained_model(drug1_f, drug2_f, drug1_a, drug2_a, g2v_a, g2v_b, y)
		MSE = torch.nn.MSELoss()
		loss = MSE(output, y)
		test_loss = test_loss + loss.item()
		outputs = output.squeeze().tolist()
		PRED_list = PRED_list+outputs


TEST_LOSS = test_loss/(batch_idx_t+1)
print("Best model TEST loss: {}".format(TEST_LOSS))
R_2_V = min(mini_df.ValLoss)
R_2_T = TEST_LOSS
R_2_1 , R_2_2 = jy_corrplot(PRED_list, Y_list,PRJ_PATH,'P3_2.M2_checkpoint' )










# 최저를 찾으려면 
# 최저를 찾으려면 
# 최저를 찾으려면 

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
# /home01/k006a01/ray_results/22.07.06.PRJ01.TRIAL3_1/RAY_MY_train_f71154ee_41_G_hiddim=512,G_layer=2,batch_size=128,dropout_1=0.2000,dropout_2=0.0100,epoch=1000,feat_size_0=64,feat_si_2022-07-08_06-25-18


mini_df = ANA_ALL_DF[TOT_key]
train_loss_1 = list(mini_df.TrainLoss)
train_loss_2 = [float(a.split('(')[1].split(',')[0]) for a in train_loss_1]
plot_loss(train_loss_2, list(mini_df.ValLoss), 
PRJ_PATH, 'TRIAL.3_2.MIN.loss' )


TOPVAL_PATH = TOT_key
my_config = ANA_DF[ANA_DF.logdir==TOT_key]

g2v_model_path = '/st06/jiyeonH/11.TOX/MY_TRIAL_GPU/PJ01.TRIAL.3_2/M4.G2V_model'
g2v_model = Doc2Vec.load(g2v_model_path)


Test_loader = torch.utils.data.DataLoader(T_test, batch_size = my_config['config/batch_size'].item(), collate_fn = graph_collate_fn, shuffle =False, num_workers=8)
Test_G2V_A, Test_G2V_B = infer_from_model(g2v_model, T_test , my_config['config/g2v_vector_size'].item())
#
#

best_trained_model = MY_GCN_G2V_MODEL(
	my_config['config/GCN_layer'].item(), T_test.gcn_drug1_F.shape[-1], my_config['config/GCN_hiddim'].item(), 
	my_config['config/g2v_vector_size'].item(), dsn1_layers, dsn2_layers, snp_layers, 1,
	inDrop, Drop)


state_dict = torch.load(os.path.join(PRJ_PATH, "M4.checkpoint"), map_location=torch.device('cpu'))
best_trained_model.load_state_dict(state_dict[0])

Test_loader = torch.utils.data.DataLoader(T_test, collate_fn = graph_collate_fn, batch_size = my_config['config/batch_size'].item(), shuffle =False)

#
best_trained_model.eval()
test_loss = 0.0
PRED_list = []
Y_list = test_data['y'].squeeze().tolist()
with torch.no_grad():
	best_trained_model.eval()
	for batch_idx_t, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB,  y) in enumerate(Test_loader):
		g2v_a = Test_G2V_A[batch_idx_t*my_config['config/batch_size'].item():(batch_idx_t+1)*my_config['config/batch_size'].item()]
		g2v_b = Test_G2V_B[batch_idx_t*my_config['config/batch_size'].item():(batch_idx_t+1)*my_config['config/batch_size'].item()]
		output = best_trained_model(drug1_f, drug2_f, drug1_a, drug2_a, g2v_a, g2v_b, y)
		MSE = torch.nn.MSELoss()
		loss = MSE(output, y)
		test_loss = test_loss + loss.item()
		outputs = output.squeeze().tolist()
		PRED_list = PRED_list+outputs

TEST_LOSS = test_loss/(batch_idx_t+1)
print("Best model TEST loss: {}".format(TEST_LOSS))
R_3_V = min(mini_df.ValLoss)
R_3_T = TEST_LOSS
R_3_1 , R_3_2 = jy_corrplot(PRED_list, Y_list,PRJ_PATH,'P3_2.M4_checkpoint' )









###############################################################################
###############################################################################
###############################################################################
###############################################################################

														iterations = iterations
														graph = graph
														features = features
														nodes = graph.nodes()
														extracted_features = [str(v) for k, v in features.items()]
														node_dict = {list(nodes)[a]:BETA_ORDER[a] for a in range(978)}
															#
														new_features = {}

														for node in nodes:
															nebs = graph.neighbors(node)
															degs = [features[node_dict[neb]] for neb in nebs]
															tmp_features = [str(features[node_dict[node]])]+sorted([str(deg) for deg in degs])
															tmp_features = "_".join(tmp_features)
															hash_object = hashlib.md5(tmp_features.encode()) # hash 값으로 그래프를 벡터화 
															hashing = hash_object.hexdigest()
															new_features[node] = hashing
														extracted_features = extracted_features + list(new_features.values())
															#
															def do_recursions(self):
																for _ in range(self.iterations):
																	self.features = self.do_a_recursion()



										def do_a_recursion(extracted_features, features):
											if extracted_features==False:
												extracted_features = [str(v) for k, v in features.items()]
											new_features = {}
											for node in nodes:
												nebs = graph.neighbors(node)
												degs = [features[node_dict[neb]] for neb in nebs]
												wl_features = [str(features[node_dict[node]])]+sorted([str(deg) for deg in degs])
												wl_features = "_".join(wl_features)
												hash_object = hashlib.md5(wl_features.encode()) # hash 값으로 그래프를 벡터화 
												hashing = hash_object.hexdigest()
												new_features[node_dict[node]] = hashing
											extracted_features = extracted_features + list(new_features.values())
											return extracted_features, new_features

										extracted_features, iter_1_result = do_a_recursion(False, tmp_feature_dict)
										extracted_features, iter_2_result = do_a_recursion(extracted_features, iter_1_result)








										wl_iterations = 2
										# rounds = wl_iterations
										# features= tmp_feature_dict


v