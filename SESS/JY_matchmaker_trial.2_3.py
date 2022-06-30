
# 화학구조 추가버전 

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

WORK_PATH='/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/my_trial_2_4/' 
# /home01/k006a01/PRJ.01/TRIAL_2.4
Tch_PATH='/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/my_trial_2_4/' 
# '/home01/k006a01/PRJ.01/TRIAL_2.4'
DC_PATH='/st06/jiyeonH/13.DD_SESS/DrugComb.1.5/' 
# '/home01/k006a01/01.DATA/DrugComb/'
IDK_PATH='/st06/jiyeonH/13.DD_SESS/ideker/'
# '/home01/k006a01/01.DATA/IDK/'
LINCS_PATH='/st06/jiyeonH/11.TOX/MY_TRIAL_5/'
# '/home01/k006a01/01.DATA/LINCS/'
TARGET_PATH= '/st06/jiyeonH/13.DD_SESS/merged_target/' 
# 

print("filtering")
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

DC_DATA7_4_ccle = DC_DATA7_3[ccle_t] # 730348
DC_DATA7_4_ccle = DC_DATA7_4_ccle[DC_DATA7_4_ccle.DrugCombCCLE != 'NA'] # 540037
DC_ccle_final = DC_DATA7_4_ccle[['drug_row_cid','drug_col_cid','DrugCombCCLE']].drop_duplicates() # 464137
DC_ccle_final_dup = DC_DATA7_4_ccle[['drug_row_cid','drug_col_cid','DrugCombCCLE', 'synergy_loewe']].drop_duplicates() # 540037


# LINCS DATA
print("LINCS")

BETA_BIND = pd.read_csv(LINCS_PATH+"BETA_DATA_for_SS_df.978.csv")
BETA_SELEC_SIG = pd.read_csv(LINCS_PATH+'SIG_INFO.220405') # cell 58가지, 129116, cid  25589
BETA_CP_info = pd.read_table(LINCS_PATH+'compoundinfo_beta.txt')
BETA_CEL_info = pd.read_table(LINCS_PATH+'cellinfo_beta.txt')
BETA_SIG_info = pd.read_table(LINCS_PATH+'siginfo_beta.txt', low_memory = False)

BETA_GENE = pd.read_table(LINCS_PATH+'geneinfo_beta.txt')
BETA_lm_genes = BETA_GENE[BETA_GENE.feature_space=='landmark'] # 978
BETA_lm_genes = BETA_lm_genes.reset_index()

BETA_CEL_info2 = BETA_CEL_info[['cell_iname','cellosaurus_id','ccle_name']] # 240 
BETA_SELEC_SIG_wCell = pd.merge(BETA_SELEC_SIG, BETA_CEL_info2, on = 'cell_iname', how = 'left') # 129116
BETA_SELEC_SIG_wCell2 = BETA_SELEC_SIG_wCell[['pubchem_cid','cellosaurus_id','ccle_name','sig_id']]
BETA_SELEC_SIG_wCell2 = BETA_SELEC_SIG_wCell2.drop_duplicates() # 129116

cello_tt=[True if type(a)==str else False for a in list(BETA_SELEC_SIG_wCell2.cellosaurus_id)]
BETA_CID_CELLO_SIG = BETA_SELEC_SIG_wCell2[cello_tt][['pubchem_cid','cellosaurus_id','sig_id']].drop_duplicates() # 111916
beta_cid_cello_sig_tf = [ True if a>0 else False for a in list(BETA_CID_CELLO_SIG.pubchem_cid)]
BETA_CID_CELLO_SIG = BETA_CID_CELLO_SIG[beta_cid_cello_sig_tf]

ccle_tt=[True if type(a)==str else False for a in list(BETA_SELEC_SIG_wCell2.ccle_name)]
BETA_CID_CCLE_SIG = BETA_SELEC_SIG_wCell2[ccle_tt][['pubchem_cid','ccle_name','sig_id']].drop_duplicates() # 110620
BETA_CID_CCLE_SIG = BETA_CID_CCLE_SIG[BETA_CID_CCLE_SIG.ccle_name!='NA']
beta_cid_ccle_sig_tf = [ True if a>0 else False for a in list(BETA_CID_CCLE_SIG.pubchem_cid)]
BETA_CID_CCLE_SIG = BETA_CID_CCLE_SIG[beta_cid_ccle_sig_tf]



# cello
BETA_CID_CELLO_SIG.columns=['drug_row_cid', 'DrugCombCello', 'BETA_sig_id']
CELLO_DC_BETA_1 = pd.merge(DC_cello_final_dup, BETA_CID_CELLO_SIG, on = ['drug_row_cid','DrugCombCello'], how = 'left') # 731051

BETA_CID_CELLO_SIG.columns=['drug_col_cid', 'DrugCombCello', 'BETA_sig_id']
CELLO_DC_BETA_2 = pd.merge(CELLO_DC_BETA_1, BETA_CID_CELLO_SIG, on = ['drug_col_cid','DrugCombCello'], how = 'left') # 731644

BETA_CID_CELLO_SIG.columns=['pubchem_cid', 'cellosaurus_id', 'sig_id']


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




# TARGET # 민지가 다시 올려준다고 함 

TARGET_LIST = pd.read_csv(TARGET_PATH+'combined_target.csv', low_memory=False)

TARGET_FILTER = TARGET_LIST[TARGET_LIST.cid.isin(CELLO_DC_BETA_cids)] # 176개 중에서 2개가 음슴 
TARGET_FILTER[TARGET_FILTER.target.isin(L_matching_list.L_gene_symbol)] # 31233
TARGET_FILTER[TARGET_FILTER.target.isin(L_matching_list.PPI_name)] # 31183

TARGET_FILTER_re = TARGET_FILTER[TARGET_FILTER.target.isin(L_matching_list.L_gene_symbol)]

# 원래 cid 에서는 2개만 안맞았는데,
# 민지가 정리한 CID 1,108,881 
MJ_CIDS = list(set(TARGET_LIST.cid))
[a for a in CELLO_DC_BETA_cids if a not in MJ_CIDS] 
그래서 내 CID 중에서 안맞는 CID 2개 ->  56655374, 2942

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
A_B_C_S_row = A_B_C_S[A_B_C_S.drug_row_cid.isin(list(TARGET_FILTER_re.cid))]
A_B_C_S_col = A_B_C_S_row[A_B_C_S_row.drug_col_cid.isin(list(TARGET_FILTER_re.cid))]
#
A_B_C_S_SET = A_B_C_S_col[['BETA_sig_id_x','BETA_sig_id_y','DrugCombCello']].drop_duplicates()
A_B_C_S_SET = A_B_C_S_SET.reset_index()


# LINCS 확인 
BETA_ORDER_pre =[list(L_matching_list.PPI_name).index(a) for a in JY_GRAPH_ORDER]
BETA_ORDER_DF = L_matching_list.iloc[BETA_ORDER_pre] # 어차피 ref 다르고 같은 애들이라 괜춘 
BETA_ORDER = list(BETA_ORDER_DF.entrez)


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


def get_targets(sig_id): 
	tmp_df1 = BETA_CID_CELLO_SIG[BETA_CID_CELLO_SIG.sig_id == sig_id]
	CID = tmp_df1.pubchem_cid.item()
	tmp_df2 = TARGET_FILTER_re[TARGET_FILTER_re.cid == CID]
	targets = list(set(tmp_df2.target))
	gene_symbols = list(BETA_ORDER_DF.L_gene_symbol)
	vec = [1 if a in targets else 0 for a in gene_symbols ]
	return vec










def convertToGraph_adj(SMILES, k):
    adj = []
    adj_norm = []
    features = []
    maxNumAtoms = 50
    # Mol
    iMol = Chem.MolFromSmiles(i.strip())
    #Adj
    iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(iMol)
    # Feature
    if( iAdjTmp.shape[0] <= maxNumAtoms):
        # Feature-preprocessing
        #iFeature = np.zeros((maxNumAtoms, 64))
        iFeatureTmp = []
        for atom in iMol.GetAtoms():
            iFeatureTmp.append( atom_feature(atom) )### atom features only
        #
        #iFeature[0:len(iFeatureTmp), 0:64] = iFeatureTmp ### 0 padding for feature-set
        #features.append(iFeature)
        # Adj-preprocessing
        iAdj = np.zeros((maxNumAtoms, maxNumAtoms))
        iAdj[0:len(iFeatureTmp), 0:len(iFeatureTmp)] = iAdjTmp + np.eye(len(iFeatureTmp))
        adj.append(adj_k(np.asarray(iAdj), k))
    return adj


def convertToGraph_fea(smiles_list, k):
    adj = []
    adj_norm = []
    features = []
    maxNumAtoms = 50
    for i in smiles_list:
        # Mol
        iMol = Chem.MolFromSmiles(i.strip())
        #Adj
        iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(iMol)
        # Feature
        if( iAdjTmp.shape[0] <= maxNumAtoms):
            # Feature-preprocessing
            iFeature = np.zeros((maxNumAtoms, 64))
            iFeatureTmp = []
            for atom in iMol.GetAtoms():
                iFeatureTmp.append( atom_feature(atom) )### atom features only
            #
            iFeature[0:len(iFeatureTmp), 0:64] = iFeatureTmp ### 0 padding for feature-set
            features.append(iFeature)
    features = np.asarray(features)
    return features



def convertToGraph(smiles_list, k):
    adj = []
    adj_norm = []
    features = []
    maxNumAtoms = 50
    for i in smiles_list:
        # Mol
        iMol = Chem.MolFromSmiles(i.strip())
        #Adj
        iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(iMol)
        # Feature
        if( iAdjTmp.shape[0] <= maxNumAtoms):
            # Feature-preprocessing
            iFeature = np.zeros((maxNumAtoms, 64))
            iFeatureTmp = []
            for atom in iMol.GetAtoms():
                iFeatureTmp.append( atom_feature(atom) )### atom features only
            #
            iFeature[0:len(iFeatureTmp), 0:64] = iFeatureTmp ### 0 padding for feature-set
            features.append(iFeature)
            # Adj-preprocessing
            iAdj = np.zeros((maxNumAtoms, maxNumAtoms))
            iAdj[0:len(iFeatureTmp), 0:len(iFeatureTmp)] = iAdjTmp + np.eye(len(iFeatureTmp))
            adj.append(adj_k(np.asarray(iAdj), k))
    features = np.asarray(features)
    return adj, features



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
                    [atom.GetIsAromatic()])    # (36, 8, 5, 5, 9, 1)
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



def adj_k(adj, k):
    ret = adj
    for i in range(0, k-1):
        ret = np.dot(ret, adj)  
    return convertAdj(ret)


import vectorize ##위에 파일 이름..        
train_list = train_all[ver]['CAN_SMILES'].tolist()
print ("train",len(train_list))
train_adj, train_features = convertToGraph(train_list, 1)
print (np.asarray(train_features).shape)
print (np.asarray(train_adj).shape)
np.save('../adj/'+"train_"+name+str(ver)+'.npy', train_adj)
np.save('../features/'+"train_"+name+str(ver)+'.npy', train_features)
train_prop = np.c_[np.array(train_all[ver]['XLOGP3']).reshape(-1,1),
np.array(train_all[ver]['TPSA']).reshape(-1,1),
np.array(train_all[ver]['MOLECULAR_WEIGHT']).reshape(-1,1),
np.array(train_all[ver]['HBOND_ACCEPTOR']).reshape(-1,1),
np.array(train_all[ver]['HBOND_DONOR']).reshape(-1,1),
np.array(train_all[ver]['ROTATABLE_BOND']).reshape(-1,1)] 


###이건 화합물 atom수로 자르는 함수 입니다
def under_number(data, num):
    print("starting to set num")
    #50개 미만으로 
    tf = list()
    z=0
    for i in data['CAN_SMILES']:
        maxNumAtoms = num
        # Mol
        iMol = Chem.MolFromSmiles(i.strip())
        #Adj
        try:
            iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(iMol)
            if( iAdjTmp.shape[0] < maxNumAtoms):
                tf.append("T")
                #print("T")
            else:
                tf.append("F")
        except:
            tf.append("error")
            print("error",z,i)
        z = z+1
    #print("tf is made",tf)
    data["tf"] = tf
    data2 = data[data.tf=="T"].reset_index(drop=True)
    return data2