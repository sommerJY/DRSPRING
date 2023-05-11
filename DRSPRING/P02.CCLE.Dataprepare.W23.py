
# only Basal & Target 



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




# Drug Comb 데이터 가져오기 
# synergy info 
DC_tmp_DF1 = pd.read_csv(DC_PATH+'TOTAL_BLOCK.csv', low_memory=False) # 1469182
DC_tmp_DF1_re = DC_tmp_DF1[['drug_row_id','drug_col_id','cell_line_id','synergy_loewe','quality']]
DC_tmp_DF1_re['drug_row_id_re'] = [float(a) for a in list(DC_tmp_DF1_re['drug_row_id'])]
DC_tmp_DF1_re['drug_col_id_re'] = [float(a) for a in list(DC_tmp_DF1_re['drug_col_id'])]

# drug len check : drug_ids = list(set(list(DC_tmp_DF1.drug_row_id_re) + list(DC_tmp_DF1.drug_col_id_re))) # 15811
# cell len check : cell_ids = list(set(DC_tmp_DF1.cell_line_id)) # 2040

DC_tmp_DF2 = DC_tmp_DF1_re[DC_tmp_DF1_re['quality'] != 'bad'] # 1457561
# drug len check : drug_ids = list(set(list(DC_tmp_DF2.drug_row_id_re) + list(DC_tmp_DF2.drug_row_id_re))) # 8333
# cell len check : cell_ids = list(set(DC_tmp_DF2.cell_line_id)) # 2040

DC_tmp_DF_ch = DC_tmp_DF2[DC_tmp_DF2.cell_line_id>0]
DC_tmp_DF_ch = DC_tmp_DF_ch.reset_index(drop=True)

DC_tmp_DF_ch_plus = DC_tmp_DF_ch[DC_tmp_DF_ch.synergy_loewe > 0 ]
DC_tmp_DF_ch_minus = DC_tmp_DF_ch[DC_tmp_DF_ch.synergy_loewe < 0 ]
DC_tmp_DF_ch_zero = DC_tmp_DF_ch[DC_tmp_DF_ch.synergy_loewe == 0 ]

DC_score_filter = pd.concat([DC_tmp_DF_ch_plus, DC_tmp_DF_ch_minus, DC_tmp_DF_ch_zero ])

DC_score_filter[['drug_row_id_re','drug_col_id_re','cell_line_id','synergy_loewe']].drop_duplicates()
# 1356208


DC_score_filter[['drug_row_id_re','drug_col_id_re','cell_line_id']].drop_duplicates()
# 1263840





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




# cid renaming
DC_DRUG_DF2 = DC_DRUG_DF_FULL[['id_re','dname','cid','CAN_SMILES']] # puibchem 공식 smiles 
																	# dnames = list(DC_DRUG_DF2['dname']) # 그래 이부분때문에 다시 다운로드 받은거였음. 이게 문제가 drug 마다 CID 가 여러개 붙는 애들이 있어서
DC_DRUG_DF2.columns = ['drug_row_id_re','drug_row','drug_row_CID', 'drug_row_sm']
DC_DATA5 = pd.merge(DC_DATA_filter4, DC_DRUG_DF2, on ='drug_row_id_re', how='left' ) # 751450 -> 740884

DC_DRUG_DF2.columns = ['drug_col_id_re','drug_col','drug_col_CID', 'drug_col_sm']
DC_DATA6 = pd.merge(DC_DATA5, DC_DRUG_DF2, on ='drug_col_id_re', how='left') # 751450 -> 740884


#  Add cell data and cid filter
DC_DATA7_1 = pd.merge(DC_DATA6, DC_CELL_DF2 , on= 'cell_line_id', how = 'left') # 751450 -> 740884
DC_DATA7_1 = DC_DATA7_1[['drug_row_CID','drug_col_CID','DrugCombCCLE','synergy_loewe']].drop_duplicates() # 740882


# filtering 
DC_DATA7_2 = DC_DATA7_1[DC_DATA7_1.drug_row_CID>0] # 740882 -> 737104
DC_DATA7_3 = DC_DATA7_2[DC_DATA7_2.drug_col_CID>0] # 737104 -> 725496
ccle_t=[True if type(a)==str else False for a in list(DC_DATA7_3.DrugCombCCLE)]
DC_DATA7_4_ccle = DC_DATA7_3[ccle_t] #  720249
TF_check = [True if np.isnan(a)==False else False for a in DC_DATA7_4_ccle.synergy_loewe] 
DC_DATA7_5_ccle = DC_DATA7_4_ccle[TF_check] # 719946
DC_DATA7_6_ccle = DC_DATA7_5_ccle[DC_DATA7_5_ccle.DrugCombCCLE != 'NA']


DC_ccle_final = DC_DATA7_6_ccle[['drug_row_CID','drug_col_CID','DrugCombCCLE']].drop_duplicates() # 455528
DC_ccle_final_dup = DC_DATA7_6_ccle[['drug_row_CID','drug_col_CID','DrugCombCCLE', 'synergy_loewe']].drop_duplicates() # 

DC_ccle_final_cids = list(set(list(DC_ccle_final_dup.drug_row_CID) + list(DC_ccle_final_dup.drug_col_CID)))
# 3146




# 공식 smiles 

for_CAN_smiles = pd.read_csv(PC_PATH+'CID_SMILES.csv', low_memory = False)
# for_CAN_smiles = copy.deepcopy(PC_FILTER)
for_CAN_smiles = for_CAN_smiles[['CID','CAN_SMILES']]
for_CAN_smiles.columns = ['drug_row_CID','ROW_CAN_SMILES']
DC_ccle_final_dup = pd.merge(DC_ccle_final_dup, for_CAN_smiles, on='drug_row_CID', how ='left' )
for_CAN_smiles.columns = ['drug_col_CID','COL_CAN_SMILES']
DC_ccle_final_dup = pd.merge(DC_ccle_final_dup, for_CAN_smiles, on='drug_col_CID', how ='left' )
for_CAN_smiles.columns = ['CID','CAN_SMILES']
DC_ccle_final_dup.shape 



# CAN_SMILES NA 있음?  
CAN_TF_1 = [True if type(a) == float else False for a in list(DC_ccle_final_dup.ROW_CAN_SMILES)]
CAN_TF_DF_1 = DC_ccle_final_dup[CAN_TF_1]
CAN_TF_2 = [True if type(a) == float else False for a in list(DC_ccle_final_dup.COL_CAN_SMILES)]
CAN_TF_DF_2 = DC_ccle_final_dup[CAN_TF_2]
# DC 기준으로는 없음. LINCS 기준에서는 있었음 







#################################################################################################
##################################################################################################


# 이게 원래 network 
print('NETWORK')
# HUMANNET 사용 



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







                # HS 다른 pathway 사용 
                print('NETWORK')
                # HUMANNET 사용 


                hunet_dir = '/st06/jiyeonH/13.DD_SESS/HumanNetV3/'

                hunet_gsp = pd.read_csv(hunet_dir+'HS-DB.tsv', sep = '\t', header = None)
                hunet_gsp.columns = ['G_A','G_B','SC']

                BETA_GENE = pd.read_table('/st06/jiyeonH/11.TOX/LINCS/L_2020/geneinfo_beta.txt')
                BETA_lm_genes = BETA_GENE[BETA_GENE.feature_space=='landmark'] # 978
                BETA_lm_genes = BETA_lm_genes.reset_index()
                lm_entrezs = list(BETA_lm_genes.gene_id)

                hnet_L1 = hunet_gsp[hunet_gsp['G_A'].isin(BETA_lm_genes.gene_id)]
                hnet_L2 = hnet_L1[hnet_L1['G_B'].isin(BETA_lm_genes.gene_id)] # 3885
                hnet_L3 = hnet_L2[hnet_L2.SC >= 3.5]


                len(set(list(hnet_L3['G_A']) + list(hnet_L3['G_B']))) # 611

                ID_G = nx.from_pandas_edgelist(hnet_L3, 'G_A', 'G_B')

                # MSSNG = [a for a in lm_entrezs if a not in list(ID_G.nodes)]

                #for nn in list(MSSNG):
                #	ID_G.add_node(nn)

                # edge 
                ID_GENE_ORDER_mini = list(ID_G.nodes()) # 978
                ID_ADJ = nx.adjacency_matrix(ID_G)
                ID_ADJ_tmp = torch.LongTensor(ID_ADJ.toarray())
                ID_ADJ_IDX = ID_ADJ_tmp.to_sparse().indices()  # [2, 7742]
                ID_WEIGHT = [] # len : 3871 -> 7742

                # 원래는 edge score 있지만 일단은...
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






#########################################################################
#########################################################################
# 이번엔 EXP 넣지 않지만 SET 유무 확인을 위해서 미리 파일읽기만 할것 

BETA_CP_info = pd.read_table('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+'compoundinfo_beta.txt') # pert 34419 , cansmiles : 
BETA_CEL_info = pd.read_table('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+'cellinfo_beta.txt')
BETA_SIG_info = pd.read_table('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+'siginfo_beta.txt', low_memory = False)

# pert type 확인 
filter1 = BETA_SIG_info[BETA_SIG_info.pert_type.isin(['ctl_vehicle', 'ctl_untrt' ,'trt_cp' ])]
filter2 = filter1[filter1.is_exemplar_sig==1]

# 한번 더 pubchem converter 로 내가 붙인 애들 
BETA_CP_info_filt = BETA_CP_info[['pert_id','canonical_smiles']].drop_duplicates() # 34419
can_sm_re = pd.read_csv('/st06/jiyeonH/11.TOX/DR_SPRING/tmp/can_sm_conv', sep = '\t', header = None)

can_sm_re.columns = ['canonical_smiles','CONV_CID']
can_sm_re = can_sm_re.drop_duplicates()

can_sm_re2 = pd.merge(BETA_CP_info_filt, can_sm_re, on = 'canonical_smiles', how = 'left') # 34419 -> 1 sm 1 cid 확인 
can_sm_re3 = can_sm_re2[['pert_id','canonical_smiles','CONV_CID']].drop_duplicates() # 

BETA_MJ = pd.read_csv('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+"Lincs_pubchem_mj.csv") # SID 필터까지 그렇게 완성한 무엇 

BETA_MJ_RE = BETA_MJ[['pert_id','SMILES_cid','canonical_smiles']].drop_duplicates() # 25903
BETA_MJ_RE_CK = BETA_MJ_RE[['pert_id','SMILES_cid']]

check = pd.merge(can_sm_re3, BETA_MJ_RE_CK, on = 'pert_id', how = 'left' )
check2 = check[check.CONV_CID !=check.SMILES_cid]
check3 = check2[check2.SMILES_cid > 0 ]
check4 = check3[check3.CONV_CID > 0 ]

pert_id_match = check[check.CONV_CID == check.SMILES_cid][['pert_id','canonical_smiles','CONV_CID']]
conv_win = check2[(check2.CONV_CID >0 ) & ( np.isnan(check2.SMILES_cid)==True)][['pert_id','canonical_smiles','CONV_CID']]
mj_win = check2[(check2.SMILES_cid >0 ) & ( np.isnan(check2.CONV_CID)==True)][['pert_id','canonical_smiles','SMILES_cid']]
nans = check2[(np.isnan(check2.SMILES_cid)==True ) & ( np.isnan(check2.CONV_CID)==True)] # 5995
nans2 = nans[nans.pert_id.isin(filter2.pert_id)]
nans3 = nans2[-nans2.canonical_smiles.isin(['restricted', np.nan])]

pert_id_match.columns = ['pert_id','canonical_smiles','CID'] # 25418,
conv_win.columns = ['pert_id','canonical_smiles','CID'] # 2521,
mj_win.columns =['pert_id','canonical_smiles','CID']

individual_check = check4.reset_index(drop =True)

individual_check_conv = individual_check.loc[[0,4,5,6,10,11,12,13,16,17,18,19]+[a for a in range(21,34)]+[36,40,54]][['pert_id','canonical_smiles','CONV_CID']]
individual_check_mj = individual_check.loc[[1,2,3,7,8,9,14,15,20,34,35,37,38,39]+[a for a in range(41,54)]+[55,56,57]][['pert_id','canonical_smiles','SMILES_cid']]
individual_check_conv.columns = ['pert_id','canonical_smiles','CID'] # 28
individual_check_mj.columns = ['pert_id','canonical_smiles','CID'] # 30 

LINCS_PERT_MATCH = pd.concat([pert_id_match, conv_win, mj_win, individual_check_conv,  individual_check_mj]) # 28424
LINCS_PERT_MATCH_cids = list(set(LINCS_PERT_MATCH.CID))

BETA_EXM = pd.merge(filter2, LINCS_PERT_MATCH, on='pert_id', how = 'left')
BETA_EXM2 = BETA_EXM[BETA_EXM.CID > 0] # 128038 # 이건 늘어났음 

BETA_CEL_info2 = BETA_CEL_info[['cell_iname','cellosaurus_id','ccle_name']] # 240 
BETA_SELEC_SIG_wCell = pd.merge(BETA_EXM2, BETA_CEL_info2, on = 'cell_iname', how = 'left') # 128038
BETA_SELEC_SIG_wCell2 = BETA_SELEC_SIG_wCell[['pert_id','CID','cellosaurus_id','ccle_name','sig_id']]
BETA_SELEC_SIG_wCell2 = BETA_SELEC_SIG_wCell2.drop_duplicates() # 128038

ccle_tt=[True if type(a)==str else False for a in list(BETA_SELEC_SIG_wCell2.ccle_name)] 
BETA_CID_CCLE_SIG = BETA_SELEC_SIG_wCell2[ccle_tt][['pert_id','CID','ccle_name','sig_id']].drop_duplicates() # 111012
beta_cid_ccle_sig_tf = [ True if a>0 else False for a in list(BETA_CID_CCLE_SIG.CID)]
BETA_CID_CCLE_SIG = BETA_CID_CCLE_SIG[beta_cid_ccle_sig_tf]
BETA_CID_CCLE_SIG['CID'] = [int(a) for a in list(BETA_CID_CCLE_SIG['CID']) ] # 111012 








#########################################################################
#########################################################################




# Graph 확인 

JY_GRAPH = MY_G
JY_GRAPH_ORDER = MY_G.nodes()
JY_ADJ = nx.adjacency_matrix(JY_GRAPH)

JY_ADJ_tmp = torch.LongTensor(JY_ADJ.toarray())
JY_ADJ_IDX = JY_ADJ_tmp.to_sparse().indices()
JY_IDX_WEIGHT = MY_WEIGHT_SCORE



A_B_C_S = DC_ccle_final_dup

A_B_C_S_SET = copy.deepcopy(A_B_C_S)
A_B_C_S_SET = A_B_C_S_SET.drop('synergy_loewe', axis = 1).drop_duplicates()
A_B_C_S_SET = A_B_C_S_SET.reset_index(drop = True)



# 50으로 제대로 자르기 위함 
# check length
def check_len_num(SMILES):
	tf = []
	re_try = []
	try :
		NUM = DC_DRUG_DF_FULL[DC_DRUG_DF_FULL.CAN_SMILES==SMILES]['leng'].item()
		tf.append(NUM)
	except : 
		re_try.append(SMILES)
	#
	if len(re_try) > 0: 
		try : 
			iMol = Chem.MolFromSmiles(SMILES.strip())
			NUM = iMol.GetNumAtoms()
			tf.append(NUM)
		except :
			tf.append("error")
			print("error",z,i)
	return tf




tf_list_A = []
tf_list_B = []
for a in range(A_B_C_S_SET.shape[0]):
	sm_1 = A_B_C_S_SET.ROW_CAN_SMILES[a]
	sm_2 = A_B_C_S_SET.COL_CAN_SMILES[a]
	tf_a = check_len_num(sm_1)
	tf_b = check_len_num(sm_2)
	tf_list_A = tf_list_A + tf_a
	tf_list_B = tf_list_B + tf_b


A_B_C_S_SET['ROW_len'] = [int(a) for a in tf_list_A]
A_B_C_S_SET['COL_len'] = [int(a) for a in tf_list_B]

max_len = max(list(A_B_C_S_SET['ROW_len'])+list(A_B_C_S_SET['COL_len']))

A_B_C_S_SET_rlen = A_B_C_S_SET[A_B_C_S_SET.ROW_len<=50]
A_B_C_S_SET_clen = A_B_C_S_SET_rlen[A_B_C_S_SET_rlen.COL_len<=50]

A_B_C_S_SET = A_B_C_S_SET_clen.reset_index(drop=True) # 


# A_B_C_S_SET.to_csv("/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V5_W23_349/ABCS_leng.csv", sep = '\t')



A_B_C_S_SET_ROW_CHECK = list(A_B_C_S_SET.drug_row_CID)
A_B_C_S_SET_COL_CHECK = list(A_B_C_S_SET.drug_col_CID)
A_B_C_S_SET_CELL_CHECK = list(A_B_C_S_SET.DrugCombCCLE)

A_B_C_S_SET['ROWCHECK'] = [str(int(A_B_C_S_SET_ROW_CHECK[i]))+'__'+A_B_C_S_SET_CELL_CHECK[i] for i in range(A_B_C_S_SET.shape[0])]
A_B_C_S_SET['COLCHECK'] = [str(int(A_B_C_S_SET_COL_CHECK[i]))+'__'+A_B_C_S_SET_CELL_CHECK[i] for i in range(A_B_C_S_SET.shape[0])]


# A_B_C_S_SET.to_csv('/st06/jiyeonH/11.TOX/DR_SPRING/trials/under_50_ABCS.csv', sep = '\t')

# LINCS exp order 따지기 


BETA_ORDER_pre = [list(LINCS_978.new_node).index(a) for a in JY_GRAPH_ORDER]
BETA_ORDER_DF = LINCS_978.iloc[BETA_ORDER_pre] # 어차피 ref 다르고 같은 애들이라 괜춘 
BETA_ENTREZ_ORDER = list(BETA_ORDER_DF.gene_id)
BETA_SYMBOL_ORDER = list(BETA_ORDER_DF.gene_symbol)
BETA_NEWNOD_ORDER = list(BETA_ORDER_DF.new_node)




def get_CHEM(cid, k=1):
	maxNumAtoms = max_len
	smiles = for_CAN_smiles[for_CAN_smiles.CID == cid]['CAN_SMILES'].item()
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




def get_synergy_data(DrugA_CID, DrugB_CID, Cell):
	ABCS1 = A_B_C_S[A_B_C_S.drug_row_CID == DrugA_CID]
	ABCS2 = ABCS1[ABCS1.drug_col_CID == DrugB_CID]
	ABCS3 = ABCS2[ABCS2.DrugCombCCLE == Cell]
	synergy_score = np.median(ABCS3.synergy_loewe) # 원래는 무조건 median
	return synergy_score



def check_lincs_ox(Drug_CID, Cell) :
    tmp_sigDF = BETA_CID_CCLE_SIG[(BETA_CID_CCLE_SIG.CID == Drug_CID) & (BETA_CID_CCLE_SIG.ccle_name == Cell)]
    if tmp_sigDF.shape[0] > 0 :
        res = True
    else :
        res = False 
    return res



# 시간이 오지게 걸리는것 같아서 아예 DC 전체에 대해서 진행한거를 가지고 하기로 했음 

DC_ALL_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/'

all_chem_DF = pd.read_csv(DC_ALL_PATH+'DC_ALL_7555_ORDER.csv')
all_chem_feat_TS = torch.load(DC_ALL_PATH+'DC_ALL.MY_chem_feat.pt')
all_chem_feat_adj = torch.load(DC_ALL_PATH+'DC_ALL.MY_chem_adj.pt')


def check_drug_f_ts(CID) :
	INDEX = all_chem_DF[all_chem_DF.CID == CID].index.item()
	adj_pre = all_chem_feat_adj[INDEX]
	# adj_proc = adj_pre.long().to_sparse().indices()
	return all_chem_feat_TS[INDEX], adj_pre




max_len = 50

MY_chem_A_feat = torch.empty(size=(A_B_C_S_SET.shape[0], max_len, 64))
MY_chem_B_feat= torch.empty(size=(A_B_C_S_SET.shape[0], max_len, 64))
MY_chem_A_adj = torch.empty(size=(A_B_C_S_SET.shape[0], max_len, max_len))
MY_chem_B_adj= torch.empty(size=(A_B_C_S_SET.shape[0], max_len, max_len))
MY_syn =  torch.empty(size=(A_B_C_S_SET.shape[0],1))

check_L_in_out_A = []
check_L_in_out_B = []


Fail_ind = []
from datetime import datetime

for IND in range(MY_chem_A_feat.shape[0]): #  100
	if IND%100 == 0 : 
		print(str(IND)+'/'+str(MY_chem_A_feat.shape[0]) )
		Fail_ind
		datetime.now()
	#
	DrugA_CID = A_B_C_S_SET.iloc[IND,]['drug_row_CID']
	DrugB_CID = A_B_C_S_SET.iloc[IND,]['drug_col_CID']
	Cell = A_B_C_S_SET.iloc[IND,]['DrugCombCCLE']
	#
	DrugA_Feat, DrugA_ADJ = check_drug_f_ts(DrugA_CID)
	DrugB_Feat, DrugB_ADJ = check_drug_f_ts(DrugB_CID)
	#
	check_L_in_out_A.append(check_lincs_ox(DrugA_CID, Cell))
	check_L_in_out_B.append(check_lincs_ox(DrugB_CID, Cell))
	# 
	AB_SYN = get_synergy_data(DrugA_CID, DrugB_CID, Cell)
	# 
	MY_chem_A_feat[IND] = torch.Tensor(DrugA_Feat)
	MY_chem_B_feat[IND] = torch.Tensor(DrugB_Feat)
	MY_chem_A_adj[IND] = torch.Tensor(DrugA_ADJ)
	MY_chem_B_adj[IND] = torch.Tensor(DrugB_ADJ)
	MY_syn[IND] = torch.Tensor([AB_SYN])




A_B_C_S_SET_2 = copy.deepcopy(A_B_C_S_SET)
A_B_C_S_SET_2['A_in_L'] = check_L_in_out_A
A_B_C_S_SET_2['B_in_L'] = check_L_in_out_B


SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V5_W23_349_FULL/' # small ver 

PRJ_NAME = 'M3V5_W23_349'

torch.save(MY_chem_A_feat, SAVE_PATH+'{}.MY_chem_A_feat.pt'.format(PRJ_NAME))
torch.save(MY_chem_B_feat, SAVE_PATH+'{}.MY_chem_B_feat.pt'.format(PRJ_NAME))
torch.save(MY_chem_A_adj, SAVE_PATH+'{}.MY_chem_A_adj.pt'.format(PRJ_NAME))
torch.save(MY_chem_B_adj, SAVE_PATH+'{}.MY_chem_B_adj.pt'.format(PRJ_NAME))
torch.save(MY_syn, SAVE_PATH+'{}.MY_syn.pt'.format(PRJ_NAME))

A_B_C_S_SET_2.to_csv(SAVE_PATH+'{}.A_B_C_S_SET.csv'.format(PRJ_NAME))
A_B_C_S.to_csv(SAVE_PATH+'{}.A_B_C_S.csv'.format(PRJ_NAME))



##########################################

# 기준 index 

A_B_C_S_SET = copy.deepcopy(A_B_C_S_SET_2)
A_B_C_S = copy.deepcopy(A_B_C_S)


CCLE_PATH = '/st06/jiyeonH/13.DD_SESS/CCLE.22Q1/'

ccle_exp = pd.read_csv(CCLE_PATH+'CCLE_expression.csv', low_memory=False)

ccle_ori_col = list(ccle_exp.columns)
ccle_new_col =['DepMap_ID'] + [int(a.split(')')[0].split('(')[1]) for a in ccle_ori_col[1:]]

ccle_exp.columns = ccle_new_col

ccle_info= pd.read_csv(CCLE_PATH+'sample_info.csv', low_memory=False)


# CCLE ver! 
ccle_cell_info = ccle_info[['DepMap_ID','CCLE_Name']]
ccle_cell_info.columns = ['DepMap_ID','DrugCombCCLE']
ccle_exp2 = pd.merge(ccle_exp, ccle_cell_info, on = 'DepMap_ID' , how='left')
ccle_exp3 = ccle_exp2[['DepMap_ID','DrugCombCCLE']+BETA_ENTREZ_ORDER]
ccle_names = [a for a in ccle_exp3.DrugCombCCLE if type(a) == str]


# 그냥 기본적인 정보 차이가 궁금 
DC_CELL_DF_ids = set(DC_CELL_DF.ccle_name) # 1475
ccle_cell_ids = set(ccle_cell_info.DrugCombCCLE) # 1827
# DC_CELL_DF_ids - ccle_cell_ids = 13
# ccle_cell_ids - DC_CELL_DF_ids = 365



# 349 
cell_basal_exp_list = []
# give vector 
for i in range(A_B_C_S_SET.shape[0]) :
    if i%100 == 0 :
        print(str(i)+'/'+str(A_B_C_S_SET.shape[0]) )
        datetime.now()
    ccle = A_B_C_S_SET.loc[i]['DrugCombCCLE']
    if ccle in ccle_names : 
        ccle_exp_df = ccle_exp3[ccle_exp3.DrugCombCCLE==ccle][BETA_ENTREZ_ORDER]
        ccle_exp_vector = ccle_exp_df.values[0].tolist()
        cell_basal_exp_list.append(ccle_exp_vector)
    else : # 'TC32_BONE', 'DU145_PROSTATE' -> 0 으로 진행하게 됨. public expression 없음 참고해야함. 
        ccle_exp_vector = [0]*349
        cell_basal_exp_list.append(ccle_exp_vector)



cell_base_tensor = torch.Tensor(cell_basal_exp_list)

torch.save(cell_base_tensor, SAVE_PATH+'{}.MY_CellBase.pt'.format(PRJ_NAME))


no_public_exp = list(set(A_B_C_S_SET['DrugCombCCLE']) - set(ccle_names))
no_p_e_list = ['X' if cell in no_public_exp else 'O' for cell in list(A_B_C_S_SET.DrugCombCCLE)]

A_B_C_S_SET_ADD = copy.deepcopy(A_B_C_S_SET)
A_B_C_S_SET_ADD = A_B_C_S_SET_ADD.reset_index(drop = True)
A_B_C_S_SET_ADD['Basal_Exp'] = no_p_e_list





#### synergy score 일관성 체크 
def get_synergy_data(DrugA_CID, DrugB_CID, Cell):
    ABCS1 = A_B_C_S[A_B_C_S.drug_row_CID == DrugA_CID]
    ABCS2 = ABCS1[ABCS1.drug_col_CID == DrugB_CID]
    ABCS3 = ABCS2[ABCS2.DrugCombCCLE == Cell]
    #
    if len(set(ABCS3.synergy_loewe>0)) ==1 : # 일관성 확인 
        OX = 'O'
    else: 
        OX = 'X'
    synergy_score = np.median(ABCS3.synergy_loewe) # 원래는 무조건 median
    return synergy_score, OX


OX_list = []

for IND in range(A_B_C_S_SET.shape[0]) :
    if IND%100 == 0 :
        print(str(IND)+'/'+str(A_B_C_S_SET.shape[0]) )
        datetime.now()
    CID_A = A_B_C_S_SET.iloc[IND,]['drug_row_CID']
    CID_B = A_B_C_S_SET.iloc[IND,]['drug_col_CID']
    Cell = A_B_C_S_SET.iloc[IND,]['DrugCombCCLE']
    score, OX = get_synergy_data(CID_A, CID_B, Cell)
    OX_list.append(OX)
    

A_B_C_S_SET_ADD['SYN_OX'] = OX_list








A_B_C_S_SET_CIDS = list(set(list(A_B_C_S_SET_ADD.drug_row_CID)+list(A_B_C_S_SET_ADD.drug_col_CID)))
gene_ids = list(BETA_ORDER_DF.gene_id)


# TARGET (1)
# TARGET (1)
# TARGET (1)

TARGET_PATH = '/st06/jiyeonH/13.DD_SESS/01.PRJ2/'
TARGET_DB = pd.read_csv(TARGET_PATH+'TARGET_CID_ENTREZ.csv', sep ='\t', index_col = 0)

TARGET_DB_RE = TARGET_DB[TARGET_DB.CID_RE.isin(A_B_C_S_SET_CIDS)]
TARGET_DB_RE = TARGET_DB_RE[TARGET_DB_RE.ENTREZ_RE.isin(gene_ids)]



# L_gene_symbol : 127501
# PPI_name : 126185
# 349
target_cids = list(set(TARGET_DB_RE.CID))
gene_ids = list(BETA_ORDER_DF.gene_id)
def get_targets(CID): # 이건 지금 필터링 한 경우임 #
	if CID in target_cids:
		tmp_df2 = TARGET_DB_RE[TARGET_DB_RE.CID == CID]
		targets = list(set(tmp_df2.EntrezID))
		vec = [1 if a in targets else 0 for a in gene_ids ]
	else :
		vec = [0] * 349
	return vec




TARGET_A = []
TARGET_B = []

for IND in range(A_B_C_S_SET_ADD.shape[0]) :
    if IND%100 == 0 :
        print(str(IND)+'/'+str(A_B_C_S_SET_ADD.shape[0]) )
        datetime.now()
    CID_A = A_B_C_S_SET_ADD.iloc[IND,]['drug_row_CID']
    CID_B = A_B_C_S_SET_ADD.iloc[IND,]['drug_col_CID']
    target_vec_A = get_targets(CID_A)
    target_vec_B = get_targets(CID_B)
    TARGET_A.append(target_vec_A)
    TARGET_B.append(target_vec_B)
    

TARGET_A_TENSOR = torch.Tensor(TARGET_A)
TARGET_B_TENSOR = torch.Tensor(TARGET_B)


torch.save(TARGET_A_TENSOR, SAVE_PATH+'{}.MY_Target_1_A.pt'.format(PRJ_NAME))
torch.save(TARGET_B_TENSOR, SAVE_PATH+'{}.MY_Target_1_B.pt'.format(PRJ_NAME))


T1_OX_list = []

for IND in range(A_B_C_S_SET_ADD.shape[0]) :
    if IND%100 == 0 :
        print(str(IND)+'/'+str(A_B_C_S_SET_ADD.shape[0]) )
        datetime.now()
    CID_A = A_B_C_S_SET_ADD.iloc[IND,]['drug_row_CID']
    CID_B = A_B_C_S_SET_ADD.iloc[IND,]['drug_col_CID']
    if (CID_A in target_cids) & (CID_B in target_cids) : 
        T1_OX_list.append('O')
    else : 
        T1_OX_list.append('X')
    

A_B_C_S_SET_ADD['T1OX']=T1_OX_list







# Tanimoto filter 

ABCS_ori_CIDs = list(set(list(A_B_C_S.drug_row_CID) + list(A_B_C_S.drug_col_CID))) # 172 
ABCS_FILT_CIDS = list(set(list(A_B_C_S_SET_ADD.drug_row_CID) + list(A_B_C_S_SET_ADD.drug_col_CID))) # 172 

ABCS_ori_SMILEs = list(set(list(A_B_C_S.ROW_CAN_SMILES) + list(A_B_C_S.COL_CAN_SMILES))) # 171
ABCS_FILT_SMILEs = list(set(list(A_B_C_S_SET_ADD.ROW_CAN_SMILES) + list(A_B_C_S_SET_ADD.COL_CAN_SMILES))) # 171 


PC_check = for_CAN_smiles[for_CAN_smiles.CID.isin(ABCS_ori_CIDs)]



def calculate_internal_pairwise_similarities(smiles_list) :
	"""
	Computes the pairwise similarities of the provided list of smiles against itself.
		Symmetric matrix of pairwise similarities. Diagonal is set to zero.
	"""
	mols = [Chem.MolFromSmiles(x.strip()) for x in smiles_list]
	fps = [Chem.RDKFingerprint(x) for x in mols]
	nfps = len(fps)
	#
	similarities = np.zeros((nfps, nfps))
	#
	for i in range(1, nfps):
		sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
		similarities[i, :i] = sims
		similarities[:i, i] = sims
	return similarities 


sim_matrix_order = list(PC_check.CAN_SMILES)
sim_matrix = calculate_internal_pairwise_similarities(sim_matrix_order)


row_means = []
for i in range(sim_matrix.shape[0]):
	indexes = [a for a in range(sim_matrix.shape[0])]
	indexes.pop(i)
	row_tmp = sim_matrix[i][indexes]
	row_mean = np.mean(row_tmp)
	row_means = row_means + [row_mean]


means_df = pd.DataFrame({ 'CIDs' : list(PC_check.CID), 'MEAN' : row_means})
means_df = means_df.sort_values('MEAN')
means_df['cat']= 'MEAN'
# means_df['filter']=['stay' if (a in cids_all) else 'nope' for a in list(means_df.CIDs)] 
means_df['dot_col']= ['IN' if (a in ABCS_FILT_CIDS) else 'OUT' for a in list(means_df.CIDs)] 


means_df['over0.1'] = ['IN' if a > 0.1  else 'OUT' for a in list(means_df.MEAN)] 
means_df['over0.2'] = ['IN' if a > 0.2  else 'OUT' for a in list(means_df.MEAN)] 
means_df['overQ'] = ['IN' if a > means_df.MEAN.describe()['25%']  else 'OUT' for a in list(means_df.MEAN)] 



# check triads num

row_cids = list(A_B_C_S_SET_ADD.drug_row_CID)
col_cids = list(A_B_C_S_SET_ADD.drug_col_CID)

tani_01 = list(means_df[(means_df['dot_col'] == 'IN') & (means_df['over0.1'] == 'IN') ]['CIDs'])
tani_02 = list(means_df[(means_df['dot_col'] == 'IN') & (means_df['over0.2'] == 'IN') ]['CIDs'])
tani_Q = list(means_df[(means_df['dot_col'] == 'IN') & (means_df['overQ'] == 'IN') ]['CIDs'])


tani_01_result = []
tani_02_result = []
tani_Q_result = []
for IND in range(A_B_C_S_SET_ADD.shape[0]) :
    if IND%100 == 0 :
        print(str(IND)+'/'+str(A_B_C_S_SET.shape[0]) )
        datetime.now()
    CID_A = A_B_C_S_SET.iloc[IND,]['drug_row_CID']
    CID_B = A_B_C_S_SET.iloc[IND,]['drug_col_CID']
    #
    if (CID_A in tani_01) & (CID_B in tani_01):
        tani_01_result.append('O')
    else : 
        tani_01_result.append('X')
    #
    if (CID_A in tani_02) & (CID_B in tani_02):
        tani_02_result.append('O')
    else : 
        tani_02_result.append('X')
        #
    if (CID_A in tani_Q) & (CID_B in tani_Q):
        tani_Q_result.append('O')
    else : 
        tani_Q_result.append('X')
    

A_B_C_S_SET_ADD['tani01'] = tani_01_result
A_B_C_S_SET_ADD['tani_02'] = tani_02_result
A_B_C_S_SET_ADD['tani_Q'] = tani_Q_result

A_B_C_S_SET_ADD.to_csv(SAVE_PATH+'{}.A_B_C_S_SET_ADD.csv'.format(PRJ_NAME))

