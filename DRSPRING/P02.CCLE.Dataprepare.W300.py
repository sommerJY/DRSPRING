

padding / no padding 





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
from matplotlib import colors as mcolors


NETWORK_PATH = '/st06/jiyeonH/13.DD_SESS/HumanNetV3/'
LINCS_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/' 
DATA_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V3_FULL/'
DC_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/'

 


# node order 고정 
# HS 다른 pathway 사용 
print('NETWORK')

hunet_dir = '/st06/jiyeonH/13.DD_SESS/HumanNetV3/'

hunet_gsp = pd.read_csv(hunet_dir+'HS-DB.tsv', sep = '\t', header = None)
hunet_gsp.columns = ['G_A','G_B','SC']

BETA_GENE = pd.read_table('/st06/jiyeonH/11.TOX/LINCS/L_2020/geneinfo_beta.txt')
BETA_lm_genes = BETA_GENE[BETA_GENE.feature_space=='landmark'] # 978
BETA_lm_genes = BETA_lm_genes.reset_index()
lm_entrezs = list(BETA_lm_genes.gene_id)

hnet_L1 = hunet_gsp[hunet_gsp['G_A'].isin(BETA_lm_genes.gene_id)] # 9757
hnet_L2 = hnet_L1[hnet_L1['G_B'].isin(BETA_lm_genes.gene_id)] # 1278
hnet_L3 = hnet_L2[hnet_L2.SC >= 3.5] # 507


len(set(list(hnet_L3['G_A']) + list(hnet_L3['G_B']))) # 349

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



# Graph 확인 

JY_GRAPH = MY_G
JY_GRAPH_ORDER = MY_G.nodes()
JY_ADJ = nx.adjacency_matrix(JY_GRAPH)

JY_ADJ_tmp = torch.LongTensor(JY_ADJ.toarray())
JY_ADJ_IDX = JY_ADJ_tmp.to_sparse().indices()
JY_IDX_WEIGHT = MY_WEIGHT_SCORE

# nx.draw(MY_G, node_size = 5)
# plt.savefig('/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V6_349_FULL/HN_G.png', dpi=300, bbox_inches='tight')
# plt.close()
# 전부가 이어져 있는 그래프는 아닌데, 그래도 나름 알아서 잘 잘려있어서 그런지 그런가봉가 
# subgraph 가 답일지도 정말로 


BETA_ORDER_pre = [list(LINCS_978.new_node).index(a) for a in JY_GRAPH_ORDER]
BETA_ORDER_DF = LINCS_978.iloc[BETA_ORDER_pre] # 어차피 ref 다르고 같은 애들이라 괜춘 
BETA_ENTREZ_ORDER = list(BETA_ORDER_DF.gene_id)
BETA_SYMBOL_ORDER = list(BETA_ORDER_DF.gene_symbol)
BETA_NEWNOD_ORDER = list(BETA_ORDER_DF.new_node)




##################################################################
##################################################################
##################################################################

BETA_CP_info = pd.read_table('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+'compoundinfo_beta.txt') # pert 34419 , cansmiles : 
BETA_CEL_info = pd.read_table('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+'cellinfo_beta.txt')
BETA_SIG_info = pd.read_table('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+'siginfo_beta.txt', low_memory = False)

LINCS_ALL_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/'
BETA_BIND_ORI = torch.load(LINCS_ALL_PATH+'BETA_BIND.349.pt')
BETA_BIND_ORI_DF = pd.read_csv(LINCS_ALL_PATH+'BETA_BIND.349.siglist.csv')
BETA_BIND_NEW = torch.load(LINCS_ALL_PATH+'BETA_BIND2.349.pt')
BETA_BIND_NEW_DF = pd.read_csv(LINCS_ALL_PATH+'BETA_BIND2.349.siglist.csv')

BETA_BIND = torch.concat([BETA_BIND_ORI, BETA_BIND_NEW])
BETA_BIND_SIG_df = pd.concat([BETA_BIND_ORI_DF, BETA_BIND_NEW_DF])
BETA_BIND_SIG_df = BETA_BIND_SIG_df.reset_index(drop = True)
						
LINCS_PERT_MATCH = pd.read_csv('/st06/jiyeonH/13.DD_SESS/01.PRJ2/LINCS_PERT_CID_MATCH.1207.csv', sep ='\t')
LINCS_PERT_MATCH = LINCS_PERT_MATCH[['pert_id','CID']]


# 지금까지는 어차피 exampler 라서 상관이 없었음 
filter1 = BETA_SIG_info[BETA_SIG_info.pert_type.isin([ 'trt_cp' ])] # 'ctl_vehicle', 'ctl_untrt' ,

dose = filter1['pert_idose']
dose = [a if type(a) == str else 'NA' for a in dose]
dose_set = list(set(dose))
dose_set.sort()

time = filter1['pert_itime']
time = [a if type(a) == str else 'NA' for a in time]
time_set = list(set(time))
time_set.sort()

dose_time = [dose[a]+'__'+time[a] for a in range(len(dose))]
dose_time_set = list(set(dose_time))
dose_time_set.sort()

filter1['dose_time'] = dose_time

freq = [dose_time.count(a) for a in dose_time_set]
freq_df = pd.DataFrame({ 'dose_time': dose_time_set, 'freq': freq })

freq_df.sort_values('freq')

filter2 = filter1[ (filter1.pert_idose == '10 uM') & (filter1.pert_itime == '24 h')] # 128077





# 한번 더 pubchem converter 로 내가 붙인 애들 
BETA_CP_info_filt = BETA_CP_info[['pert_id','canonical_smiles']].drop_duplicates() # 34419

can_sm_re = pd.read_csv('/st06/jiyeonH/11.TOX/DR_SPRING/tmp/can_sm_conv', sep = '\t', header = None)

can_sm_re.columns = ['canonical_smiles','CONV_CID']
can_sm_re = can_sm_re.drop_duplicates()
len(set([a for a in BETA_CP_info['pert_id'] if type(a) == str])) # 34419
len(set([a for a in can_sm_re['canonical_smiles'] if type(a) == str])) # 28575
len(set(can_sm_re[can_sm_re.CONV_CID>0]['CONV_CID'])) # 27841


can_sm_re2 = pd.merge(BETA_CP_info_filt, can_sm_re, on = 'canonical_smiles', how = 'left') # 34419 -> 1 sm 1 cid 확인 

can_sm_re3 = can_sm_re2[['pert_id','canonical_smiles','CONV_CID']].drop_duplicates() # 
# converter 가 더 많이 붙여주는듯 




# 민지가 pcp 로 붙인 애들 
BETA_MJ = pd.read_csv('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+"Lincs_pubchem_mj.csv") # SID 필터까지 그렇게 완성한 무엇 
# 문제는, pert id 에 붙는 cid 겹치는 애들이 있다는거.. 

BETA_MJ_RE = BETA_MJ[['pert_id','SMILES_cid','canonical_smiles']].drop_duplicates() # 25903
BETA_MJ_RE_CK = BETA_MJ_RE[['pert_id','SMILES_cid']]
len(set([a for a in BETA_MJ_RE['pert_id'] if type(a) == str])) # 25903
len(set([a for a in BETA_MJ_RE['canonical_smiles'] if type(a) == str])) # 25864
len(set(BETA_MJ_RE_CK[BETA_MJ_RE_CK.SMILES_cid>0]['SMILES_cid'])) # 25642

check = pd.merge(can_sm_re3, BETA_MJ_RE_CK, on = 'pert_id', how = 'left' )
check2 = check[check.CONV_CID !=check.SMILES_cid]
check3 = check2[check2.SMILES_cid > 0 ]
check4 = check3[check3.CONV_CID > 0 ]
# 그래서 둘이 안맞는게 58 개 정도 있어서... CID 를 최대한 붙이기 위해서 그러면.. 
# 근데 그래서 대충 drubcomb 에 있는 CID 랑 얼마나 겹치나 보니... 그냥 저냥 conv 10 개, mj 3 개 

# sum(check4.CONV_CID.isin(request_mj.CID)) # 10 
# sum(check4.SMILES_cid.isin(request_mj.CID)) # 3 

# 버리려고 했는데, 버리면 안될지도? 
# exemplar 에 다 들어가는 기염을 토하고 있음 ㅎ 


# LINCS match final 
pert_id_match = check[check.CONV_CID == check.SMILES_cid][['pert_id','canonical_smiles','CONV_CID']]
# sum((check2.CONV_CID >0 ) &( np.isnan(check2.SMILES_cid)==True)) # 2521
# sum((check2.SMILES_cid >0 ) &( np.isnan(check2.CONV_CID)==True)) # 427
conv_win = check2[(check2.CONV_CID >0 ) & ( np.isnan(check2.SMILES_cid)==True)][['pert_id','canonical_smiles','CONV_CID']]
mj_win = check2[(check2.SMILES_cid >0 ) & ( np.isnan(check2.CONV_CID)==True)][['pert_id','canonical_smiles','SMILES_cid']]
nans = check2[(np.isnan(check2.SMILES_cid)==True ) & ( np.isnan(check2.CONV_CID)==True)] # 5995
nans2 = nans[nans.pert_id.isin(filter2.pert_id)]
nans3 = nans2[-nans2.canonical_smiles.isin(['restricted', np.nan])]
# 한 162개 정도는 아예 안붙음. 심지어 우리의 exemplar 에도 있지만.. 흑... 
# 0614 기준으로는 77개

pert_id_match.columns = ['pert_id','canonical_smiles','CID'] # 25418,
conv_win.columns = ['pert_id','canonical_smiles','CID'] # 2521,
mj_win.columns =['pert_id','canonical_smiles','CID']


individual_check = check4.reset_index(drop =True)

individual_check_conv = individual_check.loc[[0,4,5,6,10,11,12,13,16,17,18,19]+[a for a in range(21,34)]+[36,40,54]][['pert_id','canonical_smiles','CONV_CID']]
individual_check_mj = individual_check.loc[[1,2,3,7,8,9,14,15,20,34,35,37,38,39]+[a for a in range(41,54)]+[55,56,57]][['pert_id','canonical_smiles','SMILES_cid']]
# [a for a in individual_check_conv.index if a in individual_check_mj.index]
individual_check_conv.columns = ['pert_id','canonical_smiles','CID'] # 28
individual_check_mj.columns = ['pert_id','canonical_smiles','CID'] # 30 


LINCS_PERT_MATCH = pd.concat([pert_id_match, conv_win, mj_win, individual_check_conv,  individual_check_mj]) # 28424
len(set([a for a in LINCS_PERT_MATCH['pert_id'] if type(a) == str])) # 34419 -> 28424
len(set([a for a in LINCS_PERT_MATCH['canonical_smiles'] if type(a) == str])) # 28575 -> 28381
len(set(LINCS_PERT_MATCH[LINCS_PERT_MATCH.CID>0]['CID'])) # 27841 -> 28154
LINCS_PERT_MATCH_cids = list(set(LINCS_PERT_MATCH.CID))

# LINCS_PERT_MATCH.to_csv('/st06/jiyeonH/13.DD_SESS/01.PRJ2/LINCS_PERT_CID_MATCH.1207.csv', sep ='\t')

# aa = list(LINCS_PERT_MATCH.CID)
# [a for a in aa if aa.count(a)>1]
# 10172943



# merge with exemplar sigid 
BETA_EXM = pd.merge(filter2, LINCS_PERT_MATCH, on='pert_id', how = 'left') # 128077
BETA_EXM2 = BETA_EXM[BETA_EXM.CID > 0] # 128038 # 이건 늘어났음 -> 122861

BETA_CEL_info2 = BETA_CEL_info[['cell_iname','cellosaurus_id','ccle_name']] # 240 
BETA_SELEC_SIG_wCell = pd.merge(BETA_EXM2, BETA_CEL_info2, on = 'cell_iname', how = 'left') # 122861
BETA_SELEC_SIG_wCell2 = BETA_SELEC_SIG_wCell[['pert_id','CID','cell_iname','cellosaurus_id','ccle_name','sig_id']]
BETA_SELEC_SIG_wCell2 = BETA_SELEC_SIG_wCell2.drop_duplicates() # 128038


# cell_iname check 

splitname = [a.split('_')[0] if type(a) == str else 'NA' for a in BETA_SELEC_SIG_wCell2.ccle_name]
tmp_tmp = BETA_SELEC_SIG_wCell2[(BETA_SELEC_SIG_wCell2.cell_iname == splitname) == False][['cell_iname','ccle_name']].drop_duplicates()
# 그래서 iname 으로 하면 CCLE_names 도 다 챙기고 새로운 애들만 더 생긴다는거 확인함 
# iname 으로만 진행해도 됨. 


# 근데 잠깐, MJ_request 에도 그런 애들 있는지 한번만 더 확인
# DC_NA = DC_CELL_DF[DC_CELL_DF.ccle_name=='NA']
# DC_syn = list(DC_NA.synonyms)
# DC_syn2 = [a.split('; ') for a in DC_syn ]
# DC_syn3 = list(set(sum(DC_syn2,[])))
# tmp_tmp = BETA_SELEC_SIG_wCell2[(BETA_SELEC_SIG_wCell2.cell_iname == splitname) == False][['cell_iname','ccle_name']].drop_duplicates()
# tmp_tmp[tmp_tmp.cell_iname.isin(DC_syn3)]
# 그렇게 나온게 TMD8, LNCAP, BJAB 인데, ccle 에 아예 처음부터 없는 애들임. 걱정 안해도 될듯 



ciname_tt = [True if type(a)==str else False for a in list(BETA_SELEC_SIG_wCell2.cell_iname)] 
# sum(ccle_tt) : 109720 -> 128038
BETA_CID_CCLE_SIG = BETA_SELEC_SIG_wCell2[ciname_tt][['pert_id','CID','cell_iname','ccle_name','sig_id']].drop_duplicates() # 111012
beta_cid_ccle_sig_tf = [ True if a>0 else False for a in list(BETA_CID_CCLE_SIG.CID)]
BETA_CID_CCLE_SIG = BETA_CID_CCLE_SIG[beta_cid_ccle_sig_tf]
BETA_CID_CCLE_SIG['CID'] = [int(a) for a in list(BETA_CID_CCLE_SIG['CID']) ] # 122861 
# 늘어남! 

# -> CCLE 필터까지 완료 / DC CID 고정 완료 






# 아예 CCLE 이름이랑 여기서부터 매치시켜줘야함 
CCLE_PATH = '/st06/jiyeonH/13.DD_SESS/CCLE.22Q1/'

ccle_exp = pd.read_csv(CCLE_PATH+'CCLE_expression.csv', low_memory=False)

ccle_ori_col = list(ccle_exp.columns)
ccle_new_col =['DepMap_ID'] + [int(a.split(')')[0].split('(')[1]) for a in ccle_ori_col[1:]]

ccle_exp.columns = ccle_new_col

# ccle_mut = pd.read_csv(CCLE_PATH+'CCLE_mutations.csv', low_memory=False)
ccle_info= pd.read_csv(CCLE_PATH+'sample_info.csv', low_memory=False)


# CCLE ver! 
ccle_cell_info = ccle_info[['DepMap_ID','stripped_cell_line_name','CCLE_Name']]
ccle_cell_info.columns = ['DepMap_ID','STR_ID','DrugCombCCLE']
ccle_exp2 = pd.merge(ccle_exp, ccle_cell_info, on = 'DepMap_ID' , how='left')

# ccle_exp3 = ccle_exp2[['DepMap_ID','DrugCombCCLE']+BETA_ENTREZ_ORDER]
# ccle_names = [a for a in ccle_exp3.DrugCombCCLE if type(a) == str]


BETA_CID_CCLE_SIG_NEW = copy.deepcopy(BETA_CID_CCLE_SIG)
BETA_CID_CCLE_SIG_NEW.columns = ['pert_id', 'CID', 'STR_ID', 'ccle_name', 'sig_id']
BETA_CID_CCLE_SIG_NEW2 = pd.merge(BETA_CID_CCLE_SIG_NEW, ccle_cell_info, on = 'STR_ID', how = 'left')

# set(BETA_CID_CCLE_SIG_NEW2.DrugCombCCLE) - set(BETA_CID_CCLE_SIG_NEW2.ccle_name)
# {'MCF10A_BREAST', 'HELA_CERVIX', 'HAP1_ENGINEERED'} # 어차피 MCF10A_BREAST 는 drugcomb 에 없음 

BETA_CID_CCLE_SIG_NEW3  = BETA_CID_CCLE_SIG_NEW2[['pert_id','CID','DrugCombCCLE','sig_id']] # 다시. 
ciname_tt2 = [True if type(a)==str else False for a in list(BETA_CID_CCLE_SIG_NEW3.DrugCombCCLE)] 
BETA_CID_CCLE_SIG_NEW4 = BETA_CID_CCLE_SIG_NEW3[ciname_tt2] 
BETA_CID_CCLE_SIG_NEW5 = BETA_CID_CCLE_SIG_NEW4[BETA_CID_CCLE_SIG_NEW4.CID>0] #  113016







# Drug Comb 데이터 가져오기 
# synergy info 
DC_PATH = '/st06/jiyeonH/13.DD_SESS/DrugComb.1.5/' 

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

for DD in range(0,len(DC_DRUG)):
	tmpdf = pd.DataFrame({k:[DC_DRUG[DD][k]] for k in DC_DRUG_K})
	DC_DRUG_DF = pd.concat([DC_DRUG_DF, tmpdf], axis = 0)

DC_DRUG_DF['id_re'] = [float(a) for a in list(DC_DRUG_DF['id'])]

DC_DRUG_DF_FULL = pd.read_csv('/st06/jiyeonH/11.TOX/MY_TRIAL_6/DC_DRUG_DF_PC.csv', sep ='\t')
# pubchem canonical smiles 쓰고 length 값 들어간 버전 


DC_lengs = list(DC_DRUG_DF_FULL.leng)
DC_lengs2 = [int(a) for a in DC_lengs if a!= 'error']



# Cell info 
with open(DC_PATH+'cell_lines.json') as json_file :
	DC_CELL =json.load(json_file)

DC_CELL_K = list(DC_CELL[0].keys())
DC_CELL_DF = pd.DataFrame(columns=DC_CELL_K)

for DD in range(0,len(DC_CELL)):
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




DC_DATA_filter_chch = DC_DATA_filter4[['drug_row_id_re','drug_col_id_re','cell_line_id']].drop_duplicates()
dc_ch_1 = list(DC_DATA_filter_chch['drug_row_id_re'])
dc_ch_2 = list(DC_DATA_filter_chch['drug_col_id_re'])
dc_ch_3 = list(DC_DATA_filter_chch['cell_line_id'])


# 순서 고려해서 unique 아이디 만들기 (크기 비교가 str 사이에서도 성립한다는거 아니?)
DC_DATA_filter_chch['id_id_cell'] = [str(int(dc_ch_1[i])) + '___' + str(int(dc_ch_2[i]))+ '___' + str(int(dc_ch_3[i])) if dc_ch_1[i] < dc_ch_2[i] else str(int(dc_ch_2[i])) + '___' + str(int(dc_ch_1[i]))+ '___' + str(int(dc_ch_3[i])) for i in range(DC_DATA_filter_chch.shape[0])]

# DC 에서는 앞뒤가 섞인적이 없음 
# CID 붙이고 나서 섞였을수는 있는거
# 근데 그건 DC 가 알바는 아니지. 



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


DC_ccle_final = DC_DATA7_6_ccle[['drug_row_CID','drug_col_CID','DrugCombCCLE']].drop_duplicates() # 554128
DC_ccle_final_dup = DC_DATA7_6_ccle[['drug_row_CID','drug_col_CID','DrugCombCCLE', 'synergy_loewe']].drop_duplicates() # 730348 -> 719946

DC_ccle_final_cids = list(set(list(DC_ccle_final_dup.drug_row_CID) + list(DC_ccle_final_dup.drug_col_CID)))
# 3146





# 한번만 더 확인 
DC_DATA_filter_chch = DC_ccle_final[['drug_row_CID','drug_col_CID','DrugCombCCLE']].drop_duplicates()
dc_ch_1 = list(DC_DATA_filter_chch['drug_row_CID'])
dc_ch_2 = list(DC_DATA_filter_chch['drug_col_CID'])
dc_ch_3 = list(DC_DATA_filter_chch['DrugCombCCLE'])


# 순서 고려해서 unique 아이디 만들기 (크기 비교가 str 사이에서도 성립한다는거 아니?)
DC_DATA_filter_chch['cid_cid_cell'] = [str(int(dc_ch_1[i])) + '___' + str(int(dc_ch_2[i]))+ '___' + str(dc_ch_3[i]) if dc_ch_1[i] < dc_ch_2[i] else str(int(dc_ch_2[i])) + '___' + str(int(dc_ch_1[i]))+ '___' + str(dc_ch_3[i]) for i in range(DC_DATA_filter_chch.shape[0])]

len(DC_DATA_filter_chch.cid_cid_cell) # 460739
len(set(DC_DATA_filter_chch.cid_cid_cell)) # 450002

# 그렇습니다. DC ID 상에서는 안겹쳤었는데 CID 로는 겹치는게 있네욤 
DC_dup_total = DC_DATA_filter_chch[DC_DATA_filter_chch['cid_cid_cell'].duplicated(keep = False)].sort_values('cid_cid_cell') # 21474 개 인거지. 
#DC_dup_total.to_csv('/st06/jiyeonH/11.TOX/DR_SPRING/trials/DC_duplicates.csv', sep = '\t', index = False)
# 그러면, CID - CCLE 가 같은건 평균 내는걸로 하고 
# sig_id 다른건 다른 세트로 취급하자 

# 근데 이게 sig_id 때문에 dup 되는거랑 구분이 안되니까
# 따로 저장해서 확인해줘야함 




# MATCH DC & LINCS 
print('DC and LINCS') 
# 무조건 pubchem 공식 파일 사용하기 (mj ver) 
# 문제는 pert_id 에 CID 가 중복되어 붙는 애들도 있다는거  -> 어떻게 해결할거? 
# CV 나눌때만 smiles 기준으로 나눠주기 
# pert id 로 나눠서 하는게 맞을것 같음.


DC_ccle_final_dup_ROW_CHECK = list(DC_ccle_final_dup.drug_row_CID)
DC_ccle_final_dup_COL_CHECK = list(DC_ccle_final_dup.drug_col_CID)
DC_ccle_final_dup_CELL_CHECK = list(DC_ccle_final_dup.DrugCombCCLE)

DC_ccle_final_dup['ROWCHECK'] = [str(int(DC_ccle_final_dup_ROW_CHECK[i]))+'__'+DC_ccle_final_dup_CELL_CHECK[i] for i in range(DC_ccle_final_dup.shape[0])]
DC_ccle_final_dup['COLCHECK'] = [str(int(DC_ccle_final_dup_COL_CHECK[i]))+'__'+DC_ccle_final_dup_CELL_CHECK[i] for i in range(DC_ccle_final_dup.shape[0])]


# 공식 smiles 
PC_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/'

for_CAN_smiles = pd.read_csv(PC_PATH+'CID_SMILES.csv', low_memory = False)
# for_CAN_smiles = copy.deepcopy(PC_FILTER)
for_CAN_smiles = for_CAN_smiles[['CID','CAN_SMILES']]
for_CAN_smiles.columns = ['drug_row_CID','ROW_CAN_SMILES']
DC_ccle_final_dup = pd.merge(DC_ccle_final_dup, for_CAN_smiles, on='drug_row_CID', how ='left' )
for_CAN_smiles.columns = ['drug_col_CID','COL_CAN_SMILES']
DC_ccle_final_dup = pd.merge(DC_ccle_final_dup, for_CAN_smiles, on='drug_col_CID', how ='left' )
for_CAN_smiles.columns = ['CID','CAN_SMILES']



# CAN_SMILES NA 있음?  
CAN_TF_1 = [True if type(a) == float else False for a in list(DC_ccle_final_dup.ROW_CAN_SMILES)]
CAN_TF_DF_1 = DC_ccle_final_dup[CAN_TF_1]
CAN_TF_2 = [True if type(a) == float else False for a in list(DC_ccle_final_dup.COL_CAN_SMILES)]
CAN_TF_DF_2 = DC_ccle_final_dup[CAN_TF_2]
# DC 기준으로는 없음. LINCS 기준에서는 있었음 


BETA_CID_CCLE_SIG_ID_CHECK = list(BETA_CID_CCLE_SIG_NEW5.CID)
BETA_CID_CCLE_SIG_CELL_CHECK = list(BETA_CID_CCLE_SIG_NEW5.DrugCombCCLE)

BETA_CID_CCLE_SIG_NEW5['IDCHECK'] = [str(int(BETA_CID_CCLE_SIG_ID_CHECK[i]))+'__'+BETA_CID_CCLE_SIG_CELL_CHECK[i] for i in range(BETA_CID_CCLE_SIG_NEW5.shape[0])]
# 이렇게 되면 , IDCHECK 에 중복이 생기긴 함. pert 때문에. 


BETA_CID_CCLE_SIG_NEW5.columns=['ROW_pert_id', 'drug_row_CID', 'DrugCombCCLE', 'ROW_BETA_sig_id',  'ROWCHECK']
CCLE_DC_BETA_1 = pd.merge(DC_ccle_final_dup, BETA_CID_CCLE_SIG_NEW5[['ROW_pert_id', 'ROW_BETA_sig_id',  'ROWCHECK']], left_on = 'ROWCHECK', right_on = 'ROWCHECK', how = 'left') # 720619

BETA_CID_CCLE_SIG_NEW5.columns=['COL_pert_id', 'drug_col_CID', 'DrugCombCCLE', 'COL_BETA_sig_id', 'COLCHECK']
CCLE_DC_BETA_2 = pd.merge(CCLE_DC_BETA_1, BETA_CID_CCLE_SIG_NEW5[['COL_pert_id', 'COL_BETA_sig_id', 'COLCHECK']], left_on = 'COLCHECK', right_on = 'COLCHECK', how = 'left') # 721206

BETA_CID_CCLE_SIG_NEW5.columns=['pert_id', 'pubchem_cid', 'DrugCombCCLE', 'sig_id', 'IDCHECK']



# DC_dup_total 고려해서 숫자세기 위함 
#ranran = list(CCLE_DC_BETA_2.index)
aa = list(CCLE_DC_BETA_2['drug_row_CID'])
bb = list(CCLE_DC_BETA_2['drug_col_CID'])
cc = list(CCLE_DC_BETA_2['DrugCombCCLE'])
aaaa = list(CCLE_DC_BETA_2['ROW_BETA_sig_id'])
bbbb = list(CCLE_DC_BETA_2['COL_BETA_sig_id'])
aaaa = [ a if type(a) == str else 'NA' for a in aaaa]
bbbb = [ a if type(a) == str else 'NA' for a in bbbb]


# 순서 고려해서 unique 아이디 만들기 (크기 비교가 str 사이에서도 성립한다는거 아니?)
CCLE_DC_BETA_2['cid_cid_cell'] = [str(int(aa[i])) + '___' + str(int(bb[i]))+ '___' + cc[i] if aa[i] < bb[i] else str(int(bb[i])) + '___' + str(int(aa[i]))+ '___' + cc[i] for i in range(CCLE_DC_BETA_2.shape[0])]
CCLE_DC_BETA_2['sig_sig_cell'] = [str(aaaa[i]) + '___' + str(bbbb[i])+ '___' + cc[i] if aaaa[i] < bbbb[i] else str(bbbb[i]) + '___' + str(aaaa[i])+ '___' + cc[i] for i in range(CCLE_DC_BETA_2.shape[0])]
# 여기서는 코드상에서 듀플 빼고 만들어달라고 했는데... 
# 사실 그냥 cid_cid_cell 로 통일하면 그걸 제거할 필요가 있나..? 
#CCLE_DC_BETA_2_rere = CCLE_DC_BETA_2[CCLE_DC_BETA_2.cid_cid_cell.isin(DC_dup_total.id_id_cell)==False]
#CCLE_DC_BETA_2 = CCLE_DC_BETA_2_rere.reset_index(drop=True)




# (1) AO BO  
FILTER_AO_BO = [a for a in range(CCLE_DC_BETA_2.shape[0]) if (type(CCLE_DC_BETA_2.ROW_BETA_sig_id[a]) == str) & (type(CCLE_DC_BETA_2.COL_BETA_sig_id[a]) == str)]
DATA_AO_BO = CCLE_DC_BETA_2.loc[FILTER_AO_BO] # 11379 
DATA_AO_BO[['drug_row_CID','drug_col_CID','DrugCombCCLE']].drop_duplicates() # 8404
DATA_AO_BO[['ROW_BETA_sig_id','COL_BETA_sig_id','DrugCombCCLE']].drop_duplicates() # 8914
DATA_AO_BO[['ROW_pert_id','COL_pert_id','DrugCombCCLE']].drop_duplicates() #  8914
DATA_AO_BO[['ROW_CAN_SMILES','COL_CAN_SMILES','DrugCombCCLE']].drop_duplicates() #  8404
DATA_AO_BO_cids = list(set(list(DATA_AO_BO.drug_row_CID) + list(DATA_AO_BO.drug_col_CID))) # 172 


# (2) AX BO 
FILTER_AX_BO = [a for a in range(CCLE_DC_BETA_2.shape[0]) if (type(CCLE_DC_BETA_2.ROW_BETA_sig_id[a]) != str) & (type(CCLE_DC_BETA_2.COL_BETA_sig_id[a]) == str)]
DATA_AX_BO = CCLE_DC_BETA_2.loc[FILTER_AX_BO] # 11967
DATA_AX_BO[['drug_row_CID','drug_col_CID','DrugCombCCLE']].drop_duplicates() # 9428
DATA_AX_BO[['ROW_BETA_sig_id','COL_BETA_sig_id','DrugCombCCLE']].drop_duplicates() # 452
DATA_AX_BO[['ROW_pert_id','COL_pert_id','DrugCombCCLE']].drop_duplicates() #  452
DATA_AX_BO[['ROW_CAN_SMILES','COL_CAN_SMILES','DrugCombCCLE']].drop_duplicates() #  9414 -> 겹치는 애들 제거해야하는걸로! 
DATA_AX_BO_cids = list(set(list(DATA_AX_BO.drug_row_CID) + list(DATA_AX_BO.drug_col_CID))) # 635 


# (3) AO BX 
FILTER_AO_BX = [a for a in range(CCLE_DC_BETA_2.shape[0]) if (type(CCLE_DC_BETA_2.ROW_BETA_sig_id[a]) == str) & (type(CCLE_DC_BETA_2.COL_BETA_sig_id[a]) != str)]
DATA_AO_BX = CCLE_DC_BETA_2.loc[FILTER_AO_BX] # 14998
DATA_AO_BX[['drug_row_CID','drug_col_CID','DrugCombCCLE']].drop_duplicates() # 12926
DATA_AO_BX[['ROW_BETA_sig_id','COL_BETA_sig_id','DrugCombCCLE']].drop_duplicates() # 449
DATA_AO_BX[['ROW_pert_id','COL_pert_id','DrugCombCCLE']].drop_duplicates() #  449
DATA_AO_BX[['ROW_CAN_SMILES','COL_CAN_SMILES','DrugCombCCLE']].drop_duplicates() #  12915 -> 겹치는 애들 제거해야하는걸로! 
DATA_AO_BX_cids = list(set(list(DATA_AO_BX.drug_row_CID) + list(DATA_AO_BX.drug_col_CID))) # 274 


# (4) AX BX 
FILTER_AX_BX = [a for a in range(CCLE_DC_BETA_2.shape[0]) if (type(CCLE_DC_BETA_2.ROW_BETA_sig_id[a]) != str) & (type(CCLE_DC_BETA_2.COL_BETA_sig_id[a]) != str)]
DATA_AX_BX = CCLE_DC_BETA_2.loc[FILTER_AX_BX] # 584465
DATA_AX_BX = DATA_AX_BX[DATA_AX_BX.DrugCombCCLE!='NA']
DATA_AX_BX[['drug_row_CID','drug_col_CID','DrugCombCCLE']].drop_duplicates() # 506710
DATA_AX_BX[['ROW_BETA_sig_id','COL_BETA_sig_id','DrugCombCCLE']].drop_duplicates() # 232 의미 없음 
DATA_AX_BX[['ROW_pert_id','COL_pert_id','DrugCombCCLE']].drop_duplicates() #  232 의미 없음 
DATA_AX_BX[['ROW_CAN_SMILES','COL_CAN_SMILES','DrugCombCCLE']].drop_duplicates() #  8404
DATA_AX_BX_cids = list(set(list(DATA_AX_BX.drug_row_CID) + list(DATA_AX_BX.drug_col_CID))) # 4280 


DATA_AO_BO['type'] = 'AOBO' # 11379
DATA_AX_BO['type'] = 'AXBO' # 11967
DATA_AO_BX['type'] = 'AOBX' # 14998
DATA_AX_BX['type'] = 'AXBX' # 584465



A_B_C_S = pd.concat([DATA_AO_BO, DATA_AX_BO, DATA_AO_BX, DATA_AX_BX])
A_B_C_S = A_B_C_S.reset_index(drop = True) # 947047

A_B_C_S_SET = copy.deepcopy(A_B_C_S)
A_B_C_S_SET = A_B_C_S_SET.drop('synergy_loewe', axis = 1).drop_duplicates() # 
A_B_C_S_SET = A_B_C_S_SET.reset_index(drop = True) # 

# A_B_C_S.to_csv('/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V7_349_FULL/A_B_C_S.0819.csv', sep = '\t', index= False)
# A_B_C_S = pd.read_csv('/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V7_349_FULL/A_B_C_S.0819.csv', sep = '\t', low_memory = False)

###################



# 50으로 제대로 자르기 위함 
# check length

DC_DRUG_DF_FULL_leng = DC_DRUG_DF_FULL[['CAN_SMILES','leng']].drop_duplicates()
DC_DRUG_DF_FULL_leng.columns = ['ROW_CAN_SMILES','ROW_len']
A_B_C_S_SET_ROWleng = pd.merge(A_B_C_S_SET, DC_DRUG_DF_FULL_leng, on ='ROW_CAN_SMILES', how = 'left')

DC_DRUG_DF_FULL_leng.columns = ['COL_CAN_SMILES','COL_len']
A_B_C_S_SET_COLleng = pd.merge(A_B_C_S_SET_ROWleng, DC_DRUG_DF_FULL_leng, on ='COL_CAN_SMILES', how = 'left')


A_B_C_S_SET['ROW_len'] = [int(a) for a in A_B_C_S_SET_COLleng.ROW_len]
A_B_C_S_SET['COL_len'] = [int(a) for a in A_B_C_S_SET_COLleng.COL_len]

max_len = max(list(A_B_C_S_SET['ROW_len'])+list(A_B_C_S_SET['COL_len']))

A_B_C_S_SET_rlen = A_B_C_S_SET[A_B_C_S_SET.ROW_len<=50]
A_B_C_S_SET_clen = A_B_C_S_SET_rlen[A_B_C_S_SET_rlen.COL_len<=50]

A_B_C_S_SET = A_B_C_S_SET_clen.reset_index(drop=True) # 


A_B_C_S_SET.columns = [
	'drug_row_CID', 'drug_col_CID', 'DrugCombCCLE', 'ROWCHECK', 'COLCHECK',
	   'ROW_CAN_SMILES', 'COL_CAN_SMILES', 'ROW_pert_id', 'ROW_BETA_sig_id',
	   'COL_pert_id', 'COL_BETA_sig_id', 'cid_cid_cell', 'sig_sig_cell','type', 'ROW_len',
	   'COL_len'
]





LINCS_ALL_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/'

# no filter ver 
BETA_BIND_MEAN = torch.load( LINCS_ALL_PATH + "10_24_sig_cell_mean.0620.pt")
BETA_BIND_M_SIG_df_CID = pd.read_csv(LINCS_ALL_PATH + '10_24_sig_cell_mean.0620.csv')


BETA_BIND_M_SIG_df_CID['CID__CELL'] = BETA_BIND_M_SIG_df_CID.CID.apply(lambda x : str(x)) + "__" + BETA_BIND_M_SIG_df_CID.CCLE_Name


# 여기에 새로 줍줍한거 들어가야함 

def get_LINCS_data(CID__CELL):
	bb_ind = BETA_BIND_M_SIG_df_CID[BETA_BIND_M_SIG_df_CID.CID_CELL == CID__CELL ].index.item()
	sig_ts = BETA_BIND_MEAN[bb_ind]
	#
	return sig_ts



A_B_C_S_syn = A_B_C_S[['cid_cid_cell','synergy_loewe']].drop_duplicates()
A_B_C_S_syn_ids = list(set(A_B_C_S_syn.cid_cid_cell))
A_B_C_S_syn_ids.sort()

A_B_C_S_syn_re = A_B_C_S_syn.groupby('cid_cid_cell').mean()


def get_synergy_data(cid_cid_cell):
	return A_B_C_S_syn_re.at[cid_cid_cell, 'synergy_loewe']



DC_ALL_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/'

all_chem_DF = pd.read_csv(DC_ALL_PATH+'DC_ALL_7555_ORDER.csv')
all_chem_feat_TS = torch.load(DC_ALL_PATH+'DC_ALL.MY_chem_feat.pt')
all_chem_feat_adj = torch.load(DC_ALL_PATH+'DC_ALL.MY_chem_adj.pt')




def check_drug_f_ts(CID) :
	INDEX = all_chem_DF[all_chem_DF.CID == CID].index.item()
	adj_pre = all_chem_feat_adj[INDEX]
	# adj_proc = adj_pre.long().to_sparse().indices()
	return all_chem_feat_TS[INDEX], adj_pre




MJ_DIR = '/st06/jiyeonH/13.DD_SESS/01.PRJ2/'

# original : MJ_request_ANS_old = pd.read_csv(MJ_DIR+'PRJ2_EXP_ccle_all_fugcn_hhhdt3_tvt.csv')
# 170 cell lines , 2984 CID


1) no target fill  # 10106  # 170 cell lines , 1343 CID
MJ_request_ANS = pd.read_csv(MJ_DIR+'PRJ2_EXP_ccle_all_fugcn_hhhdttar3.csv')

2) target fill # 15115 # 170 cell lines , 2984 CID
MJ_request_ANS = pd.read_csv(MJ_DIR+'PRJ2_EXP_ccle_all_fugcn_hhhdttf3.csv')



entrez_id = list(MJ_request_ANS['entrez_id'])
MJ_request_ANS = MJ_request_ANS.drop(['entrez_id','Unnamed: 0','CID__CELL'], axis =1)
MJ_request_ANS['entrez_id'] = entrez_id

ord = [list(MJ_request_ANS.entrez_id).index(a) for a in BETA_ENTREZ_ORDER] # ordering ok 
MJ_request_ANS_re = MJ_request_ANS.loc[ord] 


A_B_C_S_SET_MJ = A_B_C_S_SET[A_B_C_S_SET.ROWCHECK.isin(MJ_request_ANS.columns)]
A_B_C_S_SET_MJ = A_B_C_S_SET_MJ[A_B_C_S_SET_MJ.COLCHECK.isin(MJ_request_ANS.columns)]
A_B_C_S_SET_MJ = A_B_C_S_SET_MJ.reset_index(drop = True)



def get_MJ_data( CHECK ): 
	if CHECK in list(MJ_request_ANS_re.columns) :
		RES = MJ_request_ANS_re[CHECK]
		OX = 'O'
	else : 
		RES = [0]*349        ##############
		OX = 'X'
	return list(RES), OX



A_B_C_S_SET_UNIQ = copy.deepcopy(A_B_C_S_SET_MJ) # 613961 or 455856
A_B_C_S_SET_UNIQ['type'] = [ 'AXBO' if a == 'AOBX' else a for a in A_B_C_S_SET_UNIQ['type']]
A_B_C_S_SET_UNIQ_2 = A_B_C_S_SET_UNIQ[['cid_cid_cell','sig_sig_cell','type']].drop_duplicates()
A_B_C_S_SET_UNIQ_3 = A_B_C_S_SET_UNIQ_2[['cid_cid_cell','type']].drop_duplicates()
A_B_C_S_SET_UNIQ_3 = A_B_C_S_SET_UNIQ_3.reset_index(drop=True)


A_B_C_S_SET_UNIQ_3['CID_A'] = A_B_C_S_SET_UNIQ_3.cid_cid_cell.apply(lambda x : int(x.split('___')[0]))
A_B_C_S_SET_UNIQ_3['CID_B'] = A_B_C_S_SET_UNIQ_3.cid_cid_cell.apply(lambda x : int(x.split('___')[1]))
A_B_C_S_SET_UNIQ_3['CELL'] = A_B_C_S_SET_UNIQ_3.cid_cid_cell.apply(lambda x : x.split('___')[2])



max_len = 50


MY_chem_A_feat = torch.empty(size=(A_B_C_S_SET_UNIQ_3.shape[0], max_len, 64))
MY_chem_B_feat= torch.empty(size=(A_B_C_S_SET_UNIQ_3.shape[0], max_len, 64))
MY_chem_A_adj = torch.empty(size=(A_B_C_S_SET_UNIQ_3.shape[0], max_len, max_len))
MY_chem_B_adj= torch.empty(size=(A_B_C_S_SET_UNIQ_3.shape[0], max_len, max_len))
MY_syn =  torch.empty(size=(A_B_C_S_SET_UNIQ_3.shape[0],1))

MY_g_EXP_A = torch.empty(size=(A_B_C_S_SET_UNIQ_3.shape[0], 349, 1))##############
MY_g_EXP_B = torch.empty(size=(A_B_C_S_SET_UNIQ_3.shape[0], 349, 1))##############




Fail_ind = []
from datetime import datetime

for IND in range(0, A_B_C_S_SET_UNIQ_3.shape[0]): #  100
	if IND%100 == 0 : 
		print(str(IND)+'/'+str(A_B_C_S_SET_UNIQ_3.shape[0]) )
		Fail_ind
		datetime.now()
	#
	cid_cid_cell = A_B_C_S_SET_UNIQ_3.cid_cid_cell[IND]
	DrugA_CID = A_B_C_S_SET_UNIQ_3['CID_A'][IND]
	DrugB_CID = A_B_C_S_SET_UNIQ_3['CID_B'][IND]
	CELL = A_B_C_S_SET_UNIQ_3['CELL'][IND]
	dat_type = A_B_C_S_SET_UNIQ_3.type[IND]
	DrugA_CID_CELL = str(DrugA_CID) + '__' + CELL
	DrugB_CID_CELL = str(DrugB_CID) + '__' + CELL
	#
	k=1
	DrugA_Feat, DrugA_ADJ = check_drug_f_ts(DrugA_CID)
	DrugB_Feat, DrugB_ADJ = check_drug_f_ts(DrugB_CID)
	# 
	if dat_type == 'AOBO' :
		mean_ind_A = BETA_BIND_M_SIG_df_CID[BETA_BIND_M_SIG_df_CID.CID__CELL == DrugA_CID_CELL].index.item()
		EXP_A = BETA_BIND_MEAN[mean_ind_A]
		mean_ind_B = BETA_BIND_M_SIG_df_CID[BETA_BIND_M_SIG_df_CID.CID__CELL == DrugB_CID_CELL].index.item()
		EXP_B = BETA_BIND_MEAN[mean_ind_B]
	#
	else :
		DrugA_check = BETA_BIND_M_SIG_df_CID[BETA_BIND_M_SIG_df_CID.CID__CELL == DrugA_CID_CELL]
		DrugB_check = BETA_BIND_M_SIG_df_CID[BETA_BIND_M_SIG_df_CID.CID__CELL == DrugB_CID_CELL]
	#
		if len(DrugA_check) == 0 :
			EXP_A, OX = get_MJ_data(DrugA_CID_CELL)
			if 'X' in OX :
				Fail_ind.append(IND)
			EXP_A = torch.Tensor(EXP_A).unsqueeze(1)
		else : 
			mean_ind_A = BETA_BIND_M_SIG_df_CID[BETA_BIND_M_SIG_df_CID.CID__CELL == DrugA_CID_CELL].index.item()
			EXP_A = BETA_BIND_MEAN[mean_ind_A]
	#
		if len(DrugB_check) == 0 :
			EXP_B, OX = get_MJ_data(DrugB_CID_CELL)
			if 'X' in OX :
				Fail_ind.append(IND)
			EXP_B = torch.Tensor(EXP_B).unsqueeze(1)
		else : 
			mean_ind_B = BETA_BIND_M_SIG_df_CID[BETA_BIND_M_SIG_df_CID.CID__CELL == DrugB_CID_CELL].index.item()
			EXP_B = BETA_BIND_MEAN[mean_ind_B]
	#
	# 
	# 
	AB_SYN = get_synergy_data(cid_cid_cell)
	# 
	MY_chem_A_feat[IND] = torch.Tensor(DrugA_Feat)
	MY_chem_B_feat[IND] = torch.Tensor(DrugB_Feat)
	MY_chem_A_adj[IND] = torch.Tensor(DrugA_ADJ)
	MY_chem_B_adj[IND] = torch.Tensor(DrugB_ADJ)
	MY_g_EXP_A[IND] = torch.Tensor(EXP_A)
	MY_g_EXP_B[IND] = torch.Tensor(EXP_B)
	MY_syn[IND] = torch.Tensor([AB_SYN])


PRJ_NAME = 'M3V7_349_MISS2_FULL' # 1 no padding 
PRJ_NAME = 'M3V8_349_MISS2_FULL' # 2 with 0 padding 

SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V7_349_FULL/'
SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V8_349_FULL/'


torch.save(MY_chem_A_feat, SAVE_PATH+'{}.MY_chem_A_feat.pt'.format(PRJ_NAME))
torch.save(MY_chem_B_feat, SAVE_PATH+'{}.MY_chem_B_feat.pt'.format(PRJ_NAME))
torch.save(MY_chem_A_adj, SAVE_PATH+'{}.MY_chem_A_adj.pt'.format(PRJ_NAME))
torch.save(MY_chem_B_adj, SAVE_PATH+'{}.MY_chem_B_adj.pt'.format(PRJ_NAME))
torch.save(MY_g_EXP_A, SAVE_PATH+'{}.MY_g_EXP_A.pt'.format(PRJ_NAME))
torch.save(MY_g_EXP_B, SAVE_PATH+'{}.MY_g_EXP_B.pt'.format(PRJ_NAME))
torch.save(MY_syn, SAVE_PATH+'{}.MY_syn.pt'.format(PRJ_NAME))

A_B_C_S_SET_UNIQ_3.to_csv(SAVE_PATH+'{}.A_B_C_S_SET.csv'.format(PRJ_NAME))
A_B_C_S.to_csv(SAVE_PATH+'{}.A_B_C_S.csv'.format(PRJ_NAME))













##########################################
##########################################
##########################################

# 기준 index 

A_B_C_S_SET = copy.deepcopy(A_B_C_S_SET_UNIQ_3)
A_B_C_S = copy.deepcopy(A_B_C_S)



CCLE_PATH = '/st06/jiyeonH/13.DD_SESS/CCLE.22Q1/'

ccle_exp = pd.read_csv(CCLE_PATH+'CCLE_expression.csv', low_memory=False)

ccle_ori_col = list(ccle_exp.columns)
ccle_new_col =['DepMap_ID'] + [int(a.split(')')[0].split('(')[1]) for a in ccle_ori_col[1:]]

ccle_exp.columns = ccle_new_col

# ccle_mut = pd.read_csv(CCLE_PATH+'CCLE_mutations.csv', low_memory=False)
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
	ccle = A_B_C_S_SET.loc[i]['CELL']
	if ccle in ccle_names : 
		ccle_exp_df = ccle_exp3[ccle_exp3.DrugCombCCLE==ccle][BETA_ENTREZ_ORDER]
		ccle_exp_vector = ccle_exp_df.values[0].tolist()
		cell_basal_exp_list.append(ccle_exp_vector)
	else : # 'TC32_BONE', 'DU145_PROSTATE' -> 0 으로 진행하게 됨. public expression 없음 참고해야함. 
		print(ccle)
		ccle_exp_vector = [0]*349
		cell_basal_exp_list.append(ccle_exp_vector)




cell_base_tensor = torch.Tensor(cell_basal_exp_list)

torch.save(cell_base_tensor, SAVE_PATH+'{}.MY_CellBase.pt'.format(PRJ_NAME))


no_public_exp = list(set(A_B_C_S_SET['CELL']) - set(ccle_names))
no_p_e_list = ['X' if cell in no_public_exp else 'O' for cell in list(A_B_C_S_SET.CELL)]

A_B_C_S_SET_ADD = copy.deepcopy(A_B_C_S_SET)
A_B_C_S_SET_ADD = A_B_C_S_SET_ADD.reset_index(drop = True)
A_B_C_S_SET_ADD['Basal_Exp'] = no_p_e_list



abcs_synergy = A_B_C_S[['cid_cid_cell','synergy_loewe']].drop_duplicates()
abcs_synergy['TF']  = abcs_synergy.synergy_loewe.apply(lambda x : x>0)
abcs_syn_group = abcs_synergy.groupby('cid_cid_cell')

ids = []
lensets = []
for idd, groupp in abcs_syn_group:
	ids.append(idd)
	if len(set(groupp['TF'])) ==1 : # consistency check 
		lensets.append('O')
	else :
		lensets.append('X')

synOX = pd.DataFrame({'cid_cid_cell' : ids, 'SYN_OX' : lensets})


A_B_C_S_SET_ADD = pd.merge(A_B_C_S_SET_ADD, synOX, on = 'cid_cid_cell', how ='left')




A_B_C_S_SET_CIDS = list(set(list(A_B_C_S_SET_ADD.CID_A)+list(A_B_C_S_SET_ADD.CID_B)))
A_B_C_S_SET_CIDS.sort()
gene_ids = BETA_ENTREZ_ORDER



# TARGET (1) time consuming 
# TARGET (1)
# TARGET (1)

TARGET_PATH = '/st06/jiyeonH/13.DD_SESS/01.PRJ2/'
TARGET_DB = pd.read_csv(TARGET_PATH+'TARGET_CID_ENTREZ.csv', sep ='\t', index_col = 0)

TARGET_DB_RE = TARGET_DB[TARGET_DB.CID_RE.isin(A_B_C_S_SET_CIDS)]
TARGET_DB_RE = TARGET_DB_RE[TARGET_DB_RE.ENTREZ_RE.isin(gene_ids)]


target_cids = list(set(TARGET_DB_RE.CID))
target_cids.sort()
gene_ids = BETA_ENTREZ_ORDER



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
	CID_A = A_B_C_S_SET_ADD.iloc[IND,]['CID_A']
	CID_B = A_B_C_S_SET_ADD.iloc[IND,]['CID_B']
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
	if IND%1000 == 0 :
		print(str(IND)+'/'+str(A_B_C_S_SET_ADD.shape[0]) )
		datetime.now()
	CID_A = A_B_C_S_SET_ADD.iloc[IND,]['CID_A']
	CID_B = A_B_C_S_SET_ADD.iloc[IND,]['CID_B']
	if (CID_A in target_cids) & (CID_B in target_cids) : 
		T1_OX_list.append('O')
	else : 
		T1_OX_list.append('X')
	

A_B_C_S_SET_ADD['T1OX']=T1_OX_list








# Tanimoto filter -> 이거 각각 CID 로 바꿔주기 


ABCS_ori_CIDs = list(set(list(A_B_C_S.drug_row_CID) + list(A_B_C_S.drug_col_CID))) # 172 
ABCS_ori_CIDs.sort()
ABCS_FILT_CIDS = list(set(list(A_B_C_S_SET_ADD.CID_A) + list(A_B_C_S_SET_ADD.CID_B))) # 172 
ABCS_FILT_CIDS.sort()
ABCS_ori_SMILEs = list(set(list(A_B_C_S.ROW_CAN_SMILES) + list(A_B_C_S.COL_CAN_SMILES))) # 171
ABCS_ori_SMILEs.sort()


PC_check = for_CAN_smiles[for_CAN_smiles.CID.isin(ABCS_ori_CIDs)]

from rdkit import DataStructs

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

row_cids = list(A_B_C_S_SET_ADD.CID_A)
col_cids = list(A_B_C_S_SET_ADD.CID_B)

tani_01 = list(means_df[(means_df['dot_col'] == 'IN') & (means_df['over0.1'] == 'IN') ]['CIDs'])
tani_02 = list(means_df[(means_df['dot_col'] == 'IN') & (means_df['over0.2'] == 'IN') ]['CIDs'])
tani_Q = list(means_df[(means_df['dot_col'] == 'IN') & (means_df['overQ'] == 'IN') ]['CIDs'])


tani_01_result = []
tani_02_result = []
tani_Q_result = []
for IND in range(A_B_C_S_SET_ADD.shape[0]) :
	if IND%1000 == 0 :
		print(str(IND)+'/'+str(A_B_C_S_SET.shape[0]) )
		datetime.now()
	CID_A = A_B_C_S_SET.iloc[IND,]['CID_A']
	CID_B = A_B_C_S_SET.iloc[IND,]['CID_B']
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
	

A_B_C_S_SET_ADD['tani_01'] = tani_01_result
A_B_C_S_SET_ADD['tani_02'] = tani_02_result
A_B_C_S_SET_ADD['tani_Q'] = tani_Q_result

A_B_C_S_SET_ADD['tani_A_01'] = A_B_C_S_SET.CID_A.apply(lambda x : 'O' if x in tani_01 else 'X')
A_B_C_S_SET_ADD['tani_B_01'] = A_B_C_S_SET.CID_B.apply(lambda x : 'O' if x in tani_01 else 'X')

A_B_C_S_SET_ADD['tani_A_02'] = A_B_C_S_SET.CID_A.apply(lambda x : 'O' if x in tani_02 else 'X')
A_B_C_S_SET_ADD['tani_B_02'] = A_B_C_S_SET.CID_B.apply(lambda x : 'O' if x in tani_02 else 'X')

A_B_C_S_SET_ADD['tani_A_Q'] = A_B_C_S_SET.CID_A.apply(lambda x : 'O' if x in tani_Q else 'X')
A_B_C_S_SET_ADD['tani_B_Q'] = A_B_C_S_SET.CID_B.apply(lambda x : 'O' if x in tani_Q else 'X')





# final check for oneil set 
# final check for oneil set 
# final check for oneil set 

DC_oneil = pd.read_csv('/st06/jiyeonH/11.TOX/DR_SPRING/trials/DC_ONEIL_ALL.csv', sep= '\t')

o_1 = [str(int(a)) for a in DC_oneil.drug_row_CID]
o_2 = [str(int(a)) for a in DC_oneil.drug_col_CID]
o_c = [a for a in DC_oneil.DrugCombCCLE]
DC_oneil_set = [o_1[a]+'___'+o_2[a]+'___'+o_c[a] for a in range(DC_oneil.shape[0])] + [o_2[a]+'___'+o_1[a]+'___'+o_c[a] for a in range(DC_oneil.shape[0])]


abcs_set = list(A_B_C_S_SET_ADD.cid_cid_cell)

# 생각보다 시간 걸림. row 많아서 
oneil_check = []
for a in range(A_B_C_S_SET_ADD.shape[0]) :
	if a%1000 == 0 : 
		print(a)
	if abcs_set[a] in DC_oneil_set :
		oneil_check.append('O')
	else :
		oneil_check.append('X')
	

A_B_C_S_SET_ADD['ONEIL'] = oneil_check




# SM_C_CHECK 위해서 smiles 붙여주기 

for_CAN_smiles.columns = ['CID_A','ROW_CAN_SMILES']
A_B_C_S_SET_ADD = pd.merge(A_B_C_S_SET_ADD, for_CAN_smiles, on='CID_A', how ='left' )
for_CAN_smiles.columns = ['CID_B','COL_CAN_SMILES']
A_B_C_S_SET_ADD = pd.merge(A_B_C_S_SET_ADD, for_CAN_smiles, on='CID_B', how ='left' )
for_CAN_smiles.columns = ['CID','CAN_SMILES']




# sig correlation merge 하기 
sig_cor = BETA_BIND_M_SIG_df_CID[['corr_Pmean','corr_Smean','CID__CELL']]

A_B_C_S_SET_ADD['CID_A_CELL'] =  A_B_C_S_SET_ADD.CID_A.apply(lambda x : str(int(x))) + '__' + A_B_C_S_SET_ADD.CELL
A_B_C_S_SET_ADD['CID_B_CELL'] =  A_B_C_S_SET_ADD.CID_B.apply(lambda x : str(int(x))) + '__' + A_B_C_S_SET_ADD.CELL

sig_cor.columns = ['A_PC','A_SC','CID_A_CELL']
A_B_C_S_SET_ADD_Acor = pd.merge(A_B_C_S_SET_ADD, sig_cor, on = 'CID_A_CELL', how = 'left')

sig_cor.columns = ['B_PC','B_SC','CID_B_CELL']
A_B_C_S_SET_ADD_Bcor = pd.merge(A_B_C_S_SET_ADD_Acor, sig_cor, on = 'CID_B_CELL', how = 'left')



A_B_C_S_SET_ADD_Bcor.to_csv(SAVE_PATH+'{}.A_B_C_S_SET_ADD.csv'.format(PRJ_NAME))




A_B_C_S_SET_ADD_Bcor.columns =['cid_cid_cell', 'type', 'CID_A', 'CID_B', 'CELL', 'Basal_Exp', 'SYN_OX',
       'tani_01', 'tani_02', 'tani_Q', 'tani_A_01', 'tani_B_01', 'tani_A_02',
       'tani_B_02', 'tani_A_Q', 'tani_B_Q', 'ONEIL', 'ROW_CAN_SMILES',
       'COL_CAN_SMILES', 'CID_A_CELL', 'CID_B_CELL', 'T1OX', 'A_PC', 'A_SC',
       'B_PC', 'B_SC']







############### 다 안돌아갔어도 중간 체크 




import pandas as pd 
import matplotlib.pyplot as plt
from ray.tune import ExperimentAnalysis
from ray.tune import Analysis
import pickle
import math
import torch
import os 
import copy
import numpy as np


MJ_NAME = 'M3V8'
PPI_NAME = '349'
MISS_NAME = 'MIS2'

W_NAME = 'W402'
WORK_NAME = 'WORK_402' # 349
WORK_DATE = '23.08.19' # 349

anal_dir = "/home01/k040a01/ray_results/PRJ02.{}.{}.{}.{}.{}".format(WORK_DATE, MJ_NAME,  WORK_NAME, PPI_NAME, MISS_NAME)
list_dir = os.listdir(anal_dir)
exp_json = [a for a in list_dir if 'experiment_state' in a]
exp_json
anal_df = ExperimentAnalysis(os.path.join(anal_dir, exp_json[1]))
# 301 : 1  / 302 : 0  / 402 : 1 


ANA_DF_1 = anal_df.dataframe()
ANA_ALL_DF_1 = anal_df.trial_dataframes

np.max(ANA_ALL_DF_1[ANA_DF_1.at[8,'logdir']]['AV_V_PC'])


# 203 config 

OLD_PATH = '/home01/k040a01/02.M3V6/M3V6_W202_349_MIS2'
ANA_DF_CSV = pd.read_csv(os.path.join(OLD_PATH,'RAY_ANA_DF.{}.csv'.format('M3V6_W202_349_MIS2')))

with open(file='{}/RAY_ANA_DF.M3V6_W202_349_MIS2.pickle'.format(OLD_PATH), mode='rb') as f:
	ANA_all = pickle.load(f)


import numpy as np
TOT_min = np.Inf
TOT_key = ""
for key in ANA_all.keys():
	trial_min = min(ANA_all[key]['AV_V_LS'])
	if trial_min < TOT_min :
		TOT_min = trial_min
		TOT_key = key
#

mini_df = ANA_all[TOT_key]

my_config = ANA_DF_CSV[ANA_DF_CSV.trial_id=='1cf5052a'] # 349 


# 

import numpy as np
TOT_max = -np.Inf
TOT_key = ""
for key in ANA_all.keys():
	trial_max = np.max(ANA_all[key]['AV_V_PC'])
	if trial_max > TOT_max :
		TOT_max = trial_max
		TOT_key = key
#

mini_df = ANA_all[TOT_key]

round(np.max(mini_df.AV_V_PC), 4)
round(np.min(mini_df.AV_V_LS), 4)



############################





#########################################
################# GPU ###################
################# GPU ###################
################# GPU ###################
#########################################


import pandas as pd 
import matplotlib.pyplot as plt
from ray.tune import ExperimentAnalysis
from ray.tune import Analysis
import pickle
import math
import torch
import os 
import copy

MJ_NAME = 'M3V8'
PPI_NAME = '349'
MISS_NAME = 'MIS2'

W_NAME = 'W402'
WORK_NAME = 'WORK_402' # 349
WORK_DATE = '23.08.19' # 349


anal_dir = "/home01/k040a01/ray_results/PRJ02.{}.{}.{}.{}.{}".format(WORK_DATE, MJ_NAME,  WORK_NAME, PPI_NAME, MISS_NAME)


list_dir = os.listdir(anal_dir)
exp_json = [a for a in list_dir if 'experiment_state' in a]
exp_json
anal_df = ExperimentAnalysis(os.path.join(anal_dir,exp_json[1]))


ANA_DF = anal_df.dataframe()
ANA_ALL_DF = anal_df.trial_dataframes



# W321 VER
ANA_DF.to_csv('/home01/k040a01/02.M3V8/{}_{}_{}_{}/RAY_ANA_DF.{}_{}_{}_{}.csv'.format(MJ_NAME, W_NAME, PPI_NAME, MISS_NAME,  MJ_NAME, W_NAME, PPI_NAME, MISS_NAME))
import pickle
with open("/home01/k040a01/02.M3V8/{}_{}_{}_{}/RAY_ANA_DF.{}_{}_{}_{}.pickle".format(MJ_NAME, W_NAME, PPI_NAME, MISS_NAME, MJ_NAME, W_NAME, PPI_NAME, MISS_NAME), "wb") as fp:
	pickle.dump(ANA_ALL_DF,fp) 

'/home01/k020a01/02.M3V8/{}_{}_{}_{}/RAY_ANA_DF.{}_{}_{}_{}.csv'.format(MJ_NAME, W_NAME, PPI_NAME, MISS_NAME,  MJ_NAME, W_NAME, PPI_NAME, MISS_NAME)
"/home01/k020a01/02.M3V8/{}_{}_{}_{}/RAY_ANA_DF.{}_{}_{}_{}.pickle".format(MJ_NAME, W_NAME, PPI_NAME, MISS_NAME, MJ_NAME, W_NAME, PPI_NAME, MISS_NAME)



# 1) best final loss 
min(ANA_DF.sort_values('AV_V_LS')['AV_V_LS'])
DF_KEY = list(ANA_DF.sort_values('AV_V_LS')['logdir'])[0]
DF_KEY

# get /model.pth M1_model.pth


#  2) best final's best chck 
mini_df = ANA_ALL_DF[DF_KEY]
cck_num = mini_df[mini_df.AV_V_LS==min(mini_df.AV_V_LS)].index.item()
checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
TOPVAL_PATH = DF_KEY + checkpoint
TOPVAL_PATH

# get /checkpoint M2_model



# 3) total checkpoint best 
import numpy as np
TOT_min = np.Inf
TOT_key = ""
for key in ANA_ALL_DF.keys():
	trial_min = min(ANA_ALL_DF[key]['AV_V_LS'])
	if trial_min < TOT_min :
		TOT_min = trial_min
		TOT_key = key

mini_df = ANA_ALL_DF[TOT_key]
cck_num = mini_df[mini_df.AV_V_LS==min(mini_df.AV_V_LS)].index.item()
checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
TOPVAL_PATH = TOT_key + checkpoint
TOPVAL_PATH

# get /checkpoint M4_model




# 4) correlation best 
max_cor = max(ANA_DF.sort_values('AV_V_SC')['AV_V_SC'])
DF_KEY = ANA_DF[ANA_DF.AV_V_SC == max_cor]['logdir'].item()
print('best SCOR final', flush=True)
print(DF_KEY, flush=True)

# get /model.pth C1_model.pth






# 5) correlation best's best corr 
mini_df = ANA_ALL_DF[DF_KEY]
cck_num = mini_df[mini_df.AV_V_SC==max(mini_df.AV_V_SC)].index.item()
checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
TOPCOR_PATH = DF_KEY + checkpoint
TOPCOR_PATH

# get /checkpoint C2_model.pth





# 6) correlation best of all 
import numpy as np
TOT_max = -np.Inf
TOT_key = ""
for key in ANA_ALL_DF.keys():
	trial_max = max(ANA_ALL_DF[key]['AV_V_SC'])
	if trial_max > TOT_max :
		TOT_max = trial_max
		TOT_key = key
#

mini_df = ANA_ALL_DF[TOT_key]
cck_num =mini_df[mini_df.AV_V_SC==max(mini_df.AV_V_SC)].index.item()
checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
TOPCOR_PATH = TOT_key + checkpoint
TOPCOR_PATH

# get /checkpoint C4_model.pth








# 5CV check 

import pandas as pd 
import matplotlib.pyplot as plt
from ray.tune import ExperimentAnalysis
from ray.tune import Analysis
import pickle
import math
import torch
import os 
import copy
import numpy as np

MJ_NAME = 'M3V8'
PPI_NAME = '349'
MISS_NAME = 'MIS2'

W_NAME = 'W403'
WORK_NAME = 'WORK_403' # 349 WORK_203_3
WORK_DATE = '23.08.27' # 349


anal_dir = "/home01/k040a01/ray_results/PRJ02.{}.{}.{}.{}.{}".format(WORK_DATE, MJ_NAME,  WORK_NAME, PPI_NAME, MISS_NAME)
#     anal_dir = '/home01/k040a01/ray_results/PRJ02.23.06.13.M3V5.WORK_37.349.MIS22/'
list_dir = os.listdir(anal_dir)
exp_json = [a for a in list_dir if 'experiment_state' in a]
exp_json
anal_df = ExperimentAnalysis(os.path.join(anal_dir, exp_json[0]))

ANA_DF_1 = anal_df.dataframe()
ANA_ALL_DF_1 = anal_df.trial_dataframes

ANA_DF = ANA_DF_1

ANA_DF = ANA_DF.sort_values('config/CV')
ANA_DF.index = [0,1,2,3,4]
ANA_ALL_DF = ANA_ALL_DF_1



ANA_DF.to_csv('/home01/k040a01/02.M3V8/{}_{}_{}_{}/RAY_ANA_DF.{}_{}_{}_{}.csv'.format(MJ_NAME, W_NAME, PPI_NAME, MISS_NAME,  MJ_NAME, W_NAME, PPI_NAME, MISS_NAME))
import pickle
with open("/home01/k040a01/02.M3V8/{}_{}_{}_{}/RAY_ANA_DF.{}_{}_{}_{}.pickle".format(MJ_NAME, W_NAME, PPI_NAME, MISS_NAME, MJ_NAME, W_NAME, PPI_NAME, MISS_NAME), "wb") as fp:
	pickle.dump(ANA_ALL_DF,fp) 

'/home01/k040a01/02.M3V8/{}_{}_{}_{}/RAY_ANA_DF.{}_{}_{}_{}.csv'.format(MJ_NAME, W_NAME, PPI_NAME, MISS_NAME,  MJ_NAME, W_NAME, PPI_NAME, MISS_NAME)
"/home01/k040a01/02.M3V8/{}_{}_{}_{}/RAY_ANA_DF.{}_{}_{}_{}.pickle".format(MJ_NAME, W_NAME, PPI_NAME, MISS_NAME, MJ_NAME, W_NAME, PPI_NAME, MISS_NAME)


limit = 1000

cv0_key = ANA_DF['logdir'][0] ;	cv1_key = ANA_DF['logdir'][1]; 	cv2_key = ANA_DF['logdir'][2] ;	cv3_key = ANA_DF['logdir'][3];	cv4_key = ANA_DF['logdir'][4]

epc_T_LS_mean = np.mean([ANA_ALL_DF[cv0_key]['T_LS'][0:limit], ANA_ALL_DF[cv1_key]['T_LS'][0:limit],ANA_ALL_DF[cv2_key]['T_LS'][0:limit], ANA_ALL_DF[cv3_key]['T_LS'][0:limit], ANA_ALL_DF[cv4_key]['T_LS'][0:limit]], axis = 0)
epc_T_LS_std = np.std([ANA_ALL_DF[cv0_key]['T_LS'][0:limit], ANA_ALL_DF[cv1_key]['T_LS'][0:limit],ANA_ALL_DF[cv2_key]['T_LS'][0:limit], ANA_ALL_DF[cv3_key]['T_LS'][0:limit], ANA_ALL_DF[cv4_key]['T_LS'][0:limit]], axis = 0)

epc_T_PC_mean = np.mean([ANA_ALL_DF[cv0_key]['T_PC'][0:limit], ANA_ALL_DF[cv1_key]['T_PC'][0:limit],ANA_ALL_DF[cv2_key]['T_PC'][0:limit], ANA_ALL_DF[cv3_key]['T_PC'][0:limit], ANA_ALL_DF[cv4_key]['T_PC'][0:limit]], axis = 0)
epc_T_PC_std = np.std([ANA_ALL_DF[cv0_key]['T_PC'][0:limit], ANA_ALL_DF[cv1_key]['T_PC'][0:limit],ANA_ALL_DF[cv2_key]['T_PC'][0:limit], ANA_ALL_DF[cv3_key]['T_PC'][0:limit], ANA_ALL_DF[cv4_key]['T_PC'][0:limit]], axis = 0)

epc_T_SC_mean = np.mean([ANA_ALL_DF[cv0_key]['T_SC'][0:limit], ANA_ALL_DF[cv1_key]['T_SC'][0:limit],ANA_ALL_DF[cv2_key]['T_SC'][0:limit], ANA_ALL_DF[cv3_key]['T_SC'][0:limit], ANA_ALL_DF[cv4_key]['T_SC'][0:limit]], axis = 0)
epc_T_SC_std = np.std([ANA_ALL_DF[cv0_key]['T_SC'][0:limit], ANA_ALL_DF[cv1_key]['T_SC'][0:limit],ANA_ALL_DF[cv2_key]['T_SC'][0:limit], ANA_ALL_DF[cv3_key]['T_SC'][0:limit], ANA_ALL_DF[cv4_key]['T_SC'][0:limit]], axis = 0)

epc_V_LS_mean = np.mean([ANA_ALL_DF[cv0_key]['V_LS'][0:limit], ANA_ALL_DF[cv1_key]['V_LS'][0:limit],ANA_ALL_DF[cv2_key]['V_LS'][0:limit], ANA_ALL_DF[cv3_key]['V_LS'][0:limit], ANA_ALL_DF[cv4_key]['V_LS'][0:limit]], axis = 0)
epc_V_LS_std = np.std([ANA_ALL_DF[cv0_key]['V_LS'][0:limit], ANA_ALL_DF[cv1_key]['V_LS'][0:limit],ANA_ALL_DF[cv2_key]['V_LS'][0:limit], ANA_ALL_DF[cv3_key]['V_LS'][0:limit], ANA_ALL_DF[cv4_key]['V_LS'][0:limit]], axis = 0)

epc_V_PC_mean = np.mean([ANA_ALL_DF[cv0_key]['V_PC'][0:limit], ANA_ALL_DF[cv1_key]['V_PC'][0:limit],ANA_ALL_DF[cv2_key]['V_PC'][0:limit], ANA_ALL_DF[cv3_key]['V_PC'][0:limit], ANA_ALL_DF[cv4_key]['V_PC'][0:limit]], axis = 0)
epc_V_PC_std = np.std([ANA_ALL_DF[cv0_key]['V_PC'][0:limit], ANA_ALL_DF[cv1_key]['V_PC'][0:limit],ANA_ALL_DF[cv2_key]['V_PC'][0:limit], ANA_ALL_DF[cv3_key]['V_PC'][0:limit], ANA_ALL_DF[cv4_key]['V_PC'][0:limit]], axis = 0)

epc_V_SC_mean = np.mean([ANA_ALL_DF[cv0_key]['V_SC'][0:limit], ANA_ALL_DF[cv1_key]['V_SC'][0:limit],ANA_ALL_DF[cv2_key]['V_SC'][0:limit], ANA_ALL_DF[cv3_key]['V_SC'][0:limit], ANA_ALL_DF[cv4_key]['V_SC'][0:limit]], axis = 0)
epc_V_SC_std = np.std([ANA_ALL_DF[cv0_key]['V_SC'][0:limit], ANA_ALL_DF[cv1_key]['V_SC'][0:limit],ANA_ALL_DF[cv2_key]['V_SC'][0:limit], ANA_ALL_DF[cv3_key]['V_SC'][0:limit], ANA_ALL_DF[cv4_key]['V_SC'][0:limit]], axis = 0)


epc_result = pd.DataFrame({
	'T_LS_mean' : epc_T_LS_mean, 'T_PC_mean' : epc_T_PC_mean, 'T_SC_mean' : epc_T_SC_mean, 
	'T_LS_std' : epc_T_LS_std, 'T_PC_std' : epc_T_PC_std, 'T_SC_std' : epc_T_SC_std, 
	'V_LS_mean' : epc_V_LS_mean, 'V_PC_mean' : epc_V_PC_mean, 'V_SC_mean' : epc_V_SC_mean, 
	'V_LS_std' : epc_V_LS_std, 'V_PC_std' : epc_V_PC_std, 'V_SC_std' : epc_V_SC_std,
})

epc_result[[
    'T_LS_mean', 'T_LS_std', 'T_PC_mean', 'T_PC_std',
    'T_SC_mean','T_SC_std', 'V_LS_mean', 'V_LS_std', 
    'V_PC_mean', 'V_PC_std','V_SC_mean','V_SC_std']].to_csv("/home01/k040a01/02.M3V6/{}_{}_{}_{}/RAY_ANA_DF.{}_{}_{}_{}.resDF".format(MJ_NAME, W_NAME, PPI_NAME, MISS_NAME, MJ_NAME, W_NAME, PPI_NAME, MISS_NAME))

"/home01/k040a01/02.M3V6/{}_{}_{}_{}/RAY_ANA_DF.{}_{}_{}_{}.resDF".format(MJ_NAME, W_NAME, PPI_NAME, MISS_NAME, MJ_NAME, W_NAME, PPI_NAME, MISS_NAME)
        


1) min loss

min(epc_result.sort_values('V_LS_mean')['V_LS_mean']) ; min_VLS = min(epc_result.sort_values('V_LS_mean')['V_LS_mean'])
KEY_EPC = epc_result[epc_result.V_LS_mean == min_VLS].index.item()
checkpoint = "/checkpoint_"+str(KEY_EPC).zfill(6)
VLS_cv0_PATH = cv0_key + checkpoint
VLS_cv0_PATH
VLS_cv1_PATH = cv1_key + checkpoint
VLS_cv1_PATH
VLS_cv2_PATH = cv2_key + checkpoint
VLS_cv2_PATH
VLS_cv3_PATH = cv3_key + checkpoint
VLS_cv3_PATH
VLS_cv4_PATH = cv4_key + checkpoint
VLS_cv4_PATH


KEY_EPC
round(epc_result.loc[KEY_EPC].V_LS_mean,4)
round(epc_result.loc[KEY_EPC].V_LS_std,4)




get /checkpoint VLS_CV_0_model.pth 
get /checkpoint VLS_CV_1_model.pth 
get /checkpoint VLS_CV_2_model.pth 
get /checkpoint VLS_CV_3_model.pth 
get /checkpoint VLS_CV_4_model.pth 


get /home01/k040a01/ray_results/PRJ02.23.08.27.M3V8.WORK_403.349.MIS2/RAY_MY_train_1a4d8_00000_0_CV=0,G_chem_hdim=32,G_chem_layer=3,G_exp_hdim=32,G_exp_layer=3,batch_size=512,dropout_1=0.1000,dropout__2023-08-27_16-34-06/checkpoint_000992/checkpoint VLS_CV_0_model.pth 
get /home01/k040a01/ray_results/PRJ02.23.08.27.M3V8.WORK_403.349.MIS2/RAY_MY_train_1a4d8_00001_1_CV=1,G_chem_hdim=32,G_chem_layer=3,G_exp_hdim=32,G_exp_layer=3,batch_size=512,dropout_1=0.1000,dropout__2023-08-27_16-34-09/checkpoint_000992/checkpoint VLS_CV_1_model.pth 
get /home01/k040a01/ray_results/PRJ02.23.08.27.M3V8.WORK_403.349.MIS2/RAY_MY_train_1a4d8_00002_2_CV=2,G_chem_hdim=32,G_chem_layer=3,G_exp_hdim=32,G_exp_layer=3,batch_size=512,dropout_1=0.1000,dropout__2023-08-27_16-34-09/checkpoint_000992/checkpoint VLS_CV_2_model.pth 
get /home01/k040a01/ray_results/PRJ02.23.08.27.M3V8.WORK_403.349.MIS2/RAY_MY_train_1a4d8_00003_3_CV=3,G_chem_hdim=32,G_chem_layer=3,G_exp_hdim=32,G_exp_layer=3,batch_size=512,dropout_1=0.1000,dropout__2023-08-27_16-34-09/checkpoint_000992/checkpoint VLS_CV_3_model.pth 
get /home01/k040a01/ray_results/PRJ02.23.08.27.M3V8.WORK_403.349.MIS2/RAY_MY_train_1a4d8_00004_4_CV=4,G_chem_hdim=32,G_chem_layer=3,G_exp_hdim=32,G_exp_layer=3,batch_size=512,dropout_1=0.1000,dropout__2023-08-27_16-34-09/checkpoint_000992/checkpoint VLS_CV_4_model.pth 


2) PC best 

epc_result.sort_values('V_PC_mean', ascending = False) 
max(epc_result['V_PC_mean']); max_VPC = max(epc_result['V_PC_mean'])
KEY_EPC = epc_result[epc_result.V_PC_mean == max_VPC].index.item()
checkpoint = "/checkpoint_"+str(KEY_EPC).zfill(6)
VPC_cv0_PATH = cv0_key + checkpoint
VPC_cv0_PATH
VPC_cv1_PATH = cv1_key + checkpoint
VPC_cv1_PATH
VPC_cv2_PATH = cv2_key + checkpoint
VPC_cv2_PATH
VPC_cv3_PATH = cv3_key + checkpoint
VPC_cv3_PATH
VPC_cv4_PATH = cv4_key + checkpoint
VPC_cv4_PATH


KEY_EPC
round(epc_result.loc[KEY_EPC].V_PC_mean,4)
round(epc_result.loc[KEY_EPC].V_PC_std,4)




get /checkpoint VPC_CV_0_model.pth 
get /checkpoint VPC_CV_1_model.pth 
get /checkpoint VPC_CV_2_model.pth 
get /checkpoint VPC_CV_3_model.pth 
get /checkpoint VPC_CV_4_model.pth 


get /home01/k040a01/ray_results/PRJ02.23.08.27.M3V8.WORK_403.349.MIS2/RAY_MY_train_1a4d8_00000_0_CV=0,G_chem_hdim=32,G_chem_layer=3,G_exp_hdim=32,G_exp_layer=3,batch_size=512,dropout_1=0.1000,dropout__2023-08-27_16-34-06/checkpoint_000939/checkpoint VPC_CV_0_model.pth 
get /home01/k040a01/ray_results/PRJ02.23.08.27.M3V8.WORK_403.349.MIS2/RAY_MY_train_1a4d8_00001_1_CV=1,G_chem_hdim=32,G_chem_layer=3,G_exp_hdim=32,G_exp_layer=3,batch_size=512,dropout_1=0.1000,dropout__2023-08-27_16-34-09/checkpoint_000939/checkpoint VPC_CV_1_model.pth 
get /home01/k040a01/ray_results/PRJ02.23.08.27.M3V8.WORK_403.349.MIS2/RAY_MY_train_1a4d8_00002_2_CV=2,G_chem_hdim=32,G_chem_layer=3,G_exp_hdim=32,G_exp_layer=3,batch_size=512,dropout_1=0.1000,dropout__2023-08-27_16-34-09/checkpoint_000939/checkpoint VPC_CV_2_model.pth 
get /home01/k040a01/ray_results/PRJ02.23.08.27.M3V8.WORK_403.349.MIS2/RAY_MY_train_1a4d8_00003_3_CV=3,G_chem_hdim=32,G_chem_layer=3,G_exp_hdim=32,G_exp_layer=3,batch_size=512,dropout_1=0.1000,dropout__2023-08-27_16-34-09/checkpoint_000939/checkpoint VPC_CV_3_model.pth 
get /home01/k040a01/ray_results/PRJ02.23.08.27.M3V8.WORK_403.349.MIS2/RAY_MY_train_1a4d8_00004_4_CV=4,G_chem_hdim=32,G_chem_layer=3,G_exp_hdim=32,G_exp_layer=3,batch_size=512,dropout_1=0.1000,dropout__2023-08-27_16-34-09/checkpoint_000939/checkpoint VPC_CV_4_model.pth 






3) SC best 

epc_result.sort_values('V_SC_mean', ascending = False) 
max(epc_result['V_SC_mean']); max_VSC = max(epc_result['V_SC_mean'])
KEY_EPC = epc_result[epc_result.V_SC_mean == max_VSC].index.item()
checkpoint = "/checkpoint_"+str(KEY_EPC).zfill(6)
VSC_cv0_PATH = cv0_key + checkpoint
VSC_cv0_PATH
VSC_cv1_PATH = cv1_key + checkpoint
VSC_cv1_PATH
VSC_cv2_PATH = cv2_key + checkpoint
VSC_cv2_PATH
VSC_cv3_PATH = cv3_key + checkpoint
VSC_cv3_PATH
VSC_cv4_PATH = cv4_key + checkpoint
VSC_cv4_PATH

KEY_EPC
round(epc_result.loc[KEY_EPC].V_SC_mean,4)
round(epc_result.loc[KEY_EPC].V_SC_std,4)




get /checkpoint VSC_CV_0_model.pth 
get /checkpoint VSC_CV_1_model.pth 
get /checkpoint VSC_CV_2_model.pth 
get /checkpoint VSC_CV_3_model.pth 
get /checkpoint VSC_CV_4_model.pth 


get /home01/k040a01/ray_results/PRJ02.23.08.27.M3V8.WORK_403.349.MIS2/RAY_MY_train_1a4d8_00000_0_CV=0,G_chem_hdim=32,G_chem_layer=3,G_exp_hdim=32,G_exp_layer=3,batch_size=512,dropout_1=0.1000,dropout__2023-08-27_16-34-06/checkpoint_000996/checkpoint VSC_CV_0_model.pth 
get /home01/k040a01/ray_results/PRJ02.23.08.27.M3V8.WORK_403.349.MIS2/RAY_MY_train_1a4d8_00001_1_CV=1,G_chem_hdim=32,G_chem_layer=3,G_exp_hdim=32,G_exp_layer=3,batch_size=512,dropout_1=0.1000,dropout__2023-08-27_16-34-09/checkpoint_000996/checkpoint VSC_CV_1_model.pth 
get /home01/k040a01/ray_results/PRJ02.23.08.27.M3V8.WORK_403.349.MIS2/RAY_MY_train_1a4d8_00002_2_CV=2,G_chem_hdim=32,G_chem_layer=3,G_exp_hdim=32,G_exp_layer=3,batch_size=512,dropout_1=0.1000,dropout__2023-08-27_16-34-09/checkpoint_000996/checkpoint VSC_CV_2_model.pth 
get /home01/k040a01/ray_results/PRJ02.23.08.27.M3V8.WORK_403.349.MIS2/RAY_MY_train_1a4d8_00003_3_CV=3,G_chem_hdim=32,G_chem_layer=3,G_exp_hdim=32,G_exp_layer=3,batch_size=512,dropout_1=0.1000,dropout__2023-08-27_16-34-09/checkpoint_000996/checkpoint VSC_CV_3_model.pth 
get /home01/k040a01/ray_results/PRJ02.23.08.27.M3V8.WORK_403.349.MIS2/RAY_MY_train_1a4d8_00004_4_CV=4,G_chem_hdim=32,G_chem_layer=3,G_exp_hdim=32,G_exp_layer=3,batch_size=512,dropout_1=0.1000,dropout__2023-08-27_16-34-09/checkpoint_000996/checkpoint VSC_CV_4_model.pth 





# full 

get /checkpoint full_CV_0_model.pth 
get /checkpoint full_CV_1_model.pth 
get /checkpoint full_CV_2_model.pth 
get /checkpoint full_CV_3_model.pth 
get /checkpoint full_CV_4_model.pth 

get /home01/k040a01/ray_results/PRJ02.23.08.27.M3V8.WORK_403.349.MIS2/RAY_MY_train_1a4d8_00000_0_CV=0,G_chem_hdim=32,G_chem_layer=3,G_exp_hdim=32,G_exp_layer=3,batch_size=512,dropout_1=0.1000,dropout__2023-08-27_16-34-06/checkpoint_000999/checkpoint full_CV_0_model.pth 
get /home01/k040a01/ray_results/PRJ02.23.08.27.M3V8.WORK_403.349.MIS2/RAY_MY_train_1a4d8_00001_1_CV=1,G_chem_hdim=32,G_chem_layer=3,G_exp_hdim=32,G_exp_layer=3,batch_size=512,dropout_1=0.1000,dropout__2023-08-27_16-34-09/checkpoint_000999/checkpoint full_CV_1_model.pth 
get /home01/k040a01/ray_results/PRJ02.23.08.27.M3V8.WORK_403.349.MIS2/RAY_MY_train_1a4d8_00002_2_CV=2,G_chem_hdim=32,G_chem_layer=3,G_exp_hdim=32,G_exp_layer=3,batch_size=512,dropout_1=0.1000,dropout__2023-08-27_16-34-09/checkpoint_000999/checkpoint full_CV_2_model.pth 
get /home01/k040a01/ray_results/PRJ02.23.08.27.M3V8.WORK_403.349.MIS2/RAY_MY_train_1a4d8_00003_3_CV=3,G_chem_hdim=32,G_chem_layer=3,G_exp_hdim=32,G_exp_layer=3,batch_size=512,dropout_1=0.1000,dropout__2023-08-27_16-34-09/checkpoint_000999/checkpoint full_CV_3_model.pth 
get /home01/k040a01/ray_results/PRJ02.23.08.27.M3V8.WORK_403.349.MIS2/RAY_MY_train_1a4d8_00004_4_CV=4,G_chem_hdim=32,G_chem_layer=3,G_exp_hdim=32,G_exp_layer=3,batch_size=512,dropout_1=0.1000,dropout__2023-08-27_16-34-09/checkpoint_000999/checkpoint full_CV_4_model.pth 





########################################################################
########################################################################
########################################################################
							CPU 
########################################################################
########################################################################
########################################################################







NETWORK_PATH = '/st06/jiyeonH/13.DD_SESS/HumanNetV3/'
LINCS_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/' 
DATA_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V3_FULL/'
DC_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/'







# HS Drug pathway DB 활용 -> 349
print('NETWORK')
# HUMANNET 사용 

hunet_gsp = pd.read_csv(NETWORK_PATH+'HS-DB.tsv', sep = '\t', header = None)
hunet_gsp.columns = ['G_A','G_B','SC']

LINCS_gene_file = pd.read_csv(LINCS_PATH+'geneinfo_beta.txt', sep = '\t')
LINCS_978 = LINCS_gene_file[LINCS_gene_file.feature_space == 'landmark']
LINCS_978 = LINCS_978[['gene_id','gene_symbol']]
LINCS_978['new_node'] = [str(list(LINCS_978.gene_id)[i]) + "__" + list(LINCS_978.gene_symbol)[i] for i in range(978)]
LINCS_978 = LINCS_978.reset_index(drop=True)
lm_entrezs = list(LINCS_978.gene_id)

hnet_L1 = hunet_gsp[hunet_gsp['G_A'].isin(lm_entrezs)]
hnet_L2 = hnet_L1[hnet_L1['G_B'].isin(lm_entrezs)] # 3885
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



SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V8_349_FULL/'

file_name = 'M3V8_349_MISS2_FULL' # 0608

A_B_C_S_SET_ADD = pd.read_csv(SAVE_PATH+'{}.A_B_C_S_SET_ADD.csv'.format(file_name), low_memory=False)
MY_chem_A_feat = torch.load(SAVE_PATH+'{}.MY_chem_A_feat.pt'.format(file_name))
MY_chem_B_feat = torch.load(SAVE_PATH+'{}.MY_chem_B_feat.pt'.format(file_name))
MY_chem_A_adj = torch.load(SAVE_PATH+'{}.MY_chem_A_adj.pt'.format(file_name))
MY_chem_B_adj = torch.load(SAVE_PATH+'{}.MY_chem_B_adj.pt'.format(file_name))
MY_g_EXP_A = torch.load(SAVE_PATH+'{}.MY_g_EXP_A.pt'.format(file_name))
MY_g_EXP_B = torch.load(SAVE_PATH+'{}.MY_g_EXP_B.pt'.format(file_name))
MY_Target_1_A = torch.load(SAVE_PATH+'{}.MY_Target_1_A.pt'.format(file_name))
MY_Target_1_B = torch.load(SAVE_PATH+'{}.MY_Target_1_B.pt'.format(file_name))
MY_CellBase = torch.load(SAVE_PATH+'{}.MY_CellBase.pt'.format(file_name))
MY_syn = torch.load(SAVE_PATH+'{}.MY_syn.pt'.format(file_name))



A_B_C_S_SET_ADD2 = copy.deepcopy(A_B_C_S_SET_ADD)

cid_a = list(A_B_C_S_SET_ADD2['CID_A'])
cid_b = list(A_B_C_S_SET_ADD2['CID_B'])
sm_a = list(A_B_C_S_SET_ADD2['ROW_CAN_SMILES'])
sm_b = list(A_B_C_S_SET_ADD2['COL_CAN_SMILES'])
ccle = list(A_B_C_S_SET_ADD2['CELL'])

A_B_C_S_SET_ADD2['CID_CID'] = [str(int(cid_a[i])) + '___' + str(int(cid_b[i])) if cid_a[i] < cid_b[i] else str(int(cid_b[i])) + '___' + str(int(cid_a[i])) for i in range(A_B_C_S_SET_ADD2.shape[0])]
A_B_C_S_SET_ADD2['SM_C_CHECK'] = [sm_a[i] + '___' + sm_b[i]+ '___' + ccle[i] if sm_a[i] < sm_b[i] else sm_b[i] + '___' + sm_a[i]+ '___' + ccle[i] for i in range(A_B_C_S_SET_ADD2.shape[0])]

A_B_C_S_SET_ADD2['ori_index'] = list(A_B_C_S_SET_ADD2.index)




MISS_filter = ['AOBO','AXBX','AXBO','AOBX'] # 

A_B_C_S_SET = A_B_C_S_SET_ADD2[A_B_C_S_SET_ADD2.Basal_Exp == 'O']

A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.SYN_OX == 'O']

# A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.T1OX == 'O'] ####################### new targets 

A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.type.isin(MISS_filter)]





# basal node exclude -> CCLE match 만 사용
CCLE_PATH = '/st06/jiyeonH/13.DD_SESS/CCLE.22Q1/'
ccle_exp = pd.read_csv(CCLE_PATH+'CCLE_expression.csv', low_memory=False)
ccle_info= pd.read_csv(CCLE_PATH+'sample_info.csv', low_memory=False)

ccle_cell_info = ccle_info[['DepMap_ID','CCLE_Name']]
ccle_cell_info.columns = ['DepMap_ID','DrugCombCCLE']

ccle_cell_info_filt = ccle_cell_info[ccle_cell_info.DepMap_ID.isin(ccle_exp['Unnamed: 0'])]
ccle_names = [a for a in ccle_cell_info_filt.DrugCombCCLE if type(a) == str]


A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.CELL.isin(ccle_names)]




data_ind = list(A_B_C_S_SET.index)

MY_chem_A_feat_RE = MY_chem_A_feat[data_ind]
MY_chem_B_feat_RE = MY_chem_B_feat[data_ind]
MY_chem_A_adj_RE = MY_chem_A_adj[data_ind]
MY_chem_B_adj_RE = MY_chem_B_adj[data_ind]
MY_g_EXP_A_RE = MY_g_EXP_A[data_ind]
MY_g_EXP_B_RE = MY_g_EXP_B[data_ind]
MY_Target_A = copy.deepcopy(MY_Target_1_A)[data_ind] ############## NEW TARGET !!!!!! #####
MY_Target_B = copy.deepcopy(MY_Target_1_B)[data_ind] ############## NEW TARGET !!!!!! #####

MY_CellBase_RE = MY_CellBase[data_ind]
MY_syn_RE = MY_syn[data_ind]


A_B_C_S_SET = A_B_C_S_SET.reset_index(drop = True)






# cell line vector 

DC_CELL_DF2 = pd.read_csv(DC_PATH+'DC_CELL_INFO.csv', sep = '\t')
DC_CELL_DF2 = pd.concat([
	DC_CELL_DF2, 
	pd.DataFrame({'cell_line_id' : [1],'DC_cellname' : ['786O'],'DrugCombCello' : ['CVCL_1051'],'DrugCombCCLE':['786O_KIDNEY']})])

DC_CELL_info_filt = DC_CELL_DF2[DC_CELL_DF2.DrugCombCCLE.isin(A_B_C_S_SET.CELL)] # 38

DC_CELL_info_filt = DC_CELL_info_filt.drop(['Unnamed: 0'], axis = 1)
DC_CELL_info_filt.columns = ['cell_line_id', 'DC_cellname', 'DrugCombCello', 'CELL']
DC_CELL_info_filt = DC_CELL_info_filt[['CELL','DC_cellname']]

A_B_C_S_SET_COH = pd.merge(A_B_C_S_SET, DC_CELL_info_filt, on = 'CELL', how = 'left'  )








# sample number filter 

# 빈도 확인 

C_names = list(set(A_B_C_S_SET_COH.DC_cellname))
C_names.sort()

C_freq = [list(A_B_C_S_SET_COH.DC_cellname).count(a) for a in C_names]
C_cclename = [list(A_B_C_S_SET_COH[A_B_C_S_SET_COH.DC_cellname==a]['CELL'])[0] for a in C_names]

C_df = pd.DataFrame({'cell' : C_names, 'freq' : C_freq, 'ccle' : C_cclename})
C_df = C_df.sort_values('freq')


CELL_CUT = 200 ####### 이것도 그렇게 되면 바꿔야하지 않을까 ##################################################################

C_freq_filter = C_df[C_df.freq > CELL_CUT ] 

CELL_92 = ['VCAP_PROSTATE', 'NIHOVCAR3_OVARY', 'SW620_LARGE_INTESTINE', 'OVCAR4_OVARY', 'BT549_BREAST', 'A549_LUNG', 'SKMEL5_SKIN', 'A427_LUNG', 'BT474_BREAST', 'HOP92_LUNG', 'T98G_CENTRAL_NERVOUS_SYSTEM', 'NCIH23_LUNG', 'HT144_SKIN', 'RVH421_SKIN', 'MDAMB361_BREAST', 'LOVO_LARGE_INTESTINE', 'MDAMB231_BREAST', 'SKMEL28_SKIN', 'NCIH1650_LUNG', 'RKO_LARGE_INTESTINE', 'OVCAR5_OVARY', 'UACC812_BREAST', 'KPL1_BREAST', 'MSTO211H_PLEURA', 'KM12_LARGE_INTESTINE', 'IGROV1_OVARY', 'UHO1_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'NCIH520_LUNG', 'OVCAR8_OVARY', 'HCT15_LARGE_INTESTINE', 'A375_SKIN', 'CAKI1_KIDNEY', 'MDAMB468_BREAST', 'K562_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'A101D_SKIN', 'PA1_OVARY', 'UO31_KIDNEY', 'HOP62_LUNG', 'SF539_CENTRAL_NERVOUS_SYSTEM', 'MDAMB175VII_BREAST', 'U251MG_CENTRAL_NERVOUS_SYSTEM', 'HCC1500_BREAST', 'L1236_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'HCC1419_BREAST', 'NCIH460_LUNG', 'NCIH2122_LUNG', 'COLO792_SKIN', 'SR786_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'UACC257_SKIN', 'SNB75_CENTRAL_NERVOUS_SYSTEM', 'HCT116_LARGE_INTESTINE', 'PC3_PROSTATE', 'NCIH226_LUNG', 'RPMI8226_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'EKVX_LUNG', 'COLO800_SKIN', 'HT29_LARGE_INTESTINE', 'UWB1289_OVARY', 'MDAMB436_BREAST', 'SKOV3_OVARY', 'ZR751_BREAST', 'MEWO_SKIN', 'MELHO_SKIN', 'A2058_SKIN', 'RPMI7951_SKIN', 'SF268_CENTRAL_NERVOUS_SYSTEM', 'ACHN_KIDNEY', 'IPC298_SKIN', 'MALME3M_SKIN', 'A673_BONE', 'SF295_CENTRAL_NERVOUS_SYSTEM', 'CAOV3_OVARY', 'A498_KIDNEY', 'SKMEL2_SKIN', 'UACC62_SKIN', 'ES2_OVARY', 'LOXIMVI_SKIN', '786O_KIDNEY', 'MCF7_BREAST', 'WM115_SKIN', 'A2780_OVARY', 'DLD1_LARGE_INTESTINE', 'HS578T_BREAST', 'SKMES1_LUNG', 'T47D_BREAST', 'OV90_OVARY', 'G361_SKIN', 'SKMEL30_SKIN', 'COLO829_SKIN', 'SW837_LARGE_INTESTINE', 'NCIH522_LUNG', 'CAMA1_BREAST']
C_freq_filter = C_df[C_df.ccle.isin(CELL_92) ]




A_B_C_S_SET_COH = A_B_C_S_SET_COH[A_B_C_S_SET_COH.DC_cellname.isin(C_freq_filter.cell)]

DC_CELL_info_filt_re = DC_CELL_info_filt[DC_CELL_info_filt.DC_cellname.isin(C_freq_filter.cell)]
DC_CELL_info_filt_re['cell_onehot'] = [a for a in range(len(set(DC_CELL_info_filt_re.CELL)))]

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


print('CIDs', flush = True)
tmp = list(set(A_B_C_S_SET_COH2.CID_CID))
tmp2 = sum([a.split('___') for a in tmp],[])
print(len(set(tmp2)) , flush = True)


print('CID_CID', flush = True)
print(len(set(A_B_C_S_SET_COH2.CID_CID)), flush = True)



print('CID_CID_CCLE', flush = True)
print(len(set(A_B_C_S_SET_COH2.cid_cid_cell)), flush = True)

print('DrugCombCCLE', flush = True)
print(len(set(A_B_C_S_SET_COH2.CELL)), flush = True)



###########################################################################################
###########################################################################################
###########################################################################################


# 일단 생 5CV


print("LEARNING")

A_B_C_S_SET_SM = copy.deepcopy(A_B_C_S_SET_COH2) # 

# get unique values, remove duplicates, but keep original counts
data_no_dup, counts = np.unique(list(A_B_C_S_SET_SM['SM_C_CHECK']), return_counts=True)
data_no_dup_cells = [setset.split('___')[2] for setset in data_no_dup]
data_no_dup_sm_sm = [setset.split('___')[0]+'___'+setset.split('___')[1] for setset in data_no_dup]
data_nodup_df = pd.DataFrame({
	'setset' : data_no_dup.tolist(), 
	'cell' : data_no_dup_cells,
	'SM_SM' : data_no_dup_sm_sm
	 })




SM_SM_list = list(set(data_nodup_df.SM_SM))
SM_SM_list.sort()
sm_sm_list_1 = sklearn.utils.shuffle(SM_SM_list, random_state=42)

bins = [a for a in range(0, len(sm_sm_list_1), round(len(sm_sm_list_1)*0.2) )]
bins = bins[1:]
res = np.split(sm_sm_list_1, bins)

CV_1_smsm = list(res[0])
CV_2_smsm = list(res[1])
CV_3_smsm = list(res[2])
CV_4_smsm = list(res[3])
CV_5_smsm = list(res[4])
if len(res) > 5 :
	CV_5_smsm = list(res[4]) + list(res[5])

len(sm_sm_list_1)
len(CV_1_smsm) + len(CV_2_smsm) + len(CV_3_smsm) + len(CV_4_smsm) + len(CV_5_smsm)

CV_1_setset = list(data_nodup_df[data_nodup_df.SM_SM.isin(CV_1_smsm)]['setset'])
CV_2_setset = list(data_nodup_df[data_nodup_df.SM_SM.isin(CV_2_smsm)]['setset'])
CV_3_setset = list(data_nodup_df[data_nodup_df.SM_SM.isin(CV_3_smsm)]['setset'])
CV_4_setset = list(data_nodup_df[data_nodup_df.SM_SM.isin(CV_4_smsm)]['setset'])
CV_5_setset = list(data_nodup_df[data_nodup_df.SM_SM.isin(CV_5_smsm)]['setset'])




CV_ND_INDS = {
	'CV0_train' : CV_1_setset + CV_2_setset + CV_3_setset + CV_4_setset, 
	'CV0_test' : CV_5_setset,
	'CV1_train' : CV_2_setset + CV_3_setset + CV_4_setset + CV_5_setset, 
	'CV1_test' : CV_1_setset,
	'CV2_train' : CV_3_setset + CV_4_setset + CV_5_setset + CV_1_setset,
	'CV2_test' : CV_2_setset,
	'CV3_train' : CV_4_setset + CV_5_setset + CV_1_setset + CV_2_setset,
	'CV3_test' : CV_3_setset,
	'CV4_train' : CV_5_setset + CV_1_setset + CV_2_setset + CV_3_setset,
	'CV4_test' : CV_4_setset 
}

print(data_nodup_df.shape)
len( CV_1_setset + CV_2_setset + CV_3_setset + CV_4_setset + CV_5_setset)
len(set( CV_1_setset + CV_2_setset + CV_3_setset + CV_4_setset + CV_5_setset ))







WORK_NAME = 'WORK_403' # 349
W_NAME = 'W403'
PRJ_NAME = 'M3V8'
MJ_NAME = 'M3V8'
MISS_NAME = 'MIS2'
PPI_NAME = '349'

PRJ_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/{}_{}_{}_{}/'.format(PRJ_NAME, W_NAME, PPI_NAME, MISS_NAME)

# 저장해둔거랑 같은지 확인 
with open('{}/CV_SM_list.{}.pickle'.format(PRJ_PATH, WORK_NAME), 'rb') as f:
	CV_ND_INDS_ray = pickle.load(f)
 
for kk in ['CV0_train', 'CV0_test', 'CV1_train', 'CV1_test', 'CV2_train', 'CV2_test', 'CV3_train', 'CV3_test', 'CV4_train', 'CV4_test'] :
	CV_ND_INDS[kk] == CV_ND_INDS_ray[kk]

# 모두 true 



# 주어진 환경 맞춰서 5CV만 돌릴거라서 수정버전 
def prepare_data_GCN(CV_num, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, 
MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, 
MY_syn_RE2, norm ) : 
	# 
	# CV_num = 0
	train_key = 'CV{}_train'.format(CV_num)
	test_key = 'CV{}_test'.format(CV_num)
	# 
	#
	ABCS_tv = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS[train_key])]
	ABCS_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS[test_key])]
	#
	#train_ind = list(ABCS_train.index)
	#val_ind = list(ABCS_val.index)
	tv_ind = list(ABCS_tv.index)
	random.shuffle(tv_ind)
	test_ind = list(ABCS_test.index)
	# 
	chem_feat_A_tv = MY_chem_A_feat_RE2[tv_ind]; chem_feat_A_test = MY_chem_A_feat_RE2[test_ind]
	chem_feat_B_tv = MY_chem_B_feat_RE2[tv_ind]; chem_feat_B_test = MY_chem_B_feat_RE2[test_ind]
	chem_adj_A_tv = MY_chem_A_adj_RE2[tv_ind]; chem_adj_A_test = MY_chem_A_adj_RE2[test_ind]
	chem_adj_B_tv = MY_chem_B_adj_RE2[tv_ind]; chem_adj_B_test = MY_chem_B_adj_RE2[test_ind]
	gene_A_tv = MY_g_EXP_A_RE2[tv_ind];  gene_A_test = MY_g_EXP_A_RE2[test_ind]
	gene_B_tv = MY_g_EXP_B_RE2[tv_ind];  gene_B_test = MY_g_EXP_B_RE2[test_ind]
	target_A_tv = MY_Target_A2[tv_ind];  target_A_test = MY_Target_A2[test_ind]
	target_B_tv = MY_Target_B2[tv_ind];  target_B_test = MY_Target_B2[test_ind]
	cell_basal_tv = MY_CellBase_RE2[tv_ind];  cell_basal_test = MY_CellBase_RE2[test_ind]
	syn_tv = MY_syn_RE2[tv_ind];  syn_test = MY_syn_RE2[test_ind]
	#
	tv_data = {}
	test_data = {}
	#
	tv_data['drug1_feat'] = torch.concat([chem_feat_A_tv, chem_feat_B_tv], axis = 0)
	test_data['drug1_feat'] = chem_feat_A_test
	#
	tv_data['drug2_feat'] = torch.concat([chem_feat_B_tv, chem_feat_A_tv], axis = 0)
	test_data['drug2_feat'] = chem_feat_B_test
	#
	tv_data['drug1_adj'] = torch.concat([chem_adj_A_tv, chem_adj_B_tv], axis = 0)
	test_data['drug1_adj'] = chem_adj_A_test
	#
	tv_data['drug2_adj'] = torch.concat([chem_adj_B_tv, chem_adj_A_tv], axis = 0)
	test_data['drug2_adj'] = chem_adj_B_test
	#
	tv_data['GENE_A'] = torch.concat([gene_A_tv, gene_B_tv], axis = 0)
	test_data['GENE_A'] = gene_A_test
	#
	tv_data['GENE_B'] = torch.concat([gene_B_tv, gene_A_tv], axis = 0)
	test_data['GENE_B'] = gene_B_test
	#
	tv_data['TARGET_A'] = torch.concat([target_A_tv, target_B_tv], axis = 0)
	test_data['TARGET_A'] = target_A_test
	#
	tv_data['TARGET_B'] = torch.concat([target_B_tv, target_A_tv], axis = 0)
	test_data['TARGET_B'] = target_B_test
	#
	tv_data['cell_BASAL'] = torch.concat((cell_basal_tv, cell_basal_tv), axis=0)
	test_data['cell_BASAL'] = cell_basal_test
	##
	#            
	tv_data['y'] = torch.concat((syn_tv, syn_tv), axis=0)
	test_data['y'] = syn_test
	#
	print(tv_data['drug1_feat'].shape, flush=True)
	print(test_data['drug1_feat'].shape, flush=True)
	return tv_data, test_data







class DATASET_GCN_W_FT(Dataset):
	def __init__(self, gcn_drug1_F, gcn_drug2_F, gcn_drug1_ADJ, gcn_drug2_ADJ, 
	gcn_gene_A, gcn_gene_B, target_A, target_B, cell_basal, gcn_adj, gcn_adj_weight, 
	syn_ans ):
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
		return self.gcn_drug1_F[index], self.gcn_drug2_F[index],adj_re_A, adj_re_B, FEAT_A, FEAT_B, self.gcn_adj, self.gcn_adj_weight ,self.syn_ans[index]





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
	EXP_num_nodes_seen = 0
	DRUG_1_num_nodes_seen = 0
	DRUG_2_num_nodes_seen = 0
	#
	for drug1_f, drug2_f, drug1_adj, drug2_adj, expA, expB, exp_adj, exp_adj_w, y in batch :
		drug1_f_list.append(drug1_f)
		drug2_f_list.append(drug2_f)
		drug1_adj_list.append(drug1_adj+DRUG_1_num_nodes_seen)
		drug2_adj_list.append(drug2_adj+DRUG_2_num_nodes_seen)
		expA_list.append(expA)
		expB_list.append(expB)
		exp_adj_list.append(exp_adj+EXP_num_nodes_seen)
		exp_adj_w_list.append(exp_adj_w)
		y_list.append(torch.Tensor(y))
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
	return drug1_f_new, drug2_f_new, drug1_adj_new, drug2_adj_new, expA_new, expB_new, exp_adj_new, exp_adj_w_new, y_new




def weighted_mse_loss(input, target, weight):
	#return (weight * (input - target) ** 2).mean()
    return sum((weight * ((input-target)**2)).squeeze()) / sum(weight.squeeze())


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

# CV_0
train_data_0, test_data_0 = prepare_data_GCN(0, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, 
MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, 
 MY_syn_RE2, norm)

# CV_1
train_data_1, test_data_1 = prepare_data_GCN(1, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, 
MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, 
 MY_syn_RE2, norm)

# CV_2
train_data_2, test_data_2 = prepare_data_GCN(2, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, 
MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, 
 MY_syn_RE2, norm)

# CV_3
train_data_3, test_data_3 = prepare_data_GCN(3, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, 
MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, 
 MY_syn_RE2, norm)

# CV_4
train_data_4, test_data_4 = prepare_data_GCN(4, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, 
MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, 
 MY_syn_RE2, norm)



# WEIGHT 
def get_loss_weight(CV) :
	train_data = globals()['train_data_'+str(CV)]
	ys = train_data['y'].squeeze().tolist()
	min_s = np.amin(ys)
	loss_weight = np.log(train_data['y'] - min_s + np.e)
	return loss_weight


LOSS_WEIGHT_0 = get_loss_weight(0)
LOSS_WEIGHT_1 = get_loss_weight(1)
LOSS_WEIGHT_2 = get_loss_weight(2)
LOSS_WEIGHT_3 = get_loss_weight(3)
LOSS_WEIGHT_4 = get_loss_weight(4)

JY_IDX_WEIGHT_T = torch.Tensor(JY_IDX_WEIGHT).view(1,-1)





# DATA check  
def make_merged_data(CV) :
	train_data = globals()['train_data_'+str(CV)]
	test_data = globals()['test_data_'+str(CV)]
	#
	T_train = DATASET_GCN_W_FT(
		torch.Tensor(train_data['drug1_feat']), torch.Tensor(train_data['drug2_feat']), 
		torch.Tensor(train_data['drug1_adj']), torch.Tensor(train_data['drug2_adj']),
		torch.Tensor(train_data['GENE_A']), torch.Tensor(train_data['GENE_B']), 
		torch.Tensor(train_data['TARGET_A']), torch.Tensor(train_data['TARGET_B']), torch.Tensor(train_data['cell_BASAL']), 
		JY_ADJ_IDX, JY_IDX_WEIGHT_T, 
		torch.Tensor(train_data['y'])
		)
	#
	#	
	T_test = DATASET_GCN_W_FT(
		torch.Tensor(test_data['drug1_feat']), torch.Tensor(test_data['drug2_feat']), 
		torch.Tensor(test_data['drug1_adj']), torch.Tensor(test_data['drug2_adj']),
		torch.Tensor(test_data['GENE_A']), torch.Tensor(test_data['GENE_B']), 
		torch.Tensor(test_data['TARGET_A']), torch.Tensor(test_data['TARGET_B']), torch.Tensor(test_data['cell_BASAL']), 
		JY_ADJ_IDX, JY_IDX_WEIGHT_T, 
		torch.Tensor(test_data['y'])
		)
	#
	return T_train, T_test





# CV 0 
T_train_0, T_test_0 = make_merged_data(0)
RAY_test_0 = ray.put(T_test_0)


# CV 1
T_train_1, T_test_1 = make_merged_data(1)
RAY_test_1 = ray.put(T_test_1)


# CV 2 
T_train_2, T_test_2 = make_merged_data(2)
RAY_test_2 = ray.put(T_test_2)


# CV 3
T_train_3, T_test_3 = make_merged_data(3)
RAY_test_3 = ray.put(T_test_3)


# CV 4
T_train_4, T_test_4 = make_merged_data(4)
RAY_test_4 = ray.put(T_test_4)




class MY_expGCN_parallel_model(torch.nn.Module):
	def __init__(self, G_layer_chem, G_indim_chem, G_hiddim_chem, 
	G_layer_exp, G_indim_exp, G_hiddim_exp, 
	layers_1, layers_2, layers_3, 
	out_dim, inDrop, drop):
		super(MY_expGCN_parallel_model, self).__init__()
		self.G_layer_chem = G_layer_chem
		self.G_indim_chem = G_indim_chem
		self.G_hiddim_chem = G_hiddim_chem
		self.G_layer_exp = G_layer_exp
		self.G_indim_exp = G_indim_exp
		self.G_hiddim_exp = G_hiddim_exp
		self.G_Common_dim = min([G_hiddim_chem,G_hiddim_exp])
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
		self.G_convs_1_chem.extend([pyg_nn.GCNConv(self.G_hiddim_chem, self.G_Common_dim)])
		self.G_bns_1_chem = torch.nn.ModuleList([torch.nn.BatchNorm1d(self.G_hiddim_chem) for i in range(self.G_layer_chem-1)])
		##
		self.G_convs_1_exp = torch.nn.ModuleList([pyg_nn.GCNConv(self.G_indim_exp, self.G_hiddim_exp)])
		self.G_convs_1_exp.extend([pyg_nn.GCNConv(self.G_hiddim_exp, self.G_hiddim_exp) for i in range(self.G_layer_exp-2)])
		self.G_convs_1_exp.extend([pyg_nn.GCNConv(self.G_hiddim_exp, self.G_Common_dim)])
		self.G_bns_1_exp = torch.nn.ModuleList([torch.nn.BatchNorm1d(self.G_hiddim_exp) for i in range(self.G_layer_exp-1)])
		##
		self.Convs_1 = torch.nn.ModuleList([torch.nn.Linear(self.G_Common_dim+self.G_Common_dim, self.layers_1[0] )])
		self.Convs_1.extend([torch.nn.Linear(self.layers_1[a], self.layers_1[a+1]) for a in range(len(self.layers_1)-1)])
		##
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
		for conv in self.G_convs_1_exp :
			conv.reset_parameters()
		for bns in self.G_bns_1_exp :
			bns.reset_parameters()
		for conv in self.Convs_1 :
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
	def forward(self, Drug1_F, Drug2_F, Drug1_ADJ, Drug2_ADJ, EXP1, EXP2, EXP_ADJ, EXP_ADJ_WGT, syn ):
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
		for G_2_C in range(len(self.G_convs_1_chem)):
			if G_2_C == len(self.G_convs_1_chem)-1 :
				Drug2_F = self.G_convs_1_chem[G_2_C](x=Drug2_F, edge_index=Drug2_ADJ)
				Drug2_F = F.dropout(Drug2_F, p=self.inDrop, training=self.training)
				Drug2_F = self.pool(Drug2_F, Drug_batch_label )
				Drug2_F = self.tanh(Drug2_F)
				G_2_C_out = Drug2_F
			else :
				Drug2_F = self.G_convs_1_chem[G_2_C](x=Drug2_F, edge_index=Drug2_ADJ)
				Drug2_F = self.G_bns_1_chem[G_2_C](Drug2_F)
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
		for G_2_E in range(len(self.G_convs_1_exp)):
			if G_2_E == len(self.G_convs_1_exp)-1 :
				EXP2 = self.G_convs_1_exp[G_2_E](x=EXP2, edge_index=EXP_ADJ, edge_weight= EXP_ADJ_WGT)
				EXP2 = F.dropout(EXP2, p=self.inDrop, training=self.training)
				EXP2 = self.pool(EXP2, Exp_batch_label )
				EXP2 = self.tanh(EXP2)
				G_2_E_out = EXP2
			else :
				EXP2 = self.G_convs_1_exp[G_2_E](x=EXP2, edge_index=EXP_ADJ, edge_weight= EXP_ADJ_WGT)
				EXP2 = self.G_bns_1_exp[G_2_E](EXP2)
				EXP2 = F.elu(EXP2)
		#
		input_drug1 = torch.concat( (G_1_C_out, G_1_E_out), 1 ) # normalization 추가 해야할것 같은데 
		input_drug2 = torch.concat( (G_2_C_out, G_2_E_out), 1 )
		#
		for L1 in range(len(self.Convs_1)):
			if L1 != len(self.Convs_1)-1 :
				input_drug1 = self.Convs_1[L1](input_drug1)
				input_drug1 = F.dropout(input_drug1, p=self.inDrop, training = self.training)
				input_drug1 = F.elu(input_drug1)
			else :
				input_drug1 = self.Convs_1[L1](input_drug1)
		#
		for L2 in range(len(self.Convs_1)):
			if L2 != len(self.Convs_1)-1 :
				input_drug2 = self.Convs_1[L2](input_drug2)
				input_drug2 = F.dropout(input_drug2, p=self.inDrop, training = self.training)
				input_drug2 = F.elu(input_drug2)
			else :
				input_drug2 = self.Convs_1[L2](input_drug2)
		#
		X = torch.cat(( input_drug1, input_drug2 ), 1)
		for L3 in range(len(self.SNPs)):
			if L3 != len(self.SNPs)-1 :
				X = self.SNPs[L3](X)
				X = F.dropout(X, p=self.drop, training = self.training)
				X = F.elu(X)
			else :
				X = self.SNPs[L3](X)
		return X





def plot_three(big_title, train_loss, valid_loss, train_Pcorr, valid_Pcorr, train_Scorr, valid_Scorr, path, plotname, epoch = 0 ):
	fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(30, 8))
	#
	# loss plot 
	ax1.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss', color = 'Blue', linewidth=4 )
	ax1.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss', color = 'Red', linewidth=4)
	ax1.set_xlabel('epochs', fontsize=20)
	ax1.set_ylabel('loss', fontsize=20)
	ax1.tick_params(axis='both', which='major', labelsize=20 )
	ax1.set_ylim(0, math.ceil(max(train_loss+valid_loss))) # 일정한 scale
	ax1.set_xlim(0, len(train_loss)+1) # 일정한 scale
	ax1.grid(True)
	if epoch > 0 : 
		ax1.axvline(x = epoch, color = 'green', linestyle = '--', linewidth = 3)
	ax1.set_title('5CV Average Loss', fontsize=20)
	#
	# Pearson Corr 
	ax2.plot(range(1,len(train_Pcorr)+1), train_Pcorr, label='Training PCorr', color = 'Blue', linewidth=4 )
	ax2.plot(range(1,len(valid_Pcorr)+1),valid_Pcorr,label='Validation PCorr', color = 'Red', linewidth=4)
	ax2.set_xlabel('epochs', fontsize=20)
	ax2.set_ylabel('PCor', fontsize=20)
	ax2.tick_params(axis='both', which='major', labelsize=20 )
	ax2.set_ylim(0, math.ceil(max(train_Pcorr+valid_Pcorr))) # 일정한 scale
	ax2.set_xlim(0, len(train_Pcorr)+1) # 일정한 scale
	ax2.grid(True)
	if epoch > 0 : 
		ax2.axvline(x = epoch, color = 'green', linestyle = '--', linewidth = 3)
	#
	ax2.set_title('5CV Average Pearson', fontsize=20)
	#
	# Spearman Corr 
	ax3.plot(range(1,len(train_Scorr)+1), train_Scorr, label='Training SCorr', color = 'Blue', linewidth=4 )
	ax3.plot(range(1,len(valid_Scorr)+1),valid_Scorr,label='Validation SCorr', color = 'Red', linewidth=4)
	ax3.set_xlabel('epochs', fontsize=20)
	ax3.set_ylabel('SCor', fontsize=20)
	ax3.tick_params(axis='both', which='major', labelsize=20 )
	ax3.set_ylim(0, math.ceil(max(train_Scorr+valid_Scorr))) # 일정한 scale
	ax3.set_xlim(0, len(train_Scorr)+1) # 일정한 scale
	ax3.grid(True)
	if epoch > 0 : 
		ax3.axvline(x = epoch, color = 'green', linestyle = '--', linewidth = 3)
	#
	ax3.set_title('5CV Average Spearman', fontsize=20)
	#
	fig.suptitle(big_title, fontsize=18)
	plt.tight_layout()
	fig.savefig('{}/{}.three_plot.png'.format(path, plotname), bbox_inches = 'tight')



def plot_Pcorr(train_corr, valid_corr, path, plotname):
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
	fig.savefig('{}/{}.Pcorr_plot.png'.format(path, plotname), bbox_inches = 'tight')
	plt.close()

def plot_Scorr(train_corr, valid_corr, path, plotname):
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
	fig.savefig('{}/{}.Scorr_plot.png'.format(path, plotname), bbox_inches = 'tight')
	plt.close()





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






def inner_test( TEST_DATA, THIS_MODEL , use_cuda = False) :
	THIS_MODEL.eval()
	#
	running_loss = 0
	last_loss = 0 
	#
	ans_list = []
	pred_list = []
	with torch.no_grad() :
		for batch_idx_v, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y) in enumerate(TEST_DATA) :
			expA = expA.view(-1,3)#### 다른점 
			expB = expB.view(-1,3)#### 다른점 
			adj_w = adj_w.squeeze()
			# move to GPU
			if use_cuda:
				drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y = drug1_f.cuda(), drug2_f.cuda(), drug1_a.cuda(), drug2_a.cuda(), expA.cuda(), expB.cuda(), adj.cuda(), adj_w.cuda(), y.cuda()
			## update the average validation loss
			output = THIS_MODEL(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y)
			MSE = torch.nn.MSELoss()
			loss = MSE(output, y) # train 이 아니라서 weight 안넣어줌. 그냥 nn.MSE 넣어주기 
			# update average validation loss 
			running_loss = running_loss + loss.item()
			pred_list = pred_list + output.squeeze().tolist()
			ans_list = ans_list + y.squeeze().tolist()
		#
	last_loss = running_loss / (batch_idx_v+1)
	val_sc, _ = stats.spearmanr(pred_list, ans_list)
	val_pc, _ = stats.pearsonr(pred_list, ans_list)
	return last_loss, val_pc, val_sc, pred_list, ans_list    




def TEST_CPU (PRJ_PATH, CV_num, my_config, model_path, model_name, model_num) :
	use_cuda = False
	#
	CV_test_dict = { 
		'CV_0': T_test_0, 'CV_1' : T_test_1, 'CV_2' : T_test_2,
		'CV_3' : T_test_3, 'CV_4': T_test_4 }
	#
	T_test = CV_test_dict[CV_num]
	test_loader = torch.utils.data.DataLoader(T_test, batch_size = my_config["config/batch_size"].item(), collate_fn = graph_collate_fn, shuffle =False, num_workers=16) # my_config['config/n_workers'].item()
	#
	G_chem_layer = my_config['config/G_chem_layer'].item()
	G_chem_hdim = my_config['config/G_chem_hdim'].item()
	G_exp_layer = my_config['config/G_exp_layer'].item()
	G_exp_hdim = my_config['config/G_exp_hdim'].item() 
	dsn_layers = [int(a) for a in my_config["config/dsn_layer"].split('-') ]
	snp_layers = [int(a) for a in my_config["config/snp_layer"].split('-') ]
	inDrop = my_config['config/dropout_1'].item()
	Drop = my_config['config/dropout_2'].item()
	#
	best_model = MY_expGCN_parallel_model(
				G_chem_layer, T_test.gcn_drug1_F.shape[-1] , G_chem_hdim,
				G_exp_layer, 3, G_exp_hdim,
				dsn_layers, dsn_layers, snp_layers, 
				1,
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
	#
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
	last_loss, val_pc, val_sc, pred_list, ans_list = inner_test(test_loader, best_model)
	R__1 , R__2 = jy_corrplot(pred_list, ans_list, PRJ_PATH, 'P.{}.{}.{}.{}_model'.format(PRJ_NAME, MISS_NAME, CV_num, model_num) )
	return  last_loss, R__1, R__2, pred_list, ans_list




PRJ_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/{}_{}_{}_{}/'.format(PRJ_NAME, W_NAME, PPI_NAME, MISS_NAME)
ANA_DF = pd.read_csv(PRJ_PATH+'RAY_ANA_DF.{}_{}_{}_{}.csv'.format(PRJ_NAME, W_NAME, PPI_NAME, MISS_NAME))
with open(PRJ_PATH+'RAY_ANA_DF.{}_{}_{}_{}.pickle'.format(PRJ_NAME, W_NAME, PPI_NAME, MISS_NAME), 'rb') as f:
	ANA_ALL_DF = pickle.load(f)


TOPVAL_PATH = PRJ_PATH


my_config = ANA_DF.loc[0]




# 1) full 

R_1_T_CV0, R_1_1_CV0, R_1_2_CV0, pred_1_CV0, ans_1_CV0 = TEST_CPU(PRJ_PATH, 'CV_0', my_config, PRJ_PATH, 'full_CV_0_model.pth', 'FULL')
R_1_T_CV1, R_1_1_CV1, R_1_2_CV1, pred_1_CV1, ans_1_CV1 = TEST_CPU(PRJ_PATH, 'CV_1', my_config, PRJ_PATH, 'full_CV_1_model.pth', 'FULL')
R_1_T_CV2, R_1_1_CV2, R_1_2_CV2, pred_1_CV2, ans_1_CV2 = TEST_CPU(PRJ_PATH, 'CV_2', my_config, PRJ_PATH, 'full_CV_2_model.pth', 'FULL')
R_1_T_CV3, R_1_1_CV3, R_1_2_CV3, pred_1_CV3, ans_1_CV3 = TEST_CPU(PRJ_PATH, 'CV_3', my_config, PRJ_PATH, 'full_CV_3_model.pth', 'FULL')
R_1_T_CV4, R_1_1_CV4, R_1_2_CV4, pred_1_CV4, ans_1_CV4 = TEST_CPU(PRJ_PATH, 'CV_4', my_config, PRJ_PATH, 'full_CV_4_model.pth', 'FULL')


# 2) min loss 

R_2_T_CV0, R_2_1_CV0, R_2_2_CV0, pred_2_CV0, ans_2_CV0 = TEST_CPU(PRJ_PATH, 'CV_0', my_config, PRJ_PATH, 'VLS_CV_0_model.pth', 'VLS')
R_2_T_CV1, R_2_1_CV1, R_2_2_CV1, pred_2_CV1, ans_2_CV1 = TEST_CPU(PRJ_PATH, 'CV_1', my_config, PRJ_PATH, 'VLS_CV_1_model.pth', 'VLS')
R_2_T_CV2, R_2_1_CV2, R_2_2_CV2, pred_2_CV2, ans_2_CV2 = TEST_CPU(PRJ_PATH, 'CV_2', my_config, PRJ_PATH, 'VLS_CV_2_model.pth', 'VLS')
R_2_T_CV3, R_2_1_CV3, R_2_2_CV3, pred_2_CV3, ans_2_CV3 = TEST_CPU(PRJ_PATH, 'CV_3', my_config, PRJ_PATH, 'VLS_CV_3_model.pth', 'VLS')
R_2_T_CV4, R_2_1_CV4, R_2_2_CV4, pred_2_CV4, ans_2_CV4 = TEST_CPU(PRJ_PATH, 'CV_4', my_config, PRJ_PATH, 'VLS_CV_4_model.pth', 'VLS')

# 3) PC 
R_3_T_CV0, R_3_1_CV0, R_3_2_CV0, pred_3_CV0, ans_3_CV0 = TEST_CPU(PRJ_PATH, 'CV_0', my_config, PRJ_PATH, 'VPC_CV_0_model.pth', 'VPC')
R_3_T_CV1, R_3_1_CV1, R_3_2_CV1, pred_3_CV1, ans_3_CV1 = TEST_CPU(PRJ_PATH, 'CV_1', my_config, PRJ_PATH, 'VPC_CV_1_model.pth', 'VPC')
R_3_T_CV2, R_3_1_CV2, R_3_2_CV2, pred_3_CV2, ans_3_CV2 = TEST_CPU(PRJ_PATH, 'CV_2', my_config, PRJ_PATH, 'VPC_CV_2_model.pth', 'VPC')
R_3_T_CV3, R_3_1_CV3, R_3_2_CV3, pred_3_CV3, ans_3_CV3 = TEST_CPU(PRJ_PATH, 'CV_3', my_config, PRJ_PATH, 'VPC_CV_3_model.pth', 'VPC')
R_3_T_CV4, R_3_1_CV4, R_3_2_CV4, pred_3_CV4, ans_3_CV4 = TEST_CPU(PRJ_PATH, 'CV_4', my_config, PRJ_PATH, 'VPC_CV_4_model.pth', 'VPC')


# 4) SC
R_4_T_CV0, R_4_1_CV0, R_4_2_CV0, pred_4_CV0, ans_4_CV0 = TEST_CPU(PRJ_PATH, 'CV_0', my_config, PRJ_PATH, 'VSC_CV_0_model.pth', 'VSC')
R_4_T_CV1, R_4_1_CV1, R_4_2_CV1, pred_4_CV1, ans_4_CV1 = TEST_CPU(PRJ_PATH, 'CV_1', my_config, PRJ_PATH, 'VSC_CV_1_model.pth', 'VSC')
R_4_T_CV2, R_4_1_CV2, R_4_2_CV2, pred_4_CV2, ans_4_CV2 = TEST_CPU(PRJ_PATH, 'CV_2', my_config, PRJ_PATH, 'VSC_CV_2_model.pth', 'VSC')
R_4_T_CV3, R_4_1_CV3, R_4_2_CV3, pred_4_CV3, ans_4_CV3 = TEST_CPU(PRJ_PATH, 'CV_3', my_config, PRJ_PATH, 'VSC_CV_3_model.pth', 'VSC')
R_4_T_CV4, R_4_1_CV4, R_4_2_CV4, pred_4_CV4, ans_4_CV4 = TEST_CPU(PRJ_PATH, 'CV_4', my_config, PRJ_PATH, 'VSC_CV_4_model.pth', 'VSC')



# 다시 GPU 에서 해가지고 가져옴 
                def re_test(ABCS, CV_NUM, model_path, model_name, colname, use_cuda = False) :
                    CV__test = ray.get(RAY_test_list[CV_NUM])
                    #
                    THIS_MODEL = MY_expGCN_parallel_model(
                        config["G_chem_layer"], 64 , config["G_chem_hdim"],      # G_layer_chem, G_indim_chem, G_hiddim_chem, 
                        config["G_exp_layer"], 3 , config["G_exp_hdim"],      # G_layer_exp, G_indim_exp, G_hiddim_exp, 
                        dsn_layers, dsn_layers, snp_layers,      # drug 1 layers, drug 2 layers, merged layer, 
                        1,      # cell_dim ,out_dim,
                        inDrop, Drop      # inDrop, drop
                        )
                    state_dict = torch.load(os.path.join(model_path, model_name))
                    THIS_MODEL.load_state_dict(state_dict[0])
                    THIS_MODEL = THIS_MODEL.cuda()
                    THIS_MODEL.eval()
                    #
                    running_loss = 0
                    last_loss = 0 
                    #
                    ans_list = []
                    pred_list = []
                    val_loader = torch.utils.data.DataLoader(CV__test, batch_size = config["batch_size"], collate_fn = graph_collate_fn, 
                        shuffle =False, num_workers=128)
                    with torch.no_grad() :
                        for batch_idx_v, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w,  y) in enumerate(val_loader) :
                            expA = expA.view(-1,3)#### 다른점 
                            expB = expB.view(-1,3)#### 다른점 
                            adj_w = adj_w.squeeze()
                            # move to GPU
                            if use_cuda:
                                drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y= drug1_f.cuda(), drug2_f.cuda(), drug1_a.cuda(), drug2_a.cuda(), expA.cuda(), expB.cuda(), adj.cuda(), adj_w.cuda(), y.cuda()
                            ## update the average validation loss
                            output = THIS_MODEL(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w,  y)
                            MSE = torch.nn.MSELoss()
                            loss = MSE(output, y) # train 이 아니라서 weight 안넣어줌. 그냥 nn.MSE 넣어주기 
                            # update average validation loss 
                            running_loss = running_loss + loss.item()
                            pred_list = pred_list + output.squeeze().tolist()
                            ans_list = ans_list + y.squeeze().tolist()
                        #
                    last_loss = running_loss / (batch_idx_v+1)
                    val_sc, _ = stats.spearmanr(pred_list, ans_list)
                    val_pc, _ = stats.pearsonr(pred_list, ans_list)
                    #
                    ABCS['ANS'] = ans_list
                    ABCS[colname] = pred_list
                    return ABCS


                CV_NUM = 0
                test_key = 'CV{}_test'.format(CV_NUM)
                ABCS_test_CV0 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS[test_key])]

                model_path = '/home01/k040a01/ray_results/PRJ02.23.08.27.M3V8.WORK_403.349.MIS2/RAY_MY_train_1a4d8_00000_0_CV=0,G_chem_hdim=32,G_chem_layer=3,G_exp_hdim=32,G_exp_layer=3,batch_size=512,dropout_1=0.1000,dropout__2023-08-27_16-34-06/checkpoint_000999/'
                ABCS_test_CV0 = re_test(ABCS_test_CV0, 0, model_path, model_name, 'FULL', use_cuda = True)

                model_path = '/home01/k040a01/ray_results/PRJ02.23.08.27.M3V8.WORK_403.349.MIS2/RAY_MY_train_1a4d8_00000_0_CV=0,G_chem_hdim=32,G_chem_layer=3,G_exp_hdim=32,G_exp_layer=3,batch_size=512,dropout_1=0.1000,dropout__2023-08-27_16-34-06/checkpoint_000992/'
                ABCS_test_CV0 = re_test(ABCS_test_CV0, 0, model_path, model_name, 'VLS', use_cuda = True)

                model_path = '/home01/k040a01/ray_results/PRJ02.23.08.27.M3V8.WORK_403.349.MIS2/RAY_MY_train_1a4d8_00000_0_CV=0,G_chem_hdim=32,G_chem_layer=3,G_exp_hdim=32,G_exp_layer=3,batch_size=512,dropout_1=0.1000,dropout__2023-08-27_16-34-06/checkpoint_000939/'
                ABCS_test_CV0 = re_test(ABCS_test_CV0, 0, model_path, model_name, 'VPC', use_cuda = True)

                model_path = '/home01/k040a01/ray_results/PRJ02.23.08.27.M3V8.WORK_403.349.MIS2/RAY_MY_train_1a4d8_00000_0_CV=0,G_chem_hdim=32,G_chem_layer=3,G_exp_hdim=32,G_exp_layer=3,batch_size=512,dropout_1=0.1000,dropout__2023-08-27_16-34-06/checkpoint_000996/'
                ABCS_test_CV0 = re_test(ABCS_test_CV0, 0, model_path, model_name, 'VSC', use_cuda = True)

                ABCS_test_CV0.to_csv('/home01/k040a01/02.M3V8/M3V8_W403_349_MIS2/ABCS_test_CV0.csv')



                CV_NUM = 1
                test_key = 'CV{}_test'.format(CV_NUM)
                ABCS_test_CV1 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS[test_key])]

                model_path = '/home01/k040a01/ray_results/PRJ02.23.08.27.M3V8.WORK_403.349.MIS2/RAY_MY_train_1a4d8_00001_1_CV=1,G_chem_hdim=32,G_chem_layer=3,G_exp_hdim=32,G_exp_layer=3,batch_size=512,dropout_1=0.1000,dropout__2023-08-27_16-34-09/checkpoint_000999/'
                ABCS_test_CV1 = re_test(ABCS_test_CV1, 1, model_path, model_name, 'FULL', use_cuda = True)

                model_path = '/home01/k040a01/ray_results/PRJ02.23.08.27.M3V8.WORK_403.349.MIS2/RAY_MY_train_1a4d8_00001_1_CV=1,G_chem_hdim=32,G_chem_layer=3,G_exp_hdim=32,G_exp_layer=3,batch_size=512,dropout_1=0.1000,dropout__2023-08-27_16-34-09/checkpoint_000992/'
                ABCS_test_CV1 = re_test(ABCS_test_CV1, 1, model_path, model_name, 'VLS', use_cuda = True)

                model_path = '/home01/k040a01/ray_results/PRJ02.23.08.27.M3V8.WORK_403.349.MIS2/RAY_MY_train_1a4d8_00001_1_CV=1,G_chem_hdim=32,G_chem_layer=3,G_exp_hdim=32,G_exp_layer=3,batch_size=512,dropout_1=0.1000,dropout__2023-08-27_16-34-09/checkpoint_000939/'
                ABCS_test_CV1 = re_test(ABCS_test_CV1, 1, model_path, model_name, 'VPC', use_cuda = True)

                model_path = '/home01/k040a01/ray_results/PRJ02.23.08.27.M3V8.WORK_403.349.MIS2/RAY_MY_train_1a4d8_00001_1_CV=1,G_chem_hdim=32,G_chem_layer=3,G_exp_hdim=32,G_exp_layer=3,batch_size=512,dropout_1=0.1000,dropout__2023-08-27_16-34-09/checkpoint_000996/'
                ABCS_test_CV1 = re_test(ABCS_test_CV1, 1, model_path, model_name, 'VSC', use_cuda = True)

                ABCS_test_CV1.to_csv('/home01/k040a01/02.M3V8/M3V8_W403_349_MIS2/ABCS_test_CV1.csv')


                CV_NUM = 2
                test_key = 'CV{}_test'.format(CV_NUM)
                ABCS_test_CV2 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS[test_key])]

                model_path = '/home01/k040a01/ray_results/PRJ02.23.08.27.M3V8.WORK_403.349.MIS2/RAY_MY_train_1a4d8_00002_2_CV=2,G_chem_hdim=32,G_chem_layer=3,G_exp_hdim=32,G_exp_layer=3,batch_size=512,dropout_1=0.1000,dropout__2023-08-27_16-34-09/checkpoint_000999/'
                ABCS_test_CV2 = re_test(ABCS_test_CV2, 2, model_path, model_name, 'FULL', use_cuda = True)

                model_path = '/home01/k040a01/ray_results/PRJ02.23.08.27.M3V8.WORK_403.349.MIS2/RAY_MY_train_1a4d8_00002_2_CV=2,G_chem_hdim=32,G_chem_layer=3,G_exp_hdim=32,G_exp_layer=3,batch_size=512,dropout_1=0.1000,dropout__2023-08-27_16-34-09/checkpoint_000992/'
                ABCS_test_CV2 = re_test(ABCS_test_CV2, 2, model_path, model_name, 'VLS', use_cuda = True)

                model_path = '/home01/k040a01/ray_results/PRJ02.23.08.27.M3V8.WORK_403.349.MIS2/RAY_MY_train_1a4d8_00002_2_CV=2,G_chem_hdim=32,G_chem_layer=3,G_exp_hdim=32,G_exp_layer=3,batch_size=512,dropout_1=0.1000,dropout__2023-08-27_16-34-09/checkpoint_000939/'
                ABCS_test_CV2 = re_test(ABCS_test_CV2, 2, model_path, model_name, 'VPC', use_cuda = True)

                model_path = '/home01/k040a01/ray_results/PRJ02.23.08.27.M3V8.WORK_403.349.MIS2/RAY_MY_train_1a4d8_00002_2_CV=2,G_chem_hdim=32,G_chem_layer=3,G_exp_hdim=32,G_exp_layer=3,batch_size=512,dropout_1=0.1000,dropout__2023-08-27_16-34-09/checkpoint_000996/'
                ABCS_test_CV2 = re_test(ABCS_test_CV2, 2, model_path, model_name, 'VSC', use_cuda = True)


                ABCS_test_CV2.to_csv('/home01/k040a01/02.M3V8/M3V8_W403_349_MIS2/ABCS_test_CV2.csv')





                CV_NUM = 3
                test_key = 'CV{}_test'.format(CV_NUM)
                ABCS_test_CV3 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS[test_key])]

                model_path = '/home01/k040a01/ray_results/PRJ02.23.08.27.M3V8.WORK_403.349.MIS2/RAY_MY_train_1a4d8_00003_3_CV=3,G_chem_hdim=32,G_chem_layer=3,G_exp_hdim=32,G_exp_layer=3,batch_size=512,dropout_1=0.1000,dropout__2023-08-27_16-34-09/checkpoint_000999/'
                ABCS_test_CV3 = re_test(ABCS_test_CV3, 3, model_path, model_name, 'FULL', use_cuda = True)

                model_path = '/home01/k040a01/ray_results/PRJ02.23.08.27.M3V8.WORK_403.349.MIS2/RAY_MY_train_1a4d8_00003_3_CV=3,G_chem_hdim=32,G_chem_layer=3,G_exp_hdim=32,G_exp_layer=3,batch_size=512,dropout_1=0.1000,dropout__2023-08-27_16-34-09/checkpoint_000992/'
                ABCS_test_CV3 = re_test(ABCS_test_CV3, 3, model_path, model_name, 'VLS', use_cuda = True)

                model_path = '/home01/k040a01/ray_results/PRJ02.23.08.27.M3V8.WORK_403.349.MIS2/RAY_MY_train_1a4d8_00003_3_CV=3,G_chem_hdim=32,G_chem_layer=3,G_exp_hdim=32,G_exp_layer=3,batch_size=512,dropout_1=0.1000,dropout__2023-08-27_16-34-09/checkpoint_000939/'
                ABCS_test_CV3 = re_test(ABCS_test_CV3, 3, model_path, model_name, 'VPC', use_cuda = True)

                model_path = '/home01/k040a01/ray_results/PRJ02.23.08.27.M3V8.WORK_403.349.MIS2/RAY_MY_train_1a4d8_00003_3_CV=3,G_chem_hdim=32,G_chem_layer=3,G_exp_hdim=32,G_exp_layer=3,batch_size=512,dropout_1=0.1000,dropout__2023-08-27_16-34-09/checkpoint_000996/'
                ABCS_test_CV3 = re_test(ABCS_test_CV3, 3, model_path, model_name, 'VSC', use_cuda = True)

                ABCS_test_CV3.to_csv('/home01/k040a01/02.M3V8/M3V8_W403_349_MIS2/ABCS_test_CV3.csv')



                CV_NUM = 4
                test_key = 'CV{}_test'.format(CV_NUM)
                ABCS_test_CV4 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS[test_key])]

                model_path = '/home01/k040a01/ray_results/PRJ02.23.08.27.M3V8.WORK_403.349.MIS2/RAY_MY_train_1a4d8_00004_4_CV=4,G_chem_hdim=32,G_chem_layer=3,G_exp_hdim=32,G_exp_layer=3,batch_size=512,dropout_1=0.1000,dropout__2023-08-27_16-34-09/checkpoint_000999/'
                ABCS_test_CV4 = re_test(ABCS_test_CV4, 4, model_path, model_name, 'FULL', use_cuda = True)

                model_path = '/home01/k040a01/ray_results/PRJ02.23.08.27.M3V8.WORK_403.349.MIS2/RAY_MY_train_1a4d8_00004_4_CV=4,G_chem_hdim=32,G_chem_layer=3,G_exp_hdim=32,G_exp_layer=3,batch_size=512,dropout_1=0.1000,dropout__2023-08-27_16-34-09/checkpoint_000992/'
                ABCS_test_CV4 = re_test(ABCS_test_CV4, 4, model_path, model_name, 'VLS', use_cuda = True)

                model_path = '/home01/k040a01/ray_results/PRJ02.23.08.27.M3V8.WORK_403.349.MIS2/RAY_MY_train_1a4d8_00004_4_CV=4,G_chem_hdim=32,G_chem_layer=3,G_exp_hdim=32,G_exp_layer=3,batch_size=512,dropout_1=0.1000,dropout__2023-08-27_16-34-09/checkpoint_000939/'
                ABCS_test_CV4 = re_test(ABCS_test_CV4, 4, model_path, model_name, 'VPC', use_cuda = True)

                model_path = '/home01/k040a01/ray_results/PRJ02.23.08.27.M3V8.WORK_403.349.MIS2/RAY_MY_train_1a4d8_00004_4_CV=4,G_chem_hdim=32,G_chem_layer=3,G_exp_hdim=32,G_exp_layer=3,batch_size=512,dropout_1=0.1000,dropout__2023-08-27_16-34-09/checkpoint_000996/'
                ABCS_test_CV4 = re_test(ABCS_test_CV4, 4, model_path, model_name, 'VSC', use_cuda = True)

                ABCS_test_CV4.to_csv('/home01/k040a01/02.M3V8/M3V8_W403_349_MIS2/ABCS_test_CV4.csv')



ABCS_CV_0 = pd.read_csv(PRJ_PATH+'ABCS_test_CV0.csv', index_col = 0)
ABCS_CV_1 = pd.read_csv(PRJ_PATH+'ABCS_test_CV1.csv', index_col = 0)
ABCS_CV_2 = pd.read_csv(PRJ_PATH+'ABCS_test_CV2.csv', index_col = 0)
ABCS_CV_3 = pd.read_csv(PRJ_PATH+'ABCS_test_CV3.csv', index_col = 0)
ABCS_CV_4 = pd.read_csv(PRJ_PATH+'ABCS_test_CV4.csv', index_col = 0)


                        ABCS_test_0 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV0_test'])]
                        ABCS_test_0['ANS'] = ans_1_CV0 ; ABCS_test_0['PRED_1'] = pred_1_CV0; ABCS_test_0['PRED_2'] = pred_2_CV0 ; ABCS_test_0['PRED_3'] = pred_3_CV0 ;  ABCS_test_0['PRED_4'] = pred_4_CV0

                        ABCS_test_1 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV1_test'])]
                        ABCS_test_1['ANS'] = ans_1_CV1 ;ABCS_test_1['PRED_1'] = pred_1_CV1 ;ABCS_test_1['PRED_2'] = pred_2_CV1 ;ABCS_test_1['PRED_3'] = pred_3_CV1 ;ABCS_test_1['PRED_4'] = pred_4_CV1

                        ABCS_test_2 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV2_test'])]
                        ABCS_test_2['ANS'] = ans_1_CV2 ; ABCS_test_2['PRED_1'] = pred_1_CV2; ABCS_test_2['PRED_2'] = pred_2_CV2; ABCS_test_2['PRED_3'] = pred_3_CV2; ABCS_test_2['PRED_4'] = pred_4_CV2

                        ABCS_test_3 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV3_test'])]
                        ABCS_test_3['ANS'] = ans_1_CV3 ;ABCS_test_3['PRED_1'] = pred_1_CV3 ; ABCS_test_3['PRED_2'] = pred_2_CV3 ; ABCS_test_3['PRED_3'] = pred_3_CV3 ;ABCS_test_3['PRED_4'] = pred_4_CV3

                        ABCS_test_4 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV4_test'])]
                        ABCS_test_4['ANS'] = ans_1_CV4 ; ABCS_test_4['PRED_1'] = pred_1_CV4 ;ABCS_test_4['PRED_2'] = pred_2_CV4 ;ABCS_test_4['PRED_3'] = pred_3_CV4 ;ABCS_test_4['PRED_4'] = pred_4_CV4


ABCS_test_result = pd.concat([ABCS_CV_0, ABCS_CV_1, ABCS_CV_2, ABCS_CV_3, ABCS_CV_4])


ABCS_test_result.to_csv(PRJ_PATH+'ABCS_test_result.csv', index = False)


round(stats.pearsonr(ABCS_test_result['ANS'] , ABCS_test_result['FULL'])[0] , 4) # 0.7164
round(stats.pearsonr(ABCS_test_result['ANS'] , ABCS_test_result['VLS'])[0] , 4) # 0.7173
round(stats.pearsonr(ABCS_test_result['ANS'] , ABCS_test_result['VPC'])[0] , 4) # 0.7175
round(stats.pearsonr(ABCS_test_result['ANS'] , ABCS_test_result['VSC'])[0] , 4) # 0.7165







########################################
이제 bar plot 이랑 뭐든 좀 그려보자 


WORK_NAME = 'WORK_403' # 349
W_NAME = 'W403'
PRJ_NAME = 'M3V8'
MJ_NAME = 'M3V8'
MISS_NAME = 'MIS2'
PPI_NAME = '349'

PRJ_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/{}_{}_{}_{}/'.format(PRJ_NAME, W_NAME, PPI_NAME, MISS_NAME)

ABCS_test_result = pd.read_csv(PRJ_PATH+'ABCS_test_result.csv')


ABCS_test_result['tissue'] = [ '_'.join(a.split('_')[1:]) if type(a) == str else 'NA' for a in list(ABCS_test_result['CELL'])]
DC_CELL_info_filt['tissue'] = [ '_'.join(a.split('_')[1:]) if type(a) == str else 'NA' for a in list(DC_CELL_info_filt['CELL'])]


test_cell_df = pd.DataFrame({'DC_cellname' : list(set(ABCS_test_result.DC_cellname))})

cell_P = []
cell_S = []
cell_num = []


for cell in list(test_cell_df.DC_cellname) :
	tmp_test_re = ABCS_test_result[ABCS_test_result.DC_cellname == cell]
	cell_P_corr, _ = stats.pearsonr(tmp_test_re.ANS, tmp_test_re.VPC)
	cell_S_corr, _ = stats.spearmanr(tmp_test_re.ANS, tmp_test_re.VPC)
	cell_nums = tmp_test_re.shape[0]
	cell_P.append(cell_P_corr)
	cell_S.append(cell_S_corr)
	cell_num.append(cell_nums)


test_cell_df['P_COR'] = cell_P
test_cell_df['S_COR'] = cell_S
test_cell_df['cell_num'] = cell_num

test_cell_df = pd.merge(test_cell_df, DC_CELL_info_filt[['DC_cellname','tissue']], on = 'DC_cellname', how = 'left'  )

tissue_set = ['CENTRAL_NERVOUS_SYSTEM', 'LUNG', 'BREAST', 'BONE', 'OVARY', 'PROSTATE', 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'LARGE_INTESTINE', 'SKIN', 'PLEURA', 'KIDNEY' ] # list(set(test_cell_df['tissue']))
color_set = ['#CF3476','#FF7514','#025669','#308446','#84C3BE','#D53032','#4ddcfd','#ffcd36','#ac8cff',"#0000ffff","#7bff68ff"] # "#1E1E1E"
color_dict = {a : color_set[tissue_set.index(a)] for a in tissue_set}

test_cell_df['tissue_oh'] = [color_dict[a] for a in list(test_cell_df['tissue'])]




# 이쁜 그림을 위한 func 

# Pearson corr
test_cell_df = test_cell_df.sort_values('P_COR')

fig = plt.figure(figsize=(30,8))
x_pos = [a*2 for a in range(test_cell_df.shape[0])]

plt.bar(x_pos, test_cell_df['P_COR'], color=test_cell_df['tissue_oh']) # 
plt.xticks(x_pos,list(test_cell_df['DC_cellname']), rotation=90, fontsize= 18)
plt.yticks(np.arange(0, 1, step=0.2),np.round(np.arange(0, 1, step=0.2),2), fontsize= 18)
for i in range(test_cell_df.shape[0]):
	#plt.annotate(str(list(test_cell_df['cell_num'])[i]), xy=(x_pos[i],list(test_cell_df['P_COR'])[i]), ha='center', va='bottom', fontsize= 18)
	plt.annotate(str(list(np.round(test_cell_df['P_COR'],1))[i]), xy=(x_pos[i],list(test_cell_df['P_COR'])[i]), ha='center', va='bottom', fontsize= 15)

plt.legend(loc = 'upper left')
plt.tight_layout()
fig.savefig('{}/{}.png'.format(PRJ_PATH, 'new_plot_pearson'), bbox_inches = 'tight')

plt.close()





# 이쁜 그림을 위한 func  # 보고서용 다시 

# Pearson corr
test_cell_df = test_cell_df.sort_values('P_COR')

fig = plt.figure(figsize=(30,8))
x_pos = [a*2 for a in range(test_cell_df.shape[0])]

plt.bar(x_pos, test_cell_df['P_COR'], color=test_cell_df['tissue_oh']) # 
plt.xticks(x_pos,list(test_cell_df['DC_cellname']), rotation=90, fontsize= 18)
plt.yticks(np.arange(0, 1, step=0.2),np.round(np.arange(0, 1, step=0.2),2), fontsize= 18)
#plt.grid(True)
#plt.axhline(0.7)
plt.tight_layout()
fig.savefig('{}/{}.png'.format(PRJ_PATH, 'new_plot_pearson2'), bbox_inches = 'tight')
fig.savefig('{}/{}.pdf'.format(PRJ_PATH, 'new_plot_pearson2'), format="pdf", bbox_inches = 'tight')

plt.close()




max(ABCS_test_result.VPC)
min(ABCS_test_result.VPC)



max(ABCS_test_result.ANS)
min(ABCS_test_result.ANS)


# violin plot for tissue 
from matplotlib import colors as mcolors

tiss_list = tissue_set
my_order = test_cell_df.groupby(by=["tissue"])["P_COR"].mean().sort_values().iloc[::-1].index

fig, ax = plt.subplots(figsize=(25, 15))
sns.violinplot(ax = ax, data  = test_cell_df, x = 'tissue', y = 'P_COR', linewidth=1,  edgecolor="dimgrey", order=my_order,  inner = 'point') # width = 3,
violins = [c for i,c in enumerate(ax.collections) if i%2 == 0] # pollycollection 가져오기 위함 

				#'LARGE_INTESTINE', 'PROSTATE', 'OVARY', 'PLEURA', 'LUNG', 'SKIN','KIDNEY', 'CENTRAL_NERVOUS_SYSTEM', 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'BREAST', 'BONE'
				# [9, 2, 12, 1, 13, 22, 5, 6, 5, 16, 1]




# 하나밖에 없는애들은 violin 안그려져서 문제 
violins[0].set_facecolor(mcolors.to_rgba(color_dict['LARGE_INTESTINE'], 0.7))
violins[1].set_facecolor(mcolors.to_rgba(color_dict['OVARY'], 0.7))
violins[2].set_facecolor(mcolors.to_rgba(color_dict['PROSTATE'], 0.7))
violins[3].set_facecolor(mcolors.to_rgba(color_dict['LUNG'], 0.7))
violins[4].set_facecolor(mcolors.to_rgba(color_dict['KIDNEY'], 0.7))
violins[5].set_facecolor(mcolors.to_rgba(color_dict['SKIN'], 0.7))
violins[6].set_facecolor(mcolors.to_rgba(color_dict['CENTRAL_NERVOUS_SYSTEM'], 0.7))
violins[7].set_facecolor(mcolors.to_rgba(color_dict['HAEMATOPOIETIC_AND_LYMPHOID_TISSUE'], 0.7))
violins[8].set_facecolor(mcolors.to_rgba(color_dict['BREAST'], 0.7))




ax.set_xlabel('tissue names', fontsize=10)
ax.set_ylabel('Pearson Corr', fontsize=10)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
ax.tick_params(axis='both', which='major', labelsize=10 )
plt.tight_layout()
plt.grid(True)
plt.savefig(os.path.join(PRJ_PATH,'tissue_pearson.{}.png'.format(W_NAME)), dpi = 300)

plt.close()



# violin plot for tissue  22222 좀더 이쁜 버전 
from matplotlib import colors as mcolors

tiss_list = tissue_set
my_order = test_cell_df.groupby(by=["tissue"])["P_COR"].mean().sort_values().iloc[::-1].index

fig, ax = plt.subplots(figsize=(25, 10))
sns.violinplot(ax = ax, data  = test_cell_df, x = 'tissue', y = 'P_COR', linewidth=1,  edgecolor="dimgrey", order=my_order) # width = 3,,  inner = 'point'
violins = [c for i,c in enumerate(ax.collections) if i%2 == 0] # pollycollection 가져오기 위함 

				#'LARGE_INTESTINE', 'PROSTATE', 'OVARY', 'PLEURA', 'LUNG', 'SKIN','KIDNEY', 'CENTRAL_NERVOUS_SYSTEM', 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'BREAST', 'BONE'
				# [9, 2, 12, 1, 13, 22, 5, 6, 5, 16, 1]



# 하나밖에 없는애들은 violin 안그려져서 문제 
violins[0].set_facecolor(mcolors.to_rgba(color_dict['LARGE_INTESTINE'], 0.7))
violins[1].set_facecolor(mcolors.to_rgba(color_dict['OVARY'], 0.7))
violins[2].set_facecolor(mcolors.to_rgba(color_dict['PROSTATE'], 0.7))
violins[3].set_facecolor(mcolors.to_rgba(color_dict['LUNG'], 0.7))
violins[4].set_facecolor(mcolors.to_rgba(color_dict['KIDNEY'], 0.7))
violins[5].set_facecolor(mcolors.to_rgba(color_dict['SKIN'], 0.7))
violins[6].set_facecolor(mcolors.to_rgba(color_dict['CENTRAL_NERVOUS_SYSTEM'], 0.7))
violins[7].set_facecolor(mcolors.to_rgba(color_dict['HAEMATOPOIETIC_AND_LYMPHOID_TISSUE'], 0.7))
violins[8].set_facecolor(mcolors.to_rgba(color_dict['BREAST'], 0.7))



avail_cell_dict = {'PROSTATE': ['VCAP', 'PC3'], 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE': ['UHO1', 'K562', 'L1236', 'SR786', 'RPMI8226'], 'BREAST': ['BT549', 'BT474', 'MDAMB361', 'MDAMB231', 'UACC812', 'KPL1', 'MDAMB468', 'MDAMB175VII', 'HCC1500', 'HCC1419', 'MDAMB436', 'ZR751', 'MCF7', 'HS578T', 'T47D', 'CAMA1'], 'CENTRAL_NERVOUS_SYSTEM': ['T98G', 'SF539', 'U251MG', 'SNB75', 'SF268', 'SF295'], 'LUNG': ['A549', 'A427', 'HOP92', 'NCIH23', 'NCIH1650', 'NCIH520', 'HOP62', 'NCIH460', 'NCIH2122', 'NCIH226', 'EKVX', 'SKMES1', 'NCIH522'], 'LARGE_INTESTINE': ['SW620', 'LOVO', 'RKO', 'KM12', 'HCT15', 'HCT116', 'HT29', 'DLD1', 'SW837'], 'OVARY': ['NIHOVCAR3', 'OVCAR4', 'OVCAR5', 'IGROV1', 'OVCAR8', 'PA1', 'UWB1289', 'SKOV3', 'CAOV3', 'ES2', 'A2780', 'OV90'], 'SKIN': ['SKMEL5', 'HT144', 'RVH421', 'SKMEL28', 'A375', 'A101D', 'COLO792', 'UACC257', 'COLO800', 'MEWO', 'MELHO', 'A2058', 'RPMI7951', 'IPC298', 'MALME3M', 'SKMEL2', 'UACC62', 'LOXIMVI', 'WM115', 'G361', 'SKMEL30', 'COLO829'], 'BONE': ['A673'], 'KIDNEY': ['CAKI1', 'UO31', 'ACHN', 'A498', '786O'], 'PLEURA': ['MSTO211H']}
breast_check = ccle_info[ccle_info.stripped_cell_line_name.isin(avail_cell_dict['BREAST'])][['cell_line_name','lineage_molecular_subtype']]
breast_check.columns = ['DC_cellname','subclass']


test_cell_df2 = pd.merge(test_cell_df, breast_check, on = 'DC_cellname', how = 'left')
test_cell_df2['subclass'] = test_cell_df2.subclass.apply(lambda x : "NA" if type(x) != str else x)

test_cell_df2.at[test_cell_df2[test_cell_df2.DC_cellname=='ZR751'].index.item(),'subclass'] = 'luminal'
test_cell_df2.at[test_cell_df2[test_cell_df2.DC_cellname=='HS 578T'].index.item(),'subclass'] = 'basal_B'
test_cell_df2.at[test_cell_df2[test_cell_df2.DC_cellname=='KPL1'].index.item(),'subclass'] = 'luminal'
test_cell_df2.at[test_cell_df2[test_cell_df2.DC_cellname=='MDAMB436'].index.item(),'subclass'] = 'basal_B'

# test_cell_df2.subclass.factorize()
sns.swarmplot(ax = ax, data  = test_cell_df2, x = 'tissue', y = 'P_COR', hue='subclass', palette=sns.color_palette(['grey', 'yellow', 'pink', 'white','green']), order=my_order)



ax.set_xlabel('tissue names', fontsize=10)
ax.set_ylabel('Pearson Corr', fontsize=10)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, fontsize=25)
ax.tick_params(axis='both', which='major', labelsize=10 )
plt.tight_layout()
plt.grid(True)
plt.savefig(os.path.join(PRJ_PATH,'tissue_pearson2.{}.png'.format(W_NAME)), dpi = 300)

plt.close()



# violin plot for tissue  33333 좀더 이쁜 버전 
from matplotlib import colors as mcolors

tiss_list = tissue_set
my_order = test_cell_df.groupby(by=["tissue"])["P_COR"].mean().sort_values().iloc[::-1].index

fig, ax = plt.subplots(figsize=(23, 13))
sns.violinplot(ax = ax, data  = test_cell_df, x = 'tissue', y = 'P_COR', linewidth=2,  edgecolor="black", order=my_order, width = 1, inner = None) # width = 3,,  
violins = [c for i,c in enumerate(ax.collections) ] # pollycollection 가져오기 위함 # if i%2 == 0

				#'LARGE_INTESTINE', 'PROSTATE', 'OVARY', 'PLEURA', 'LUNG', 'SKIN','KIDNEY', 'CENTRAL_NERVOUS_SYSTEM', 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'BREAST', 'BONE'
				# [9, 2, 12, 1, 13, 22, 5, 6, 5, 16, 1]


# 하나밖에 없는애들은 violin 안그려져서 문제 
violins[0].set_facecolor(mcolors.to_rgba(color_dict['LARGE_INTESTINE'], 0.8))
violins[1].set_facecolor(mcolors.to_rgba(color_dict['OVARY'], 0.8))
violins[2].set_facecolor(mcolors.to_rgba(color_dict['PROSTATE'], 0.8))
violins[3].set_facecolor(mcolors.to_rgba(color_dict['LUNG'], 0.8))
violins[4].set_facecolor(mcolors.to_rgba(color_dict['KIDNEY'], 0.8))
violins[5].set_facecolor(mcolors.to_rgba(color_dict['SKIN'], 0.8))
violins[6].set_facecolor(mcolors.to_rgba(color_dict['CENTRAL_NERVOUS_SYSTEM'], 0.8))
violins[7].set_facecolor(mcolors.to_rgba(color_dict['HAEMATOPOIETIC_AND_LYMPHOID_TISSUE'], 0.8))
violins[8].set_facecolor(mcolors.to_rgba(color_dict['BREAST'], 0.8))


avail_cell_dict = {'PROSTATE': ['VCAP', 'PC3'], 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE': ['UHO1', 'K562', 'L1236', 'SR786', 'RPMI8226'], 'BREAST': ['BT549', 'BT474', 'MDAMB361', 'MDAMB231', 'UACC812', 'KPL1', 'MDAMB468', 'MDAMB175VII', 'HCC1500', 'HCC1419', 'MDAMB436', 'ZR751', 'MCF7', 'HS578T', 'T47D', 'CAMA1'], 'CENTRAL_NERVOUS_SYSTEM': ['T98G', 'SF539', 'U251MG', 'SNB75', 'SF268', 'SF295'], 'LUNG': ['A549', 'A427', 'HOP92', 'NCIH23', 'NCIH1650', 'NCIH520', 'HOP62', 'NCIH460', 'NCIH2122', 'NCIH226', 'EKVX', 'SKMES1', 'NCIH522'], 'LARGE_INTESTINE': ['SW620', 'LOVO', 'RKO', 'KM12', 'HCT15', 'HCT116', 'HT29', 'DLD1', 'SW837'], 'OVARY': ['NIHOVCAR3', 'OVCAR4', 'OVCAR5', 'IGROV1', 'OVCAR8', 'PA1', 'UWB1289', 'SKOV3', 'CAOV3', 'ES2', 'A2780', 'OV90'], 'SKIN': ['SKMEL5', 'HT144', 'RVH421', 'SKMEL28', 'A375', 'A101D', 'COLO792', 'UACC257', 'COLO800', 'MEWO', 'MELHO', 'A2058', 'RPMI7951', 'IPC298', 'MALME3M', 'SKMEL2', 'UACC62', 'LOXIMVI', 'WM115', 'G361', 'SKMEL30', 'COLO829'], 'BONE': ['A673'], 'KIDNEY': ['CAKI1', 'UO31', 'ACHN', 'A498', '786O'], 'PLEURA': ['MSTO211H']}
breast_check = ccle_info[ccle_info.stripped_cell_line_name.isin(avail_cell_dict['BREAST'])][['cell_line_name','lineage_molecular_subtype']]
breast_check.columns = ['DC_cellname','subclass']


test_cell_df2 = pd.merge(test_cell_df, breast_check, on = 'DC_cellname', how = 'left')
test_cell_df2['subclass'] = test_cell_df2.subclass.apply(lambda x : "NA" if type(x) != str else x)

test_cell_df2.at[test_cell_df2[test_cell_df2.DC_cellname=='ZR751'].index.item(),'subclass'] = 'luminal'
test_cell_df2.at[test_cell_df2[test_cell_df2.DC_cellname=='HS 578T'].index.item(),'subclass'] = 'basal_B'
test_cell_df2.at[test_cell_df2[test_cell_df2.DC_cellname=='KPL1'].index.item(),'subclass'] = 'luminal'
test_cell_df2.at[test_cell_df2[test_cell_df2.DC_cellname=='MDAMB436'].index.item(),'subclass'] = 'basal_B'

# test_cell_df2.subclass.factorize()
sns.swarmplot(ax = ax, data  = test_cell_df2, x = 'tissue', y = 'P_COR', order=my_order, 
hue='subclass', linewidth=0.1, edgecolor="white", palette=sns.color_palette(['black', 'black', 'black', 'black','black']))


ax.set_xlabel('tissue names', fontsize=15)
ax.set_ylabel('Pearson Corr', fontsize=15)
ax.set_xticks(ax.get_xticks())
ax.set_yticks(ax.get_yticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, fontsize=25)

ytick_names = np.array(['', 0.0 , 0.2, 0.4, 0.6, 0.8, 1.0, ''])
ax.set_yticklabels(ytick_names,  fontsize=25)
ax.tick_params(axis='both', which='major', labelsize=20 )
ax.get_legend().remove()
plt.tight_layout()
plt.grid(True, linewidth = 0.2, linestyle = '--')
plt.savefig(os.path.join(PRJ_PATH,'tissue_pearson3.{}.png'.format(W_NAME)), dpi = 300)
plt.savefig('{}/{}.pdf'.format(PRJ_PATH, 'tissue_pearson3.{}.pdf'.format(W_NAME)), format="pdf", bbox_inches = 'tight')

plt.close()










