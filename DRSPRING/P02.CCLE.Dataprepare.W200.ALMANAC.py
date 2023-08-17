
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


NETWORK_PATH = '/st06/jiyeonH/13.DD_SESS/HumanNetV3/'
LINCS_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/' 
DATA_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V3_FULL/'
DC_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/'


'CCLE', 'GCSI', 'FRIEDMAN', 'DYALL', 
'NCATS_ES(NAMPT+PARP)', 'MATHEWS', 'SCHMIDT', 
'FALLAHI-SICHANI', 'FRIEDMAN2', 'GRAY', 'MILLER', 
'ALMANAC', 'WILSON', 'BOBROWSKI', 'BEATAML', 
'YOHE', 'ONEIL', 'NCATS_MDR_CS', 'UHNBREAST', 
'MOTT', 'NCATS_2D_3D', 'NCATS_ATL', 'NCATS_HL', 
'FIMM', 'FLOBAK', 'GDSC1', 'PHELAN', 'ASTRAZENECA', 
'NCATS_DIPG', 'CTRPV2', 'CLOUD', 'NCATS_ES(FAKI/AURKI)', 
'NCATS_SARS-COV-2DPI', 'FORCINA'






#######################################################




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

######################################


# LINCS 에서 주는 내용들 하나로 잘 정리해두기 
#BETA_BIND_ORI = pd.read_csv('/st06/jiyeonH/11.TOX/MY_TRIAL_5/'+"BETA_DATA_for_20220705_978.csv")
#BETA_BIND_ADD = pd.read_csv('/st06/jiyeonH/11.TOX/MY_TRIAL_5/BETA_DATA_for_20230614_978.csv')

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


###################################################
###################################################
###################################################
###################################################

# 여기서부터 drugcomb 데이터 필터링 

DC_PATH = '/st06/jiyeonH/13.DD_SESS/DrugComb.1.5/' 

# sumary info 
summ =  pd.read_csv('/st06/jiyeonH/13.DD_SESS/DrugComb.1.5/summary_v_1_5.csv', low_memory=False)
summ2 = summ[['block_id','study_name']]

summ3 = summ2[summ2.study_name=='ONEIL'] # 92208 blocks 
summ3 = summ2[summ2.study_name=='ALMANAC'] # 311604 blocks
summ3 = summ2[summ2.study_name=='FORCINA'] # 1818 blocks
summ3 = summ2[summ2.study_name=='ASTRAZENECA'] # 19475 blocks
summ3 = summ2[summ2.study_name=='YOHE'] # 270 blocks
summ3 = summ2[summ2.study_name=='FLOBAK'] # 9984 blocks 


DC_tmp_DF1 = pd.read_csv(DC_PATH+'TOTAL_BLOCK.csv', low_memory=False) # 1469182

Oneil_block = DC_tmp_DF1[DC_tmp_DF1.block_id.isin(summ3.block_id)]
Oneil_block_1 = Oneil_block[['drug_row_id', 'drug_col_id', 'cell_line_id','synergy_loewe','quality']]
# -> 92208

DC_tmp_DF1_re = Oneil_block_1[['drug_row_id','drug_col_id','cell_line_id','synergy_loewe','quality']]
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
# 92208

DC_score_filter[['drug_row_id_re','drug_col_id_re','cell_line_id','synergy_loewe']].drop_duplicates()
# 92208


DC_score_filter[['drug_row_id_re','drug_col_id_re','cell_line_id']].drop_duplicates()
# 22737



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

DC_DATA_filter = DC_tmp_DF2[['drug_row_id_re','drug_col_id_re','cell_line_id','synergy_loewe']] 
DC_DATA_filter2 = DC_DATA_filter.drop_duplicates() 

DC_DATA_filter3 = DC_DATA_filter2[DC_DATA_filter2.drug_row_id_re > 0] 
DC_DATA_filter4 = DC_DATA_filter3[DC_DATA_filter3.drug_col_id_re > 0] 
DC_DATA_filter4.cell_line_id # unique 295
DC_DATA_filter4[['drug_row_id_re','drug_col_id_re','cell_line_id']].drop_duplicates() # 22737
DC_DATA_filter4[['drug_row_id_re','drug_col_id_re']].drop_duplicates() # 583
len(list(set(list(DC_DATA_filter4.drug_row_id_re) + list(DC_DATA_filter4.drug_col_id_re)))) # 38




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
DC_DATA5 = pd.merge(DC_DATA_filter4, DC_DRUG_DF2, on ='drug_row_id_re', how='left' ) # 92208

DC_DRUG_DF2.columns = ['drug_col_id_re','drug_col','drug_col_CID', 'drug_col_sm']
DC_DATA6 = pd.merge(DC_DATA5, DC_DRUG_DF2, on ='drug_col_id_re', how='left') # 92208


#  Add cell data and cid filter
DC_DATA7_1 = pd.merge(DC_DATA6, DC_CELL_DF2 , on= 'cell_line_id', how = 'left') # 
DC_DATA7_1 = DC_DATA7_1[['drug_row_CID','drug_col_CID','DrugCombCCLE','synergy_loewe']].drop_duplicates() # 92208






# filtering 
DC_DATA7_2 = DC_DATA7_1[DC_DATA7_1.drug_row_CID>0] # 740882 -> 737104
DC_DATA7_3 = DC_DATA7_2[DC_DATA7_2.drug_col_CID>0] # 737104 -> 725496
ccle_t=[True if type(a)==str else False for a in list(DC_DATA7_3.DrugCombCCLE)]
DC_DATA7_4_ccle = DC_DATA7_3[ccle_t] #  720249
TF_check = [True if np.isnan(a)==False else False for a in DC_DATA7_4_ccle.synergy_loewe] 
DC_DATA7_5_ccle = DC_DATA7_4_ccle[TF_check] # 719946
DC_DATA7_6_ccle = DC_DATA7_5_ccle[DC_DATA7_5_ccle.DrugCombCCLE != 'NA'] # 82712


DC_ccle_final = DC_DATA7_6_ccle[['drug_row_CID','drug_col_CID','DrugCombCCLE']].drop_duplicates() # 20405
DC_ccle_final_dup = DC_DATA7_6_ccle[['drug_row_CID','drug_col_CID','DrugCombCCLE', 'synergy_loewe']].drop_duplicates() # 82712

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
# 196388
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


tmp = DATA_AX_BO[['drug_row_CID','DrugCombCCLE']].drop_duplicates() # 1274
tmp1 = [str(a) for a in tmp['drug_row_CID']]
tmp2 = [tmp1[a] + "__" + list(tmp['DrugCombCCLE'])[a] for a in range(tmp.shape[0])] # 1274
len(tmp2)



# (3) AO BX 
FILTER_AO_BX = [a for a in range(CCLE_DC_BETA_2.shape[0]) if (type(CCLE_DC_BETA_2.ROW_BETA_sig_id[a]) == str) & (type(CCLE_DC_BETA_2.COL_BETA_sig_id[a]) != str)]
DATA_AO_BX = CCLE_DC_BETA_2.loc[FILTER_AO_BX] # 14998
DATA_AO_BX[['drug_row_CID','drug_col_CID','DrugCombCCLE']].drop_duplicates() # 12926
DATA_AO_BX[['ROW_BETA_sig_id','COL_BETA_sig_id','DrugCombCCLE']].drop_duplicates() # 449
DATA_AO_BX[['ROW_pert_id','COL_pert_id','DrugCombCCLE']].drop_duplicates() #  449
DATA_AO_BX[['ROW_CAN_SMILES','COL_CAN_SMILES','DrugCombCCLE']].drop_duplicates() #  12915 -> 겹치는 애들 제거해야하는걸로! 
DATA_AO_BX_cids = list(set(list(DATA_AO_BX.drug_row_CID) + list(DATA_AO_BX.drug_col_CID))) # 274 


tmp = DATA_AO_BX[['drug_col_CID','DrugCombCCLE']].drop_duplicates()
tmp1 = [str(a) for a in tmp['drug_col_CID']]
tmp2 = [tmp1[a] + "__" + list(tmp['DrugCombCCLE'])[a] for a in range(len(tmp1))] # 900


miss_2 = pd.concat([DATA_AX_BO, DATA_AO_BX])



# (4) AX BX 
FILTER_AX_BX = [a for a in range(CCLE_DC_BETA_2.shape[0]) if (type(CCLE_DC_BETA_2.ROW_BETA_sig_id[a]) != str) & (type(CCLE_DC_BETA_2.COL_BETA_sig_id[a]) != str)]
DATA_AX_BX = CCLE_DC_BETA_2.loc[FILTER_AX_BX] # 584465
DATA_AX_BX = DATA_AX_BX[DATA_AX_BX.DrugCombCCLE!='NA']
DATA_AX_BX[['drug_row_CID','drug_col_CID','DrugCombCCLE']].drop_duplicates() # 506710
DATA_AX_BX[['ROW_BETA_sig_id','COL_BETA_sig_id','DrugCombCCLE']].drop_duplicates() # 232 의미 없음 
DATA_AX_BX[['ROW_pert_id','COL_pert_id','DrugCombCCLE']].drop_duplicates() #  232 의미 없음 
DATA_AX_BX[['ROW_CAN_SMILES','COL_CAN_SMILES','DrugCombCCLE']].drop_duplicates() #  8404
DATA_AX_BX_cids = list(set(list(DATA_AX_BX.drug_row_CID) + list(DATA_AX_BX.drug_col_CID))) # 4280 


tmp_r = DATA_AX_BX[['drug_row_CID','DrugCombCCLE']].drop_duplicates() # 18162
tmp_c = DATA_AX_BX[['drug_col_CID','DrugCombCCLE']].drop_duplicates() # 12702
tmp1 = [str(a) for a in tmp_r['drug_row_CID']]
tmp2 = [str(a) for a in tmp_c['drug_col_CID']]
tmp3 = [tmp1[a] + "__" + list(tmp_r['DrugCombCCLE'])[a] for a in range(len(tmp1))] # 
tmp4 = [tmp2[a] + "__" + list(tmp_c['DrugCombCCLE'])[a] for a in range(len(tmp2))] # 
len(set(tmp3+tmp4)) # 19429


DATA_AO_BO['type'] = 'AOBO' # 94360
DATA_AX_BO['type'] = 'AXBO' # 17376
DATA_AO_BX['type'] = 'AOBX' # 11188
DATA_AX_BX['type'] = 'AXBX' # 73464

# 아? HELA 랑 HAP1 이... 상관이 없네? 
# 왜냐면 이미 synergy 실험내용이 없음

len(set(DATA_AO_BO.cid_cid_cell))
len(set(DATA_AO_BO.DrugCombCCLE))
len(set(list(DATA_AO_BO.drug_row_CID) + list(DATA_AO_BO.drug_col_CID)))


len(set(list(DATA_AX_BO.cid_cid_cell) + list(DATA_AO_BX.cid_cid_cell)))
len(set(list(DATA_AX_BO.DrugCombCCLE) + list(DATA_AO_BX.DrugCombCCLE)))
len(set(list(DATA_AX_BO.drug_row_CID) + list(DATA_AX_BO.drug_col_CID)+list(DATA_AO_BX.drug_row_CID) + list(DATA_AO_BX.drug_col_CID)))


len(set(DATA_AX_BX.cid_cid_cell))
len(set(DATA_AX_BX.DrugCombCCLE))
len(set(list(DATA_AX_BX.drug_row_CID) + list(DATA_AX_BX.drug_col_CID)))

###############################################

A_B_C_S = pd.concat([DATA_AO_BO, DATA_AX_BO, DATA_AO_BX, DATA_AX_BX])
A_B_C_S = A_B_C_S.reset_index(drop = True) # 947047

A_B_C_S_SET = copy.deepcopy(A_B_C_S)
A_B_C_S_SET = A_B_C_S_SET.drop('synergy_loewe', axis = 1).drop_duplicates() # 776134
A_B_C_S_SET = A_B_C_S_SET.reset_index(drop = True) # 456422 


# A_B_C_S.to_csv('/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V6_349_FULL/A_B_C_S.ALMANAC.csv', sep = '\t', index= False)
# A_B_C_S = pd.read_csv('/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V6_349_FULL/A_B_C_S.ALMANAC.csv', sep = '\t', low_memory = False)

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

# A_B_C_S_SET.to_csv('/st06/jiyeonH/11.TOX/DR_SPRING/trials/A_B_C_S_SET_ALL.ALMANAC.csv', sep = '\t', index = False )
# A_B_C_S_SET = pd.read_csv('/st06/jiyeonH/11.TOX/DR_SPRING/trials/A_B_C_S_SET_ALL.ALMANAC.csv', sep = '\t', low_memory = False)

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


MJ_request_ANS = pd.read_csv(MJ_DIR+'PRJ2_EXP_ccle_all_fugcn_hhhdt3_tvt.csv')




entrez_id = list(MJ_request_ANS['entrez_id'])
MJ_request_ANS = MJ_request_ANS.drop(['entrez_id','Unnamed: 0','CID__CELL'], axis =1)
MJ_request_ANS['entrez_id'] = entrez_id

ord = [list(MJ_request_ANS.entrez_id).index(a) for a in BETA_ENTREZ_ORDER]
MJ_request_ANS_re = MJ_request_ANS.loc[ord] 



# fu (M3 & M33 & M3V3 & M3V4) 
A_B_C_S_SET_MJ = A_B_C_S_SET[A_B_C_S_SET.ROWCHECK.isin(MJ_request_ANS.columns)]
A_B_C_S_SET_MJ = A_B_C_S_SET_MJ[A_B_C_S_SET_MJ.COLCHECK.isin(MJ_request_ANS.columns)]
A_B_C_S_SET_MJ = A_B_C_S_SET_MJ.reset_index(drop = True)



# fu (M3 & M33)
def get_MJ_data( CHECK ): 
	if CHECK in list(MJ_request_ANS_re.columns) :
		RES = MJ_request_ANS_re[CHECK]
		OX = 'O'
	else : 
		RES = [0]*349        ##############
		OX = 'X'
	return list(RES), OX



# 여기서부터는 그냥 줄여서 할거임. 시간적인 문제도 있고해서. 

A_B_C_S_SET_UNIQ = copy.deepcopy(A_B_C_S_SET_MJ) # 613961
A_B_C_S_SET_UNIQ['type'] = [ 'AXBO' if a == 'AOBX' else a for a in A_B_C_S_SET_UNIQ['type']]
A_B_C_S_SET_UNIQ_2 = A_B_C_S_SET_UNIQ[['cid_cid_cell','sig_sig_cell','type']].drop_duplicates()
A_B_C_S_SET_UNIQ_3 = A_B_C_S_SET_UNIQ_2[['cid_cid_cell','type']].drop_duplicates()
A_B_C_S_SET_UNIQ_3 = A_B_C_S_SET_UNIQ_3.reset_index(drop=True)

	# 실험을 위해 조금만 일단! 
	# A_B_C_S_SET_UNIQ_3 = A_B_C_S_SET_UNIQ_3[A_B_C_S_SET_UNIQ_3.type.isin(['AOBO', 'AXBO'])]
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
		#DrugA_SIGS = list(BETA_SELEC_SIG_wCell2[(BETA_SELEC_SIG_wCell2.CID == DrugA_CID) & (BETA_SELEC_SIG_wCell2.ccle_name == CELL)]['sig_id'])
		#DrugB_SIGS = list(BETA_SELEC_SIG_wCell2[(BETA_SELEC_SIG_wCell2.CID == DrugB_CID) & (BETA_SELEC_SIG_wCell2.ccle_name == CELL)]['sig_id'])
		#EXP_A_list = [get_LINCS_data(AA) for AA in DrugA_SIGS]
		#EXP_A = torch.mean(torch.concat(EXP_A_list, axis =1), axis =1).view(-1,1)
		#EXP_B_list = [get_LINCS_data(BB) for BB in DrugB_SIGS]
		#EXP_B = torch.mean(torch.concat(EXP_B_list, axis =1), axis =1).view(-1,1)
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


PRJ_NAME = 'M3V6_349_MISS2_ALMANAC' # save the original ver 

SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V6_349_FULL/'


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
	if len(set(groupp['TF'])) ==1 :
		lensets.append('O')
	else :
		lensets.append('X')

synOX = pd.DataFrame({'cid_cid_cell' : ids, 'SYN_OX' : lensets})


A_B_C_S_SET_ADD = pd.merge(A_B_C_S_SET_ADD, synOX, on = 'cid_cid_cell', how ='left')




A_B_C_S_SET_CIDS = list(set(list(A_B_C_S_SET_ADD.CID_A)+list(A_B_C_S_SET_ADD.CID_B)))
A_B_C_S_SET_CIDS.sort()
gene_ids = list(BETA_ORDER_DF.gene_id)





# TARGET (1) time consuming 
# TARGET (1)
# TARGET (1)

TARGET_PATH = '/st06/jiyeonH/13.DD_SESS/01.PRJ2/'
TARGET_DB = pd.read_csv(TARGET_PATH+'TARGET_CID_ENTREZ.csv', sep ='\t', index_col = 0)

TARGET_DB_RE = TARGET_DB[TARGET_DB.CID_RE.isin(A_B_C_S_SET_CIDS)]
TARGET_DB_RE = TARGET_DB_RE[TARGET_DB_RE.ENTREZ_RE.isin(gene_ids)]


target_cids = list(set(TARGET_DB_RE.CID))
target_cids.sort()
gene_ids = list(BETA_ORDER_DF.gene_id)
gene_ids.sort()


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



# PRJ_NAME = 'check_the_full'
A_B_C_S_SET_ADD_Bcor.to_csv(SAVE_PATH+'{}.A_B_C_S_SET_ADD.csv'.format(PRJ_NAME))




