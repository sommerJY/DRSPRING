
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




# 아예 처음부터 오닐로 진행하기 



DC_PATH = '/st06/jiyeonH/13.DD_SESS/DrugComb.1.5/' 
LINCS_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/' 
TARGET_PATH = '/st06/jiyeonH/13.DD_SESS/01.PRJ2/'
PC_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/'


# sumary info 
summ =  pd.read_csv('/st06/jiyeonH/13.DD_SESS/DrugComb.1.5/summary_v_1_5.csv', low_memory=False)
summ2 = summ[['block_id','study_name']]
summ3 = summ2[summ2.study_name=='ONEIL']



# Drug Comb 데이터 가져오기 
# synergy info 
DC_tmp_DF1 = pd.read_csv(DC_PATH+'TOTAL_BLOCK.csv', low_memory=False) # 1469182

Oneil_block = DC_tmp_DF1[DC_tmp_DF1.block_id.isin(summ3.block_id)]
Oneil_block_1 = Oneil_block[['drug_row_id', 'drug_col_id', 'cell_line_id','synergy_loewe','quality']]
# -> 92208

DC_tmp_DF1_re = Oneil_block_1[['drug_row_id','drug_col_id','cell_line_id','synergy_loewe','quality']]
DC_tmp_DF1_re['drug_row_id_re'] = [float(a) for a in list(Oneil_block_1['drug_row_id'])]
DC_tmp_DF1_re['drug_col_id_re'] = [float(a) for a in list(Oneil_block_1['drug_col_id'])]


DC_tmp_DF2 = DC_tmp_DF1_re[DC_tmp_DF1_re['quality'] != 'bad'] # 1457561


DC_tmp_DF_ch = DC_tmp_DF2[DC_tmp_DF2.cell_line_id>0]
DC_tmp_DF_ch = DC_tmp_DF_ch.reset_index(drop=True)

DC_tmp_DF_ch_plus = DC_tmp_DF_ch[DC_tmp_DF_ch.synergy_loewe > 0 ]
DC_tmp_DF_ch_minus = DC_tmp_DF_ch[DC_tmp_DF_ch.synergy_loewe < 0 ]
DC_tmp_DF_ch_zero = DC_tmp_DF_ch[DC_tmp_DF_ch.synergy_loewe == 0 ]

DC_score_filter = pd.concat([DC_tmp_DF_ch_plus, DC_tmp_DF_ch_minus, DC_tmp_DF_ch_zero ])

DC_score_filter[['drug_row_id_re','drug_col_id_re','cell_line_id','synergy_loewe']].drop_duplicates()
# 변화 없음 


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

DC_DATA_filter = DC_tmp_DF2[['drug_row_id_re','drug_col_id_re','cell_line_id','synergy_loewe']] # 1457561
DC_DATA_filter2 = DC_DATA_filter.drop_duplicates() # 1374958 -> 1363698

DC_DATA_filter3 = DC_DATA_filter2[DC_DATA_filter2.drug_row_id_re > 0] # 1374958 -> 1363698
DC_DATA_filter4 = DC_DATA_filter3[DC_DATA_filter3.drug_col_id_re > 0] # 751450 -> 740884
DC_DATA_filter4.cell_line_id # unique 295
DC_DATA_filter4[['drug_row_id_re','drug_col_id_re','cell_line_id']].drop_duplicates() # 648516
DC_DATA_filter4[['drug_row_id_re','drug_col_id_re']].drop_duplicates() # 75174
len(list(set(list(DC_DATA_filter4.drug_row_id_re) + list(DC_DATA_filter4.drug_col_id_re)))) # 4327
# 38


DC_DATA_filter_chch = DC_DATA_filter4[['drug_row_id_re','drug_col_id_re','cell_line_id']].drop_duplicates()
dc_ch_1 = list(DC_DATA_filter_chch['drug_row_id_re'])
dc_ch_2 = list(DC_DATA_filter_chch['drug_col_id_re'])
dc_ch_3 = list(DC_DATA_filter_chch['cell_line_id'])


# 순서 고려해서 unique 아이디 만들기 (크기 비교가 str 사이에서도 성립한다는거 아니?)
DC_DATA_filter_chch['id_id_cell'] = [str(int(dc_ch_1[i])) + '___' + str(int(dc_ch_2[i]))+ '___' + str(int(dc_ch_3[i])) if dc_ch_1[i] < dc_ch_2[i] else str(int(dc_ch_2[i])) + '___' + str(int(dc_ch_1[i]))+ '___' + str(int(dc_ch_3[i])) for i in range(DC_DATA_filter_chch.shape[0])]




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
# 38








# 한번만 더 확인 
DC_DATA_filter_chch = DC_ccle_final[['drug_row_CID','drug_col_CID','DrugCombCCLE']].drop_duplicates()
dc_ch_1 = list(DC_DATA_filter_chch['drug_row_CID'])
dc_ch_2 = list(DC_DATA_filter_chch['drug_col_CID'])
dc_ch_3 = list(DC_DATA_filter_chch['DrugCombCCLE'])


# 순서 고려해서 unique 아이디 만들기 (크기 비교가 str 사이에서도 성립한다는거 아니?)
DC_DATA_filter_chch['cid_cid_cell'] = [str(int(dc_ch_1[i])) + '___' + str(int(dc_ch_2[i]))+ '___' + str(dc_ch_3[i]) if dc_ch_1[i] < dc_ch_2[i] else str(int(dc_ch_2[i])) + '___' + str(int(dc_ch_1[i]))+ '___' + str(dc_ch_3[i]) for i in range(DC_DATA_filter_chch.shape[0])]

len(DC_DATA_filter_chch.cid_cid_cell) # 460739
len(set(DC_DATA_filter_chch.cid_cid_cell)) # 450002

# 그렇습니다. DC ID 상에서는 안겹쳤었는데 CID 로는 겹치는게 있네욤 (오닐은 없음)
DC_dup_total = DC_DATA_filter_chch[DC_DATA_filter_chch['cid_cid_cell'].duplicated(keep = False)].sort_values('cid_cid_cell') # 21474 개 인거지. 



# LINCS data filter 
# LINCS data filter 
# LINCS data filter 


BETA_BIND = pd.read_csv('/st06/jiyeonH/11.TOX/MY_TRIAL_5/'+"BETA_DATA_for_20220705_978.csv")
BETA_CP_info = pd.read_table('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+'compoundinfo_beta.txt') # pert 34419 , cansmiles : 
BETA_CEL_info = pd.read_table('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+'cellinfo_beta.txt')
BETA_SIG_info = pd.read_table('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+'siginfo_beta.txt', low_memory = False)








# pert type 확인 
filter1 = BETA_SIG_info[BETA_SIG_info.pert_type.isin([ 'ctl_vehicle', 'ctl_untrt' ,'trt_cp' ])]
# 764996 
filter2 = filter1[filter1.is_exemplar_sig==1]
# 136460 samples, 58 cells, 30456 compounds 



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
BETA_EXM = pd.merge(filter2, LINCS_PERT_MATCH, on='pert_id', how = 'left')
BETA_EXM2 = BETA_EXM[BETA_EXM.CID > 0] # 128038 # 이건 늘어났음 

BETA_CEL_info2 = BETA_CEL_info[['cell_iname','cellosaurus_id','ccle_name']] # 240 
BETA_SELEC_SIG_wCell = pd.merge(BETA_EXM2, BETA_CEL_info2, on = 'cell_iname', how = 'left') # 128038
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
BETA_CID_CCLE_SIG['CID'] = [int(a) for a in list(BETA_CID_CCLE_SIG['CID']) ] # 111012 
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

# 순서 고려해서 unique 아이디 만들기 (크기 비교가 str 사이에서도 성립한다는거 아니?)
CCLE_DC_BETA_2['cid_cid_cell'] = [str(int(aa[i])) + '___' + str(int(bb[i]))+ '___' + cc[i] if aa[i] < bb[i] else str(int(bb[i])) + '___' + str(int(aa[i]))+ '___' + cc[i] for i in range(CCLE_DC_BETA_2.shape[0])]



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


DATA_AO_BO['type'] = 'AOBO' # 3032
DATA_AX_BO['type'] = 'AXBO' # 11967
DATA_AO_BX['type'] = 'AOBX' # 14998
DATA_AX_BX['type'] = 'AXBX' # 584465

# 아? HELA 랑 HAP1 이... 상관이 없네? 
# 왜냐면 이미 synergy 실험내용이 없음

############################################################






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



# Graph 확인 

JY_GRAPH = MY_G
JY_GRAPH_ORDER = MY_G.nodes()
JY_ADJ = nx.adjacency_matrix(JY_GRAPH)

JY_ADJ_tmp = torch.LongTensor(JY_ADJ.toarray())
JY_ADJ_IDX = JY_ADJ_tmp.to_sparse().indices()
JY_IDX_WEIGHT = MY_WEIGHT_SCORE




# data construction 

(1) NONE MISSING : MISS_0

A_B_C_S = DATA_AO_BO.reset_index(drop = True) # 11379
A_B_C_S[['drug_row_CID','drug_col_CID', 'DrugCombCCLE']].drop_duplicates()


(2) one missing : MISS_1
A_B_C_S = pd.concat([DATA_AO_BO, DATA_AX_BO, DATA_AO_BX])
A_B_C_S = A_B_C_S.reset_index(drop = True) # 38344 
A_B_C_S[['drug_row_CID','drug_col_CID', 'DrugCombCCLE']].drop_duplicates()


(3) two missing : MISS_2
A_B_C_S = pd.concat([DATA_AO_BO, DATA_AX_BO, DATA_AO_BX, DATA_AX_BX])
A_B_C_S = A_B_C_S.reset_index(drop = True) # 622809
A_B_C_S[['drug_row_CID','drug_col_CID', 'DrugCombCCLE']].drop_duplicates()


A_B_C_S_SET = copy.deepcopy(A_B_C_S)
A_B_C_S_SET = A_B_C_S_SET.drop('synergy_loewe', axis = 1).drop_duplicates()
A_B_C_S_SET = A_B_C_S_SET.reset_index(drop = True) # 456422 






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

A_B_C_S_SET.to_csv('/st06/jiyeonH/11.TOX/DR_SPRING/trials/A_B_C_S_SET_ALL_ONEIL.csv', sep = '\t', index = False )
# A_B_C_S_SET = pd.read_csv('/st06/jiyeonH/11.TOX/DR_SPRING/trials/A_B_C_S_SET_ALL_ONEIL.csv', sep = '\t')


max_len = max(list(A_B_C_S_SET['ROW_len'])+list(A_B_C_S_SET['COL_len']))

A_B_C_S_SET_rlen = A_B_C_S_SET[A_B_C_S_SET.ROW_len<=50]
A_B_C_S_SET_clen = A_B_C_S_SET_rlen[A_B_C_S_SET_rlen.COL_len<=50]

A_B_C_S_SET = A_B_C_S_SET_clen.reset_index(drop=True) # 
A_B_C_S_SET.columns = [
	'drug_row_CID', 'drug_col_CID', 'DrugCombCCLE', 'ROWCHECK', 'COLCHECK',
	   'ROW_CAN_SMILES', 'COL_CAN_SMILES', 'ROW_pert_id', 'ROW_BETA_sig_id',
	   'COL_pert_id', 'COL_BETA_sig_id', 'cid_cid_cell', 'type', 'ROW_len',
	   'COL_len'
]




# LINCS exp order 따지기 


BETA_ORDER_pre = [list(LINCS_978.new_node).index(a) for a in JY_GRAPH_ORDER]
BETA_ORDER_DF = LINCS_978.iloc[BETA_ORDER_pre] # 어차피 ref 다르고 같은 애들이라 괜춘 
BETA_ENTREZ_ORDER = list(BETA_ORDER_DF.gene_id)
BETA_SYMBOL_ORDER = list(BETA_ORDER_DF.gene_symbol)
BETA_NEWNOD_ORDER = list(BETA_ORDER_DF.new_node)


def get_LINCS_data(DRUG_SIG):
	Drug_EXP = BETA_BIND[['id',DRUG_SIG]]
	BIND_ORDER =[list(BETA_BIND.id).index(a) for a in BETA_ORDER_DF.gene_id]
	Drug_EXP_ORD = Drug_EXP.iloc[BIND_ORDER]
	#
	return list(Drug_EXP_ORD[DRUG_SIG])



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




def get_synergy_data(cid_cid_cell):
	ABCS1 = A_B_C_S[A_B_C_S.cid_cid_cell == cid_cid_cell]
	synergy_score = np.median(ABCS1.synergy_loewe) # 원래는 무조건 median
	return synergy_score





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



MJ_DIR = '/st06/jiyeonH/13.DD_SESS/01.PRJ2/'

MJ_request_ANS_PRE = pd.read_csv(MJ_DIR+'/PRJ2_EXP_ccle_all_fugcn_hhh3.csv')
MJ_request_ANS_786O = pd.read_csv(MJ_DIR+'/PRJ2_EXP_ccle_cell786O_fugcn_hhh3.csv')

MJ_request_ANS = pd.concat([MJ_request_ANS_PRE, MJ_request_ANS_786O], axis =1)
entrez_id = list(MJ_request_ANS.entrez_id.iloc[:,1])
MJ_request_ANS = MJ_request_ANS.drop(['entrez_id'], axis =1)
MJ_request_ANS['entrez_id'] = entrez_id




# fu (M3 & M33 & M3V3 & M3V4) 
A_B_C_S_SET_MJ = A_B_C_S_SET[A_B_C_S_SET.ROWCHECK.isin(MJ_request_ANS.columns)]
A_B_C_S_SET_MJ = A_B_C_S_SET_MJ[A_B_C_S_SET_MJ.COLCHECK.isin(MJ_request_ANS.columns)]
A_B_C_S_SET_MJ = A_B_C_S_SET_MJ.reset_index(drop = True)


# fu (M3 & M33)
def get_MJ_data( CHECK ): 
	if CHECK in list(MJ_request_ANS.columns) :
		MJ_DATA = MJ_request_ANS[['entrez_id', CHECK]]
		ord = [list(MJ_DATA.entrez_id).index(a) for a in BETA_ENTREZ_ORDER]
		MJ_DATA_re = MJ_DATA.loc[ord] 
		RES = MJ_DATA_re[CHECK]
		OX = 'O'
	else : 
		#RES = [0]*978        ##############
		RES = [0]*349        ##############
		#RES = [0]*845        ##############
		OX = 'X'
	return list(RES), OX


max_len = 50

MY_chem_A_feat = torch.empty(size=(A_B_C_S_SET_MJ.shape[0], max_len, 64))
MY_chem_B_feat= torch.empty(size=(A_B_C_S_SET_MJ.shape[0], max_len, 64))
MY_chem_A_adj = torch.empty(size=(A_B_C_S_SET_MJ.shape[0], max_len, max_len))
MY_chem_B_adj= torch.empty(size=(A_B_C_S_SET_MJ.shape[0], max_len, max_len))
MY_syn =  torch.empty(size=(A_B_C_S_SET_MJ.shape[0],1))

#MY_g_EXP_A = torch.empty(size=(A_B_C_S_SET_MJ.shape[0], 978, 1)) ##############
#MY_g_EXP_B = torch.empty(size=(A_B_C_S_SET_MJ.shape[0], 978, 1))##############

MY_g_EXP_A = torch.empty(size=(A_B_C_S_SET_MJ.shape[0], 349, 1))##############
MY_g_EXP_B = torch.empty(size=(A_B_C_S_SET_MJ.shape[0], 349, 1))##############

#MY_g_EXP_A = torch.empty(size=(A_B_C_S_SET_MJ.shape[0], 845, 1))##############
#MY_g_EXP_B = torch.empty(size=(A_B_C_S_SET_MJ.shape[0], 845, 1))##############



Fail_ind = []
from datetime import datetime

for IND in range(MY_chem_A_feat.shape[0]): #  100
	if IND%100 == 0 : 
		print(str(IND)+'/'+str(MY_chem_A_feat.shape[0]) )
		Fail_ind
		datetime.now()
	#
	DrugA_SIG = A_B_C_S_SET_MJ.iloc[IND,]['ROW_BETA_sig_id']
	DrugB_SIG = A_B_C_S_SET_MJ.iloc[IND,]['COL_BETA_sig_id']
	DrugA_CID = A_B_C_S_SET_MJ.iloc[IND,]['drug_row_CID']
	DrugB_CID = A_B_C_S_SET_MJ.iloc[IND,]['drug_col_CID']
	DrugA_CID_CELL = A_B_C_S_SET_MJ.iloc[IND,]['ROWCHECK']
	DrugB_CID_CELL = A_B_C_S_SET_MJ.iloc[IND,]['COLCHECK']	
	Cell = A_B_C_S_SET_MJ.iloc[IND,]['DrugCombCCLE']
	cid_cid_cell = A_B_C_S_SET_MJ.iloc[IND,]['cid_cid_cell']
	dat_type = A_B_C_S_SET_MJ.iloc[IND,]['type']
	#
	k=1
	DrugA_Feat, DrugA_ADJ = check_drug_f_ts(DrugA_CID)
	DrugB_Feat, DrugB_ADJ = check_drug_f_ts(DrugB_CID)
	# 
	if dat_type == 'AOBO' :
		EXP_A = get_LINCS_data(DrugA_SIG)
		EXP_B = get_LINCS_data(DrugB_SIG)
	elif dat_type == "AXBO" : 
		EXP_A, OX = get_MJ_data(DrugA_CID_CELL)
		EXP_B = get_LINCS_data(DrugB_SIG)
		if OX == 'X':
			Fail_ind.append(IND)
	elif dat_type == "AOBX" :
		EXP_A = get_LINCS_data(DrugA_SIG)
		EXP_B, OX = get_MJ_data(DrugB_CID_CELL)
		if OX == 'X':
			Fail_ind.append(IND)
	elif dat_type == "AXBX" :
		EXP_A, OX1 = get_MJ_data(DrugA_CID_CELL)
		EXP_B, OX2 = get_MJ_data(DrugB_CID_CELL)
		if 'X' in [OX1, OX2]:
			Fail_ind.append(IND)   
	# 
	# 
	AB_SYN = get_synergy_data(cid_cid_cell)
	# 
	MY_chem_A_feat[IND] = torch.Tensor(DrugA_Feat)
	MY_chem_B_feat[IND] = torch.Tensor(DrugB_Feat)
	MY_chem_A_adj[IND] = torch.Tensor(DrugA_ADJ)
	MY_chem_B_adj[IND] = torch.Tensor(DrugB_ADJ)
	MY_g_EXP_A[IND] = torch.Tensor(EXP_A).unsqueeze(1)	
	MY_g_EXP_B[IND] = torch.Tensor(EXP_B).unsqueeze(1)
	MY_syn[IND] = torch.Tensor([AB_SYN])





selec_ind = A_B_C_S_SET_MJ.index.isin(Fail_ind)==False
A_B_C_S_SET_MJ_FAFILT = A_B_C_S_SET_MJ[selec_ind]
MY_chem_A_feat_re = MY_chem_A_feat[selec_ind]
MY_chem_B_feat_re = MY_chem_B_feat[selec_ind]
MY_chem_A_adj_re = MY_chem_A_adj[selec_ind]
MY_chem_B_adj_re = MY_chem_B_adj[selec_ind]
MY_g_EXP_A_re = MY_g_EXP_A[selec_ind]
MY_g_EXP_B_re = MY_g_EXP_B[selec_ind]
MY_syn_re = MY_syn[selec_ind]


PRJ_NAME = 'M3V5_349_MISS2_ONEIL'
SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V5_W32_349_FULL/'


torch.save(MY_chem_A_feat_re, SAVE_PATH+'{}.MY_chem_A_feat.pt'.format(PRJ_NAME))
torch.save(MY_chem_B_feat_re, SAVE_PATH+'{}.MY_chem_B_feat.pt'.format(PRJ_NAME))
torch.save(MY_chem_A_adj_re, SAVE_PATH+'{}.MY_chem_A_adj.pt'.format(PRJ_NAME))
torch.save(MY_chem_B_adj_re, SAVE_PATH+'{}.MY_chem_B_adj.pt'.format(PRJ_NAME))
torch.save(MY_g_EXP_A_re, SAVE_PATH+'{}.MY_g_EXP_A.pt'.format(PRJ_NAME))
torch.save(MY_g_EXP_B_re, SAVE_PATH+'{}.MY_g_EXP_B.pt'.format(PRJ_NAME))
torch.save(MY_syn_re, SAVE_PATH+'{}.MY_syn.pt'.format(PRJ_NAME))

A_B_C_S_SET_MJ_FAFILT.to_csv(SAVE_PATH+'{}.A_B_C_S_SET.csv'.format(PRJ_NAME))
A_B_C_S.to_csv(SAVE_PATH+'{}.A_B_C_S.csv'.format(PRJ_NAME))




##########################################
##########################################
##########################################

# 기준 index 

A_B_C_S_SET = copy.deepcopy(A_B_C_S_SET_MJ_FAFILT)
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
def get_synergy_data(cid_cid_cell):
    ABCS1 = A_B_C_S[A_B_C_S.cid_cid_cell == cid_cid_cell]
    #
    if len(set(ABCS1.synergy_loewe>0)) ==1 : # 일관성 확인 
        OX = 'O'
    else: 
        OX = 'X'
    return  OX


OX_list = []

for IND in range(A_B_C_S_SET_ADD.shape[0]) :
    if IND%100 == 0 :
        print(str(IND)+'/'+str(A_B_C_S_SET_ADD.shape[0]) )
        datetime.now()
    cid_cid_cell = A_B_C_S_SET_ADD.iloc[IND,]['cid_cid_cell']
    OX = get_synergy_data(cid_cid_cell)
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


A_B_C_S_SET_ADD.to_csv(SAVE_PATH+'{}.A_B_C_S_SET_ADD.csv'.format(PRJ_NAME))
