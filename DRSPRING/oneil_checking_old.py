


# check for Oneil data 


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









# sumary info 
summ =  pd.read_csv('/st06/jiyeonH/13.DD_SESS/DrugComb.1.5/summary_v_1_5.csv', low_memory=False)
summ2 = summ[['block_id','study_name']]
summ3 = summ2[summ2.study_name=='ONEIL']





# Drug Comb 데이터 가져오기 
# synergy info 
DC_tmp_DF1 = pd.read_csv(DC_PATH+'TOTAL_BLOCK.csv', low_memory=False) # 1469182
DC_tmp_DF1_O = DC_tmp_DF1[DC_tmp_DF1.block_id.isin(summ3.block_id)]
DC_tmp_DF1_re = DC_tmp_DF1_O[['drug_row_id','drug_col_id','cell_line_id','synergy_loewe','quality']]
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


DC_ccle_final = DC_DATA7_6_ccle[['drug_row_CID','drug_col_CID','DrugCombCCLE']].drop_duplicates() # 554128
DC_ccle_final_dup = DC_DATA7_6_ccle[['drug_row_CID','drug_col_CID','DrugCombCCLE', 'synergy_loewe']].drop_duplicates() # 730348 -> 719946

DC_ccle_final_cids = list(set(list(DC_ccle_final_dup.drug_row_CID) + list(DC_ccle_final_dup.drug_col_CID)))
# 3146



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
BETA_SELEC_SIG_wCell2 = BETA_SELEC_SIG_wCell[['pert_id','CID','cellosaurus_id','ccle_name','sig_id']]
BETA_SELEC_SIG_wCell2 = BETA_SELEC_SIG_wCell2.drop_duplicates() # 128038


ccle_tt=[True if type(a)==str else False for a in list(BETA_SELEC_SIG_wCell2.ccle_name)] 
# sum(ccle_tt) : 109720
BETA_CID_CCLE_SIG = BETA_SELEC_SIG_wCell2[ccle_tt][['pert_id','CID','ccle_name','sig_id']].drop_duplicates() # 111012
beta_cid_ccle_sig_tf = [ True if a>0 else False for a in list(BETA_CID_CCLE_SIG.CID)]
BETA_CID_CCLE_SIG = BETA_CID_CCLE_SIG[beta_cid_ccle_sig_tf]
BETA_CID_CCLE_SIG['CID'] = [int(a) for a in list(BETA_CID_CCLE_SIG['CID']) ] # 111012 
# 늘어남! 

# -> CCLE 필터까지 완료 / DC CID 고정 완료 







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


BETA_CID_CCLE_SIG_ID_CHECK = list(BETA_CID_CCLE_SIG.CID)
BETA_CID_CCLE_SIG_CELL_CHECK = list(BETA_CID_CCLE_SIG.ccle_name)

BETA_CID_CCLE_SIG['IDCHECK'] = [str(int(BETA_CID_CCLE_SIG_ID_CHECK[i]))+'__'+BETA_CID_CCLE_SIG_CELL_CHECK[i] for i in range(BETA_CID_CCLE_SIG.shape[0])]
# 이렇게 되면 , IDCHECK 에 중복이 생기긴 함. pert 때문에. 


BETA_CID_CCLE_SIG.columns=['ROW_pert_id', 'drug_row_CID', 'DrugCombCCLE', 'ROW_BETA_sig_id',  'ROWCHECK']
CCLE_DC_BETA_1 = pd.merge(DC_ccle_final_dup, BETA_CID_CCLE_SIG[['ROW_pert_id', 'ROW_BETA_sig_id',  'ROWCHECK']], left_on = 'ROWCHECK', right_on = 'ROWCHECK', how = 'left') # 720619

BETA_CID_CCLE_SIG.columns=['COL_pert_id', 'drug_col_CID', 'DrugCombCCLE', 'COL_BETA_sig_id', 'COLCHECK']
CCLE_DC_BETA_2 = pd.merge(CCLE_DC_BETA_1, BETA_CID_CCLE_SIG[['COL_pert_id', 'COL_BETA_sig_id', 'COLCHECK']], left_on = 'COLCHECK', right_on = 'COLCHECK', how = 'left') # 721206

BETA_CID_CCLE_SIG.columns=['pert_id', 'pubchem_cid', 'ccle_name', 'sig_id', 'IDCHECK']








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
DATA_AX_BO['type'] = 'AXBO' # 2352
DATA_AO_BX['type'] = 'AOBX' # 1856
DATA_AX_BX['type'] = 'AXBX' # 75932




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
#########################################################################
#########################################################################




# Graph 확인 

JY_GRAPH = MY_G
JY_GRAPH_ORDER = MY_G.nodes()
JY_ADJ = nx.adjacency_matrix(JY_GRAPH)

JY_ADJ_tmp = torch.LongTensor(JY_ADJ.toarray())
JY_ADJ_IDX = JY_ADJ_tmp.to_sparse().indices()
JY_IDX_WEIGHT = MY_WEIGHT_SCORE


(3) two missing : MISS_2
A_B_C_S = pd.concat([DATA_AO_BO, DATA_AX_BO, DATA_AO_BX, DATA_AX_BX])
A_B_C_S = A_B_C_S.reset_index(drop = True) # 622809
A_B_C_S[['drug_row_CID','drug_col_CID', 'DrugCombCCLE']].drop_duplicates()

